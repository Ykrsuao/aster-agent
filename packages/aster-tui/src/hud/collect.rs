// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Bounded host reads and atomic per-stream ledgers; never called by the renderer.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, Local, NaiveDate};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Config, Daily, Host, Limit, Telemetry};

pub struct Request {
    pub workspace: PathBuf,
    pub config: Config,
    pub ledger: Option<PathBuf>,
    pub records: Vec<Telemetry>,
    pub generation: u64,
}

pub struct Collected {
    pub host: Host,
    pub generation: u64,
    pub saved: BTreeMap<String, u64>,
}

#[derive(Serialize, Deserialize)]
struct LedgerRecord {
    sequence: u64,
    daily: Daily,
}

fn record(root: &Path, metrics: &Telemetry) -> Result<()> {
    // Streams have unique file names across processes. Serialize same-process
    // writes too: a cancelled collector's blocking write may outlive its task.
    static WRITES: Mutex<()> = Mutex::new(());
    let _guard = WRITES
        .lock()
        .map_err(|_| anyhow::anyhow!("Usage ledger writer stopped"))?;
    fs::create_dir_all(root)?;
    for (day, daily) in &metrics.days {
        NaiveDate::parse_from_str(day, "%Y-%m-%d").context("Invalid usage date")?;
        if day.len() != 10 {
            bail!("Invalid usage date");
        }
        let path = root.join(format!("{day}_{}.json", metrics.stream_id));
        if let Ok(bytes) = crate::file_io::read_regular(&path, 4096) {
            let previous: LedgerRecord = serde_json::from_slice(&bytes)?;
            if previous.sequence >= metrics.sequence {
                continue;
            }
        } else if path.exists() {
            bail!("Cannot replace an unreadable usage ledger");
        }
        let mut temp = tempfile::NamedTempFile::new_in(root)?;
        serde_json::to_writer(
            &mut temp,
            &LedgerRecord {
                sequence: metrics.sequence,
                daily: daily.clone(),
            },
        )?;
        temp.flush()?;
        temp.as_file().sync_all()?;
        temp.persist(&path).map_err(|error| error.error)?;
    }
    Ok(())
}

pub(super) fn save_records(
    root: Option<&Path>,
    records: Vec<Telemetry>,
) -> (BTreeMap<String, u64>, Vec<String>) {
    let mut saved = BTreeMap::new();
    let mut errors = Vec::new();
    for metrics in records {
        match root.map_or(Ok(()), |root| record(root, &metrics)) {
            Ok(()) => {
                saved.insert(metrics.stream_id, metrics.sequence);
            }
            Err(error) => errors.push(format!("Usage was not saved: {error:#}")),
        }
    }
    (saved, errors)
}

#[derive(Default)]
struct CostTotal {
    value: f64,
    partial: bool,
    known: bool,
}

impl CostTotal {
    fn add(&mut self, daily: &Daily) {
        if daily.cost.is_finite() && daily.cost >= 0.0 {
            self.value += daily.cost;
            self.known |= daily.complete || daily.priced_calls.unwrap_or(0) > 0 || daily.cost > 0.0;
        } else {
            self.partial = true;
        }
        self.partial |= !daily.complete;
    }

    fn amount(&self) -> Option<f64> {
        (self.value.is_finite() && (!self.partial || self.known)).then_some(self.value)
    }
}

fn costs(root: &Path, today: NaiveDate) -> Result<(CostTotal, CostTotal)> {
    let week = today - chrono::Duration::days(i64::from(today.weekday().num_days_from_monday()));
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((CostTotal::default(), CostTotal::default()))
        }
        Err(error) => return Err(error.into()),
    };
    let (mut day_cost, mut week_cost) = (CostTotal::default(), CostTotal::default());
    for (index, entry) in entries.enumerate() {
        if index >= 100_000 {
            bail!("Usage ledger directory exceeds the read limit");
        }
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.ends_with(".json") {
            continue;
        }
        let Some((date, _)) = name.split_once('_') else {
            continue;
        };
        let Ok(date) = NaiveDate::parse_from_str(date, "%Y-%m-%d") else {
            continue;
        };
        if date < week || date > today {
            continue;
        }
        let bytes = crate::file_io::read_regular(&entry.path(), 4096)
            .map_err(|error| anyhow::anyhow!("Cannot read usage ledger: {error:?}"))?;
        let record: LedgerRecord = serde_json::from_slice(&bytes)?;
        week_cost.add(&record.daily);
        if date == today {
            day_cost.add(&record.daily);
        }
    }
    Ok((day_cost, week_cost))
}

fn epoch(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .or_else(|| {
            value
                .as_str()
                .and_then(|text| chrono::DateTime::parse_from_rfc3339(text).ok())
                .map(|time| time.timestamp() as f64)
        })
}

fn external(path: &Path, ttl: u64, now: f64) -> Result<(Vec<Limit>, Option<f64>, f64)> {
    let bytes = crate::file_io::read_regular(path, 64 * 1024)
        .map_err(|error| anyhow::anyhow!("Cannot read usage snapshot: {error:?}"))?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let updated = epoch(&value["updated_at"]).context("Usage snapshot requires updated_at")?;
    if updated > now + 60.0 || now - updated > ttl as f64 {
        bail!("Usage snapshot is stale");
    }
    let mut limits = Vec::new();
    for (key, label) in [
        ("five_hour", "5h"),
        ("seven_day", "7d"),
        ("spend_limit", "spend"),
    ] {
        if let Some(percent) = value[key]["used_percentage"]
            .as_f64()
            .filter(|value| value.is_finite() && (0.0..=100.0).contains(value))
        {
            limits.push(Limit {
                label: label.into(),
                percent,
                resets_at: epoch(&value[key]["resets_at"]),
            });
        }
    }
    if let Some(windows) = value["windows"].as_array() {
        for window in windows.iter().take(8 - limits.len()) {
            if let Some(percent) = window["percent"]
                .as_f64()
                .filter(|value| value.is_finite() && (0.0..=100.0).contains(value))
            {
                limits.push(Limit {
                    label: crate::state::clean(window["label"].as_str().unwrap_or("usage"))
                        .chars()
                        .take(32)
                        .collect(),
                    percent,
                    resets_at: epoch(&window["resets_at"]),
                });
            }
        }
    }
    limits.retain(|limit| limit.resets_at.is_none_or(|reset| reset > now));
    Ok((limits, epoch(&value["cache_expires_at"]), updated))
}

fn memory() -> Option<(u64, u64)> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: the initialized structure supplies the ABI size expected by Windows.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 {
            return Some((
                status.ullTotalPhys.saturating_sub(status.ullAvailPhys),
                status.ullTotalPhys,
            ));
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        let text = fs::read_to_string("/proc/meminfo").ok()?;
        let read = |name: &str| {
            text.lines().find_map(|line| {
                line.strip_prefix(name)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
        };
        let total = read("MemTotal:")?.checked_mul(1024)?;
        let available = read("MemAvailable:")?.checked_mul(1024)?;
        Some((total.saturating_sub(available), total))
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

fn instruction_files(workspace: &Path) -> Vec<String> {
    let ancestors: Vec<_> = workspace.ancestors().take(64).collect();
    let directories = ancestors
        .iter()
        .position(|path| path.join(".git").exists())
        .map_or(&ancestors[..1], |index| &ancestors[..=index]);
    directories
        .iter()
        .rev()
        .filter_map(|directory| {
            let path = directory.join("AGENTS.md");
            path.is_file().then(|| path.display().to_string())
        })
        .collect()
}

pub async fn sample(request: Request) -> Collected {
    let git = if request.config.git && request.config.layout != super::Layout::Hidden {
        crate::workspace_changes::summary(&request.workspace, request.config.git_details).await
    } else {
        Ok(None)
    };
    let generation = request.generation;
    let result = tokio::task::spawn_blocking(move || {
        let mut host = Host::default();
        if request
            .config
            .elements
            .iter()
            .any(|element| element == "skills")
        {
            host.instruction_files = instruction_files(&request.workspace);
        }
        match git {
            Ok(git) => host.git = git,
            Err(error) => host.notices.push(format!("Git: {error:#}")),
        }
        if request
            .config
            .elements
            .iter()
            .any(|element| element == "memory")
        {
            host.memory = memory();
        }
        let (saved, errors) = save_records(request.ledger.as_deref(), request.records);
        host.notices.extend(errors);
        if let Some(root) = request.ledger {
            if request.config.daily_cost
                || request.config.weekly_cost
                || request
                    .config
                    .elements
                    .iter()
                    .any(|element| element == "cost")
            {
                match costs(&root, Local::now().date_naive()) {
                    Ok((daily, weekly)) => {
                        host.daily_cost = daily.amount();
                        host.daily_cost_partial = daily.partial;
                        host.weekly_cost = weekly.amount();
                        host.weekly_cost_partial = weekly.partial;
                    }
                    Err(error) => host
                        .notices
                        .push(format!("Usage totals unavailable: {error:#}")),
                }
            }
        }
        if !request.config.external_usage_path.is_empty() {
            let path = request.workspace.join(&request.config.external_usage_path);
            match external(
                &path,
                request.config.external_usage_ttl_seconds,
                Local::now().timestamp() as f64,
            ) {
                Ok((limits, expiry, updated)) => {
                    host.limits = limits;
                    host.cache_expires_at = expiry;
                    host.usage_updated_at = Some(updated);
                }
                Err(error) => host.notices.push(format!("External usage: {error:#}")),
            }
        }
        Collected {
            host,
            generation,
            saved,
        }
    })
    .await;
    result.unwrap_or_else(|error| Collected {
        host: Host {
            notices: vec![format!("HUD collector stopped: {error}")],
            ..Host::default()
        },
        generation,
        saved: BTreeMap::new(),
    })
}
