// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Configurable status display. Collection runs outside drawing and accounting survives view changes.

pub mod collect;
pub mod config;
mod render;

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use config::{Config, Layout};
pub use render::{rows, Row};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Tokens {
    pub input: Option<u64>,
    pub output: Option<u64>,
    pub cache_read: Option<u64>,
    pub cache_write: Option<u64>,
    pub reasoning: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Latest {
    pub model: String,
    pub responded_at: Option<f64>,
    pub duration: Option<f64>,
    pub speed: Option<f64>,
    pub tokens: Tokens,
    pub cache_expires_at: Option<f64>,
    pub limits: Vec<Limit>,
    pub limits_updated_at: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Environment {
    pub skills: Vec<String>,
    pub available_skills: usize,
    pub mcp_servers: Vec<String>,
    pub provider: String,
    pub effort: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Daily {
    pub cost: f64,
    pub complete: bool,
    pub calls: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priced_calls: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Telemetry {
    pub stream_id: String,
    pub sequence: u64,
    pub started_at: Option<f64>,
    pub calls: u64,
    pub restored_calls: u64,
    pub failed_calls: u64,
    pub tokens: Tokens,
    pub days: BTreeMap<String, Daily>,
    pub latest: Latest,
    pub environment: Environment,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Limit {
    pub label: String,
    pub percent: f64,
    pub resets_at: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct Host {
    pub git: Option<crate::workspace_changes::GitSummary>,
    pub memory: Option<(u64, u64)>,
    pub limits: Vec<Limit>,
    pub cache_expires_at: Option<f64>,
    pub daily_cost: Option<f64>,
    pub daily_cost_partial: bool,
    pub usage_updated_at: Option<f64>,
    pub weekly_cost: Option<f64>,
    pub weekly_cost_partial: bool,
    pub notices: Vec<String>,
    pub instruction_files: Vec<String>,
}

#[derive(Default)]
pub struct Hud {
    pub config: Config,
    pub telemetry: Telemetry,
    pub host: Host,
    pub problem: Option<String>,
    pub config_path: Option<PathBuf>,
    pub ledger_path: Option<PathBuf>,
    previous: Option<Config>,
    pending: VecDeque<Telemetry>,
    last_poll: Option<Instant>,
    pub generation: u64,
}

impl Hud {
    pub fn attach(&mut self, workspace: &Path) {
        let local = workspace.join(".aster/tui-hud.json");
        let mut paths = Vec::new();
        if let Some(user) = config::user_directory() {
            paths.push(user.join("tui-hud.json"));
            self.ledger_path = Some(user.join("hud-usage"));
        }
        paths.push(local.clone());
        match config::load(&paths) {
            Ok(config) => self.config = config,
            Err(error) => self.problem = Some(format!("{error:#}")),
        }
        let previous = local.with_extension("previous.json");
        self.previous = previous
            .exists()
            .then(|| config::load(&[previous]).ok())
            .flatten();
        self.config_path = Some(local);
    }

    /// Accounting already queued for disk is independent of the visible conversation.
    pub fn reset_session(&mut self) {
        self.telemetry = Telemetry::default();
    }

    pub async fn flush(&mut self) -> anyhow::Result<()> {
        let records = self.pending.iter().cloned().collect();
        let ledger = self.ledger_path.clone();
        let (saved, errors) =
            tokio::task::spawn_blocking(move || collect::save_records(ledger.as_deref(), records))
                .await?;
        self.acknowledge(&saved);
        if !errors.is_empty() {
            anyhow::bail!("{}", errors.join("\n"));
        }
        Ok(())
    }

    pub fn configure(&mut self, command: &str) -> anyhow::Result<()> {
        let next = if command == "undo" {
            self.previous
                .clone()
                .ok_or_else(|| anyhow::anyhow!("No previous HUD configuration"))?
        } else if command == "reload" {
            let local = self
                .config_path
                .clone()
                .ok_or_else(|| anyhow::anyhow!("No HUD settings path"))?;
            let mut paths: Vec<_> = config::user_directory()
                .map(|root| root.join("tui-hud.json"))
                .into_iter()
                .collect();
            paths.push(local);
            config::load(&paths)?
        } else {
            self.config.changed(command)?
        };
        if let Some(path) = self.config_path.as_ref().filter(|_| command != "reload") {
            config::save(&path.with_extension("previous.json"), &self.config)?;
            config::save(path, &next)?;
        }
        self.previous = Some(std::mem::replace(&mut self.config, next));
        self.problem = None;
        self.last_poll = None;
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    pub fn ingest(&mut self, value: &Value) {
        let Some(metrics) = Self::decode(value) else {
            return;
        };
        if self.telemetry.stream_id == metrics.stream_id
            && self.telemetry.sequence >= metrics.sequence
        {
            return;
        }
        self.queue(metrics.clone());
        self.telemetry = metrics;
    }

    /// Source-session activity is billable during replay and until its runtime finishes closing.
    pub fn account(&mut self, value: &Value) {
        if let Some(metrics) = Self::decode(value) {
            self.queue(metrics);
        }
    }

    fn decode(value: &Value) -> Option<Telemetry> {
        let metrics = serde_json::from_value::<Telemetry>(value.clone()).ok()?;
        if metrics.stream_id.len() != 32
            || !metrics.stream_id.bytes().all(|c| c.is_ascii_hexdigit())
            || metrics.days.len() > 366
            || metrics.environment.skills.len() > 100
            || metrics.environment.mcp_servers.len() > 100
            || metrics.latest.limits.len() > 8
            || metrics
                .days
                .values()
                .any(|day| !day.cost.is_finite() || day.cost < 0.0)
        {
            return None;
        }
        Some(metrics)
    }

    fn queue(&mut self, metrics: Telemetry) {
        if let Some(previous) = self
            .pending
            .iter_mut()
            .find(|item| item.stream_id == metrics.stream_id)
        {
            if previous.sequence < metrics.sequence {
                *previous = metrics;
            }
        } else {
            self.pending.push_back(metrics);
        }
    }

    /// One request owns a copy; new events can arrive while its filesystem work is running.
    pub fn collect_request(&mut self, workspace: &str, force: bool) -> Option<collect::Request> {
        let interval = if self.pending.is_empty() {
            self.config.refresh_seconds
        } else {
            1
        };
        if !force
            && self
                .last_poll
                .is_some_and(|last| last.elapsed() < Duration::from_secs(interval))
        {
            return None;
        }
        self.last_poll = Some(Instant::now());
        Some(collect::Request {
            workspace: PathBuf::from(workspace),
            config: self.config.clone(),
            ledger: self.ledger_path.clone(),
            records: self.pending.iter().cloned().collect(),
            generation: self.generation,
        })
    }

    pub fn collected(&mut self, result: collect::Collected) {
        self.acknowledge(&result.saved);
        if self.generation == result.generation {
            self.host = result.host;
        }
    }

    fn acknowledge(&mut self, saved: &BTreeMap<String, u64>) {
        self.pending.retain(|item| {
            saved
                .get(&item.stream_id)
                .is_none_or(|sequence| *sequence < item.sequence)
        });
    }

    pub fn choices(&self) -> Vec<crate::state::Choice> {
        let mut options = vec![
            (
                "Live metrics and data sources".to_owned(),
                "Inspect tokens, cost, cache, provider facts and collection errors".to_owned(),
                "info".to_owned(),
            ),
            (
                "Expanded layout".to_owned(),
                "Model, workspace and usage with activity rows".to_owned(),
                "expanded".to_owned(),
            ),
            (
                "Full details".to_owned(),
                "Show every metric; includes Git details and recorded day/week cost".to_owned(),
                "full".to_owned(),
            ),
            (
                "Compact layout".to_owned(),
                "Combine status on wide terminals; keep important values when narrow".to_owned(),
                "compact".to_owned(),
            ),
            (
                "Hide HUD".to_owned(),
                "Use /hud to show it again".to_owned(),
                "off".to_owned(),
            ),
        ];
        for element in config::ELEMENTS {
            let shown = self.config.elements.iter().any(|item| item == element);
            options.push((
                format!("{} {element}", if shown { "[✓]" } else { "[ ]" }),
                "Enter toggles this row; /hud set elements [...] changes its order".into(),
                format!("{} {element}", if shown { "hide" } else { "show" }),
            ));
        }
        options.extend([
            (
                "Last-request token and cache counts".into(),
                "Use /hud set token_scope session for conversation totals".into(),
                "set token_scope request".into(),
            ),
            (
                "Show full workspace path".into(),
                "Default: folder name; /hud set path_levels 1 restores it".into(),
                "set path_levels 0".into(),
            ),
            (
                "Restore defaults".into(),
                "Replace HUD settings with the defaults".into(),
                "reset".into(),
            ),
            (
                "Reload settings files".into(),
                "Apply manual edits to global and workspace JSON".into(),
                "reload".into(),
            ),
        ]);
        if self.previous.is_some() {
            options.push((
                "Undo settings change".into(),
                "Restore the previous saved configuration".into(),
                "undo".into(),
            ));
        }
        options
            .into_iter()
            .map(|(label, detail, value)| crate::state::Choice {
                label,
                detail,
                value,
            })
            .collect()
    }

    pub fn diagnostics(&self) -> String {
        format!("HUD settings: {}\n{}\n\nSession usage (including restored model responses):\n{}\n\nGit: {:?}\nHost memory (used/total bytes): {:?}\nRecorded cost today: {:?} USD{} · this week: {:?} USD{}\nExternal quota windows: {:?}\nUsage ledger: {}\nPending usage records: {}\n{}\n{}\n\nMissing provider values are unavailable, including cache expiry and account quotas.\nDay/week costs include usage recorded by this TUI, across workspaces.\n≥ marks a known subtotal with unpriced calls.\n\n/hud set KEY VALUE · /hud show ELEMENT · /hud hide ELEMENT\n/hud compact · /hud expanded · /hud full · /hud reset · /hud undo",
            self.config_path.as_ref().map_or("in-memory".into(), |path| path.display().to_string()),
            serde_json::to_string_pretty(&self.config).unwrap_or_default(),
            serde_json::to_string_pretty(&self.telemetry).unwrap_or_default(),
            self.host.git, self.host.memory, self.host.daily_cost,
            if self.host.daily_cost_partial { " (partial)" } else { "" },
            self.host.weekly_cost,
            if self.host.weekly_cost_partial { " (partial)" } else { "" },
            self.host.limits, self.ledger_path.as_ref().map_or("disabled".into(), |path| path.display().to_string()), self.pending.len(),
            self.problem.as_deref().unwrap_or("Settings valid"), self.host.notices.join("\n"))
    }
}
