// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Compact,
    #[default]
    Expanded,
    Hidden,
}

pub const ELEMENTS: &[&str] = &[
    "project", "usage", "tokens", "cache", "cost", "tools", "tasks", "agents", "skills", "memory",
    "limits", "session", "custom",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub layout: Layout,
    pub elements: Vec<String>,
    /// Zero shows the full path; otherwise retain this many trailing components.
    pub path_levels: usize,
    pub max_rows: usize,
    pub max_width: usize,
    pub context_value: String,
    pub context_bar: bool,
    pub token_scope: String,
    pub warning_percent: u8,
    pub critical_percent: u8,
    pub git: bool,
    pub git_details: bool,
    pub refresh_seconds: u64,
    pub activity_limit: usize,
    pub daily_cost: bool,
    pub weekly_cost: bool,
    pub model_format: String,
    pub model_override: String,
    pub provider: String,
    pub custom_text: String,
    pub external_usage_path: String,
    pub external_usage_ttl_seconds: u64,
    pub language: String,
    /// Colors are #rrggbb, ANSI indices (0..255), or reset for the terminal default.
    pub colors: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            layout: Layout::Expanded,
            elements: ["project", "usage", "tools", "tasks", "agents"]
                .map(str::to_owned)
                .into(),
            path_levels: 1,
            max_rows: 8,
            max_width: 160,
            context_value: "percent".into(),
            context_bar: true,
            token_scope: "session".into(),
            warning_percent: 70,
            critical_percent: 85,
            git: true,
            git_details: false,
            refresh_seconds: 5,
            activity_limit: 3,
            daily_cost: false,
            weekly_cost: false,
            model_format: "full".into(),
            model_override: String::new(),
            provider: String::new(),
            custom_text: String::new(),
            external_usage_path: String::new(),
            external_usage_ttl_seconds: 300,
            language: "en".into(),
            colors: BTreeMap::new(),
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.path_levels > 8
            || !(1..=16).contains(&self.max_rows)
            || !(20..=400).contains(&self.max_width)
            || !(2..=120).contains(&self.refresh_seconds)
            || !(1..=8).contains(&self.activity_limit)
            || !(1..=86400).contains(&self.external_usage_ttl_seconds)
        {
            bail!("HUD limits: path_levels 0–8, max_rows 1–16, max_width 20–400, refresh_seconds 2–120, activity_limit 1–8, usage TTL 1–86400");
        }
        if self.warning_percent >= self.critical_percent || self.critical_percent > 100 {
            bail!("HUD thresholds must satisfy 0 <= warning_percent < critical_percent <= 100");
        }
        if !["percent", "tokens", "remaining", "both"].contains(&self.context_value.as_str())
            || !["full", "short"].contains(&self.model_format.as_str())
            || !["session", "request"].contains(&self.token_scope.as_str())
            || !["en", "zh"].contains(&self.language.as_str())
        {
            bail!("Invalid HUD context_value, token_scope, model_format or language");
        }
        let mut seen = std::collections::HashSet::new();
        if self
            .elements
            .iter()
            .any(|key| !ELEMENTS.contains(&key.as_str()) || !seen.insert(key))
        {
            bail!(
                "HUD elements must be unique names from {}",
                ELEMENTS.join(", ")
            );
        }
        if self.model_override.len() > 256
            || self.provider.len() > 128
            || self.custom_text.len() > 512
            || self.external_usage_path.len() > 4096
        {
            bail!("HUD text setting is too long");
        }
        for (key, color) in &self.colors {
            if ![
                "model", "project", "text", "muted", "context", "warning", "critical",
            ]
            .contains(&key.as_str())
                || !(color == "reset"
                    || color.parse::<u8>().is_ok()
                    || color.len() == 7
                        && color.starts_with('#')
                        && color[1..].bytes().all(|c| c.is_ascii_hexdigit()))
            {
                bail!("Invalid HUD color {key}; use #rrggbb, 0–255, or reset");
            }
        }
        Ok(())
    }

    pub fn changed(&self, command: &str) -> Result<Self> {
        let mut next = self.clone();
        let (verb, args) = command
            .trim()
            .split_once(char::is_whitespace)
            .unwrap_or((command.trim(), ""));
        let args = args.trim();
        if matches!(
            verb,
            "compact" | "expanded" | "on" | "full" | "off" | "reset"
        ) && !args.is_empty()
        {
            bail!("HUD {verb} takes no arguments");
        }
        match verb {
            "compact" => next.layout = Layout::Compact,
            "expanded" | "on" => next.layout = Layout::Expanded,
            "full" => {
                next.layout = Layout::Expanded;
                next.elements = ELEMENTS
                    .iter()
                    .map(|element| (*element).to_owned())
                    .collect();
                next.max_rows = 16;
                next.daily_cost = true;
                next.weekly_cost = true;
                next.git_details = true;
            }
            "off" => next.layout = Layout::Hidden,
            "reset" => next = Self::default(),
            "show" | "hide" => {
                if !ELEMENTS.contains(&args.trim()) {
                    bail!("Unknown HUD element: {}", args.trim());
                }
                if verb == "hide" {
                    next.elements.retain(|key| key != args);
                } else if !next.elements.iter().any(|key| key == args) {
                    next.elements.push(args.to_owned());
                }
            }
            "set" => {
                let (key, raw) = args
                    .split_once(char::is_whitespace)
                    .context("Use /hud set KEY VALUE")?;
                let raw = raw.trim();
                let value: Value =
                    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()));
                let mut object = serde_json::to_value(&next)?;
                if !object.as_object().is_some_and(|map| map.contains_key(key)) {
                    bail!("Unknown HUD setting: {key}");
                }
                object[key] = value;
                next = serde_json::from_value(object).context("Invalid HUD setting value")?;
            }
            _ => bail!(
                "Use /hud compact|expanded|off|show ELEMENT|hide ELEMENT|set KEY VALUE|reset|undo"
            ),
        }
        next.validate()?;
        Ok(next)
    }
}

pub fn user_directory() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ASTER_USER_DIR").filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
                .map(|path| PathBuf::from(path).join(".config"))
        })
        .map(|root| root.join("aster"))
}

fn merge(base: &mut Value, patch: Value) {
    if let (Some(base), Value::Object(patch)) = (base.as_object_mut(), &patch) {
        for (key, value) in patch {
            let entry = base.entry(key).or_insert(Value::Null);
            if value.as_object().is_some_and(|object| object.is_empty()) {
                // An explicit empty map clears inherited values, including reset colors.
                *entry = value.clone();
            } else {
                merge(entry, value.clone());
            }
        }
    } else {
        *base = patch;
    }
}

pub fn load(paths: &[PathBuf]) -> Result<Config> {
    let mut value = serde_json::to_value(Config::default())?;
    for path in paths {
        let bytes = match crate::file_io::read_regular(path, 64 * 1024) {
            Ok(bytes) => bytes,
            Err(crate::file_io::ReadError::Inspect(error))
                if error.kind() == std::io::ErrorKind::NotFound =>
            {
                continue
            }
            Err(error) => bail!("Cannot read HUD settings {}: {error:?}", path.display()),
        };
        merge(
            &mut value,
            serde_json::from_slice(&bytes).context("Invalid HUD settings JSON")?,
        );
    }
    let config: Config = serde_json::from_value(value)?;
    config.validate()?;
    Ok(config)
}

pub fn save(path: &Path, config: &Config) -> Result<()> {
    config.validate()?;
    let parent = path.parent().context("HUD settings path has no parent")?;
    fs::create_dir_all(parent)?;
    if let Ok(meta) = fs::symlink_metadata(path) {
        if !meta.is_file() || crate::file_io::is_link(&meta) {
            bail!("HUD settings must be a regular file");
        }
    }
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temp, config)?;
    temp.write_all(b"\n")?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .map_err(|error| error.error)
        .context("Could not save HUD settings")?;
    Ok(())
}
