// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Pure, width-bounded status rows, with real data shared by the screen and previews.

use chrono::{Local, TimeZone};
use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::{Config, Layout};
use crate::state::{clean, Entry, State};

pub struct Row {
    pub line: Line<'static>,
    pub command: &'static str,
    pub element: &'static str,
}

struct Theme<'a> {
    config: &'a Config,
    monochrome: bool,
    columns: usize,
}

impl Theme<'_> {
    fn style(&self, role: &str) -> Style {
        if self.monochrome {
            return Style::default();
        }
        let default = match role {
            "model" => Color::Rgb(177, 185, 249),
            "project" => Color::Rgb(190, 190, 190),
            "text" => Color::Rgb(225, 225, 225),
            "context" => Color::Rgb(147, 186, 133),
            "warning" => Color::Rgb(220, 186, 115),
            "critical" => Color::Rgb(224, 130, 130),
            _ => Color::Rgb(128, 128, 128),
        };
        let color = self
            .config
            .colors
            .get(role)
            .and_then(|text| {
                if text == "reset" {
                    Some(Color::Reset)
                } else if let Ok(index) = text.parse::<u8>() {
                    Some(Color::Indexed(index))
                } else {
                    u32::from_str_radix(text.strip_prefix('#')?, 16)
                        .ok()
                        .map(|rgb| Color::Rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8))
                }
            })
            .unwrap_or(default);
        Style::default().fg(color)
    }
    fn usage(&self, ratio: f64) -> Style {
        self.style(
            if ratio * 100.0 >= f64::from(self.config.critical_percent) {
                "critical"
            } else if ratio * 100.0 >= f64::from(self.config.warning_percent) {
                "warning"
            } else {
                "context"
            },
        )
    }
    fn label<'a>(&self, en: &'a str, zh: &'a str) -> &'a str {
        if self.config.language == "zh" {
            zh
        } else {
            en
        }
    }

    fn token_label(&self) -> &'static str {
        if self.config.token_scope == "request" {
            self.label("Tokens (last)", "Token（本次）")
        } else {
            self.label("Tokens", "Token")
        }
    }

    fn label_width(&self) -> usize {
        if self.config.layout != Layout::Expanded || self.columns < 60 {
            return 0;
        }
        let width = self.label("Reasoning", "思考强度").width();
        if self.config.elements.iter().any(|name| name == "tokens") {
            width.max(self.token_label().width())
        } else {
            width
        }
    }

    fn metric_label(&self, label: &str) -> Span<'static> {
        let padding = self.label_width().saturating_sub(label.width()) + 2;
        Span::styled(
            format!("{label}{}", " ".repeat(padding)),
            self.style("muted"),
        )
    }

    fn value(&self, value: impl Into<String>) -> Span<'static> {
        let value = value.into();
        let style = self.style(if matches!(value.as_str(), "n/a" | "—") {
            "muted"
        } else {
            "text"
        });
        Span::styled(value, style)
    }

    fn separate(&self, spans: &mut Vec<Span<'static>>) {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", self.style("muted")));
        }
    }

    fn field(&self, spans: &mut Vec<Span<'static>>, label: &str, value: impl Into<String>) {
        self.separate(spans);
        spans.push(Span::styled(format!("{label} "), self.style("muted")));
        spans.push(self.value(value));
    }
}

fn fit(text: &str, columns: usize) -> String {
    let text = clean(text).replace(['\n', '\r', '\t'], " ");
    if text.width() <= columns {
        return text;
    }
    if columns == 0 {
        return String::new();
    }
    let mut result = String::new();
    for glyph in text.graphemes(true) {
        if result.width() + glyph.width() >= columns {
            break;
        }
        result.push_str(glyph);
    }
    result.push('…');
    result
}

fn clip(line: Line<'static>, columns: usize) -> Line<'static> {
    let style = line.style;
    let mut remaining = columns;
    let mut spans = Vec::new();
    for span in line.spans {
        if remaining == 0 {
            break;
        }
        let text = fit(&span.content, remaining);
        remaining = remaining.saturating_sub(text.width());
        spans.push(Span::styled(text, span.style));
    }
    Line::from(spans).style(style)
}

fn metric_row(
    element: &'static str,
    command: &'static str,
    label: &str,
    details: Vec<Span<'static>>,
    theme: &Theme<'_>,
) -> Row {
    let mut spans = vec![theme.metric_label(label)];
    spans.extend(details);
    Row {
        element,
        command,
        line: Line::from(spans),
    }
}

fn count(value: Option<u64>) -> String {
    value.map_or("n/a".into(), |value| {
        if value >= 1_000_000 {
            format!("{:.1}m", value as f64 / 1_000_000.0)
        } else if value >= 10_000 {
            format!("{:.1}k", value as f64 / 1000.0)
        } else {
            value.to_string()
        }
    })
}

fn count_k(value: Option<u64>) -> String {
    match value {
        None => "n/a".into(),
        Some(0) => "0k".into(),
        Some(value) => format!("{:.1}k", value as f64 / 1000.0),
    }
}

fn money(value: Option<f64>, partial: bool) -> String {
    value
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map_or("n/a".into(), |value| {
            format!("{}${value:.4}", if partial { "≥" } else { "" })
        })
}

fn elapsed(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    if seconds >= 3600 {
        format!("{}h {}m", seconds / 3600, seconds / 60 % 60)
    } else if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn path_name(path: &str, levels: usize) -> String {
    if levels == 0 {
        return clean(path);
    }
    let path = std::path::Path::new(path);
    let components: Vec<_> = path.components().collect();
    components[components.len().saturating_sub(levels)..]
        .iter()
        .collect::<std::path::PathBuf>()
        .display()
        .to_string()
}

fn project_model(state: &State) -> &str {
    let cfg = &state.hud.config;
    let model = if cfg.model_override.is_empty() {
        &state.model
    } else {
        &cfg.model_override
    };
    if cfg.model_format == "short" {
        model.rsplit(':').next().unwrap_or(model)
    } else {
        model
    }
}

fn project_controls(
    state: &State,
    columns: usize,
    theme: &Theme<'_>,
    inline_effort: bool,
) -> Line<'static> {
    let mut spans = Vec::new();
    if state.runtime_supported {
        spans.push(Span::styled(
            format!(
                " · {}{}",
                if columns >= 60 { "permissions: " } else { "" },
                clean(state.permission_label())
            ),
            theme.style("muted"),
        ));
    }
    if inline_effort {
        if let Some(effort) = reasoning_level(state) {
            spans.push(Span::styled(" · ", theme.style("muted")));
            spans.push(Span::styled(
                clean(effort),
                theme.style("model").add_modifier(Modifier::BOLD),
            ));
        }
    }
    Line::from(spans)
}

fn project(state: &State, columns: usize, theme: &Theme<'_>, inline_effort: bool) -> Row {
    let cfg = &state.hud.config;
    let controls = project_controls(state, columns, theme, inline_effort);
    let model = fit(project_model(state), columns.saturating_sub(controls.width() + 2));
    let mut spans = vec![
        Span::styled("[", theme.style("muted")),
        Span::styled(model, theme.style("model").add_modifier(Modifier::BOLD)),
        Span::styled("]", theme.style("muted")),
    ];
    spans.extend(controls.spans);
    let mut details = vec![Span::styled(
        clean(&path_name(&state.workspace, cfg.path_levels)),
        theme.style("project"),
    )];
    if let Some(git) = state.hud.host.git.as_ref().filter(|_| cfg.git) {
        let mut label = format!(
            "  {}{}",
            clean(&git.branch),
            if git.dirty { "*" } else { "" }
        );
        if git.ahead > 0 {
            label.push_str(&format!(" ↑{}", git.ahead));
        }
        if git.behind > 0 {
            label.push_str(&format!(" ↓{}", git.behind));
        }
        if git.worktree {
            label.push_str(" · worktree");
        }
        if cfg.git_details {
            label.push_str(&format!(
                " · ~{} +{} -{} ?{} (+{} -{})",
                git.modified,
                git.added,
                git.deleted,
                git.untracked,
                count(git.line_counts_available.then_some(git.lines_added)),
                count(git.line_counts_available.then_some(git.lines_deleted))
            ));
        }
        if git.conflicts > 0 {
            label.push_str(&format!(" · {} conflicts", git.conflicts));
        }
        details.push(Span::styled(
            label,
            theme.style(if git.conflicts > 0 {
                "critical"
            } else {
                "muted"
            }),
        ));
    }
    let provider = if cfg.provider.is_empty() {
        &state.hud.telemetry.environment.provider
    } else {
        &cfg.provider
    };
    if !provider.is_empty() {
        details.push(Span::styled(
            format!(" · {}", clean(provider)),
            theme.style("muted"),
        ));
    }
    if state.sandbox {
        details.push(Span::styled(" · Python sandbox", theme.style("muted")));
    }
    let remaining = columns.saturating_sub(spans.iter().map(Span::width).sum::<usize>());
    if remaining > 3 {
        spans.push(Span::styled(" │ ", theme.style("muted")));
        spans.extend(clip(Line::from(details), remaining - 3).spans);
    }
    Row {
        element: "project",
        command: "/status",
        line: Line::from(spans),
    }
}

fn reasoning_level(state: &State) -> Option<&str> {
    match state.runtime.get("reasoningLevel") {
        Some(level) => level.as_str().filter(|level| !level.trim().is_empty()),
        None => {
            let level = &state.hud.telemetry.environment.effort;
            (!level.trim().is_empty()).then_some(level.as_str())
        }
    }
}

fn usage(state: &State, columns: usize, theme: &Theme<'_>) -> Row {
    let label = theme.label(if columns < 60 { "ctx" } else { "context" }, "上下文");
    let label = if theme.label_width() > 0 {
        theme.metric_label(label)
    } else {
        Span::styled(format!("{label} "), theme.style("muted"))
    };
    let cost = if state.cost.is_empty() {
        "cost —"
    } else {
        &state.cost
    };
    let context_width = columns.saturating_sub(cost.width().min(columns / 2) + 3);
    let mut spans = Vec::new();
    let ratio = state.context_ratio.filter(|ratio| ratio.is_finite());
    let pending_window = state
        .context_window
        .filter(|_| ratio.is_none() && state.usage.is_empty());
    // A known capacity can show an empty track while usage is pending. Keep
    // this placeholder local to rendering; the measured ratio stays unknown.
    if let Some(ratio) = ratio.or(pending_window.map(|_| 0.0)) {
        let snapshot = state.context_snapshot.as_ref();
        let value =
            pending_window
                .map(|window| format!("0k / {}", count(Some(window))))
                .or_else(|| match state.hud.config.context_value.as_str() {
                    "tokens" => snapshot
                        .map(|value| format!("{} / {}", count_k(value.used), count(value.window))),
                    "remaining" => snapshot.and_then(|value| value.used.zip(value.window)).map(
                        |(used, total)| format!("{} left", count(Some(total.saturating_sub(used)))),
                    ),
                    "both" => snapshot.map(|value| {
                        format!(
                            "{:.0}% · {} / {}",
                            ratio * 100.0,
                            count_k(value.used),
                            count(value.window)
                        )
                    }),
                    _ => None,
                })
                .unwrap_or_else(|| format!("{:.0}%", ratio * 100.0));
        let bar_width = if columns >= 100 { 10 } else { 6 };
        let label_width = label.width();
        spans.push(label);
        if state.hud.config.context_bar
            && columns >= 40
            && label_width + bar_width + value.width() < context_width
        {
            let filled = (ratio.clamp(0.0, 1.0) * bar_width as f64).round() as usize;
            spans.push(Span::styled("█".repeat(filled), theme.usage(ratio)));
            spans.push(Span::styled(
                "░".repeat(bar_width - filled),
                theme.style("muted"),
            ));
            spans.push(Span::raw(" "));
        }
        spans.push(Span::styled(
            value,
            if pending_window.is_some() {
                theme.style("muted")
            } else {
                theme.usage(ratio).add_modifier(Modifier::BOLD)
            },
        ));
    } else {
        if state.usage.is_empty() || state.usage == "context n/a" {
            spans.push(label);
            spans.push(theme.value(if state.usage.is_empty() { "—" } else { "n/a" }));
        } else {
            spans.push(Span::styled(
                state
                    .usage
                    .replace(" tokens", if columns < 60 { " tok" } else { " tokens" }),
                theme.style("muted"),
            ));
        }
    }
    let mut line = clip(Line::from(spans), context_width);
    line.spans.push(Span::styled(" │ ", theme.style("muted")));
    if let Some(value) = cost.strip_prefix("cost ") {
        line.spans.push(Span::styled("cost ", theme.style("muted")));
        line.spans.push(theme.value(value));
    } else {
        line.spans.push(theme.value(cost));
    }
    if let Some(speed) = state
        .hud
        .telemetry
        .latest
        .speed
        .filter(|value| value.is_finite() && *value >= 0.0)
    {
        theme.separate(&mut line.spans);
        line.spans.push(theme.value(format!("{speed:.1}")));
        line.spans
            .push(Span::styled(" tok/s", theme.style("muted")));
    }
    Row {
        element: "usage",
        command: "/context",
        line,
    }
}

fn element(state: &State, name: &str, columns: usize, theme: &Theme<'_>, now: f64) -> Option<Row> {
    let hud = &state.hud;
    let metrics = &hud.telemetry;
    let fresh = |updated: Option<f64>| {
        updated.is_some_and(|time| {
            time.is_finite()
                && time <= now + 60.0
                && now - time <= hud.config.external_usage_ttl_seconds as f64
        })
    };
    let tokens = if hud.config.token_scope == "request" {
        &metrics.latest.tokens
    } else {
        &metrics.tokens
    };
    let row = match name {
        "project" => project(state, columns, theme, false),
        "usage" => usage(state, columns, theme),
        "reasoning" => {
            let level = reasoning_level(state);
            metric_row(
                "reasoning",
                "/effort",
                theme.label("Reasoning", "思考强度"),
                vec![Span::styled(
                    clean(level.unwrap_or(theme.label("model default", "模型默认"))),
                    if level.is_some() {
                        theme.style("model").add_modifier(Modifier::BOLD)
                    } else {
                        theme.style("muted")
                    },
                )],
                theme,
            )
        }
        "tokens" => {
            let mut details = Vec::new();
            theme.field(&mut details, "in", count(tokens.input));
            theme.field(&mut details, "out", count(tokens.output));
            theme.field(&mut details, "reasoning", count(tokens.reasoning));
            if hud.config.token_scope != "request" {
                theme.separate(&mut details);
                details.push(theme.value(metrics.calls.to_string()));
                details.push(Span::styled(
                    if metrics.failed_calls > 0 {
                        " completed"
                    } else {
                        " calls"
                    },
                    theme.style("muted"),
                ));
                if metrics.failed_calls > 0 {
                    theme.separate(&mut details);
                    details.push(Span::styled(
                        format!("{} failed", metrics.failed_calls),
                        theme.style("critical"),
                    ));
                }
            }
            metric_row("tokens", "/hud info", theme.token_label(), details, theme)
        }
        "cache" => {
            let hit = tokens
                .cache_read
                .zip(tokens.input)
                .filter(|(cached, input)| *input > 0 && cached <= input)
                .map_or("n/a".into(), |(cached, input)| {
                    format!("{:.1}%", cached as f64 / input as f64 * 100.0)
                });
            let expiry = hud
                .host
                .cache_expires_at
                .filter(|_| {
                    !hud.config.external_usage_path.is_empty() && fresh(hud.host.usage_updated_at)
                })
                .or_else(|| {
                    metrics
                        .latest
                        .cache_expires_at
                        .filter(|_| fresh(metrics.latest.responded_at))
                })
                .filter(|value| value.is_finite() && *value >= 0.0)
                .map(|value| {
                    if value <= now {
                        " · expired".into()
                    } else {
                        Local
                            .timestamp_opt(value as i64, 0)
                            .single()
                            .map_or(String::new(), |date| {
                                format!(" · until {}", date.format("%H:%M"))
                            })
                    }
                })
                .unwrap_or_default();
            let mut details = Vec::new();
            theme.field(&mut details, "read", count(tokens.cache_read));
            theme.field(&mut details, "write", count(tokens.cache_write));
            theme.field(&mut details, "hit", hit);
            details.push(Span::styled(expiry, theme.style("muted")));
            metric_row(
                "cache",
                "/hud info",
                theme.label("Cache", "缓存"),
                details,
                theme,
            )
        }
        "cost" => {
            let mut details = Vec::new();
            theme.field(
                &mut details,
                "session",
                if state.cost.is_empty() {
                    "—"
                } else {
                    state.cost.strip_prefix("cost ").unwrap_or(&state.cost)
                },
            );
            if hud.config.daily_cost {
                theme.field(
                    &mut details,
                    "today",
                    money(hud.host.daily_cost, hud.host.daily_cost_partial),
                );
            }
            if hud.config.weekly_cost {
                theme.field(
                    &mut details,
                    "week",
                    money(hud.host.weekly_cost, hud.host.weekly_cost_partial),
                );
            }
            metric_row(
                "cost",
                "/hud info",
                theme.label("Cost", "费用"),
                details,
                theme,
            )
        }
        "tools" => {
            let tools: Vec<_> = state
                .entries
                .iter()
                .filter_map(|entry| {
                    if let Entry::Tool(tool) = entry {
                        Some(tool)
                    } else {
                        None
                    }
                })
                .collect();
            if tools.is_empty() {
                return None;
            }
            let active: Vec<_> = tools
                .iter()
                .filter(|tool| matches!(tool.status.as_str(), "pending" | "in_progress"))
                .collect();
            let completed = tools
                .iter()
                .filter(|tool| tool.status == "completed")
                .count();
            let failed = tools
                .iter()
                .filter(|tool| matches!(tool.status.as_str(), "failed" | "cancelled"))
                .count();
            let mut details = Vec::new();
            for tool in active.iter().take(hud.config.activity_limit) {
                theme.separate(&mut details);
                details.push(Span::styled("◐ ", theme.style("model")));
                details.push(theme.value(fit(&tool.title, 32)));
                if let Some(seconds) = tool.elapsed_seconds() {
                    details.push(Span::styled(
                        format!(" ({})", elapsed(seconds)),
                        theme.style("muted"),
                    ));
                }
            }
            theme.separate(&mut details);
            details.push(Span::styled("✓ ", theme.style("context")));
            details.push(theme.value(completed.to_string()));
            if failed > 0 {
                theme.separate(&mut details);
                details.push(Span::styled(format!("! {failed}"), theme.style("critical")));
            }
            metric_row(
                "tools",
                "/tools",
                theme.label("Tools", "工具"),
                details,
                theme,
            )
        }
        "tasks" => {
            let steps = state.tasks["steps"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default();
            let waiting = state.tasks["waitReason"]
                .as_str()
                .filter(|reason| !reason.trim().is_empty());
            let current = state.tasks["currentTask"]
                .as_str()
                .filter(|task| !task.trim().is_empty());
            if steps.is_empty() && waiting.is_none() && current.is_none() {
                return None;
            }
            let done = steps
                .iter()
                .filter(|step| step["status"] == "completed")
                .count();
            let blocked = steps
                .iter()
                .filter(|step| {
                    step["blockedBy"]
                        .as_array()
                        .is_some_and(|items| !items.is_empty())
                })
                .count();
            let mut details = Vec::new();
            if !steps.is_empty() {
                details.push(theme.value(format!("{done}/{}", steps.len())));
            }
            if let Some(reason) = waiting {
                theme.separate(&mut details);
                details.push(Span::styled(clean(reason), theme.style("warning")));
            }
            if let Some(current) = current {
                theme.separate(&mut details);
                details.push(theme.value(clean(current)));
            } else if waiting.is_none() {
                theme.separate(&mut details);
                details.push(theme.value(theme.label(
                    if done == steps.len() {
                        "complete"
                    } else {
                        "pending"
                    },
                    if done == steps.len() {
                        "已完成"
                    } else {
                        "待处理"
                    },
                )));
            }
            if blocked > 0 {
                theme.separate(&mut details);
                details.push(Span::styled(
                    format!("{blocked} blocked"),
                    theme.style("warning"),
                ));
            }
            metric_row(
                "tasks",
                "/tasks",
                if steps.is_empty() {
                    theme.label("Status", "状态")
                } else {
                    theme.label("Plan", "计划")
                },
                details,
                theme,
            )
        }
        "agents" => {
            let jobs = state.tasks["jobs"]
                .as_array()
                .filter(|jobs| !jobs.is_empty())?;
            let active = jobs.iter().filter(|job| job["status"] == "running").count();
            let mut details = vec![
                theme.value(format!("{active}/{}", jobs.len())),
                Span::styled(" running", theme.style("muted")),
            ];
            for job in jobs.iter().take(hud.config.activity_limit) {
                let live_elapsed = (job["status"] == "running")
                    .then(|| job["startedAt"].as_f64())
                    .flatten()
                    .filter(|started| started.is_finite() && *started >= 0.0 && *started <= now)
                    .map(|started| now - started);
                let seconds = live_elapsed
                    .or_else(|| job["elapsedSeconds"].as_f64())
                    .filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
                let (marker, role) = match job["status"].as_str() {
                    Some("running") => ("◐ ", "model"),
                    Some("failed") => ("! ", "critical"),
                    Some("cancelled") => ("× ", "muted"),
                    Some("done" | "completed") => ("✓ ", "context"),
                    _ => ("? ", "muted"),
                };
                theme.separate(&mut details);
                details.push(Span::styled(marker, theme.style(role)));
                details.push(theme.value(fit(job["title"].as_str().unwrap_or("job"), 32)));
                if let Some(seconds) = seconds {
                    details.push(Span::styled(
                        format!(" ({})", elapsed(seconds)),
                        theme.style("muted"),
                    ));
                }
            }
            metric_row(
                "agents",
                "/tasks",
                theme.label("Jobs", "后台作业"),
                details,
                theme,
            )
        }
        "skills" => {
            let environment = &metrics.environment;
            let details = if metrics.stream_id.is_empty() {
                vec![theme.value("n/a")]
            } else {
                let mut details = vec![
                    theme.value(format!(
                        "{}/{}",
                        environment.skills.len(),
                        environment.available_skills,
                    )),
                    Span::styled(" active", theme.style("muted")),
                ];
                theme.field(
                    &mut details,
                    "MCP",
                    environment.mcp_servers.len().to_string(),
                );
                theme.separate(&mut details);
                details.push(theme.value(hud.host.instruction_files.len().to_string()));
                details.push(Span::styled(" AGENTS.md", theme.style("muted")));
                if !environment.skills.is_empty() {
                    theme.separate(&mut details);
                    details.push(Span::styled(
                        environment
                            .skills
                            .iter()
                            .take(hud.config.activity_limit)
                            .map(|name| clean(name))
                            .collect::<Vec<_>>()
                            .join(", "),
                        theme.style("muted"),
                    ));
                    let hidden = environment.skills.len().saturating_sub(hud.config.activity_limit);
                    if hidden > 0 {
                        details.push(Span::styled(format!(" +{hidden}"), theme.style("muted")));
                    }
                }
                details
            };
            metric_row(
                "skills",
                "/skills",
                theme.label("Skills", "技能"),
                details,
                theme,
            )
        }
        "memory" => metric_row(
            "memory",
            "/hud info",
            theme.label("Host RAM", "主机内存"),
            hud.host.memory.map_or_else(
                || vec![theme.value("n/a")],
                |(used, total)| {
                    vec![
                        theme.value(format!(
                            "{:.1}/{:.1}",
                            used as f64 / 1073741824.0,
                            total as f64 / 1073741824.0,
                        )),
                        Span::styled(" GiB · ", theme.style("muted")),
                        theme.value(format!("{:.0}%", used as f64 / total.max(1) as f64 * 100.0)),
                    ]
                },
            ),
            theme,
        ),
        "limits" => {
            let limits = if !hud.config.external_usage_path.is_empty() {
                &hud.host.limits
            } else {
                &metrics.latest.limits
            };
            let updated = if hud.config.external_usage_path.is_empty() {
                metrics
                    .latest
                    .limits_updated_at
                    .or(metrics.latest.responded_at)
            } else {
                hud.host.usage_updated_at
            };
            let fresh = fresh(updated);
            // Gateway snapshots have their own observation clock. Retain an
            // older observation as explicitly historical, never as fresh usage.
            let last_known = metrics.latest.limits_updated_at.filter(|time| {
                hud.config.external_usage_path.is_empty()
                    && !fresh
                    && time.is_finite()
                    && *time >= 0.0
                    && *time <= now + 60.0
            });
            let parts: Vec<_> = limits
                .iter()
                .filter(|limit| {
                    (fresh || last_known.is_some())
                        && limit.percent.is_finite()
                        && (0.0..=100.0).contains(&limit.percent)
                        && limit.resets_at.is_none_or(|reset| reset > now)
                })
                .map(|limit| {
                    let percent = if limit.percent > 0.0 && limit.percent < 0.01 {
                        "<0.01%".into()
                    } else if limit.percent > 0.0 && limit.percent < 1.0 {
                        format!("{:.2}%", limit.percent)
                    } else {
                        format!("{:.0}%", limit.percent)
                    };
                    format!(
                        "{} {percent}{}",
                        clean(&limit.label),
                        limit.resets_at.map_or(String::new(), |reset| format!(
                            " · resets {}",
                            elapsed(reset - now)
                        ))
                    )
                })
                .collect();
            let details = if parts.is_empty() {
                vec![theme.value("n/a")]
            } else if let Some(observed) = last_known {
                vec![Span::styled(
                    format!(
                        "{} {}{} · {}",
                        theme.label("last known", "上次数据"),
                        elapsed(now - observed),
                        theme.label(" ago", "前"),
                        parts.join(" │ ")
                    ),
                    theme.style("muted"),
                )]
            } else {
                vec![theme.value(parts.join(" │ "))]
            };
            metric_row(
                "limits",
                "/hud info",
                theme.label("Limits", "额度"),
                details,
                theme,
            )
        }
        "session" => {
            let mut details = vec![theme.value(
                metrics
                    .started_at
                    .map_or("—".into(), |time| elapsed(now - time)),
            )];
            theme.field(
                &mut details,
                "last reply",
                metrics
                    .latest
                    .responded_at
                    .map_or("—".into(), |time| format!("{} ago", elapsed(now - time))),
            );
            metric_row(
                "session",
                "/hud info",
                theme.label("Session", "会话"),
                details,
                theme,
            )
        }
        "custom" => {
            if hud.config.custom_text.is_empty() {
                return None;
            }
            Row {
                element: "custom",
                command: "/hud",
                line: Line::styled(clean(&hud.config.custom_text), theme.style("muted")),
            }
        }
        _ => return None,
    };
    Some(row)
}

pub fn rows(state: &State, columns: usize, max_rows: usize) -> Vec<Row> {
    let cfg = &state.hud.config;
    if cfg.layout == Layout::Hidden || max_rows == 0 {
        return vec![];
    }
    let columns = columns.min(cfg.max_width);
    let theme = Theme {
        config: cfg,
        monochrome: state.monochrome,
        columns,
    };
    let now = Local::now().timestamp() as f64;
    let mut rows: Vec<_> = cfg
        .elements
        .iter()
        .filter_map(|name| element(state, name, columns, &theme, now))
        .collect();
    let budget = max_rows.min(cfg.max_rows);
    let reasoning_visible = |merged: bool| {
        if budget == 1 && rows.iter().any(|row| row.element == "usage") {
            return false;
        }
        let visible = rows.len().saturating_sub(usize::from(merged)).min(budget)
            .saturating_sub(usize::from(state.hud.problem.is_some()));
        rows.iter()
            .filter(|row| !merged || row.element != "usage")
            .take(visible)
            .any(|row| row.element == "reasoning")
    };
    let inline_effort = !reasoning_visible(false);
    let merged_inline_effort = !reasoning_visible(true);
    if let Some(index) = rows.iter().position(|row| row.element == "project") {
        rows[index] = project(state, columns, &theme, inline_effort);
    }
    if cfg.layout == Layout::Compact {
        if let (Some(project_index), Some(usage_index)) = (
            rows.iter().position(|row| row.element == "project"),
            rows.iter().position(|row| row.element == "usage"),
        ) {
            let project_width = columns.saturating_sub(rows[usage_index].line.width() + 3);
            let permission_reserve = if state.runtime_supported {
                "Accept edits".width().saturating_sub(clean(state.permission_label()).width())
            } else {
                0
            };
            let core_width = clean(project_model(state)).width()
                + project_controls(state, project_width, &theme, merged_inline_effort).width()
                + permission_reserve
                + 2;
            if project_width >= core_width.max(18) {
                let usage_row = rows.remove(usage_index);
                let project_index = project_index - usize::from(usage_index < project_index);
                rows[project_index].line = clip(
                    project(state, project_width, &theme, merged_inline_effort).line,
                    project_width,
                );
                rows[project_index]
                    .line
                    .spans
                    .push(Span::styled(" │ ", theme.style("muted")));
                rows[project_index].line.spans.extend(usage_row.line.spans);
                rows[project_index].element = "usage";
                rows[project_index].command = "/context";
            }
        }
    }
    if budget == 1 {
        if let Some(index) = rows.iter().position(|row| row.element == "usage") {
            let usage = rows.remove(index);
            rows.insert(0, usage);
        }
    }
    rows.truncate(budget);
    if let Some(problem) = &state.hud.problem {
        if !rows.is_empty() {
            rows.pop();
        }
        rows.push(Row {
            element: "problem",
            command: "/hud info",
            line: Line::styled(
                format!("HUD: {problem} · /hud info"),
                theme.style("critical"),
            ),
        });
    }
    for row in &mut rows {
        row.line = clip(std::mem::take(&mut row.line), columns);
    }
    rows
}
