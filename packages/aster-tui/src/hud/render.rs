// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Pure, width-bounded status rows, with real data shared by the screen and previews.

use chrono::{Local, TimeZone};
use ratatui::{
    style::{Color, Style},
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

fn text_row(element: &'static str, command: &'static str, text: String, theme: &Theme<'_>) -> Row {
    Row {
        element,
        command,
        line: Line::styled(text, theme.style("muted")),
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

fn project(state: &State, columns: usize, theme: &Theme<'_>) -> Row {
    let cfg = &state.hud.config;
    let raw_model = if cfg.model_override.is_empty() {
        &state.model
    } else {
        &cfg.model_override
    };
    let raw_model = if cfg.model_format == "short" {
        raw_model.rsplit(':').next().unwrap_or(raw_model)
    } else {
        raw_model
    };
    let model = fit(raw_model, columns.saturating_sub(8).min(columns * 2 / 3));
    let workspace = fit(
        &path_name(&state.workspace, cfg.path_levels),
        columns.saturating_sub(model.width() + 5),
    );
    let mut spans = vec![
        Span::styled(format!("[{model}]"), theme.style("model")),
        Span::styled(" │ ", theme.style("muted")),
        Span::styled(workspace, theme.style("project")),
    ];
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
        spans.push(Span::styled(
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
        spans.push(Span::styled(
            format!(" · {}", clean(provider)),
            theme.style("muted"),
        ));
    }
    let effort = state.runtime["reasoningLevel"]
        .as_str()
        .unwrap_or(&state.hud.telemetry.environment.effort);
    if !effort.is_empty() {
        spans.push(Span::styled(
            format!(" · {}", clean(effort)),
            theme.style("muted"),
        ));
    }
    if state.runtime_supported {
        spans.push(Span::styled(
            format!(
                " · {} · approval: {}",
                clean(&state.mode),
                clean(&state.approval_policy)
            ),
            theme.style("muted"),
        ));
    }
    if state.sandbox {
        spans.push(Span::styled(" · sandbox", theme.style("muted")));
    }
    Row {
        element: "project",
        command: "/status",
        line: Line::from(spans),
    }
}

fn usage(state: &State, columns: usize, theme: &Theme<'_>) -> Row {
    let label = theme.label(if columns < 60 { "ctx" } else { "context" }, "上下文");
    let cost = if state.cost.is_empty() {
        "cost —"
    } else {
        &state.cost
    };
    let context_width = columns.saturating_sub(cost.width().min(columns / 2) + 3);
    let mut spans = Vec::new();
    if let Some(ratio) = state.context_ratio.filter(|ratio| ratio.is_finite()) {
        let snapshot = state.context_snapshot.as_ref();
        let value = match state.hud.config.context_value.as_str() {
            "tokens" => {
                snapshot.map(|value| format!("{} / {}", count(value.used), count(value.window)))
            }
            "remaining" => snapshot
                .and_then(|value| value.used.zip(value.window))
                .map(|(used, total)| format!("{} left", count(Some(total.saturating_sub(used))))),
            "both" => snapshot.map(|value| {
                format!(
                    "{:.0}% · {} / {}",
                    ratio * 100.0,
                    count(value.used),
                    count(value.window)
                )
            }),
            _ => None,
        }
        .unwrap_or_else(|| format!("{:.0}%", ratio * 100.0));
        let bar_width = if columns >= 100 { 10 } else { 6 };
        spans.push(Span::styled(format!("{label} "), theme.style("muted")));
        if state.hud.config.context_bar
            && columns >= 40
            && label.width() + bar_width + value.width() + 2 <= context_width
        {
            let filled = (ratio.clamp(0.0, 1.0) * bar_width as f64).round() as usize;
            spans.push(Span::styled("█".repeat(filled), theme.usage(ratio)));
            spans.push(Span::styled(
                "░".repeat(bar_width - filled),
                theme.style("muted"),
            ));
            spans.push(Span::raw(" "));
        }
        spans.push(Span::styled(value, theme.usage(ratio)));
    } else {
        let value = if state.usage.is_empty() {
            format!("{label} —")
        } else if state.usage == "context n/a" {
            format!("{label} n/a")
        } else {
            state
                .usage
                .replace(" tokens", if columns < 60 { " tok" } else { " tokens" })
        };
        spans.push(Span::styled(value, theme.style("muted")));
    }
    let mut line = clip(Line::from(spans), context_width);
    line.spans.push(Span::styled(" │ ", theme.style("muted")));
    line.spans
        .push(Span::styled(cost.to_owned(), theme.style("muted")));
    if let Some(speed) = state
        .hud
        .telemetry
        .latest
        .speed
        .filter(|value| value.is_finite() && *value >= 0.0)
    {
        line.spans.push(Span::styled(
            format!(" · {speed:.1} tok/s"),
            theme.style("muted"),
        ));
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
        "project" => project(state, columns, theme),
        "usage" => usage(state, columns, theme),
        "tokens" => text_row(
            "tokens",
            "/hud info",
            format!(
                "{}{}  in {} · out {} · reasoning {}{}",
                theme.label("Tokens", "Token"),
                if hud.config.token_scope == "request" {
                    theme.label(" (last)", "（本次）")
                } else {
                    ""
                },
                count(tokens.input),
                count(tokens.output),
                count(tokens.reasoning),
                if hud.config.token_scope == "request" {
                    String::new()
                } else if metrics.failed_calls > 0 {
                    format!(
                        " · {} completed · {} failed",
                        metrics.calls, metrics.failed_calls
                    )
                } else {
                    format!(" · {} calls", metrics.calls)
                }
            ),
            theme,
        ),
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
            text_row(
                "cache",
                "/hud info",
                format!(
                    "{}  read {} · write {} · hit {hit}{expiry}",
                    theme.label("Cache", "缓存"),
                    count(tokens.cache_read),
                    count(tokens.cache_write)
                ),
                theme,
            )
        }
        "cost" => text_row(
            "cost",
            "/hud info",
            format!(
                "{}  session {}{}{}",
                theme.label("Cost", "费用"),
                if state.cost.is_empty() {
                    "—"
                } else {
                    state.cost.strip_prefix("cost ").unwrap_or(&state.cost)
                },
                if hud.config.daily_cost {
                    format!(
                        " · today {}",
                        money(hud.host.daily_cost, hud.host.daily_cost_partial)
                    )
                } else {
                    String::new()
                },
                if hud.config.weekly_cost {
                    format!(
                        " · week {}",
                        money(hud.host.weekly_cost, hud.host.weekly_cost_partial)
                    )
                } else {
                    String::new()
                }
            ),
            theme,
        ),
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
            let labels = active
                .iter()
                .take(hud.config.activity_limit)
                .map(|tool| {
                    format!(
                        "◐ {}{}",
                        fit(&tool.title, 32),
                        tool.elapsed_seconds()
                            .map_or(String::new(), |seconds| format!(" ({})", elapsed(seconds)))
                    )
                })
                .collect::<Vec<_>>()
                .join(" · ");
            text_row(
                "tools",
                "/tools",
                format!(
                    "{}  {}{}✓ {completed}{}",
                    theme.label("Tools", "工具"),
                    labels,
                    if labels.is_empty() { "" } else { " · " },
                    if failed > 0 {
                        format!(" · ! {failed}")
                    } else {
                        String::new()
                    }
                ),
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
                details.push(format!("{done}/{}", steps.len()));
            }
            if let Some(reason) = waiting {
                details.push(clean(reason));
            }
            if let Some(current) = current {
                details.push(clean(current));
            } else if waiting.is_none() {
                details.push(
                    theme
                        .label(
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
                        )
                        .into(),
                );
            }
            if blocked > 0 {
                details.push(format!("{blocked} blocked"));
            }
            text_row(
                "tasks",
                "/tasks",
                format!(
                    "{}  {}",
                    if steps.is_empty() {
                        theme.label("Status", "状态")
                    } else {
                        theme.label("Plan", "计划")
                    },
                    details.join(" · ")
                ),
                theme,
            )
        }
        "agents" => {
            let jobs = state.tasks["jobs"]
                .as_array()
                .filter(|jobs| !jobs.is_empty())?;
            let active = jobs.iter().filter(|job| job["status"] == "running").count();
            let labels = jobs
                .iter()
                .take(hud.config.activity_limit)
                .map(|job| {
                    let live_elapsed = (job["status"] == "running")
                        .then(|| job["startedAt"].as_f64())
                        .flatten()
                        .filter(|started| started.is_finite() && *started >= 0.0 && *started <= now)
                        .map(|started| now - started);
                    let seconds = live_elapsed
                        .or_else(|| job["elapsedSeconds"].as_f64())
                        .filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
                    format!(
                        "{} {}{}",
                        if job["status"] == "running" {
                            "◐"
                        } else if job["status"] == "failed" {
                            "!"
                        } else if job["status"] == "cancelled" {
                            "×"
                        } else if matches!(job["status"].as_str(), Some("done" | "completed")) {
                            "✓"
                        } else {
                            "?"
                        },
                        fit(job["title"].as_str().unwrap_or("job"), 32),
                        seconds.map_or(String::new(), |seconds| format!(" ({})", elapsed(seconds)))
                    )
                })
                .collect::<Vec<_>>()
                .join(" · ");
            text_row(
                "agents",
                "/tasks",
                format!(
                    "{}  {active}/{} running · {labels}",
                    theme.label("Jobs", "后台作业"),
                    jobs.len()
                ),
                theme,
            )
        }
        "skills" => {
            let environment = &metrics.environment;
            let details = if metrics.stream_id.is_empty() {
                "n/a".into()
            } else {
                format!(
                    "{}/{} active · MCP {} · {} AGENTS.md{}",
                    environment.skills.len(),
                    environment.available_skills,
                    environment.mcp_servers.len(),
                    hud.host.instruction_files.len(),
                    if environment.skills.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " · {}",
                            environment
                                .skills
                                .iter()
                                .take(hud.config.activity_limit)
                                .map(|name| clean(name))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                )
            };
            text_row(
                "skills",
                "/hud info",
                format!("{}  {details}", theme.label("Skills", "技能")),
                theme,
            )
        }
        "memory" => text_row(
            "memory",
            "/hud info",
            hud.host.memory.map_or_else(
                || format!("{}  n/a", theme.label("Host RAM", "主机内存")),
                |(used, total)| {
                    format!(
                        "{}  {:.1}/{:.1} GiB · {:.0}%",
                        theme.label("Host RAM", "主机内存"),
                        used as f64 / 1073741824.0,
                        total as f64 / 1073741824.0,
                        used as f64 / total.max(1) as f64 * 100.0
                    )
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
            text_row(
                "limits",
                "/hud info",
                format!(
                    "{}  {}",
                    theme.label("Limits", "额度"),
                    if parts.is_empty() {
                        "n/a".into()
                    } else if let Some(observed) = last_known {
                        format!(
                            "{} {}{} · {}",
                            theme.label("last known", "上次数据"),
                            elapsed(now - observed),
                            theme.label(" ago", "前"),
                            parts.join(" │ ")
                        )
                    } else {
                        parts.join(" │ ")
                    }
                ),
                theme,
            )
        }
        "session" => text_row(
            "session",
            "/hud info",
            format!(
                "{}  {} · last reply {}",
                theme.label("Session", "会话"),
                metrics
                    .started_at
                    .map_or("—".into(), |time| elapsed(now - time)),
                metrics
                    .latest
                    .responded_at
                    .map_or("—".into(), |time| format!("{} ago", elapsed(now - time)))
            ),
            theme,
        ),
        "custom" => {
            if hud.config.custom_text.is_empty() {
                return None;
            }
            text_row("custom", "/hud", clean(&hud.config.custom_text), theme)
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
    };
    let now = Local::now().timestamp() as f64;
    let mut rows: Vec<_> = cfg
        .elements
        .iter()
        .filter_map(|name| element(state, name, columns, &theme, now))
        .collect();
    if cfg.layout == Layout::Compact {
        if let (Some(project_index), Some(usage_index)) = (
            rows.iter().position(|row| row.element == "project"),
            rows.iter().position(|row| row.element == "usage"),
        ) {
            let project_width = columns.saturating_sub(rows[usage_index].line.width() + 3);
            if project_width >= 18 {
                let usage_row = rows.remove(usage_index);
                let project_index = project_index - usize::from(usage_index < project_index);
                rows[project_index].line =
                    clip(project(state, project_width, &theme).line, project_width);
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
    let budget = max_rows.min(cfg.max_rows);
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
