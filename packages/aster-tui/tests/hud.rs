// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Exercise the HUD through its public configuration, accounting and rendering interfaces.

use aster_tui::{
    controller::Controller,
    hud::{self, Hud, Layout},
    state::{Entry, Overlay, Phase, State, Tool},
    ui,
};
use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, style::Color, Terminal};
use serde_json::{json, Value};

fn metrics(stream: &str, sequence: u64, cost: f64) -> Value {
    let day = Local::now().format("%Y-%m-%d").to_string();
    json!({
        "streamId": stream.repeat(32), "sequence": sequence,
        "startedAt": Local::now().timestamp() - 60, "calls": sequence,
        "tokens": {"input": 12000, "output": 2400, "cacheRead": 8000, "cacheWrite": 0, "reasoning": 300},
        "latest": {"model": "test-model", "speed": 48.5, "respondedAt": Local::now().timestamp(),
                   "tokens": {"input": 6000, "output": 200}},
        "environment": {"skills": ["coding"], "availableSkills": 3, "mcpServers": ["docs"], "effort": "high"},
        "days": {day: {"cost": cost, "complete": true, "calls": sequence}}
    })
}

fn state() -> State {
    let mut state = State::new("E:/工作区/project".into(), "test-model".into(), false);
    state.title = "Conversation title".into();
    state.phase = Phase::Ready;
    state.context_ratio = Some(0.86);
    state.cost = "$0.1234".into();
    state.hud.ingest(&metrics("a", 2, 0.1234));
    state
}

fn lines(state: &State, width: usize, budget: usize) -> Vec<String> {
    hud::rows(state, width, budget)
        .into_iter()
        .map(|row| row.line.to_string())
        .collect()
}

#[test]
fn layered_settings_validate_merge_colors_and_round_trip_without_losing_defaults() {
    let temp = tempfile::tempdir().unwrap();
    let global = temp.path().join("global.json");
    let project = temp.path().join("local.json");
    std::fs::write(
        &global,
        r##"{"layout":"compact","colors":{"model":"#abcdef","warning":"3"},"max_rows":4}"##,
    )
    .unwrap();
    std::fs::write(
        &project,
        r##"{"path_levels":2,"colors":{"model":"#123456"}}"##,
    )
    .unwrap();
    let config = hud::config::load(&[global, project.clone()]).unwrap();
    assert_eq!(config.layout, Layout::Compact);
    assert_eq!(config.path_levels, 2);
    assert_eq!(config.max_rows, 4);
    assert_eq!(config.colors["warning"], "3");
    assert_eq!(config.colors["model"], "#123456");
    assert!(config.git);
    hud::config::save(&project, &config).unwrap();
    assert_eq!(
        hud::config::load(std::slice::from_ref(&project)).unwrap(),
        config
    );
    std::fs::write(&project, r#"{"max_rows":0}"#).unwrap();
    assert!(hud::config::load(std::slice::from_ref(&project)).is_err());
    std::fs::write(&project, r#"{"unknown_option":true}"#).unwrap();
    assert!(hud::config::load(&[project]).is_err());
}

#[test]
fn commands_preserve_element_order_validate_values_and_persist_undo() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(".aster/tui-hud.json");
    let mut hud = Hud::default();
    hud.config_path = Some(path.clone());
    // Construct through Default: a new installation has no invented undo state.
    assert!(!hud.choices().iter().any(|choice| choice.value == "undo"));
    hud.configure("compact").unwrap();
    let before = std::fs::read(&path).unwrap();
    assert!(hud.configure("set warning_percent 99").is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(hud.config.changed("show project").unwrap(), hud.config);
    hud.configure("set   path_levels  0").unwrap();
    assert_eq!(hud.config.path_levels, 0);
    hud.configure("undo").unwrap();
    assert_eq!(hud.config.layout, Layout::Compact);
    assert_eq!(hud.config.path_levels, 1);
    assert_eq!(hud::config::load(&[path]).unwrap(), hud.config);
    hud.configure("full").unwrap();
    assert_eq!(hud.config.elements.len(), hud::config::ELEMENTS.len());
    assert!(hud.config.daily_cost && hud.config.weekly_cost && hud.config.git_details);
    assert!(hud.configure("set elements [\"usage\",\"usage\"]").is_err());
    assert!(hud.configure("off extra").is_err());
}

#[test]
fn hud_resetting_colors_survives_reloading_layered_settings_and_undo() {
    let temp = tempfile::tempdir().unwrap();
    let global = temp.path().join("global.json");
    let local = temp.path().join("local.json");
    std::fs::write(
        &global,
        r##"{"layout":"compact","colors":{"model":"#abcdef","warning":"3"}}"##,
    )
    .unwrap();
    std::fs::write(&local, "{}").unwrap();
    let paths = [global, local.clone()];
    let original = hud::config::load(&paths).unwrap();
    assert_eq!(original.colors.len(), 2);
    let mut hud = Hud::default();
    hud.config_path = Some(local.clone());
    hud.config = original.clone();
    hud.configure("reset").unwrap();
    assert_eq!(hud.config, hud::Config::default());
    assert_eq!(hud::config::load(&paths).unwrap(), hud.config);

    hud.configure("undo").unwrap();
    assert_eq!(hud::config::load(&paths).unwrap(), original);
    hud.configure("set colors {}").unwrap();
    assert!(hud::config::load(&paths).unwrap().colors.is_empty());
}

#[test]
fn hud_cache_expiry_expires_with_its_observation_before_the_next_poll() {
    for external in [false, true] {
        let mut state = state();
        let now = Local::now().timestamp() as f64;
        state.hud.config.elements = vec!["cache".into()];
        state.hud.config.external_usage_ttl_seconds = 1;
        state.hud.config.refresh_seconds = 120;
        if external {
            state.hud.config.external_usage_path = "usage.json".into();
            state.hud.host.cache_expires_at = Some(now + 3600.0);
            state.hud.host.usage_updated_at = Some(now);
        } else {
            state.hud.telemetry.latest.cache_expires_at = Some(now + 3600.0);
            state.hud.telemetry.latest.responded_at = Some(now);
        }
        assert!(lines(&state, 160, 2).join("\n").contains(" · until "));
        for updated in [None, Some(now - 60.0), Some(now + 120.0), Some(f64::NAN)] {
            if external {
                state.hud.host.usage_updated_at = updated;
            } else {
                state.hud.telemetry.latest.responded_at = updated;
            }
            let rendered = lines(&state, 160, 2).join("\n");
            assert!(!rendered.contains(" · until "), "{rendered}");
            assert!(!rendered.contains(" · expired"), "{rendered}");
        }
    }
}

#[test]
fn settings_work_while_busy_without_sending_a_prompt_or_touching_the_draft() {
    let mut controller = Controller::new(state(), None, false);
    controller.state.phase = Phase::Busy;
    controller.state.editor.insert_str("draft 中文");
    let cursor = controller.state.editor.cursor();
    assert!(controller.input("/hud".into()).is_empty());
    assert_eq!(controller.state.overlay, Some(Overlay::Hud));
    controller.state.selected = controller
        .state
        .choices()
        .iter()
        .position(|choice| choice.value == "compact")
        .unwrap();
    assert!(controller
        .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .is_empty());
    assert_eq!(controller.state.hud.config.layout, Layout::Compact);
    assert_eq!(controller.state.overlay, Some(Overlay::Hud));
    controller.input("/hud info".into());
    assert_eq!(controller.state.overlay, Some(Overlay::HudInfo));
    assert_eq!(controller.state.editor.lines(), &["draft 中文"]);
    assert_eq!(controller.state.editor.cursor(), cursor);
}

#[test]
fn responsive_hud_keeps_usage_and_cost_below_composer_and_preserves_the_title() {
    let mut state = state();
    for columns in 24..=160 {
        for layout in [Layout::Compact, Layout::Expanded] {
            state.hud.config.layout = layout;
            let rows = hud::rows(&state, columns, 1);
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].command, "/context");
            assert!(rows[0].line.width() <= columns);
            assert!(rows[0].line.to_string().contains("86%"));
            assert!(rows[0].line.to_string().contains("$0.1234"));
        }
    }
    for (columns, height) in [(24, 8), (48, 12), (80, 24), (140, 40)] {
        state.hud.config.layout = Layout::Expanded;
        let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
        let mut hits = ui::HitAreas::default();
        terminal
            .draw(|frame| hits = ui::draw(frame, &mut state))
            .unwrap();
        assert!(hits
            .hud_rows
            .iter()
            .all(|(area, _)| area.y >= hits.composer.bottom() && area.bottom() <= height));
        let title: String = (0..columns)
            .map(|x| terminal.backend().buffer()[(x, hits.composer.y)].symbol())
            .collect();
        assert!(title.contains("Conversation") || columns < 30);
        let top: String = (0..columns)
            .map(|x| terminal.backend().buffer()[(x, 0)].symbol())
            .collect();
        assert!(!top.contains("project") && !top.contains("$0.1234"));
    }
    state.hud.configure("off").unwrap();
    assert!(hud::rows(&state, 120, 10).is_empty());
}

#[test]
fn full_details_use_real_values_cancelled_jobs_and_unknown_fields() {
    let mut state = state();
    state.hud.configure("full").unwrap();
    state.hud.host.daily_cost = Some(1.5);
    state.hud.host.weekly_cost = Some(4.0);
    state.hud.host.memory = Some((4 << 30, 16 << 30));
    state.hud.host.instruction_files.push("AGENTS.md".into());
    state.tasks = json!({"steps":[{"status":"completed"},{"status":"pending","blockedBy":["one"]}],
                        "currentTask":"Run checks", "jobs":[{"status":"cancelled","title":"old job","elapsedSeconds":3.0}]});
    state.entries.push(Entry::Tool(Tool {
        id: "one".into(),
        title: "Inspect source".into(),
        status: "completed".into(),
        content: String::new(),
        diff: String::new(),
        expanded: false,
        started: None,
        elapsed: Some(1.0),
    }));
    let full = lines(&state, 160, 16).join("\n");
    for value in [
        "48.5 tok/s",
        "in 12.0k",
        "write 0",
        "hit 66.7%",
        "today $1.5000",
        "week $4.0000",
        "✓ 1",
        "1/2",
        "1 blocked",
        "× old job (3s)",
        "MCP 1",
        "1 AGENTS.md",
        "4.0/16.0 GiB",
        "Limits  n/a",
    ] {
        assert!(full.contains(value), "{value}: {full}");
    }
    state.hud.telemetry.tokens.cache_write = None;
    state.hud.host.daily_cost = None;
    let unknown = lines(&state, 160, 16).join("\n");
    assert!(unknown.contains("write n/a") && unknown.contains("today n/a"));
}

#[test]
fn task_waiting_and_live_job_time_survive_without_a_plan_or_new_snapshot() {
    let mut state = state();
    let now = Local::now().timestamp();
    state.hud.config.elements = vec!["tasks".into(), "agents".into()];
    state.hud.config.activity_limit = 4;
    state.tasks = json!({
        "steps": [], "waitReason": "Waiting for approval",
        "jobs": [
            {"status":"running", "title":"review", "startedAt":now - 125, "elapsedSeconds":1},
            {"status":"done", "title":"saved", "startedAt":now - 125, "elapsedSeconds":3},
            {"status":"failed", "title":"check", "elapsedSeconds":2},
            {"status":"unknown", "title":"legacy"}
        ]
    });
    let rendered = lines(&state, 160, 8).join("\n");
    assert!(rendered.contains("Status  Waiting for approval"));
    assert!(rendered.contains("◐ review (2m"), "{rendered}");
    assert!(rendered.contains("✓ saved (3s)"));
    assert!(rendered.contains("! check (2s)"));
    assert!(rendered.contains("? legacy"));
    state.tasks["waitReason"] = json!("");
    assert!(hud::rows(&state, 160, 8)
        .iter()
        .all(|row| row.element != "tasks"));
}

#[test]
fn color_overrides_respect_monochrome_and_untrusted_text_cannot_inject_controls() {
    let mut state = state();
    state.monochrome = false;
    state
        .hud
        .configure(r##"set colors {"critical":"#123456"}"##)
        .unwrap();
    state
        .hud
        .configure("set custom_text \"中文 👩‍💻 \\u001b[31m injected\\n next\"")
        .unwrap();
    state.hud.configure("show custom").unwrap();
    let rows = hud::rows(&state, 80, 8);
    assert!(rows
        .iter()
        .flat_map(|row| &row.line.spans)
        .any(|span| span.style.fg == Some(Color::Rgb(0x12, 0x34, 0x56))));
    assert!(!rows
        .iter()
        .any(|row| row.line.to_string().contains(['\x1b', '\n'])));
    state.monochrome = true;
    for row in hud::rows(&state, 80, 8) {
        assert!(row.line.style.fg.is_none());
        assert!(row.line.spans.iter().all(|span| span.style.fg.is_none()));
    }
}

#[test]
fn request_scope_and_compact_layout_use_the_same_live_metrics() {
    let mut state = state();
    state.hud.configure("set token_scope request").unwrap();
    state.hud.configure("show tokens").unwrap();
    state.hud.configure("show cache").unwrap();
    let rendered = lines(&state, 120, 8).join("\n");
    assert!(rendered.contains("Tokens (last)  in 6000 · out 200"));
    assert!(rendered.contains("read n/a"));
    state.hud.configure("compact").unwrap();
    let first = &hud::rows(&state, 120, 8)[0];
    assert!(first.line.to_string().contains("project"));
    assert!(first.line.to_string().contains("86%"));
    assert!(first.line.to_string().contains("$0.1234"));
}

#[test]
fn full_hud_and_long_drafts_share_short_screens_without_overlapping_controls() {
    let mut state = state();
    state.hud.configure("full").unwrap();
    state.phase = Phase::Busy;
    state.attachment_loading = true;
    state.queued.push_back("queued message".into());
    state.tasks = json!({"steps":[{"status":"pending","title":"check"}], "jobs":[{"status":"running","title":"review"}]});
    state.editor.insert_str("long draft line\n".repeat(20));
    let cursor = state.editor.cursor();
    for height in 3..=36 {
        for columns in [18, 48, 100] {
            let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
            terminal
                .draw(|frame| {
                    let hits = ui::draw(frame, &mut state);
                    assert!(hits.composer.bottom() <= height);
                    assert!(hits.transcript.bottom() <= hits.composer.y);
                    assert!(hits.hud_rows.iter().all(
                        |(area, _)| area.y >= hits.composer.bottom() && area.bottom() <= height
                    ));
                })
                .unwrap();
        }
    }
    assert_eq!(state.editor.cursor(), cursor);
}

#[tokio::test]
async fn in_flight_collection_cannot_drop_new_usage_or_replace_reconfigured_host_state() {
    let temp = tempfile::tempdir().unwrap();
    let mut hud = Hud::default();
    hud.ledger_path = Some(temp.path().join("usage"));
    hud.config.git = false;
    hud.ingest(&metrics("a", 1, 1.0));
    let request = hud.collect_request(".", true).unwrap();
    hud.ingest(&metrics("a", 2, 2.0));
    hud.configure("compact").unwrap();
    hud.host.daily_cost = Some(99.0);
    hud.collected(hud::collect::sample(request).await);
    assert_eq!(hud.host.daily_cost, Some(99.0));
    assert_eq!(
        hud.collect_request(".", true).unwrap().records[0].sequence,
        2
    );
    hud.reset_session();
    assert!(hud.telemetry.stream_id.is_empty());
    hud.flush().await.unwrap();
    assert!(hud.collect_request(".", true).unwrap().records.is_empty());
    let mut request = hud.collect_request(".", true).unwrap();
    request.config.daily_cost = true;
    hud.collected(hud::collect::sample(request).await);
    assert_eq!(hud.host.daily_cost, Some(2.0));
}

#[tokio::test]
async fn separate_streams_and_duplicate_notifications_do_not_clobber_or_double_bill() {
    let temp = tempfile::tempdir().unwrap();
    let mut first = Hud::default();
    first.ledger_path = Some(temp.path().join("usage"));
    first.config.git = false;
    let mut second = Hud::default();
    second.ledger_path = first.ledger_path.clone();
    second.config.git = false;
    first.ingest(&metrics("a", 2, 1.25));
    first.ingest(&metrics("a", 1, 8.0));
    first.ingest(&metrics("a", 2, 8.0));
    second.ingest(&metrics("b", 1, 2.5));
    let (one, two) = tokio::join!(
        hud::collect::sample(first.collect_request(".", true).unwrap()),
        hud::collect::sample(second.collect_request(".", true).unwrap())
    );
    first.collected(one);
    second.collected(two);
    first.flush().await.unwrap();
    first.config.daily_cost = true;
    first.config.weekly_cost = true;
    let result = hud::collect::sample(first.collect_request(".", true).unwrap()).await;
    assert_eq!(result.host.daily_cost, Some(3.75));
    assert_eq!(result.host.weekly_cost, Some(3.75));
}

#[tokio::test]
async fn unreadable_ledger_retains_the_outbox_and_unknown_prices_never_look_free() {
    let temp = tempfile::tempdir().unwrap();
    let bad = temp.path().join("file");
    std::fs::write(&bad, "occupied").unwrap();
    let mut hud = Hud::default();
    hud.config.git = false;
    hud.ledger_path = Some(bad);
    let mut value = metrics("a", 1, 0.0);
    value["days"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap()["complete"] = json!(false);
    hud.ingest(&value);
    assert!(hud.flush().await.is_err());
    assert_eq!(hud.collect_request(".", true).unwrap().records.len(), 1);
    hud.ledger_path = Some(temp.path().join("recovered"));
    hud.flush().await.unwrap();
    hud.config.daily_cost = true;
    hud.config.weekly_cost = true;
    let result = hud::collect::sample(hud.collect_request(".", true).unwrap()).await;
    assert_eq!(result.host.daily_cost, None);
    assert_eq!(result.host.weekly_cost, None);
}

#[tokio::test]
async fn external_quota_expiry_freshness_and_errors_are_visible_without_network_access() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("usage.json");
    let now = Local::now().timestamp();
    std::fs::write(
        &path,
        json!({"updated_at": now, "five_hour":{"used_percentage":25,"resets_at":now+3600},
        "seven_day":{"used_percentage":80,"resets_at":now-1}, "cache_expires_at":now+300})
        .to_string(),
    )
    .unwrap();
    let mut state = state();
    state.workspace = temp.path().display().to_string();
    state.hud.config.git = false;
    state.hud.config.external_usage_path = "usage.json".into();
    state.hud.config.elements = vec!["limits".into(), "cache".into()];
    let request = state.hud.collect_request(&state.workspace, true).unwrap();
    state.hud.collected(hud::collect::sample(request).await);
    assert_eq!(state.hud.host.limits.len(), 1);
    assert!(lines(&state, 100, 3).join("\n").contains("5h 25%"));
    std::fs::write(
        &path,
        json!({"updated_at": now-1000,"five_hour":{"used_percentage":99}}).to_string(),
    )
    .unwrap();
    let request = state.hud.collect_request(&state.workspace, true).unwrap();
    state.hud.collected(hud::collect::sample(request).await);
    assert!(state.hud.host.limits.is_empty());
    assert!(state.hud.diagnostics().contains("stale"));
    assert!(lines(&state, 100, 3).join("\n").contains("Limits  n/a"));
    std::fs::write(&path, "not json").unwrap();
    let request = state.hud.collect_request(&state.workspace, true).unwrap();
    state.hud.collected(hud::collect::sample(request).await);
    assert!(!state.hud.host.notices.is_empty());
}

#[tokio::test]
async fn partially_priced_days_keep_known_amounts_with_a_visible_marker() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = state();
    state.hud = Hud::default();
    state.hud.ledger_path = Some(temp.path().join("usage"));
    state.hud.config.git = false;
    state.hud.config.daily_cost = true;
    state.hud.config.weekly_cost = true;
    state.hud.config.elements = vec!["cost".into()];
    state.hud.ingest(&metrics("a", 1, 0.25));
    let mut unknown = metrics("b", 1, 0.0);
    for day in unknown["days"].as_object_mut().unwrap().values_mut() {
        day["complete"] = json!(false);
    }
    state.hud.ingest(&unknown);
    let result = hud::collect::sample(state.hud.collect_request(".", true).unwrap()).await;
    state.hud.collected(result);
    let rendered = lines(&state, 140, 4).join("\n");
    assert!(rendered.contains("today ≥$0.2500"), "{rendered}");
    assert!(rendered.contains("week ≥$0.2500"), "{rendered}");
}

#[test]
fn gateway_limits_use_their_own_observation_clock_and_keep_small_usage_visible() {
    let mut state = state();
    state.hud.config.elements = vec!["limits".into()];
    let now = Local::now().timestamp();
    let mut value = metrics("a", 3, 0.25);
    value["latest"]["respondedAt"] = json!(now - 3600);
    value["latest"]["limitsUpdatedAt"] = json!(now);
    value["latest"]["limits"] = json!([
        {"label": "workbuddy Credits", "percent": 0.0433, "resetsAt": now + 3600}
    ]);
    state.hud.ingest(&value);
    let rendered = lines(&state, 140, 4).join("\n");
    assert!(rendered.contains("workbuddy Credits 0.04%"), "{rendered}");
    value["sequence"] = json!(4);
    value["latest"]["limitsUpdatedAt"] = json!(now - 3600);
    value["latest"]["respondedAt"] = json!(now);
    state.hud.ingest(&value);
    let rendered = lines(&state, 140, 4).join("\n");
    assert!(rendered.contains("last known 1h"), "{rendered}");
    assert!(rendered.contains("workbuddy Credits 0.04%"), "{rendered}");
    assert!(!rendered.contains("n/a"), "{rendered}");
    for observed in [json!(now + 120), json!(-1), json!(null)] {
        value["sequence"] = json!(value["sequence"].as_u64().unwrap() + 1);
        value["latest"]["limitsUpdatedAt"] = observed;
        value["latest"]["respondedAt"] = json!(now - 3600);
        state.hud.ingest(&value);
        assert!(lines(&state, 140, 4).join("\n").contains("Limits  n/a"));
    }
    value["sequence"] = json!(8);
    value["latest"]["limitsUpdatedAt"] = json!(now - 3600);
    value["latest"]["limits"][0]["resetsAt"] = json!(now - 1);
    state.hud.ingest(&value);
    assert!(lines(&state, 140, 4).join("\n").contains("Limits  n/a"));
}
