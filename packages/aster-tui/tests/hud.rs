// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Exercise the HUD through its public configuration, accounting and rendering interfaces.

use aster_tui::{
    controller::{Action, Controller, Pending},
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
fn skills_hud_links_to_the_browser_and_marks_additional_active_skills() {
    let mut state = state();
    state.hud.config.elements = vec!["skills".into()];
    state.hud.telemetry.environment.skills = [
        "aster.libwriting",
        "aster.repo",
        "aster.shell",
        "aster.todo",
    ]
    .map(str::to_owned)
    .into();
    let rows = hud::rows(&state, 160, 1);
    assert_eq!(rows[0].command, "/skills");
    assert!(rows[0].line.to_string().contains("+1"));
}

#[test]
fn project_hud_shows_the_confirmed_permission_mode_in_plain_language() {
    for (mode, approval, permission, label) in [
        ("code", "session", "auto", "Auto"),
        ("code", "ask", "manual", "Manual"),
        ("code", "accept_edits", "accept_edits", "Accept edits"),
        ("plan", "ask", "plan", "Plan"),
    ] {
        let mut state = state();
        state.runtime_supported = true;
        state.mode = mode.into();
        state.approval_policy = approval.into();
        state.hud.config.layout = Layout::Expanded;
        state.hud.config.elements = vec!["project".into()];
        for metadata in [Value::Null, json!({"permissionMode":permission})] {
            state.runtime = metadata;
            let rendered = lines(&state, 160, 1).join("\n");
            assert!(
                rendered.contains(&format!("permissions: {label}")),
                "{rendered}"
            );
            assert!(!rendered.contains("approval: ask"), "{rendered}");
            assert!(!rendered.contains("approval: session"), "{rendered}");
        }
    }
}

#[test]
fn compact_hud_keeps_permissions_and_effort_before_long_project_details() {
    for (permission, label) in [
        ("auto", "Auto"),
        ("manual", "Manual"),
        ("accept_edits", "Accept edits"),
        ("plan", "Plan"),
    ] {
        let mut state = state();
        state.model = "magpie:linxi/gpt-6.1-sol-with-a-long-route-name".into();
        state.workspace = "E:/项目/long-workspace-directory-name".into();
        state.runtime_supported = true;
        state.runtime = json!({"permissionMode": permission, "reasoningLevel": "high"});
        state.hud.host.git = Some(aster_tui::workspace_changes::GitSummary {
            branch: "feature/long-branch-name-with-more-details".into(),
            dirty: true,
            ahead: 12,
            behind: 3,
            ..Default::default()
        });
        for columns in [48, 80, 120] {
            let rows = hud::rows(&state, columns, 3);
            let rendered = rows
                .iter()
                .map(|row| row.line.to_string())
                .collect::<Vec<_>>();
            let project = &rendered[0];
            assert!(project.contains(label), "{columns}: {rendered:?}");
            assert!(project.contains("high"), "{columns}: {rendered:?}");
            assert!(project.starts_with("[magpie:"), "{columns}: {rendered:?}");
            assert!(rows.iter().all(|row| row.line.width() <= columns));
            assert!(rendered
                .iter()
                .any(|line| line.contains("86%") && line.contains("$0.1234")));
            if columns <= 80 {
                assert_eq!(rows[0].command, "/status");
                assert_eq!(rows[1].command, "/context");
            }
            let single = hud::rows(&state, columns, 1);
            assert_eq!(single.len(), 1);
            assert_eq!(single[0].command, "/context");
            assert!(single[0].line.to_string().contains("86%"));
        }
    }
}

#[test]
fn project_keeps_effort_when_the_configured_reasoning_row_does_not_fit() {
    let mut state = state();
    state.model = "magpie:linxi/gpt-6.1-sol-with-a-long-route-name".into();
    state.runtime_supported = true;
    state.runtime = json!({"permissionMode": "manual", "reasoningLevel": "high"});
    state.hud.config.elements = ["project", "usage", "tools", "reasoning"]
        .map(str::to_owned)
        .into();
    state.entries.push(Entry::Tool(Tool {
        id: "active".into(),
        title: "Read source".into(),
        status: "in_progress".into(),
        content: String::new(),
        diff: String::new(),
        expanded: false,
        started: None,
        elapsed: None,
    }));
    for columns in [48, 80, 120, 160] {
        for budget in [2, 3, 4] {
            state.hud.config.max_rows = budget;
            for problem in [None, Some("Collection unavailable".into())] {
                state.hud.problem = problem;
                let rows = hud::rows(&state, columns, budget);
                let rendered = rows
                    .iter()
                    .map(|row| row.line.to_string())
                    .collect::<Vec<_>>();
                assert_eq!(
                    rendered
                        .iter()
                        .map(|line| line.matches("high").count())
                        .sum::<usize>(),
                    1,
                    "{columns} columns, {budget} rows: {rendered:?}"
                );
                assert!(rows.iter().all(|row| row.line.width() <= columns));
                assert!(rows.len() <= budget);
            }
        }
    }
}

#[test]
fn changing_permissions_keeps_the_compact_hud_row_layout_stable() {
    let mut state = state();
    state.model = "linxi/gpt-6.1-sol".into();
    state.runtime_supported = true;
    for columns in 48..=140 {
        let mut expected = None;
        for permission in ["manual", "accept_edits", "plan", "auto"] {
            state.runtime = json!({"permissionMode": permission, "reasoningLevel": "high"});
            let layout = hud::rows(&state, columns, 3)
                .into_iter()
                .map(|row| (row.element, row.command))
                .collect::<Vec<_>>();
            if let Some(expected) = &expected {
                assert_eq!(
                    &layout, expected,
                    "{columns} columns, permission {permission}"
                );
            } else {
                expected = Some(layout);
            }
        }
    }
}

#[test]
fn startup_context_shows_zero_and_known_capacity_while_measurement_stays_pending() {
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.session_id = Some("session".into());
    state.phase = Phase::Ready;
    state.hud.config.layout = Layout::Expanded;
    state.hud.config.elements = vec!["usage".into()];
    state.hud.config.context_value = "tokens".into();
    let mut controller = Controller::new(state, None, false);
    assert!(lines(&controller.state, 80, 1)[0].contains("context    —"));
    let update = |value| {
        json!({"method":"session/update", "params":{
            "sessionId":"session", "update":value
        }})
    };
    controller.message(update(json!({
        "sessionUpdate":"usage_update", "used":0, "size":1_048_576,
        "_meta":{"aster":{"usagePending":true, "usageAvailable":false}}
    })));
    assert_eq!(controller.state.context_window, Some(1_048_576));
    assert!(controller.state.context_snapshot.is_none());
    assert!(controller.state.context_ratio.is_none());
    for width in [24, 48, 80, 120] {
        let rendered = lines(&controller.state, width, 1)[0].clone();
        assert!(rendered.contains("0k / 1.0m"), "{rendered}");
        assert_eq!(rendered.contains('░'), width >= 40, "{rendered}");
        assert!(!rendered.contains('█'), "{rendered}");
    }
    controller.state.hud.config.context_bar = false;
    assert!(!lines(&controller.state, 80, 1)[0].contains('░'));
    controller.state.hud.config.context_bar = true;
    controller.state.hud.config.language = "zh".into();
    assert!(lines(&controller.state, 80, 1)[0].contains("上下文    ░░░░░░ 0k / 1.0m"));
    controller.message(update(json!({
        "sessionUpdate":"usage_update", "used":1234, "size":1_048_576,
        "cost":{"amount":0.25, "currency":"USD"}
    })));
    assert_eq!(
        controller.state.context_snapshot.as_ref().unwrap().used,
        Some(1234)
    );
    assert!(lines(&controller.state, 80, 1)[0].contains("1.2k / 1.0m"));
    controller.message(update(json!({
        "sessionUpdate":"usage_update", "used":0, "size":0,
        "_meta":{"aster":{"usageAvailable":false}}
    })));
    let rendered = lines(&controller.state, 80, 1)[0].clone();
    assert!(rendered.contains("上下文    n/a"), "{rendered}");
    assert!(!rendered.contains("0k /"));
    assert!(!rendered.contains('░'));
}

#[test]
fn model_switch_uses_new_capacity_before_its_first_reply_and_keeps_cost() {
    let mut state = state();
    state.session_id = Some("session".into());
    state.hud.config.layout = Layout::Expanded;
    state.hud.config.elements = vec!["usage".into()];
    let mut controller = Controller::new(state, None, false);
    controller.message(json!({"method":"session/update", "params":{
        "sessionId":"session", "update":{
            "sessionUpdate":"current_mode_update", "currentModeId":"code",
            "_meta":{"aster":{"runtime":{
                "model":"new-model", "contextWindow":2_000_000,
                "reasoningLevel":null, "reasoningLevels":null
            }}}
        }
    }}));
    assert_eq!(controller.state.context_window, Some(2_000_000));
    assert!(controller.state.context_snapshot.is_none());
    assert!(controller.state.context_ratio.is_none());
    let rendered = lines(&controller.state, 80, 1)[0].clone();
    assert!(rendered.contains("0k / 2.0m"), "{rendered}");
    assert!(rendered.contains("░░░░░░"), "{rendered}");
    assert!(rendered.contains("$0.1234"), "{rendered}");
}

#[test]
fn clearing_reasoning_does_not_reuse_stale_effort_telemetry() {
    let mut state = state();
    state.hud.config.elements = vec!["reasoning".into()];
    state.runtime = json!({"reasoningLevel":"low"});
    assert!(lines(&state, 80, 1)[0].contains("Reasoning  low"));
    state.runtime["reasoningLevel"] = Value::Null;
    assert!(lines(&state, 80, 1)[0].contains("Reasoning  model default"));
    assert_eq!(hud::rows(&state, 80, 1)[0].command, "/effort");
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
fn full_hud_shows_all_available_rows_after_sending_message() {
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.phase = Phase::Ready;
    state.session_id = Some("session".into());
    state.hud.configure("full").unwrap();
    let mut controller = Controller::new(state, None, false);
    controller.paste("检查布局".into());
    let actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(matches!(
        actions.as_slice(),
        [Action::Request {
            method: "session/prompt",
            ..
        }]
    ));
    assert!(matches!(controller.state.entries[0], Entry::User(_)));
    assert_eq!(controller.state.hud.config.layout, Layout::Expanded);
    assert_eq!(controller.state.hud.config.max_rows, 16);
    let expected = hud::rows(&controller.state, 76, 16);
    for height in [96, 24, 18] {
        let mut terminal = Terminal::new(TestBackend::new(80, height)).unwrap();
        let mut hits = ui::HitAreas::default();
        terminal
            .draw(|frame| hits = ui::draw(frame, &mut controller.state))
            .unwrap();
        assert_eq!(
            hits.hud_rows.len(),
            expected.len(),
            "sending a message reduced the full HUD to {} rows in 80x{height}",
            hits.hud_rows.len(),
        );
    }
}

#[test]
fn full_hud_stays_expanded_through_input_and_response() {
    for (columns, height) in [(80, 24), (120, 36)] {
        let mut state = State::new("workspace".into(), "model".into(), false);
        state.phase = Phase::Ready;
        state.session_id = Some("session".into());
        state.hud.configure("full").unwrap();
        let mut controller = Controller::new(state, None, false);
        let check = |controller: &mut Controller, stage: &str| {
            let expected = hud::rows(&controller.state, columns as usize - 4, 16);
            let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
            let mut hits = ui::HitAreas::default();
            terminal
                .draw(|frame| hits = ui::draw(frame, &mut controller.state))
                .unwrap();
            assert_eq!(controller.state.hud.config.layout, Layout::Expanded);
            assert_eq!(controller.state.hud.config.max_rows, 16);
            assert_eq!(
                hits.hud_rows.len(),
                expected.len(),
                "full HUD collapsed during {stage} in {columns}x{height}",
            );
            for ((_, command), row) in hits.hud_rows.iter().zip(expected) {
                assert_eq!(*command, row.command);
            }
            assert!(hits.transcript.height >= 6);
            assert!(hits.transcript.bottom() <= hits.composer.y);
            assert!(hits
                .hud_rows
                .iter()
                .all(|(area, _)| { area.y >= hits.composer.bottom() && area.bottom() < height }));
        };
        check(&mut controller, "welcome");
        controller.paste("检查布局".into());
        check(&mut controller, "typing");
        let actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(
            actions.as_slice(),
            [Action::Request {
                method: "session/prompt",
                ..
            }]
        ));
        controller.registered(1, Pending::Prompt);
        check(&mut controller, "sending");
        for (stage, update) in [
            (
                "running a tool",
                json!({"sessionUpdate":"tool_call", "toolCallId":"tool",
                    "title":"Check layout", "status":"in_progress"}),
            ),
            (
                "finishing a tool",
                json!({"sessionUpdate":"tool_call_update", "toolCallId":"tool",
                    "status":"completed"}),
            ),
            (
                "receiving a response",
                json!({"sessionUpdate":"agent_message_chunk",
                    "content":{"type":"text", "text":"布局检查完成。"}}),
            ),
        ] {
            controller.message(json!({"method":"session/update",
                "params":{"sessionId":"session", "update":update}}));
            check(&mut controller, stage);
        }
        controller.message(json!({"id":1, "result":{"stopReason":"end_turn"}}));
        assert_eq!(controller.state.phase, Phase::Ready);
        check(&mut controller, "response complete");
    }
}

#[test]
fn full_hud_preserves_conversation_space_in_short_windows() {
    let mut state = state();
    state.hud.configure("full").unwrap();
    state.entries.push(Entry::Assistant(
        (1..=30)
            .map(|row| format!("Conversation line {row}\n"))
            .collect(),
    ));
    for layout in [Layout::Compact, Layout::Expanded] {
        state.hud.config.layout = layout;
        for (height, max_hud_rows) in [(12, 1), (18, 2), (24, 3), (32, 4), (48, 6), (96, 12)] {
            for columns in [24, 48, 80, 120] {
                let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
                let mut hits = ui::HitAreas::default();
                terminal
                    .draw(|frame| hits = ui::draw(frame, &mut state))
                    .unwrap();
                let body_reserve = if layout == Layout::Compact {
                    height / 2
                } else if height >= 24 {
                    6
                } else {
                    2
                };
                assert!(
                    hits.transcript.height >= body_reserve,
                    "only {} conversation rows remain with {} HUD rows in {columns}x{height}",
                    hits.transcript.height,
                    hits.hud_rows.len(),
                );
                if layout == Layout::Compact {
                    assert!(hits.hud_rows.len() <= max_hud_rows);
                } else if height >= 18 {
                    assert_eq!(
                        hits.hud_rows.len(),
                        hud::rows(&state, hits.transcript.width as usize, 16).len(),
                    );
                }
                assert!(hits
                    .hud_rows
                    .iter()
                    .any(|(_, command)| *command == "/context"));
                assert!(hits.transcript.bottom() <= hits.composer.y);
                assert_eq!(hits.hud_rows.last().unwrap().0.bottom(), height - 1);
            }
        }
    }
}

#[test]
fn welcome_with_full_hud_keeps_the_complete_wordmark_visible() {
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.phase = Phase::Ready;
    state.hud.configure("full").unwrap();
    let logo = [
        " █████╗  ███████╗ ████████╗ ███████╗ ██████╗ ",
        "██╔══██╗ ██╔════╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗",
        "███████║ ███████╗    ██║    █████╗   ██████╔╝",
        "██╔══██║ ╚════██║    ██║    ██╔══╝   ██╔══██╗",
        "██║  ██║ ███████║    ██║    ███████╗ ██║  ██║",
        "╚═╝  ╚═╝ ╚══════╝    ╚═╝    ╚══════╝ ╚═╝  ╚═╝",
    ];
    for layout in [Layout::Compact, Layout::Expanded] {
        state.hud.config.layout = layout;
        for monochrome in [false, true] {
            state.monochrome = monochrome;
            for (columns, height) in [(80, 24), (48, 26), (80, 26), (120, 32), (160, 48)] {
                let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
                let mut hits = ui::HitAreas::default();
                terminal
                    .draw(|frame| hits = ui::draw(frame, &mut state))
                    .unwrap();
                let text: Vec<String> = terminal
                    .backend()
                    .buffer()
                    .content
                    .chunks(columns as usize)
                    .map(|row| row.iter().map(|cell| cell.symbol()).collect())
                    .collect();
                let welcome = &text[hits.transcript.y as usize..hits.transcript.bottom() as usize];
                for logo_line in logo {
                    assert!(
                        welcome.iter().any(|line| line.trim() == logo_line.trim()),
                        "missing logo row with full HUD in {columns}x{height}:\n{}",
                        text.join("\n"),
                    );
                }
                assert_eq!(hits.welcome_rows.len(), 4);
                assert!(hits
                    .welcome_rows
                    .iter()
                    .all(|(area, _)| area.x >= hits.transcript.x
                        && area.y >= hits.transcript.y
                        && area.right() <= hits.transcript.right()
                        && area.bottom() <= hits.transcript.bottom()));
                assert!(hits
                    .hud_rows
                    .iter()
                    .all(|(area, _)| area.y >= hits.composer.bottom()));
                assert_eq!(state.hud.config.elements.len(), hud::config::ELEMENTS.len());
                assert_eq!(state.hud.config.max_rows, 16);
                assert_eq!(state.hud.config.layout, layout);
            }
        }
    }
}

#[test]
fn welcome_with_full_hud_shows_every_available_status_row() {
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.phase = Phase::Ready;
    state.hud.configure("full").unwrap();
    let expected = hud::rows(&state, 76, 16);
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    let mut hits = ui::HitAreas::default();
    terminal
        .draw(|frame| hits = ui::draw(frame, &mut state))
        .unwrap();
    assert_eq!(hits.hud_rows.len(), expected.len());
    for ((area, command), row) in hits.hud_rows.iter().zip(expected) {
        let text: String = (area.x..area.right())
            .map(|x| terminal.backend().buffer()[(x, area.y)].symbol())
            .collect();
        assert!(text.starts_with(&row.line.to_string()));
        assert_eq!(*command, row.command);
    }
    assert_eq!(hits.welcome_rows.len(), 4);
    assert_eq!(hits.welcome_rows[0].0.y, hits.welcome_rows[1].0.y);
    assert_eq!(hits.welcome_rows[2].0.y, hits.welcome_rows[3].0.y);
    assert!(hits.welcome_rows[0].0.right() <= hits.welcome_rows[1].0.x);
    assert!(hits.welcome_rows[2].0.right() <= hits.welcome_rows[3].0.x);
}

#[test]
fn small_welcome_keeps_a_compact_wordmark_and_all_menu_actions() {
    let mut state = State::new("workspace".into(), "model".into(), false);
    state.phase = Phase::Ready;
    state.hud.configure("full").unwrap();
    for (columns, height) in [(24, 12), (32, 18), (44, 24), (80, 12), (80, 16), (80, 18)] {
        let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
        let mut hits = ui::HitAreas::default();
        terminal
            .draw(|frame| hits = ui::draw(frame, &mut state))
            .unwrap();
        let text: Vec<String> = terminal
            .backend()
            .buffer()
            .content
            .chunks(columns as usize)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();
        let welcome = &text[hits.transcript.y as usize..hits.transcript.bottom() as usize];
        assert!(welcome.iter().any(|line| line.trim() == "ASTER"));
        assert_eq!(hits.welcome_rows.len(), 4);
        assert!(hits
            .welcome_rows
            .iter()
            .all(|(area, _)| area.bottom() <= hits.transcript.bottom()));
    }
}

#[test]
fn default_hud_stays_compact_when_tools_and_jobs_are_active() {
    let mut state = state();
    state.phase = Phase::Busy;
    state.tasks = json!({"steps":[{"status":"pending","title":"check"}],
        "jobs":[{"status":"running","title":"review"}]});
    state.entries.push(Entry::Tool(Tool {
        id: "one".into(),
        title: "Inspect source".into(),
        status: "running".into(),
        content: "Source contents".into(),
        diff: String::new(),
        expanded: false,
        started: None,
        elapsed: None,
    }));
    for (columns, height) in [(48, 24), (80, 24), (120, 24), (160, 40)] {
        let mut terminal = Terminal::new(TestBackend::new(columns, height)).unwrap();
        let mut hits = ui::HitAreas::default();
        terminal
            .draw(|frame| hits = ui::draw(frame, &mut state))
            .unwrap();
        assert!(hits.hud_rows.len() <= 3);
        assert!(hits.transcript.height >= height / 2);
        assert!(height - hits.composer.bottom() <= 4);
        if columns >= 120 {
            assert_eq!(hits.hud_rows[0].1, "/context");
            let y = hits.hud_rows[0].0.y;
            let summary: String = (0..columns)
                .map(|x| terminal.backend().buffer()[(x, y)].symbol())
                .collect();
            assert!(summary.contains("project") && summary.contains("86%"));
        }
    }
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
        "Limits     n/a",
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
    state.hud.configure("full").unwrap();
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
    let rows = hud::rows(&state, 80, 16);
    assert!(rows
        .iter()
        .flat_map(|row| &row.line.spans)
        .any(|span| span.style.fg == Some(Color::Rgb(0x12, 0x34, 0x56))));
    assert!(!rows
        .iter()
        .any(|row| row.line.to_string().contains(['\x1b', '\n'])));
    state.monochrome = true;
    for row in hud::rows(&state, 80, 16) {
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
