// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Usage indicators through the same session messages and renderer as the terminal.

use aster_tui::{
    controller::{Action, Controller},
    state::{Overlay, Phase, State},
    ui,
};
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::{
    backend::TestBackend,
    buffer::Buffer,
    layout::{Position, Rect},
    style::{Color, Modifier},
    Terminal,
};
use serde_json::{json, Value};
use unicode_width::UnicodeWidthStr;

struct Client {
    controller: Controller,
    next_id: u64,
}

impl Client {
    fn new() -> Self {
        Self {
            controller: Controller::new(
                State::new("C:/test-workspace".into(), "model-a".into(), false),
                None,
                false,
            ),
            next_id: 1,
        }
    }

    fn ready() -> Self {
        let mut client = Self::new();
        let action = client.controller.initialize();
        let id = client.register(vec![action], "initialize");
        let actions = client.answer(
            id,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "close": {}}
                },
                "_meta": {"aster": {"runtimeControls": true}}
            }),
        );
        let id = client.register(actions, "session/list");
        let actions = client.answer(id, json!({"sessions": []}));
        let id = client.register(actions, "session/new");
        let actions = client.answer(id, json!({"sessionId": "session-a"}));
        client.finish_runtime(actions, "model-a");
        assert_eq!(client.controller.state.phase, Phase::Ready);
        client
    }

    fn register(&mut self, actions: Vec<Action>, expected: &str) -> u64 {
        assert_eq!(actions.len(), 1, "expected {expected}: {actions:?}");
        let Action::Request {
            method, pending, ..
        } = actions.into_iter().next().unwrap()
        else {
            panic!("expected {expected} request");
        };
        assert_eq!(method, expected);
        let id = self.next_id;
        self.next_id += 1;
        self.controller.registered(id, pending);
        id
    }

    fn answer(&mut self, id: u64, result: Value) -> Vec<Action> {
        self.controller.message(json!({"id": id, "result": result}))
    }

    fn fail(&mut self, id: u64) {
        assert!(self
            .controller
            .message(json!({
                "id": id,
                "error": {"code": -32603, "message": "fixture failure"}
            }))
            .is_empty());
    }

    fn finish_runtime(&mut self, actions: Vec<Action>, model: &str) {
        let id = self.register(actions, "_aster/session/runtime");
        assert!(self.answer(id, runtime(model, "code", "ask")).is_empty());
    }

    fn update(&mut self, session: &str, update: Value) {
        assert!(self
            .controller
            .message(json!({
                "method": "session/update",
                "params": {"sessionId": session, "update": update}
            }))
            .is_empty());
    }

    fn header(&mut self, columns: u16, rows: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
        let mut usage_row = None;
        terminal
            .draw(|frame| {
                usage_row = ui::draw(frame, &mut self.controller.state)
                    .hud_rows
                    .into_iter()
                    .find(|(_, command)| *command == "/context")
                    .map(|(area, _)| area.y);
            })
            .unwrap();
        let start = usage_row.expect("visible HUD usage row") as usize * columns as usize;
        terminal.backend().buffer().content[start..start + columns as usize]
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    fn seed_usage(&mut self) {
        self.update("session-a", usage(18_000, 100_000, Some(0.0024)));
    }

    fn press(&mut self, code: KeyCode) -> Vec<Action> {
        self.controller.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn inspect_selected(&mut self, request_id: &str, category: &str, key: Option<&str>) -> u64 {
        let actions = self.press(KeyCode::Enter);
        let Some(Action::Request { params, .. }) = actions.first() else {
            panic!("expected a context content request: {actions:?}");
        };
        assert_eq!(
            params,
            &json!({
                "sessionId": "session-a", "requestId": request_id,
                "category": category, "key": key
            })
        );
        self.register(actions, "_aster/session/context")
    }

    fn screen(&mut self, columns: u16, rows: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
        terminal
            .draw(|frame| {
                ui::draw(frame, &mut self.controller.state);
            })
            .unwrap();
        terminal.backend().buffer().clone()
    }
}

fn runtime(model: &str, mode: &str, approval: &str) -> Value {
    json!({
        "model": model,
        "mode": mode,
        "approval": approval,
        "executionBackend": "native"
    })
}

fn usage(used: u64, size: u64, cost: Option<f64>) -> Value {
    let mut value = json!({"sessionUpdate": "usage_update", "used": used, "size": size});
    if let Some(cost) = cost {
        value["cost"] = json!({"amount": cost, "currency": "USD"});
    }
    value
}

fn composition(used: u64, label: &str) -> Value {
    let mut value = usage(used, 100_000, Some(0.2));
    value["_meta"] = json!({"aster": {"context": {
        "version": 1, "model": "model-a", "windowTokens": 100_000,
        "reservedOutputTokens": 5000, "attribution": "characters",
        "categories": [
            {"id": "system_prompt", "tokens": used / 4},
            {"id": "messages", "tokens": used - used / 4}
        ],
        "items": [
            {"category": "system_prompt", "tokens": used / 4, "key": "system_prompt"},
            {"category": "messages", "tokens": used - used / 4, "key": label}
        ]
    }}});
    value
}

fn inspectable_composition(request_id: &str) -> Value {
    let mut value = composition(20_000, "Conversation text");
    let metadata = &mut value["_meta"]["aster"]["context"];
    metadata["requestId"] = json!(request_id);
    metadata["contentItems"] = json!([
        {"category": "system_prompt", "key": "system_prompt", "chars": 80},
        {"category": "messages", "key": "Conversation text", "chars": 240}
    ]);
    metadata["outputAvailable"] = json!(true);
    value
}

fn content_response(request_id: &str, category: &str, key: Option<&str>, text: &str) -> Value {
    json!({
        "requestId": request_id, "category": category, "key": key,
        "source": if category == "output" { "model_response" } else { "model_request" },
        "sections": [{
            "key": key.unwrap_or(if category == "output" { "Assistant text" } else { category }),
            "text": text, "characters": text.chars().count(), "truncated": false
        }]
    })
}

fn screen_text(buffer: &Buffer, columns: u16) -> String {
    buffer
        .content
        .chunks(columns as usize)
        .map(|row| {
            let mut text = String::new();
            let mut continuation = 0;
            for cell in row {
                if continuation > 0 {
                    continuation -= 1;
                    continue;
                }
                text.push_str(cell.symbol());
                if !cell.symbol().contains('\x1b') {
                    continuation = cell.symbol().width().saturating_sub(1);
                }
            }
            text
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_context_label(header: &str) {
    assert!(
        header.contains("context") || header.contains("ctx"),
        "{header}"
    );
}

fn assert_pending(header: &str) {
    assert_context_label(header);
    assert!(header.contains("cost"), "{header}");
    assert!(header.matches('—').count() >= 2, "{header}");
    assert!(!header.contains('%') && !header.contains('$'), "{header}");
    assert!(!header.contains("n/a"), "{header}");
}

fn assert_known(header: &str, percent: &str, cost: &str) {
    assert_context_label(header);
    assert!(header.contains(percent), "{header}");
    assert!(header.contains(cost), "{header}");
}

#[test]
fn startup_and_ready_headers_show_pending_indicators_at_supported_widths() {
    for mut client in [Client::new(), Client::ready()] {
        for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
            assert_pending(&client.header(columns, rows));
        }
    }
}

#[test]
fn usage_notifications_render_real_context_and_cost_even_on_narrow_screens() {
    let mut client = Client::ready();
    client.seed_usage();
    for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        assert_known(&client.header(columns, rows), "18%", "$0.0024");
    }

    client.update("session-a", usage(0, 100_000, Some(0.0)));
    assert_known(&client.header(80, 24), "0%", "$0.0000");
}

#[test]
fn unknown_context_window_preserves_real_tokens_and_cost() {
    let mut client = Client::ready();
    client.update("session-a", usage(18_000, 0, Some(0.0024)));
    for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        let header = client.header(columns, rows);
        assert!(header.contains("18000"), "{header}");
        assert!(header.contains("tok"), "{header}");
        assert!(header.contains("$0.0024"), "{header}");
        assert!(!header.contains('%') && !header.contains("/ 0"), "{header}");
    }
    assert!(client.controller.state.context_ratio.is_none());
}

#[test]
fn missing_or_invalid_cost_replaces_the_previous_amount_with_unavailable() {
    for cost in [
        None,
        Some(Value::Null),
        Some(json!({"amount": -0.25, "currency": "USD"})),
        Some(json!({"amount": "unknown", "currency": "USD"})),
        Some(json!({"amount": null, "currency": "USD"})),
    ] {
        let mut client = Client::ready();
        client.seed_usage();
        let mut update = usage(25_000, 100_000, None);
        if let Some(cost) = cost {
            update["cost"] = cost;
        }
        client.update("session-a", update);
        let header = client.header(80, 24);
        assert!(header.contains("25%"), "{header}");
        assert!(header.contains("cost n/a"), "{header}");
        assert!(!header.contains('$'), "{header}");
    }
}

#[test]
fn known_session_subtotals_have_a_marker_until_accounting_is_complete() {
    let mut client = Client::ready();
    let mut update = usage(100, 1000, None);
    update["_meta"] = json!({"aster": {"hud": {"costComplete": false, "costSubtotal": 0.25}}});
    client.update("session-a", update);
    let header = client.header(100, 24);
    assert!(header.contains("≥$0.2500"), "{header}");
    client.update("session-a", usage(100, 1000, Some(0.5)));
    let header = client.header(100, 24);
    assert!(header.contains("$0.5000"), "{header}");
    assert!(!header.contains('≥'), "{header}");
}

#[test]
fn session_subtotals_require_a_valid_amount_and_explicit_incompleteness() {
    for hud in [
        json!({"costComplete": false}),
        json!({"costComplete": false, "costSubtotal": -1}),
        json!({"costComplete": false, "costSubtotal": "0.25"}),
        json!({"costSubtotal": 0.25}),
        json!({"costComplete": true, "costSubtotal": 0.25}),
    ] {
        let mut client = Client::ready();
        let mut update = usage(100, 1000, None);
        update["_meta"] = json!({"aster": {"hud": hud}});
        client.update("session-a", update);
        let header = client.header(100, 24);
        assert!(header.contains("cost n/a"), "{header}");
        assert!(!header.contains('$'), "{header}");
    }
}

#[test]
fn responses_without_usage_show_unavailable_instead_of_fake_zero_or_stale_values() {
    let mut client = Client::ready();
    client.seed_usage();
    let mut update = usage(0, 0, None);
    update["_meta"] = json!({"aster": {"usageAvailable": false}});
    client.update("session-a", update);
    for (columns, rows) in [(24, 6), (40, 12), (80, 24), (120, 36)] {
        let header = client.header(columns, rows);
        assert_context_label(&header);
        assert!(header.matches("n/a").count() >= 2, "{header}");
        assert!(!header.contains('%') && !header.contains('$'), "{header}");
        assert!(!header.contains("tok"), "{header}");
    }
    assert!(client.controller.state.context_ratio.is_none());
}

#[test]
fn malformed_token_counts_do_not_become_a_zero_percent_context() {
    for used in [Value::Null, json!(-1), json!("unknown")] {
        let mut client = Client::ready();
        client.seed_usage();
        let mut update = usage(0, 100_000, Some(0.003));
        update["used"] = used;
        client.update("session-a", update);
        let header = client.header(80, 24);
        assert_context_label(&header);
        assert!(header.contains("n/a"), "{header}");
        assert!(header.contains("$0.0030"), "{header}");
        assert!(!header.contains('%'), "{header}");
        assert!(client.controller.state.context_ratio.is_none());
    }
}

#[test]
fn new_conversation_clears_usage_and_ignores_notifications_from_the_closed_session() {
    let mut client = Client::ready();
    client.seed_usage();
    let actions = client.controller.input("/new".into());
    let close = client.register(actions, "session/close");
    assert_known(&client.header(80, 24), "18%", "$0.0024");

    let actions = client.answer(close, json!({}));
    let new = client.register(actions, "session/new");
    assert_pending(&client.header(80, 24));
    client.update("session-a", usage(99_000, 100_000, Some(9.0)));
    assert_pending(&client.header(80, 24));

    let actions = client.answer(new, json!({"sessionId": "session-b"}));
    client.finish_runtime(actions, "model-a");
    assert_pending(&client.header(80, 24));
    client.update("session-a", usage(99_000, 100_000, Some(9.0)));
    assert_pending(&client.header(80, 24));
    client.update("session-b", usage(5_000, 100_000, Some(0.001)));
    assert_known(&client.header(80, 24), "5%", "$0.0010");
}

#[test]
fn successful_load_keeps_only_target_usage_and_never_the_sources_buffered_values() {
    for target_has_usage in [false, true] {
        let mut client = Client::ready();
        client.seed_usage();
        let actions = client.controller.resume("session-b".into());
        let load = client.register(actions, "session/load");
        assert_pending(&client.header(80, 24));
        client.update("session-a", usage(25_000, 100_000, Some(0.5)));
        assert_pending(&client.header(80, 24));
        if target_has_usage {
            client.update("session-b", usage(5_000, 100_000, Some(0.75)));
        }

        let actions = client.answer(load, json!({}));
        let close = client.register(actions, "session/close");
        let actions = client.answer(close, json!({}));
        client.finish_runtime(actions, "model-a");
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("session-b")
        );
        client.update("session-a", usage(99_000, 100_000, Some(9.0)));
        client.update("unrelated-session", usage(99_000, 100_000, Some(9.0)));
        let header = client.header(80, 24);
        if target_has_usage {
            assert_known(&header, "5%", "$0.7500");
        } else {
            assert_pending(&header);
        }
    }
}

#[test]
fn failed_load_restores_source_usage_including_updates_received_during_replay() {
    for source_updates in [false, true] {
        let mut client = Client::ready();
        client.seed_usage();
        let actions = client.controller.resume("session-b".into());
        let load = client.register(actions, "session/load");
        client.update("session-b", usage(5_000, 100_000, Some(0.75)));
        assert_known(&client.header(80, 24), "5%", "$0.7500");
        if source_updates {
            client.update("session-a", usage(25_000, 100_000, Some(0.5)));
            assert_known(&client.header(80, 24), "5%", "$0.7500");
        }

        client.fail(load);
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("session-a")
        );
        let (percent, cost) = if source_updates {
            ("25%", "$0.5000")
        } else {
            ("18%", "$0.0024")
        };
        assert_known(&client.header(80, 24), percent, cost);
        client.update("session-b", usage(99_000, 100_000, Some(9.0)));
        assert_known(&client.header(80, 24), percent, cost);
    }
}

#[test]
fn successful_model_change_clears_context_but_keeps_session_cost() {
    let mut client = Client::ready();
    client.seed_usage();
    let actions = client.controller.input("/model model-b".into());
    let configure = client.register(actions, "_aster/session/configure");
    assert_known(&client.header(80, 24), "18%", "$0.0024");
    assert!(client
        .answer(configure, runtime("model-b", "code", "ask"))
        .is_empty());

    let header = client.header(80, 24);
    assert_context_label(&header);
    assert!(header.contains('—'), "{header}");
    assert!(header.contains("$0.0024"), "{header}");
    assert!(!header.contains('%'), "{header}");
    assert!(client.controller.state.context_ratio.is_none());

    client.update("session-a", usage(10_000, 200_000, Some(0.004)));
    assert_known(&client.header(80, 24), "5%", "$0.0040");
}

fn hud_usage(stream: &str, sequence: u64) -> Value {
    let mut update = usage(18_000, 100_000, Some(sequence as f64));
    update["_meta"] = json!({"aster":{"hud":{
        "streamId":stream.repeat(32), "sequence":sequence,
        "tokens":{"input":100 * sequence},
        "days":{"2026-11-15":{"cost":sequence,"complete":true,"calls":sequence}},
        "latest":{"model":"model-a","speed":12.0}
    }}});
    update
}

#[test]
fn hud_switching_sessions_restores_the_right_metrics_and_keeps_source_accounting() {
    for succeeds in [false, true] {
        let mut client = Client::ready();
        client.update("session-a", hud_usage("a", 1));
        let actions = client.controller.resume("session-b".into());
        let load = client.register(actions, "session/load");
        assert!(client.controller.state.hud.telemetry.stream_id.is_empty());
        client.update("session-b", hud_usage("b", 1));
        client.update("session-a", hud_usage("a", 2));
        assert_eq!(
            client.controller.state.hud.telemetry.stream_id,
            "b".repeat(32)
        );
        let queued = client
            .controller
            .state
            .hud
            .collect_request(".", true)
            .unwrap()
            .records;
        assert_eq!(
            queued
                .iter()
                .find(|record| record.stream_id == "a".repeat(32))
                .unwrap()
                .sequence,
            2
        );
        if succeeds {
            let actions = client.answer(load, json!({}));
            let close = client.register(actions, "session/close");
            let actions = client.answer(close, json!({}));
            client.finish_runtime(actions, "model-a");
        } else {
            client.fail(load);
        }
        assert_eq!(
            client.controller.state.hud.telemetry.stream_id,
            if succeeds { "b" } else { "a" }.repeat(32)
        );
        assert_eq!(
            client.controller.state.hud.telemetry.sequence,
            if succeeds { 1 } else { 2 }
        );
        client.update("unrelated", hud_usage("c", 99));
        assert_ne!(
            client.controller.state.hud.telemetry.stream_id,
            "c".repeat(32)
        );
    }
}

#[test]
fn hud_model_change_clears_request_rate_while_preserving_session_tokens() {
    let mut client = Client::ready();
    client.update("session-a", hud_usage("a", 2));
    let actions = client.controller.input("/model model-b".into());
    let configure = client.register(actions, "_aster/session/configure");
    client.answer(configure, runtime("model-b", "code", "ask"));
    assert_eq!(
        client.controller.state.hud.telemetry.tokens.input,
        Some(200)
    );
    assert!(client.controller.state.hud.telemetry.latest.speed.is_none());
}

#[test]
fn hud_accounts_for_source_usage_until_its_close_completes() {
    let mut client = Client::ready();
    client.controller.state.tasks = json!({"jobs":[{
        "id":"background-call", "status":"running", "title":"Background model call"
    }]});
    client.update("session-a", hud_usage("a", 1));
    let actions = client.controller.resume("session-b".into());
    let load = client.register(actions, "session/load");
    client.update("session-b", hud_usage("b", 1));
    client.update("session-a", hud_usage("a", 2));
    let actions = client.answer(load, json!({}));
    let close = client.register(actions, "session/close");

    client.update("session-a", hud_usage("a", 3));
    client.update("unrelated", hud_usage("c", 99));
    let actions = client.answer(close, json!({}));
    client.finish_runtime(actions, "model-a");
    // Once the close has completed, stale notifications must remain ignored.
    client.update("session-a", hud_usage("a", 4));

    let hud = &mut client.controller.state.hud;
    assert_eq!(hud.telemetry.stream_id, "b".repeat(32));
    assert_eq!(hud.telemetry.sequence, 1);
    let records = hud.collect_request(".", true).unwrap().records;
    assert_eq!(records.len(), 2);
    let source = records
        .iter()
        .find(|record| record.stream_id == "a".repeat(32))
        .unwrap();
    assert_eq!(source.sequence, 3);
    assert_eq!(source.days["2026-11-15"].cost, 3.0);
    assert_known(&client.header(80, 24), "18%", "$1.0000");
}

#[test]
fn failed_model_change_and_non_model_settings_preserve_current_metrics() {
    let mut client = Client::ready();
    client.seed_usage();
    let actions = client.controller.input("/model unavailable".into());
    let configure = client.register(actions, "_aster/session/configure");
    client.fail(configure);
    assert_eq!(client.controller.state.model, "model-a");
    assert_known(&client.header(80, 24), "18%", "$0.0024");

    for (command, mode, approval) in [
        ("/mode plan", "plan", "ask"),
        ("/permissions session", "plan", "session"),
        ("/model model-a", "plan", "session"),
    ] {
        let actions = client.controller.input(command.into());
        let configure = client.register(actions, "_aster/session/configure");
        assert!(client
            .answer(configure, runtime("model-a", mode, approval))
            .is_empty());
        assert_known(&client.header(80, 24), "18%", "$0.0024");
    }
}

#[test]
fn context_is_local_and_navigation_preserves_the_unsubmitted_draft() {
    for working in [false, true] {
        let mut client = Client::ready();
        client.update("session-a", composition(20_000, "Latest messages"));
        if working {
            let actions = client.controller.input("Run a task".into());
            client.register(actions, "session/prompt");
            assert!(client.controller.state.busy());
        }
        client.controller.state.editor.insert_str("未发送的草稿");
        assert!(client.controller.input("/context".into()).is_empty());
        assert_eq!(client.controller.state.overlay, Some(Overlay::Context));
        assert!(!client.controller.state.context_expanded);
        for code in [
            KeyCode::Tab,
            KeyCode::PageDown,
            KeyCode::Down,
            KeyCode::Up,
            KeyCode::Esc,
        ] {
            assert!(client
                .controller
                .key(KeyEvent::new(code, KeyModifiers::NONE))
                .is_empty());
        }
        assert!(client.controller.state.context_expanded);
        assert_eq!(client.controller.state.overlay, None);
        assert_eq!(client.controller.state.editor.lines(), ["未发送的草稿"]);
        assert_eq!(client.controller.state.busy(), working);
        assert!(client.controller.input("/context all".into()).is_empty());
        assert!(client.controller.state.context_expanded);
        assert_eq!(client.controller.state.editor.lines(), ["未发送的草稿"]);
    }
}

#[test]
fn context_can_be_opened_from_the_composer_while_a_request_is_running() {
    let mut client = Client::ready();
    let actions = client.controller.input("Run a task".into());
    client.register(actions, "session/prompt");
    client.controller.state.editor.insert_str("/context all");
    assert!(client
        .controller
        .key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .is_empty());
    assert_eq!(client.controller.state.overlay, Some(Overlay::Context));
    assert!(client.controller.state.context_expanded);
    assert!(client.controller.state.busy());
}

#[test]
fn context_grid_renders_colors_legend_details_and_a_fixed_capacity() {
    let mut client = Client::ready();
    client.controller.state.monochrome = false;
    client.update("session-a", composition(20_000, "Conversation text"));
    client.controller.input("/context".into());
    let buffer = client.screen(120, 40);
    let text = screen_text(&buffer, 120);
    let cell_count = |row: &str| row.chars().filter(|c| *c == '■').count();
    assert_eq!(
        text.lines()
            .filter(|row| cell_count(row) > 1)
            .map(cell_count)
            .sum::<usize>(),
        100
    );
    assert!(
        text.contains("■■■■■"),
        "character cells have no inserted gaps"
    );
    for label in [
        "Context usage",
        "20,000 / 100,000 tokens",
        "System prompt",
        "Messages",
        "~5,000",
        "Output reserve",
        "Free space",
        "Tab details",
    ] {
        assert!(text.contains(label), "{label}: {text}");
    }
    let mut colors = Vec::new();
    for cell in buffer
        .content
        .iter()
        .filter(|cell| cell.symbol() == "■")
        .take(20)
    {
        if !colors.contains(&cell.fg) {
            colors.push(cell.fg);
        }
    }
    assert_eq!(colors.len(), 2);
    client
        .controller
        .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    let text = screen_text(&client.screen(120, 40), 120);
    assert!(text.contains("Context blocks and messages"), "{text}");
    assert!(text.contains("Conversation text"), "{text}");
    client.controller.state.monochrome = true;
    for expanded in [false, true] {
        client.controller.state.context_expanded = expanded;
        let buffer = client.screen(120, 40);
        for cell in &buffer.content {
            assert_eq!(cell.fg, Color::Reset, "monochrome foreground: {cell:?}");
            assert_eq!(cell.bg, Color::Reset, "monochrome background: {cell:?}");
        }
    }
}

#[test]
fn context_scrolls_and_handles_small_terminals_and_unavailable_usage() {
    let mut client = Client::ready();
    client.update("session-a", composition(20_000, "Conversation text"));
    client.controller.input("/context all".into());
    for (columns, rows) in [(1, 1), (8, 4), (24, 6), (40, 12), (80, 24), (120, 36)] {
        client.screen(columns, rows);
        client
            .controller
            .key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
        client.screen(columns, rows);
        client
            .controller
            .key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(client.controller.state.detail_scroll, 0);
    }
    client.controller.state.detail_scroll = 0;
    client.update(
        "session-a",
        json!({"sessionUpdate": "usage_update", "used": 0, "size": 0,
        "_meta": {"aster": {"usageAvailable": false}}}),
    );
    let text = screen_text(&client.screen(120, 36), 120);
    assert!(text.contains("Input-token usage unavailable"), "{text}");
    assert!(!text.contains('■'), "{text}");
}

fn native_context_client() -> Client {
    let mut client = Client::ready();
    client.controller.state.monochrome = false;
    client.controller.state.terminal_graphics = aster_tui::terminal_input::GraphicsCapabilities {
        sixel: true,
        cell_pixels: aster_tui::terminal_image::CellPixels::new(10, 23),
        sixel_display_mode: Some(false),
    };
    let mut update = composition(20_000, "Conversation text");
    let mut items =
        vec![json!({"category": "system_prompt", "tokens": 5000, "key": "system_prompt"})];
    items.extend((0..32).map(|index| {
        json!({
            "category": "messages", "tokens": 468 + u64::from(index < 24),
            "key": format!("Conversation text {index}")
        })
    }));
    update["_meta"]["aster"]["context"]["items"] = json!(items);
    client.update("session-a", update);
    client.controller.input("/context all".into());
    client.controller.state.editor.insert_str("未发送的草稿");
    client
}

fn native_frame(terminal: &mut Terminal<TestBackend>, state: &mut State) -> (Buffer, ui::HitAreas) {
    let mut hits = ui::HitAreas::default();
    let frame = terminal
        .draw(|frame| hits = ui::draw(frame, state))
        .unwrap();
    (frame.buffer.clone(), hits)
}

fn reader_client(category: &str, text: &str) -> Client {
    let mut client = Client::ready();
    client.update("session-a", inspectable_composition("request-one"));
    client.controller.input("/context".into());
    for _ in 0..if category == "output" { 2 } else { 0 } {
        client.press(KeyCode::Down);
    }
    let pending = client.inspect_selected("request-one", category, None);
    client.answer(
        pending,
        content_response("request-one", category, None, text),
    );
    client.controller.state.editor.insert_str("未发送的草稿");
    client.controller.state.editor.select_all();
    client
}

fn reader_position(buffer: &Buffer, hits: &ui::HitAreas, needle: &str) -> Position {
    let area = hits.context_content.expect("text viewport");
    let screen = screen_text(buffer, buffer.area.width);
    for (y, row) in screen.lines().enumerate() {
        if (area.y..area.bottom()).contains(&(y as u16)) {
            if let Some(byte) = row.find(needle) {
                return Position::new(row[..byte].width() as u16, y as u16);
            }
        }
    }
    panic!("missing {needle:?} in reader: {screen}");
}

fn reader_mouse(
    client: &mut Client,
    area: Rect,
    kind: MouseEventKind,
    at: Position,
) -> Vec<Action> {
    client
        .controller
        .context_mouse(
            Some(area),
            MouseEvent {
                kind,
                column: at.x,
                row: at.y,
                modifiers: KeyModifiers::NONE,
            },
        )
        .expect("reader handled pointer event")
}

fn assert_copied(actions: &[Action], expected: &str) {
    assert!(
        matches!(actions, [Action::CopySelection { text, cut: false }] if text == expected),
        "expected only a read-only copy of {expected:?}: {actions:?}"
    );
}

#[test]
fn context_drag_copies_on_release_with_unicode_highlighting_and_no_draft_or_exit_changes() {
    let chosen = "中文\t e\u{301} 👩‍💻";
    for (reverse, monochrome) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut client = reader_client("system_prompt", &format!("COPY_BEGIN {chosen} COPY_END"));
        client.controller.state.monochrome = monochrome;
        client.controller.state.phase = Phase::Busy;
        let draft = client.controller.state.editor.lines().to_vec();
        let draft_selection = client.controller.state.editor.selection_range();
        let draft_cursor = client.controller.state.editor.cursor();
        let revision = client.controller.state.draft_revision;
        let mut terminal = Terminal::new(TestBackend::new(80, 28)).unwrap();
        let (before, hits) = native_frame(&mut terminal, &mut client.controller.state);
        assert!(hits.choice_rows.is_empty());
        assert!(hits.context_grid.is_none());
        let area = hits.context_content.unwrap();
        let mut start = reader_position(&before, &hits, "COPY_BEGIN ");
        start.x += "COPY_BEGIN ".width() as u16;
        let end = reader_position(&before, &hits, " COPY_END");
        // Either cell of a CJK glyph selects the same whole character.
        let wide_start = Position::new(start.x + 1, start.y);
        let (anchor, focus) = if reverse {
            (end, wide_start)
        } else {
            (wide_start, end)
        };
        assert!(reader_mouse(
            &mut client,
            area,
            MouseEventKind::Down(MouseButton::Left),
            anchor
        )
        .is_empty());
        assert!(reader_mouse(
            &mut client,
            area,
            MouseEventKind::Drag(MouseButton::Left),
            focus
        )
        .is_empty());
        let (selected, hits) = native_frame(&mut terminal, &mut client.controller.state);
        assert_eq!(selected[(start.x, start.y)].symbol(), "中");
        if monochrome {
            assert!(selected[(start.x, start.y)]
                .modifier
                .contains(Modifier::REVERSED));
        } else {
            assert_ne!(
                selected[(start.x, start.y)].bg,
                before[(start.x, start.y)].bg
            );
        }
        let emoji = reader_position(&selected, &hits, "👩‍💻");
        assert_eq!(selected[(emoji.x, emoji.y)].symbol(), "👩‍💻");
        assert_eq!(selected[(end.x, end.y)].bg, before[(end.x, end.y)].bg);
        let copied = reader_mouse(
            &mut client,
            area,
            MouseEventKind::Up(MouseButton::Left),
            focus,
        );
        assert_copied(&copied, chosen);
        assert!(client
            .controller
            .context_mouse(
                Some(area),
                MouseEvent {
                    kind: MouseEventKind::Up(MouseButton::Left),
                    column: focus.x,
                    row: focus.y,
                    modifiers: KeyModifiers::NONE,
                }
            )
            .is_none());
        for _ in 0..2 {
            assert_copied(
                &client
                    .controller
                    .key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
                chosen,
            );
        }
        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            assert!(client
                .controller
                .key(KeyEvent::new_with_kind(
                    KeyCode::Char('c'),
                    KeyModifiers::CONTROL,
                    kind
                ))
                .is_empty());
        }
        assert!(client
            .controller
            .key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL))
            .is_empty());
        assert!(client.controller.state.exit_confirmation.is_none());
        assert_eq!(client.controller.state.editor.lines(), draft);
        assert_eq!(client.controller.state.editor.cursor(), draft_cursor);
        assert_eq!(
            client.controller.state.editor.selection_range(),
            draft_selection
        );
        assert_eq!(client.controller.state.draft_revision, revision);
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(
            client.controller.state.session_id.as_deref(),
            Some("session-a")
        );
        assert_eq!(client.controller.state.overlay, Some(Overlay::Context));
    }
}

#[test]
fn context_copy_preserves_hard_newlines_tabs_and_graphemes_across_soft_wraps_and_resize() {
    let text = format!(
        "WRAP_BEGIN {}\n硬换行\toriginal indentation\n\nWRAP_END",
        "中文 e\u{301} 👩‍💻 long wrapped words. ".repeat(6)
    );
    let mut client = reader_client("output", &text);
    let mut terminal = Terminal::new(TestBackend::new(40, 32)).unwrap();
    let (buffer, hits) = native_frame(&mut terminal, &mut client.controller.state);
    let area = hits.context_content.unwrap();
    let start = reader_position(&buffer, &hits, "WRAP_BEGIN");
    let mut end = reader_position(&buffer, &hits, "WRAP_END");
    end.x += "WRAP_END".width() as u16;
    assert!(end.y > start.y + 4, "fixture must wrap on screen");
    reader_mouse(
        &mut client,
        area,
        MouseEventKind::Down(MouseButton::Left),
        start,
    );
    reader_mouse(
        &mut client,
        area,
        MouseEventKind::Drag(MouseButton::Left),
        end,
    );
    assert_copied(
        &reader_mouse(
            &mut client,
            area,
            MouseEventKind::Up(MouseButton::Left),
            end,
        ),
        &text,
    );
    let (selected, hits) = native_frame(&mut terminal, &mut client.controller.state);
    let emoji = reader_position(&selected, &hits, "👩‍💻");
    assert_eq!(selected[(emoji.x, emoji.y)].symbol(), "👩‍💻");
    let mut resized = Terminal::new(TestBackend::new(80, 28)).unwrap();
    let (wide, hits) = native_frame(&mut resized, &mut client.controller.state);
    let wide_start = reader_position(&wide, &hits, "WRAP_BEGIN");
    assert_eq!(
        wide[(wide_start.x, wide_start.y)].bg,
        selected[(start.x, start.y)].bg
    );
    assert_copied(
        &client
            .controller
            .key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        &text,
    );
}

#[test]
fn context_drag_scrolls_outside_the_viewport_and_keeps_selection_pinned_until_navigation() {
    let text = format!(
        "SCROLL_BEGIN\n{}",
        (0..120)
            .map(|row| format!("row {row:03}\n"))
            .collect::<String>()
    );
    let mut client = reader_client("system_prompt", &text);
    let mut terminal = Terminal::new(TestBackend::new(80, 18)).unwrap();
    let (buffer, hits) = native_frame(&mut terminal, &mut client.controller.state);
    let area = hits.context_content.unwrap();
    let start = reader_position(&buffer, &hits, "SCROLL_BEGIN");
    reader_mouse(
        &mut client,
        area,
        MouseEventKind::Down(MouseButton::Left),
        start,
    );
    reader_mouse(&mut client, area, MouseEventKind::ScrollDown, start);
    let outside = Position::new(area.x + "row 000".width() as u16, area.bottom() + 2);
    reader_mouse(
        &mut client,
        area,
        MouseEventKind::Drag(MouseButton::Left),
        outside,
    );
    native_frame(&mut terminal, &mut client.controller.state);
    let view = client.controller.state.context_view.as_ref().unwrap();
    let selected = view.selected_text().unwrap();
    let last: usize = selected
        .lines()
        .last()
        .unwrap()
        .strip_prefix("row ")
        .unwrap()
        .parse()
        .unwrap();
    let expected = format!(
        "SCROLL_BEGIN\n{}",
        (0..=last)
            .map(|row| format!("row {row:03}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(selected, expected);
    assert!(last >= usize::from(area.height));
    let scroll = view.scroll;
    assert_copied(
        &reader_mouse(
            &mut client,
            area,
            MouseEventKind::Up(MouseButton::Left),
            outside,
        ),
        &expected,
    );
    assert_eq!(
        client
            .controller
            .state
            .context_view
            .as_ref()
            .unwrap()
            .scroll,
        scroll
    );
    client.update("session-a", inspectable_composition("request-two"));
    let mut resized = Terminal::new(TestBackend::new(40, 32)).unwrap();
    native_frame(&mut resized, &mut client.controller.state);
    client.press(KeyCode::End);
    native_frame(&mut resized, &mut client.controller.state);
    assert_eq!(
        client
            .controller
            .state
            .context_view
            .as_ref()
            .unwrap()
            .request_id,
        "request-one"
    );
    assert_copied(
        &client
            .controller
            .key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        &expected,
    );
    client.press(KeyCode::Esc);
    client.press(KeyCode::Down);
    client.press(KeyCode::Down);
    let pending = client.inspect_selected("request-two", "output", None);
    client.answer(
        pending,
        content_response("request-two", "output", None, "NEW_OUTPUT"),
    );
    let (_, hits) = native_frame(&mut resized, &mut client.controller.state);
    assert!(!client
        .controller
        .state
        .context_view
        .as_ref()
        .unwrap()
        .has_selection());
    assert!(client
        .controller
        .context_mouse(
            hits.context_content,
            MouseEvent {
                kind: MouseEventKind::Up(MouseButton::Left),
                column: outside.x,
                row: outside.y,
                modifiers: KeyModifiers::NONE,
            }
        )
        .is_none());
    assert!(client
        .controller
        .key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .is_empty());
    assert!(client.controller.state.exit_confirmation.is_some());
    assert_eq!(client.controller.state.editor.lines(), ["未发送的草稿"]);
}

#[test]
fn context_clicks_shift_extension_and_selection_clear_do_not_copy_empty_text() {
    let mut client = reader_client("system_prompt", "CLICK_TEXT first second");
    let mut terminal = Terminal::new(TestBackend::new(80, 28)).unwrap();
    let (buffer, hits) = native_frame(&mut terminal, &mut client.controller.state);
    let area = hits.context_content.unwrap();
    let start = reader_position(&buffer, &hits, "first");
    let mut end = reader_position(&buffer, &hits, "second");
    end.x += "second".width() as u16;
    assert!(reader_mouse(
        &mut client,
        area,
        MouseEventKind::Down(MouseButton::Left),
        start
    )
    .is_empty());
    assert!(reader_mouse(
        &mut client,
        area,
        MouseEventKind::Up(MouseButton::Left),
        start
    )
    .is_empty());
    assert!(client
        .controller
        .context_mouse(
            Some(area),
            MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: end.x,
                row: end.y,
                modifiers: KeyModifiers::SHIFT,
            }
        )
        .unwrap()
        .is_empty());
    assert_copied(
        &reader_mouse(
            &mut client,
            area,
            MouseEventKind::Up(MouseButton::Left),
            end,
        ),
        "first second",
    );
    // Clicking outside clears the highlight and is marked handled so the UI redraws.
    assert!(reader_mouse(
        &mut client,
        area,
        MouseEventKind::Down(MouseButton::Left),
        Position::new(area.x - 1, area.y)
    )
    .is_empty());
    assert!(!client
        .controller
        .state
        .context_view
        .as_ref()
        .unwrap()
        .has_selection());
    let (cleared, _) = native_frame(&mut terminal, &mut client.controller.state);
    assert_eq!(
        cleared[(start.x, start.y)].bg,
        buffer[(start.x, start.y)].bg
    );
}

#[test]
fn native_context_grid_preserves_notes_scroll_bounds_and_draft_on_close() {
    for (columns, rows) in [(18, 32), (40, 32), (80, 28), (100, 44)] {
        let mut client = native_context_client();
        let state = &mut client.controller.state;
        let mut terminal = Terminal::new(TestBackend::new(columns, rows)).unwrap();
        let (pending, hits) = native_frame(&mut terminal, state);
        assert!(hits.context_grid.is_none());
        assert!(!pending
            .content
            .iter()
            .any(|cell| cell.symbol().contains("\x1bP")));
        let request = state
            .terminal_image
            .take_request()
            .expect("context queues a native grid");
        assert!(state.terminal_image.complete(request.render()));
        let (complete, hits) = native_frame(&mut terminal, state);
        let area = hits.context_grid.expect("complete grid fits the panel");
        let image = state.terminal_image.ready().unwrap();
        assert_eq!((area.width, area.height), (image.columns, image.rows));
        assert!(area.right() < columns && area.bottom() < rows);
        assert_eq!(
            complete
                .content
                .iter()
                .filter(|cell| cell.symbol().contains("\x1bP"))
                .count(),
            1
        );
        let text = screen_text(&complete, columns);
        if columns >= 40 {
            assert!(text.contains("20,000 / 100,000 tokens"), "{text}");
            assert!(text.contains("Usage by category"), "{text}");
        }
        assert!(
            state.terminal_image.take_request().is_none(),
            "redraw reuses the encoding"
        );
        state.detail_scroll = 1;
        let (_, scrolled_hits) = native_frame(&mut terminal, state);
        if columns >= 80 {
            assert!(
                scrolled_hits.context_grid.is_none(),
                "a partially visible grid is never painted"
            );
        } else if let Some(scrolled) = scrolled_hits.context_grid {
            assert_eq!(scrolled.y + 1, area.y);
        }
        state.detail_scroll = u16::MAX;
        let (bottom, _) = native_frame(&mut terminal, state);
        assert!(!bottom
            .content
            .iter()
            .any(|cell| cell.symbol().contains("\x1bP")));
        state.detail_scroll = 0;
        assert!(native_frame(&mut terminal, state).1.context_grid.is_some());
        state.overlay = None;
        let (closed, hits) = native_frame(&mut terminal, state);
        assert!(hits.context_grid.is_none());
        assert!(!closed.content.iter().any(|cell| cell.diff_option
            == ratatui::buffer::CellDiffOption::Skip
            || cell.symbol().contains("\x1bP")));
        assert!(state.terminal_image.ready().is_none());
        assert!(
            state.attachments.is_empty(),
            "the grid is never attached to a model request"
        );
        assert_eq!(state.editor.lines(), ["未发送的草稿"]);
    }
}

#[test]
fn native_context_grid_rejects_stale_resizes_and_falls_back_after_errors() {
    let mut client = native_context_client();
    let state = &mut client.controller.state;
    let mut wide = Terminal::new(TestBackend::new(80, 28)).unwrap();
    native_frame(&mut wide, state);
    let old = state.terminal_image.take_request().unwrap();
    let mut narrow = Terminal::new(TestBackend::new(18, 32)).unwrap();
    native_frame(&mut narrow, state);
    let resized = state.terminal_image.take_request().unwrap();
    assert!(
        !state.terminal_image.complete(old.render()),
        "resize rejects the old footprint"
    );
    assert!(state.terminal_image.complete(resized.render()));
    assert!(native_frame(&mut narrow, state).1.context_grid.is_some());

    state.context_snapshot.as_mut().unwrap().used = Some(50_000);
    state.context_snapshot.as_mut().unwrap().categories.clear();
    native_frame(&mut wide, state);
    let changed = state.terminal_image.take_request().unwrap();
    assert!(state
        .terminal_image
        .complete(changed.key().failed("fixture encoder failure".into())));
    let (fallback, hits) = native_frame(&mut wide, state);
    assert!(hits.context_grid.is_none());
    assert!(screen_text(&fallback, 80).contains("■■■■■"));
    assert!(
        state.terminal_image.take_request().is_none(),
        "failed encodings are not retried every frame"
    );

    state.monochrome = true;
    assert!(native_frame(&mut wide, state).1.context_grid.is_none());
    assert!(state.terminal_image.ready().is_none());
    state.monochrome = false;
    state.context_snapshot.as_mut().unwrap().used = Some(70_000);
    native_frame(&mut wide, state);
    let pending = state.terminal_image.take_request().unwrap();
    state.overlay = Some(Overlay::Help);
    native_frame(&mut wide, state);
    assert!(
        !state.terminal_image.complete(pending.render()),
        "closing rejects a late worker"
    );
    assert!(state.terminal_image.ready().is_none());

    state.overlay = Some(Overlay::Context);
    state.terminal_graphics.sixel_display_mode = None;
    let (fallback, hits) = native_frame(&mut wide, state);
    assert!(hits.context_grid.is_none());
    assert!(screen_text(&fallback, 80).contains("■■■■■"));
    assert!(state.terminal_image.take_request().is_none());
    state.terminal_graphics.sixel_display_mode = Some(false);
    let mut tiny = Terminal::new(TestBackend::new(24, 6)).unwrap();
    assert!(native_frame(&mut tiny, state).1.context_grid.is_none());
    assert!(state.terminal_image.take_request().is_none());
    state.context_snapshot = None;
    assert!(native_frame(&mut wide, state).1.context_grid.is_none());
    assert!(state.terminal_image.take_request().is_none());
}

#[test]
fn context_composition_restores_source_or_keeps_target_on_session_load() {
    for success in [false, true] {
        for source_updates in [false, true] {
            let mut client = Client::ready();
            client.update("session-a", composition(20_000, "source"));
            let actions = client.controller.resume("session-b".into());
            let load = client.register(actions, "session/load");
            assert!(client.controller.state.context_snapshot.is_none());
            client.update("session-b", composition(5000, "target"));
            if source_updates {
                client.update("session-a", composition(25_000, "source-latest"));
            }
            if success {
                let actions = client.answer(load, json!({}));
                let close = client.register(actions, "session/close");
                let actions = client.answer(close, json!({}));
                client.finish_runtime(actions, "model-a");
            } else {
                client.fail(load);
            }
            let snapshot = client.controller.state.context_snapshot.as_ref().unwrap();
            let (used, label) = if success {
                (5000, "target")
            } else if source_updates {
                (25_000, "source-latest")
            } else {
                (20_000, "source")
            };
            assert_eq!(snapshot.used, Some(used));
            assert_eq!(snapshot.items[1].key, label);
        }
    }
}

#[test]
fn changing_the_model_clears_composition_after_success_and_preserves_it_on_failure() {
    for success in [false, true] {
        let mut client = Client::ready();
        client.update("session-a", composition(20_000, "source"));
        let actions = client.controller.input("/model model-b".into());
        let configure = client.register(actions, "_aster/session/configure");
        if success {
            client.answer(configure, runtime("model-b", "code", "ask"));
            assert!(client.controller.state.context_snapshot.is_none());
        } else {
            client.fail(configure);
            assert_eq!(
                client
                    .controller
                    .state
                    .context_snapshot
                    .as_ref()
                    .unwrap()
                    .items[1]
                    .key,
                "source"
            );
        }
    }
}

#[test]
fn context_reads_system_messages_and_output_and_scrolls_past_65535_lines() {
    let mut client = Client::ready();
    client.update("session-a", inspectable_composition("request-one"));
    client.controller.input("/context".into());
    client.controller.state.editor.insert_str("未发送的草稿");
    let pending = client.inspect_selected("request-one", "system_prompt", None);
    assert!(screen_text(&client.screen(80, 28), 80).contains("Loading content"));
    let text = format!("VIEW_SYSTEM 中文\n{}END_OF_SYSTEM", "line\n".repeat(70_000));
    assert!(client
        .answer(
            pending,
            content_response("request-one", "system_prompt", None, &text)
        )
        .is_empty());
    let screen = screen_text(&client.screen(80, 28), 80);
    assert!(screen.contains("VIEW_SYSTEM 中文"), "{screen}");
    client.press(KeyCode::End);
    let screen = screen_text(&client.screen(80, 28), 80);
    assert!(screen.contains("END_OF_SYSTEM"), "{screen}");
    assert!(
        client
            .controller
            .state
            .context_view
            .as_ref()
            .unwrap()
            .scroll
            > u16::MAX as usize
    );
    client.press(KeyCode::PageUp);
    let end = client
        .controller
        .state
        .context_view
        .as_ref()
        .unwrap()
        .scroll;
    client.press(KeyCode::PageDown);
    assert_eq!(
        client
            .controller
            .state
            .context_view
            .as_ref()
            .unwrap()
            .scroll,
        end + 10
    );
    client.press(KeyCode::Home);
    assert!(screen_text(&client.screen(40, 32), 40).contains("VIEW_SYSTEM 中文"));
    client.press(KeyCode::Esc);
    assert_eq!(client.controller.state.overlay, Some(Overlay::Context));
    assert!(client.controller.state.context_view.is_none());
    client.press(KeyCode::Down);
    let pending = client.inspect_selected("request-one", "messages", None);
    client.answer(
        pending,
        content_response(
            "request-one",
            "messages",
            None,
            "USER_QUESTION\nTOOL_OUTPUT: 42",
        ),
    );
    let screen = screen_text(&client.screen(80, 28), 80);
    assert!(
        screen.contains("USER_QUESTION") && screen.contains("TOOL_OUTPUT: 42"),
        "{screen}"
    );
    client.press(KeyCode::Left);
    client.press(KeyCode::Down);
    let pending = client.inspect_selected("request-one", "output", None);
    client.answer(
        pending,
        content_response("request-one", "output", None, "MODEL_REPLY 中文"),
    );
    let screen = screen_text(&client.screen(80, 28), 80);
    assert!(
        screen.contains("Latest model output") && screen.contains("MODEL_REPLY 中文"),
        "{screen}"
    );
    client.press(KeyCode::Esc);
    client.press(KeyCode::Esc);
    assert!(client.controller.state.overlay.is_none());
    assert_eq!(client.controller.state.editor.lines(), ["未发送的草稿"]);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn context_rejects_late_views_and_keeps_open_content_pinned_to_its_request() {
    let mut client = Client::ready();
    client.update("session-a", inspectable_composition("request-one"));
    client.controller.input("/context".into());
    let old = client.inspect_selected("request-one", "system_prompt", None);
    client.press(KeyCode::Esc);
    let current = client.inspect_selected("request-one", "system_prompt", None);
    client.answer(
        old,
        content_response("request-one", "system_prompt", None, "LATE_OLD"),
    );
    let screen = screen_text(&client.screen(80, 28), 80);
    assert!(
        screen.contains("Loading content") && !screen.contains("LATE_OLD"),
        "{screen}"
    );
    client.update("session-a", inspectable_composition("request-two"));
    client.answer(
        current,
        content_response("request-one", "system_prompt", None, "PINNED_SYSTEM"),
    );
    assert!(screen_text(&client.screen(80, 28), 80).contains("PINNED_SYSTEM"));
    assert_eq!(
        client
            .controller
            .state
            .context_view
            .as_ref()
            .unwrap()
            .request_id,
        "request-one"
    );
    client.press(KeyCode::Esc);
    let pending = client.inspect_selected("request-two", "system_prompt", None);
    client.answer(
        pending,
        content_response("request-one", "system_prompt", None, "WRONG_REQUEST"),
    );
    let screen = screen_text(&client.screen(80, 28), 80);
    assert!(
        screen.contains("different request") && !screen.contains("WRONG_REQUEST"),
        "{screen}"
    );
    client.press(KeyCode::Esc);
    let pending = client.inspect_selected("request-two", "system_prompt", None);
    client.controller.message(json!({
        "id": pending, "error": {
            "code": -32602, "message": "Invalid params",
            "data": {"reason": "That request is no longer available."}
        }
    }));
    assert!(screen_text(&client.screen(80, 28), 80).contains("no longer available"));
}

#[test]
fn context_opens_individual_blocks_without_token_usage_and_handles_small_windows() {
    let mut client = Client::ready();
    let mut update = inspectable_composition("request-one");
    update["_meta"]["aster"]["usageAvailable"] = json!(false);
    update["_meta"]["aster"]["context"]["attribution"] = json!("unavailable");
    client.update("session-a", update);
    client.controller.input("/context".into());
    assert_eq!(client.controller.state.choices().len(), 3);
    client.press(KeyCode::Tab);
    assert_eq!(client.controller.state.choices().len(), 5);
    for _ in 0..4 {
        client.press(KeyCode::Down);
    }
    let pending = client.inspect_selected("request-one", "messages", Some("Conversation text"));
    client.answer(
        pending,
        content_response(
            "request-one",
            "messages",
            Some("Conversation text"),
            "BLOCK_TEXT 中文\n    original indentation\nsecond line",
        ),
    );
    for (columns, rows) in [(1, 1), (8, 4), (24, 6), (40, 12), (40, 32), (80, 28)] {
        client.press(KeyCode::End);
        client.screen(columns, rows);
        client.press(KeyCode::Home);
        let screen = screen_text(&client.screen(columns, rows), columns);
        if rows >= 28 {
            assert!(screen.contains("BLOCK_TEXT 中文"), "{screen}");
            assert!(screen.contains("    original indentation"), "{screen}");
        }
    }
    client.press(KeyCode::Esc);
    client.press(KeyCode::Tab);
    assert_eq!(client.controller.state.choices().len(), 3);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn context_errors_do_not_render_partial_or_oversized_content() {
    let mut invalid = content_response("request-one", "system_prompt", None, "PARTIAL_TEXT");
    invalid["sections"]
        .as_array_mut()
        .unwrap()
        .push(json!({"key": "invalid"}));
    let mut oversized =
        content_response("request-one", "system_prompt", None, &"x".repeat(1_000_001));
    oversized["sections"][0]["truncated"] = json!(true);
    for response in [invalid, oversized] {
        let mut client = Client::ready();
        client.update("session-a", inspectable_composition("request-one"));
        client.controller.input("/context".into());
        let pending = client.inspect_selected("request-one", "system_prompt", None);
        client.answer(pending, response);
        let screen = screen_text(&client.screen(80, 28), 80);
        assert!(
            screen.contains("Invalid") && !screen.contains("PARTIAL_TEXT"),
            "{screen}"
        );
    }
}

#[test]
fn context_content_responses_do_not_follow_a_session_switch_or_replay_on_load_failure() {
    for success in [false, true] {
        let mut client = Client::ready();
        client.update("session-a", inspectable_composition("request-one"));
        client.controller.input("/context".into());
        let pending = client.inspect_selected("request-one", "system_prompt", None);
        let actions = client.controller.resume("session-b".into());
        let load = client.register(actions, "session/load");
        client.answer(
            pending,
            content_response("request-one", "system_prompt", None, "OLD_SESSION_CONTENT"),
        );
        if success {
            let actions = client.answer(load, json!({}));
            let close = client.register(actions, "session/close");
            let actions = client.answer(close, json!({}));
            client.finish_runtime(actions, "model-a");
        } else {
            client.fail(load);
        }
        assert!(client.controller.state.context_view.is_none());
        assert!(!screen_text(&client.screen(80, 28), 80).contains("OLD_SESSION_CONTENT"));
    }
}

#[test]
fn opening_context_content_cancels_the_native_grid_and_restores_it_on_back() {
    let mut client = native_context_client();
    client.update("session-a", inspectable_composition("request-one"));
    client.controller.input("/context".into());
    let mut terminal = Terminal::new(TestBackend::new(80, 28)).unwrap();
    native_frame(&mut terminal, &mut client.controller.state);
    let old_grid = client
        .controller
        .state
        .terminal_image
        .take_request()
        .unwrap();
    let pending = client.inspect_selected("request-one", "system_prompt", None);
    assert!(!client
        .controller
        .state
        .terminal_image
        .complete(old_grid.render()));
    client.answer(
        pending,
        content_response("request-one", "system_prompt", None, "ACTUAL_SYSTEM_TEXT"),
    );
    let (body, hits) = native_frame(&mut terminal, &mut client.controller.state);
    assert!(hits.context_grid.is_none() && hits.choice_rows.is_empty());
    assert!(screen_text(&body, 80).contains("ACTUAL_SYSTEM_TEXT"));
    assert!(!body
        .content
        .iter()
        .any(|cell| cell.symbol().contains("\x1bP")));
    client.press(KeyCode::Esc);
    native_frame(&mut terminal, &mut client.controller.state);
    let grid = client
        .controller
        .state
        .terminal_image
        .take_request()
        .unwrap();
    assert!(client
        .controller
        .state
        .terminal_image
        .complete(grid.render()));
    assert!(native_frame(&mut terminal, &mut client.controller.state)
        .1
        .context_grid
        .is_some());
    assert_eq!(client.controller.state.editor.lines(), ["未发送的草稿"]);
    assert!(client.controller.state.attachments.is_empty());
}
