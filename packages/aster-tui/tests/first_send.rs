// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! First-send feedback through the same input, RPC and rendering boundaries as the terminal.

use aster_tui::{
    attachments::Attachment,
    controller::{Action, Controller},
    state::{Entry, Phase, State},
    ui,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, layout::Rect, Terminal};
use serde_json::{json, Value};
use unicode_width::UnicodeWidthStr;

fn controller() -> Controller {
    Controller::new(
        State::new("workspace".into(), "test-model".into(), false),
        None,
        false,
    )
}

fn key(controller: &mut Controller, code: KeyCode) -> Vec<Action> {
    controller.key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn only_request(actions: Vec<Action>, expected_method: &str) -> Action {
    assert_eq!(actions.len(), 1, "{actions:?}");
    let action = actions.into_iter().next().unwrap();
    assert!(
        matches!(&action, Action::Request { method, .. } if *method == expected_method),
        "expected {expected_method}, got {action:?}"
    );
    action
}

fn reply(controller: &mut Controller, action: Action, result: Value) -> Vec<Action> {
    let Action::Request { pending, .. } = action else {
        panic!("expected RPC, got {action:?}");
    };
    controller.registered(1, pending);
    controller.message(json!({"id": 1, "result": result}))
}

fn begin_preparing(controller: &mut Controller) -> Action {
    let initialize = controller.initialize();
    let listing = only_request(
        reply(
            controller,
            initialize,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "close": {}},
                    "promptCapabilities": {"image": true}
                },
                "_meta": {"aster": {"attachments": true}}
            }),
        ),
        "session/list",
    );
    let preparing = only_request(
        reply(controller, listing, json!({"sessions": []})),
        "session/new",
    );
    assert_eq!(controller.state.phase, Phase::Preparing);
    preparing
}

fn render(controller: &mut Controller, width: u16, height: u16) -> (String, String, usize) {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    let mut hits = ui::HitAreas::default();
    terminal
        .draw(|frame| hits = ui::draw(frame, &mut controller.state))
        .unwrap();
    let text = |area: Rect| {
        (area.y..area.bottom())
            .map(|y| {
                let mut line = String::new();
                let mut continuation = 0;
                for x in area.x..area.right() {
                    if continuation > 0 {
                        continuation -= 1;
                        continue;
                    }
                    let symbol = terminal.backend().buffer()[(x, y)].symbol();
                    line.push_str(symbol);
                    continuation = UnicodeWidthStr::width(symbol).saturating_sub(1);
                }
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    (
        text(hits.transcript),
        text(Rect::new(0, 0, width, height)),
        hits.welcome_rows.len(),
    )
}

fn assert_pending_visible(controller: &mut Controller, message: &str, width: u16, height: u16) {
    let (transcript, screen, welcome_rows) = render(controller, width, height);
    for line in message.lines() {
        assert!(
            transcript.contains(line),
            "submitted message must remain in the transcript at {width}x{height}:\n{screen}"
        );
    }
    assert_eq!(
        welcome_rows, 0,
        "sending must leave the welcome screen:\n{screen}"
    );
}

fn assert_sent_once(controller: &mut Controller, preparing: Action, message: &str) -> Value {
    let prompt = only_request(
        reply(controller, preparing, json!({"sessionId": "first-session"})),
        "session/prompt",
    );
    let Action::Request { params, .. } = &prompt else {
        unreachable!();
    };
    let params = params.clone();
    assert_eq!(params["prompt"][0]["text"], message);
    assert_eq!(params["sessionId"], "first-session");
    assert!(controller.state.queued.is_empty());
    assert_eq!(
        controller
            .state
            .entries
            .iter()
            .filter(|entry| matches!(entry, Entry::User(_)))
            .count(),
        1
    );
    let (transcript, screen, welcome_rows) = render(controller, 80, 24);
    assert_eq!(
        transcript.matches(message.lines().next().unwrap()).count(),
        1,
        "{screen}"
    );
    assert_eq!(welcome_rows, 0);
    assert!(reply(controller, prompt, json!({"stopReason": "end_turn"})).is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
    params
}

#[test]
fn first_enter_while_connecting_keeps_message_visible_until_sent_once() {
    for (width, height) in [(80, 24), (40, 12)] {
        let mut controller = controller();
        assert_eq!(controller.state.phase, Phase::Connecting);
        controller.paste("first question".into());
        assert!(key(&mut controller, KeyCode::Enter).is_empty());
        assert_eq!(controller.state.editor.lines(), &[""]);
        assert_pending_visible(&mut controller, "first question", width, height);

        let preparing = begin_preparing(&mut controller);
        assert_pending_visible(&mut controller, "first question", width, height);
        assert_sent_once(&mut controller, preparing, "first question");
    }
}

#[test]
fn first_enter_while_preparing_keeps_message_visible_on_a_narrow_screen() {
    let mut controller = controller();
    let preparing = begin_preparing(&mut controller);
    controller.paste("检查代码".into());
    assert!(key(&mut controller, KeyCode::Enter).is_empty());
    assert_pending_visible(&mut controller, "检查代码", 32, 12);
    assert_sent_once(&mut controller, preparing, "检查代码");
}

#[test]
fn image_only_first_enter_waits_for_session_and_sends_once() {
    for connecting in [true, false] {
        let mut controller = controller();
        let mut preparing = if connecting {
            None
        } else {
            Some(begin_preparing(&mut controller))
        };
        let image = Attachment::from_image(
            "clipboard.png",
            &base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1QAAAAASUVORK5CYII=",
            )
            .unwrap(),
        )
        .unwrap();
        controller.add_attachment(controller.state.conversation_generation, Ok(image));
        assert!(key(&mut controller, KeyCode::Enter).is_empty());
        assert_pending_visible(&mut controller, "clipboard.png", 80, 24);
        if connecting {
            preparing = Some(begin_preparing(&mut controller));
        }
        let prompt = only_request(
            reply(
                &mut controller,
                preparing.unwrap(),
                json!({"sessionId": "first-session"}),
            ),
            "session/prompt",
        );
        let Action::Request { params, .. } = &prompt else {
            unreachable!();
        };
        assert_eq!(params["prompt"].as_array().unwrap().len(), 1);
        assert_eq!(params["prompt"][0]["type"], "image");
        assert!(controller.state.queued.is_empty());
        render(&mut controller, 80, 24);
        assert!(reply(&mut controller, prompt, json!({"stopReason": "end_turn"})).is_empty());
        assert_eq!(controller.state.phase, Phase::Ready);
    }
}

#[test]
fn waiting_first_message_preserves_multiline_text_and_attachment_preview() {
    let mut controller = controller();
    let preparing = begin_preparing(&mut controller);
    let attachment = Attachment::Text {
        name: "notes.txt".into(),
        uri: "file:///notes.txt".into(),
        text: "captured file contents".into(),
    };
    controller.add_attachment(controller.state.conversation_generation, Ok(attachment));
    let message = "inspect this file\n保留第二行";
    controller.paste(message.into());
    assert!(key(&mut controller, KeyCode::Enter).is_empty());
    assert_pending_visible(&mut controller, message, 80, 24);
    let (transcript, screen, _) = render(&mut controller, 80, 24);
    assert!(transcript.contains("notes.txt"), "{screen}");

    let params = assert_sent_once(&mut controller, preparing, message);
    assert_eq!(params["prompt"].as_array().unwrap().len(), 2);
    assert_eq!(
        params["prompt"][1]["resource"]["text"],
        "captured file contents"
    );
}

#[test]
fn escape_recalls_waiting_first_message_and_requires_enter_before_sending() {
    let mut controller = controller();
    let preparing = begin_preparing(&mut controller);
    let message = "recall this question\nkeep this line";
    controller.paste(message.into());
    assert!(key(&mut controller, KeyCode::Enter).is_empty());
    assert!(key(&mut controller, KeyCode::Esc).is_empty());
    assert_eq!(controller.state.editor.lines().join("\n"), message);
    assert!(controller.state.queued.is_empty());
    let (transcript, _, _) = render(&mut controller, 80, 24);
    assert!(!transcript.contains("recall this question"));
    assert!(reply(
        &mut controller,
        preparing,
        json!({"sessionId": "first-session"})
    )
    .is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
    assert!(!controller
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(_))));
    let prompt = only_request(key(&mut controller, KeyCode::Enter), "session/prompt");
    let Action::Request { params, .. } = prompt else {
        unreachable!();
    };
    assert_eq!(params["prompt"][0]["text"], message);
}

#[test]
fn failed_preparation_keeps_its_reason_visible_after_a_long_first_message() {
    let mut controller = controller();
    let preparing = begin_preparing(&mut controller);
    let message = (1..=30)
        .map(|line| format!("Keep this requested change, line {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    controller.paste(message.clone());
    assert!(key(&mut controller, KeyCode::Enter).is_empty());
    render(&mut controller, 80, 24);

    let Action::Request { pending, .. } = preparing else {
        unreachable!();
    };
    controller.registered(1, pending);
    let reason = "Model connection refused; check the endpoint.";
    assert!(controller
        .message(json!({"id": 1, "error": {"code": -32603, "message": reason}}))
        .is_empty());

    let (transcript, screen, welcome_rows) = render(&mut controller, 80, 24);
    assert!(
        transcript.contains(reason),
        "the preparation error must stay visible without scrolling:\n{screen}"
    );
    assert_eq!(welcome_rows, 0);
    assert_eq!(controller.state.queued.len(), 1);
    assert_eq!(controller.state.queued.front().unwrap().text, message);
    assert!(controller.state.queue_paused);
    assert_eq!(controller.state.phase, Phase::Ready);
    assert!(!controller
        .state
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(_))));
}

#[test]
fn multiple_waiting_messages_send_once_each_in_order_after_preparation() {
    let mut controller = controller();
    for message in ["first request", "second request"] {
        controller.paste(message.into());
        assert!(key(&mut controller, KeyCode::Enter).is_empty());
    }
    assert_pending_visible(&mut controller, "first request\nsecond request", 80, 24);
    let preparing = begin_preparing(&mut controller);
    let first_prompt = only_request(
        reply(
            &mut controller,
            preparing,
            json!({"sessionId": "first-session"}),
        ),
        "session/prompt",
    );
    let Action::Request { params, .. } = &first_prompt else {
        unreachable!();
    };
    assert_eq!(params["prompt"][0]["text"], "first request");
    assert_eq!(controller.state.queued.len(), 1);
    assert_eq!(
        controller.state.queued.front().unwrap().text,
        "second request"
    );
    assert_eq!(
        controller
            .state
            .entries
            .iter()
            .filter(|entry| matches!(entry, Entry::User(_)))
            .count(),
        1
    );

    let second_prompt = only_request(
        reply(
            &mut controller,
            first_prompt,
            json!({"stopReason": "end_turn"}),
        ),
        "session/prompt",
    );
    let Action::Request { params, .. } = &second_prompt else {
        unreachable!();
    };
    assert_eq!(params["prompt"][0]["text"], "second request");
    assert!(controller.state.queued.is_empty());
    let (transcript, screen, welcome_rows) = render(&mut controller, 80, 24);
    for message in ["first request", "second request"] {
        assert_eq!(transcript.matches(message).count(), 1, "{screen}");
    }
    assert_eq!(welcome_rows, 0);
    assert!(reply(
        &mut controller,
        second_prompt,
        json!({"stopReason": "end_turn"})
    )
    .is_empty());
    assert_eq!(controller.state.phase, Phase::Ready);
}
