// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Exercise session controls through the same controller and wire boundaries as the frontend.

use std::path::Path;

use aster_tui::attachments::{load_file, Attachment};
use aster_tui::controller::{Action, Controller};
use aster_tui::state::{Entry, Overlay, Phase, State};
use base64::Engine;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};
use serde_json::{json, Value};

struct Client {
    controller: Controller,
    next_id: u64,
}

impl Client {
    fn ready() -> Self {
        let mut client = Self {
            controller: Controller::new(
                State::new("C:/test-workspace".into(), "model-a".into(), false),
                None,
                false,
            ),
            next_id: 1,
        };
        let action = client.controller.initialize();
        let (id, _) = client.register(vec![action], "initialize");
        let actions = client.answer(
            id,
            json!({
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": {"list": {}, "close": {}},
                    "promptCapabilities": {"image": true}
                },
                "_meta": {"aster": {
                    "attachments": true, "runtimeControls": true, "structuredProgress": true, "skills": true
                }}
            }),
        );
        let (id, _) = client.register(actions, "session/list");
        let actions = client.answer(id, json!({"sessions": []}));
        let (id, _) = client.register(actions, "session/new");
        let actions = client.answer(id, json!({"sessionId": "session-a"}));
        let (id, _) = client.register(actions, "_aster/session/runtime");
        assert!(client
            .answer(id, runtime("model-a", "code", "ask"))
            .is_empty());
        assert_eq!(client.controller.state.phase, Phase::Ready);
        client
    }

    fn register(&mut self, actions: Vec<Action>, expected_method: &str) -> (u64, Value) {
        assert_eq!(
            actions.len(),
            1,
            "expected one {expected_method} request: {actions:?}"
        );
        let Action::Request {
            method,
            params,
            pending,
        } = actions.into_iter().next().unwrap()
        else {
            panic!("expected a {expected_method} request");
        };
        assert_eq!(method, expected_method);
        let id = self.next_id;
        self.next_id += 1;
        self.controller.registered(id, pending);
        (id, params)
    }

    fn answer(&mut self, id: u64, result: Value) -> Vec<Action> {
        self.controller.message(json!({"id": id, "result": result}))
    }

    fn start_prompt(&mut self) -> u64 {
        let actions = self.controller.input("Inspect the supplied context".into());
        self.register(actions, "session/prompt").0
    }

    fn update(&mut self, update: Value) {
        assert!(self
            .controller
            .message(json!({
                "method": "session/update",
                "params": {"sessionId": "session-a", "update": update}
            }))
            .is_empty());
    }
}

fn key(controller: &mut Controller, code: KeyCode) -> Vec<Action> {
    controller.key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn text_attachment(uri: &str, text: &str) -> Attachment {
    Attachment::Text {
        name: "README.md".into(),
        uri: uri.into(),
        text: text.into(),
    }
}

fn clipboard_image() -> Attachment {
    let bytes = base64::engine::general_purpose::STANDARD.decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1QAAAAASUVORK5CYII="
    ).unwrap();
    Attachment::from_image("clipboard.png", &bytes).unwrap()
}

fn runtime(model: &str, mode: &str, approval: &str) -> Value {
    json!({
        "model": model,
        "mode": mode,
        "approval": approval,
        "permissionMode": match (mode, approval) {
            ("plan", _) => "plan",
            (_, "session") => "auto",
            (_, "accept_edits") => "accept_edits",
            _ => "manual",
        },
        "contextWindow": 1_048_576,
        "reasoningLevel": "low",
        "reasoningSelection": null,
        "reasoningDefault": "low",
        "reasoningLevels": ["low", "high"],
        "executionBackend": "native",
        "executionBackends": ["native", "sandbox"],
        "models": [{"id": "model-a", "name": "Model A"}, {"id": "model-b", "name": "Model B"}],
        "modes": [{"id": "code", "name": "Code"}, {"id": "plan", "name": "Plan"}]
    })
}

fn permission(id: &str, session: &str) -> Value {
    json!({
        "id": id,
        "method": "session/request_permission",
        "params": {
            "sessionId": session,
            "toolCall": {"toolCallId": "cell-1", "title": "Execute Python", "rawInput": {"code": "print('checked')"}},
            "options": [
                {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                {"optionId": "allow_session", "name": "Allow this session", "kind": "allow_always"},
                {"optionId": "reject", "name": "Reject", "kind": "reject_once"}
            ]
        }
    })
}

fn permission_for_code(id: &str, code: &str) -> Value {
    let mut request = permission(id, "session-a");
    request["params"]["toolCall"]["rawInput"]["code"] = json!(code);
    request
}

#[test]
fn approval_is_cancelled_when_the_dialog_could_not_show_the_code_exactly() {
    let escape = char::from(27u8);
    let line_feed = char::from(10u8);
    let carriage_return = char::from(13u8);
    for hidden in [
        format!("print('ok') # {escape}]{line_feed}import os"),
        format!("print('ok') # {escape}[2K{line_feed}import os"),
        format!("print('ok') # {carriage_return}import os"),
        format!(
            "print('ok') # {}{line_feed}import os",
            char::from_u32(0x202e).unwrap()
        ),
    ] {
        let mut client = Client::ready();
        client.start_prompt();

        let actions = client
            .controller
            .permission_request(&permission_for_code("hidden-code", &hidden));

        assert_cancelled_permission(&actions, "hidden-code");
        assert!(client.controller.state.approvals.is_empty());
        assert_eq!(client.controller.state.overlay, None);
    }
}

#[test]
fn approval_shows_tabs_and_windows_line_endings_unchanged() {
    let tab = char::from(9u8);
    let line_feed = char::from(10u8);
    let carriage_return = char::from(13u8);
    let mut client = Client::ready();
    client.start_prompt();

    let actions = client.controller.permission_request(&permission_for_code(
        "plain-code",
        &format!("if True:{carriage_return}{line_feed}{tab}print('ok')"),
    ));

    assert!(actions.is_empty());
    assert_eq!(
        client.controller.state.approvals[0].code,
        format!("if True:{line_feed}{tab}print('ok')")
    );
}

fn assert_cancelled_permission(actions: &[Action], expected_id: &str) {
    assert!(
        actions.iter().any(|action| matches!(action,
            Action::Reply { id, result }
            if id == expected_id && result == &json!({"outcome": {"outcome": "cancelled"}})
        )),
        "expected cancellation for {expected_id}: {actions:?}"
    );
}

#[test]
fn completing_a_prompt_dismisses_its_pending_execution_approval() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    assert!(client
        .controller
        .permission_request(&permission("approval-1", "session-a"))
        .is_empty());
    assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));

    let actions = client.answer(prompt, json!({"stopReason": "end_turn"}));

    assert_cancelled_permission(&actions, "approval-1");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[tokio::test]
async fn same_named_files_keep_both_reviewed_snapshots_when_sent() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("one")).unwrap();
    std::fs::create_dir(directory.path().join("two")).unwrap();
    std::fs::write(directory.path().join("one/README.md"), "first snapshot").unwrap();
    std::fs::write(directory.path().join("two/README.md"), "second snapshot").unwrap();
    let mut first = load_file(directory.path(), Path::new("one/README.md"))
        .await
        .unwrap();
    let mut second = load_file(directory.path(), Path::new("two/README.md"))
        .await
        .unwrap();
    assert_eq!(first.name(), "one/README.md");
    assert_eq!(second.name(), "two/README.md");
    // Relative labels distinguish these files, but display labels are not identity.
    // Keep the real file snapshots and make the labels coincide to cover deduplication.
    for attachment in [&mut first, &mut second] {
        let Attachment::Text { name, .. } = attachment else {
            panic!("expected a text snapshot");
        };
        *name = "README.md".into();
    }

    let mut client = Client::ready();
    let generation = client.controller.state.conversation_generation;
    client.controller.add_attachment(generation, Ok(first));
    client.controller.add_attachment(generation, Ok(second));
    // Sending must use the content the user attached, even if its source changes.
    std::fs::write(directory.path().join("one/README.md"), "later edit").unwrap();
    client.controller.paste("Compare both files".into());
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "session/prompt");

    assert_eq!(params["prompt"].as_array().unwrap().len(), 3);
    assert_eq!(params["prompt"][1]["resource"]["text"], "first snapshot");
    assert_eq!(params["prompt"][2]["resource"]["text"], "second snapshot");
    assert_ne!(
        params["prompt"][1]["resource"]["uri"],
        params["prompt"][2]["resource"]["uri"]
    );
    assert!(client.controller.state.attachments.is_empty());
}

#[test]
fn pasted_image_without_text_sends_one_image_block() {
    let mut client = Client::ready();
    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
    let generation = client.controller.state.conversation_generation;
    client
        .controller
        .add_attachment(generation, Ok(clipboard_image()));

    let actions = key(&mut client.controller, KeyCode::Enter);
    let (id, params) = client.register(actions, "session/prompt");
    let blocks = params["prompt"].as_array().unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "image");
    assert_eq!(blocks[0]["mimeType"], "image/png");
    assert!(!blocks[0]["data"].as_str().unwrap().is_empty());
    assert!(client.controller.state.attachments.is_empty());
    let Some(Entry::User(shown)) = client.controller.state.entries.first() else {
        panic!("sent image must have a visible user message");
    };
    assert!(shown.starts_with("[Image] clipboard.png · PNG · "), "{shown}");
    assert_eq!(shown.lines().count(), 1);
    assert!(client
        .answer(id, json!({"stopReason": "end_turn"}))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn rejected_image_request_shows_model_reason_and_keeps_the_complete_draft() {
    for caption in ["", "hi"] {
        let mut client = Client::ready();
        let image = clipboard_image();
        let generation = client.controller.state.conversation_generation;
        client
            .controller
            .add_attachment(generation, Ok(image.clone()));
        client.controller.paste(caption.into());
        let actions = key(&mut client.controller, KeyCode::Enter);
        let (id, _) = client.register(actions, "session/prompt");
        let reason = "The selected model is not configured to receive images. Select an image-capable model or remove the image attachment.";
        assert!(client
            .controller
            .message(json!({
                "id": id,
                "error": {"code": -32602, "message": "Invalid params", "data": {"reason": reason}}
            }))
            .is_empty());
        assert!(client
            .controller
            .state
            .entries
            .iter()
            .any(|entry| matches!(entry, Entry::Notice {text, error: true} if text == reason)));
        assert_eq!(client.controller.state.phase, Phase::Ready);
        let draft = client.controller.state.recovery.as_ref().unwrap();
        assert_eq!(draft.text, caption);
        assert_eq!(draft.attachments, vec![image.clone()]);

        assert!(client.controller.input("/retry".into()).is_empty());
        assert_eq!(client.controller.state.editor.lines().join("\n"), caption);
        assert_eq!(client.controller.state.attachments, vec![image]);
    }
}

#[test]
fn pasted_image_cannot_release_the_pending_file_send_guard() {
    let mut client = Client::ready();
    let actions = client.controller.input("/attach pending.txt".into());
    let [Action::AttachFile { generation, .. }] = actions.as_slice() else {
        panic!("expected an asynchronous file read: {actions:?}");
    };
    let generation = *generation;
    client.controller.paste("Review both attachments".into());
    client
        .controller
        .add_attachment(generation, Ok(clipboard_image()));

    assert!(client.controller.state.attachment_loading);
    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["Review both attachments"]
    );
    client.controller.attachment_result(
        generation,
        Ok(text_attachment(
            "file:///workspace/pending.txt",
            "complete file",
        )),
    );
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "session/prompt");

    assert!(!client.controller.state.attachment_loading);
    assert_eq!(params["prompt"].as_array().unwrap().len(), 3);
    assert_eq!(params["prompt"][1]["type"], "image");
    assert_eq!(params["prompt"][1]["mimeType"], "image/png");
    assert_eq!(params["prompt"][2]["resource"]["text"], "complete file");
}

#[test]
fn backend_slash_prompts_wait_for_the_selected_file_before_sending() {
    let mut client = Client::ready();
    client.update(json!({
        "sessionUpdate": "available_commands_update",
        "availableCommands": [{"name": "review", "description": "Review supplied context"}]
    }));
    assert!(client
        .controller
        .state
        .commands
        .iter()
        .any(|command| command.name == "/review"));
    let actions = client.controller.input("/attach pending.txt".into());
    let [Action::AttachFile { generation, .. }] = actions.as_slice() else {
        panic!("expected an asynchronous file read");
    };
    let generation = *generation;
    client.controller.paste("/review attached context".into());

    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
    assert_eq!(
        client.controller.state.editor.lines(),
        &["/review attached context"]
    );
    assert!(client.controller.state.attachment_loading);
    client.controller.attachment_result(
        generation,
        Ok(text_attachment(
            "file:///workspace/pending.txt",
            "complete file",
        )),
    );
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "session/prompt");
    assert_eq!(params["prompt"][0]["text"], "/review attached context");
    assert_eq!(params["prompt"][1]["resource"]["text"], "complete file");
}

#[test]
fn file_completion_from_another_conversation_does_not_attach_to_the_new_one() {
    let mut client = Client::ready();
    let actions = client.controller.input("/attach old.txt".into());
    let [Action::AttachFile { generation, .. }] = actions.as_slice() else {
        panic!("expected an asynchronous file read");
    };
    let generation = *generation;
    let actions = client.controller.resume("session-b".into());
    client.register(actions, "session/load");
    client.controller.attachment_result(
        generation,
        Ok(text_attachment(
            "file:///workspace/old.txt",
            "old conversation",
        )),
    );

    assert!(client.controller.state.attachments.is_empty());
    assert!(!client.controller.state.attachment_loading);
}

#[test]
fn approval_choices_reply_to_the_request_and_keep_the_running_prompt_and_draft() {
    for (choice, expected) in [
        (KeyCode::Char('1'), "allow_once"),
        (KeyCode::Char('2'), "allow_session"),
        (KeyCode::Char('3'), "reject"),
        (KeyCode::Enter, "allow_once"),
    ] {
        let mut client = Client::ready();
        client.start_prompt();
        let turn = client.controller.state.active_turn_id.clone();
        client.controller.paste("unfinished message".into());
        client
            .controller
            .permission_request(&permission("approval-1", "session-a"));
        assert_eq!(
            client.controller.state.approvals[0].code,
            "print('checked')"
        );

        let actions = key(&mut client.controller, choice);

        assert!(matches!(actions.as_slice(), [Action::Reply { id, result }]
            if id == "approval-1" && result == &json!({"outcome": {"outcome": "selected", "optionId": expected}})
        ));
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.active_turn_id, turn);
        assert_eq!(
            client.controller.state.editor.lines(),
            &["unfinished message"]
        );
        assert!(client.controller.state.approvals.is_empty());
        assert_eq!(client.controller.state.overlay, None);
    }
}

#[test]
fn approval_default_uses_option_kind_and_falls_back_to_reject() {
    for (kinds, expected) in [
        (
            vec!["allow_always", "reject_once", "allow_once"],
            "option-2",
        ),
        (vec!["allow_always", "reject_once"], "option-1"),
    ] {
        let mut client = Client::ready();
        client.start_prompt();
        let mut request = permission("approval-1", "session-a");
        request["params"]["options"] = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                json!({"optionId": format!("option-{index}"), "name": kind, "kind": kind})
            })
            .collect();
        assert!(client.controller.permission_request(&request).is_empty());

        let actions = key(&mut client.controller, KeyCode::Enter);

        assert!(matches!(actions.as_slice(), [Action::Reply { id, result }]
            if id == "approval-1" && result == &json!({"outcome": {"outcome": "selected", "optionId": expected}})
        ));
        assert!(client.controller.state.approvals.is_empty());
    }
}

#[test]
fn approval_key_repeats_do_not_confirm_the_next_request() {
    for code in [
        KeyCode::Enter,
        KeyCode::Char('1'),
        KeyCode::Char('2'),
        KeyCode::Char('3'),
    ] {
        let mut client = Client::ready();
        client.start_prompt();
        for id in ["first", "second"] {
            assert!(client
                .controller
                .permission_request(&permission(id, "session-a"))
                .is_empty());
        }
        let actions = key(&mut client.controller, code);
        assert!(matches!(actions.as_slice(), [Action::Reply { id, .. }] if id == "first"));

        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            assert!(client
                .controller
                .key(KeyEvent::new_with_kind(code, KeyModifiers::NONE, kind))
                .is_empty());
        }
        assert_eq!(client.controller.state.approvals.len(), 1);
        assert_eq!(client.controller.state.approvals[0].id, "second");

        let actions = key(&mut client.controller, code);
        assert!(matches!(actions.as_slice(), [Action::Reply { id, .. }] if id == "second"));
        assert!(client.controller.state.approvals.is_empty());
    }
}

#[test]
fn escape_rejects_the_execution_request_without_cancelling_the_whole_prompt() {
    let mut client = Client::ready();
    client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    let actions = key(&mut client.controller, KeyCode::Esc);

    assert_eq!(actions.len(), 1);
    assert_cancelled_permission(&actions, "approval-1");
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert_eq!(client.controller.state.phase, Phase::Busy);
}

#[test]
fn permissions_for_another_session_or_a_finished_turn_are_rejected() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    let actions = client
        .controller
        .permission_request(&permission("wrong-session", "session-b"));
    assert_cancelled_permission(&actions, "wrong-session");
    assert!(client.controller.state.approvals.is_empty());
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());

    let actions = client
        .controller
        .permission_request(&permission("late-request", "session-a"));

    assert_cancelled_permission(&actions, "late-request");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn failed_or_cancelled_prompts_dismiss_pending_approvals_and_pause_queued_work() {
    for result in [
        json!({"stopReason": "cancelled"}),
        json!({"stopReason": "max_tokens"}),
    ] {
        let mut client = Client::ready();
        let prompt = client.start_prompt();
        client.controller.paste("keep this draft".into());
        client
            .controller
            .permission_request(&permission("approval-1", "session-a"));

        let actions = client.answer(prompt, result);

        assert_cancelled_permission(&actions, "approval-1");
        assert!(client.controller.state.approvals.is_empty());
        assert_eq!(client.controller.state.overlay, None);
        assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
        assert!(client.controller.state.queue_paused);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn a_prompt_error_dismisses_its_permission_instead_of_leaving_a_stale_dialog() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    let actions = client
        .controller
        .message(json!({"id": prompt, "error": {"code": -32603, "message": "provider stopped"}}));

    assert_cancelled_permission(&actions, "approval-1");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn a_disconnected_backend_cannot_leave_or_open_an_execution_dialog() {
    let mut client = Client::ready();
    client.start_prompt();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    client.controller.failure("Protocol stream closed");
    let actions = client
        .controller
        .permission_request(&permission("after-close", "session-a"));

    assert_cancelled_permission(&actions, "after-close");
    assert!(client.controller.state.approvals.is_empty());
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Failed);
}

#[test]
fn changing_runtime_settings_uses_the_backend_result_before_showing_the_new_value() {
    for (command, field, value, model, mode, approval) in [
        (
            "/model model-b",
            "model",
            "model-b",
            "model-b",
            "code",
            "ask",
        ),
        ("/mode plan", "mode", "plan", "model-a", "plan", "ask"),
        (
            "/permissions session",
            "approval",
            "session",
            "model-a",
            "code",
            "session",
        ),
        (
            "/permissions auto",
            "permissionMode",
            "auto",
            "model-a",
            "code",
            "session",
        ),
        (
            "/permissions manual",
            "permissionMode",
            "manual",
            "model-a",
            "code",
            "ask",
        ),
        (
            "/permissions accept_edits",
            "permissionMode",
            "accept_edits",
            "model-a",
            "code",
            "accept_edits",
        ),
        (
            "/permissions plan",
            "permissionMode",
            "plan",
            "model-a",
            "plan",
            "ask",
        ),
        (
            "/permissions ask",
            "approval",
            "ask",
            "model-a",
            "code",
            "ask",
        ),
    ] {
        let mut client = Client::ready();
        let actions = client.controller.input(command.into());
        let (id, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(params["sessionId"], "session-a");
        assert_eq!(params[field], value);
        assert_eq!(client.controller.state.model, "model-a");
        assert_eq!(client.controller.state.mode, "code");
        assert_eq!(client.controller.state.approval_policy, "ask");

        assert!(client.answer(id, runtime(model, mode, approval)).is_empty());

        assert_eq!(client.controller.state.model, model);
        assert_eq!(client.controller.state.mode, mode);
        assert_eq!(client.controller.state.approval_policy, approval);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn shift_tab_cycles_permissions_after_backend_confirmation_without_changing_the_draft() {
    for shortcut in [
        KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE),
        KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
        KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT),
    ] {
        let mut client = Client::ready();
        client.controller.paste("keep this\ndraft here".into());
        key(&mut client.controller, KeyCode::Left);
        let draft = client.controller.state.editor.lines().to_vec();
        let cursor = client.controller.state.editor.cursor();
        for (permission, mode, approval) in [
            ("accept_edits", "code", "accept_edits"),
            ("plan", "plan", "ask"),
            ("auto", "code", "session"),
            ("manual", "code", "ask"),
        ] {
            let before = client.controller.state.runtime.clone();
            let actions = client.controller.key(shortcut);
            let (id, params) = client.register(actions, "_aster/session/configure");
            assert_eq!(
                params,
                json!({"sessionId": "session-a", "permissionMode": permission})
            );
            assert_eq!(client.controller.state.runtime, before);
            assert_eq!(client.controller.state.editor.lines(), draft);
            assert_eq!(client.controller.state.editor.cursor(), cursor);
            assert_eq!(client.controller.state.phase, Phase::Busy);

            assert!(client
                .answer(id, runtime("model-a", mode, approval))
                .is_empty());
            assert_eq!(client.controller.state.permission_mode(), permission);
            assert_eq!(client.controller.state.editor.lines(), draft);
            assert_eq!(client.controller.state.editor.cursor(), cursor);
            assert_eq!(client.controller.state.phase, Phase::Ready);
        }
    }
}

#[test]
fn permission_switch_keeps_footer_positions_while_waiting_for_the_backend() {
    fn footer(state: &mut State, columns: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(columns, 24)).unwrap();
        terminal
            .draw(|frame| {
                aster_tui::ui::draw(frame, state);
            })
            .unwrap();
        (0..columns)
            .map(|x| terminal.backend().buffer()[(x, 23)].symbol())
            .collect()
    }

    for columns in [48, 80, 120] {
        for draft in ["keep this draft", ""] {
            for rejected in [false, true] {
                let mut client = Client::ready();
                client.controller.paste(draft.into());
                let before = footer(&mut client.controller.state, columns);
                let actions = client
                    .controller
                    .key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
                let (id, _) = client.register(actions, "_aster/session/configure");
                let pending = footer(&mut client.controller.state, columns);
                assert_eq!(
                    pending.replace("Enter:wait", "Enter:send"),
                    before,
                    "footer moved at {columns} columns with draft {draft:?}"
                );
                if !draft.is_empty() {
                    assert!(pending.contains("Enter:wait"), "{pending}");
                    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
                    assert!(client.controller.state.queued.is_empty());
                }
                assert_eq!(client.controller.state.permission_mode(), "manual");
                assert_eq!(client.controller.state.editor.lines(), &[draft]);

                if rejected {
                    client.controller.message(json!({
                        "id": id,
                        "error": {"code": -32602, "message": "Permission change unavailable"}
                    }));
                } else {
                    client.answer(id, runtime("model-a", "code", "accept_edits"));
                }
                assert_eq!(client.controller.state.phase, Phase::Ready);
                if !draft.is_empty() || !rejected {
                    assert_eq!(footer(&mut client.controller.state, columns), before);
                }
            }
        }
    }
}

#[test]
fn shift_tab_ignores_repeated_modified_keys_and_open_overlays() {
    let mut client = Client::ready();
    client.controller.paste("keep this draft".into());
    let before = client.controller.state.runtime.clone();
    for code in [KeyCode::BackTab, KeyCode::Tab] {
        let draft = client.controller.state.editor.lines().to_vec();
        for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
            assert!(client
                .controller
                .key(KeyEvent::new_with_kind(code, KeyModifiers::SHIFT, kind))
                .is_empty());
        }
        assert_eq!(client.controller.state.editor.lines(), draft);
        for extra in [KeyModifiers::CONTROL, KeyModifiers::ALT, KeyModifiers::SUPER] {
            assert!(client
                .controller
                .key(KeyEvent::new(code, KeyModifiers::SHIFT | extra))
                .is_empty());
        }
    }
    assert_eq!(client.controller.state.runtime, before);
    assert_eq!(client.controller.state.phase, Phase::Ready);

    let draft = client.controller.state.editor.lines().to_vec();
    for overlay in [Overlay::Help, Overlay::Commands, Overlay::Permissions] {
        client.controller.state.open(overlay);
        for shortcut in [
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE),
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT),
        ] {
            assert!(client.controller.key(shortcut).is_empty());
            assert_eq!(client.controller.state.runtime, before);
            assert_eq!(client.controller.state.phase, Phase::Ready);
            assert_eq!(client.controller.state.overlay, Some(overlay));
        }
    }
    assert_eq!(client.controller.state.editor.lines(), draft);
}

#[test]
fn shift_tab_respects_backend_support_active_work_and_pending_approvals() {
    for blocker in ["legacy", "configure", "approval"] {
        let mut client = Client::ready();
        let expected_notice = match blocker {
            "legacy" => {
                client
                    .controller
                    .state
                    .runtime
                    .as_object_mut()
                    .unwrap()
                    .remove("permissionMode");
                "Permissions are unchanged"
            }
            "configure" => {
                let actions = client.controller.input("/model model-b".into());
                client.register(actions, "_aster/session/configure");
                "Finish or stop the active task"
            }
            _ => {
                publish_tasks(
                    &mut client,
                    task_snapshot(json!([
                        {"id": "background", "status": "running", "canStop": true}
                    ])),
                );
                assert!(client
                    .controller
                    .permission_request(&permission("approval-1", "session-a"))
                    .is_empty());
                assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));
                assert!(client
                    .controller
                    .key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT))
                    .is_empty());
                assert_eq!(client.controller.state.approvals.len(), 1);
                "Resolve or cancel the pending approval"
            }
        };
        // Exercise configure's guards even when a pending approval is not the visible panel.
        client.controller.state.overlay = None;
        client.controller.paste("keep this draft".into());
        let before = client.controller.state.runtime.clone();
        let phase = client.controller.state.phase;
        assert!(
            client
                .controller
                .key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT))
                .is_empty(),
            "{blocker}"
        );
        assert_eq!(client.controller.state.runtime, before, "{blocker}");
        assert_eq!(client.controller.state.phase, phase, "{blocker}");
        assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
        assert!(
            client.controller.state.entries.iter().any(|entry| {
                matches!(entry, Entry::Notice {text, ..} if text.contains(expected_notice))
            }),
            "{blocker}"
        );
    }
}

#[test]
fn shift_tab_keeps_the_confirmed_permission_and_draft_when_configuration_fails() {
    let mut client = Client::ready();
    client.controller.paste("keep this draft".into());
    key(&mut client.controller, KeyCode::Left);
    let cursor = client.controller.state.editor.cursor();
    let before = client.controller.state.runtime.clone();
    let actions = client
        .controller
        .key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
    let (id, _) = client.register(actions, "_aster/session/configure");
    assert!(client
        .controller
        .message(json!({
            "id": id,
            "error": {"code": -32602, "message": "Permission change unavailable"}
        }))
        .is_empty());
    assert_eq!(client.controller.state.runtime, before);
    assert_eq!(client.controller.state.permission_label(), "Manual");
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
    assert_eq!(client.controller.state.editor.cursor(), cursor);
}

#[test]
fn sandbox_commands_wait_for_confirmation_and_preserve_the_previous_backend_on_failure() {
    let mut client = Client::ready();
    for (command, requested, previously_sandboxed) in [
        ("/sandbox on", "sandbox", false),
        ("/sandbox off", "native", true),
    ] {
        let before = client.controller.state.runtime.clone();
        let actions = client.controller.input(command.into());
        let (id, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(
            params,
            json!({"sessionId":"session-a", "executionBackend":requested})
        );
        assert_eq!(client.controller.state.sandbox, previously_sandboxed);
        assert_eq!(client.controller.state.runtime, before);
        assert_eq!(client.controller.state.phase, Phase::Busy);
        let reason = "Sandbox could not start; settings are unchanged";
        assert!(client
            .controller
            .message(json!({
                "id":id, "error":{"code":-32603, "message":"Failed", "data":{"reason":reason}}
            }))
            .is_empty());
        assert_eq!(client.controller.state.sandbox, previously_sandboxed);
        assert_eq!(client.controller.state.runtime, before);
        assert_eq!(client.controller.state.phase, Phase::Ready);
        assert!(client
            .controller
            .state
            .entries
            .iter()
            .any(|entry| { matches!(entry, Entry::Notice {text, error:true} if text == reason) }));

        let actions = client.controller.input(command.into());
        let (id, _) = client.register(actions, "_aster/session/configure");
        let mut accepted = before;
        accepted["executionBackend"] = json!(requested);
        client.answer(id, accepted.clone());
        assert_eq!(client.controller.state.runtime, accepted);
        assert_eq!(client.controller.state.sandbox, !previously_sandboxed);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Notice {text, ..} if text.contains("Shell/file tools stay on host") && text.contains("not migrated"))
    }));
}

#[test]
fn sandbox_picker_focuses_the_confirmed_backend_and_preserves_the_draft() {
    let mut client = Client::ready();
    client.controller.paste("keep this draft".into());
    assert!(client
        .controller
        .state
        .commands
        .iter()
        .any(|command| command.name == "/sandbox"));
    let actions = client.controller.input("/sandbox".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");
    assert_eq!(client.controller.state.overlay, Some(Overlay::Sandbox));
    assert_eq!(client.controller.state.selected, 0);
    let mut status = runtime("model-a", "code", "ask");
    status["executionBackend"] = json!("sandbox");
    client.answer(inspection, status);
    assert_eq!(client.controller.state.selected, 1);
    let choices = client.controller.state.choices();
    assert_eq!(
        choices
            .iter()
            .map(|choice| (choice.label.as_str(), choice.value.as_str()))
            .collect::<Vec<_>>(),
        [("Native", "native"), ("Python sandbox", "sandbox")]
    );
    assert!(choices[1].detail.contains("Current"));
    key(&mut client.controller, KeyCode::Up);
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["executionBackend"], "native");
    assert!(client.controller.state.sandbox);
    assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
}

#[test]
fn escape_does_not_claim_to_cancel_a_model_configuration_request() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model model-b".into());
    let (id, _) = client.register(actions, "_aster/session/configure");
    assert!(key(&mut client.controller, KeyCode::Esc).is_empty());
    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.status, "Applying session settings…");
    assert_eq!(client.controller.state.model, "model-a");
    assert!(client
        .answer(id, runtime("model-b", "code", "ask"))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert_eq!(client.controller.state.model, "model-b");
}

#[test]
fn escape_cancels_sandbox_preparation_and_waits_for_the_backend_to_finish() {
    let mut client = Client::ready();
    let before = client.controller.state.runtime.clone();
    let actions = client.controller.input("/sandbox on".into());
    let (id, _) = client.register(actions, "_aster/session/configure");
    assert_eq!(
        client.controller.state.status,
        "Preparing Python sandbox… Esc cancels"
    );
    client.controller.paste("keep this draft".into());

    let actions = key(&mut client.controller, KeyCode::Esc);
    assert!(matches!(
        actions.as_slice(),
        [Action::Notify { method: "session/cancel", params }]
            if params == &json!({"sessionId":"session-a"})
    ));
    assert_eq!(client.controller.state.phase, Phase::Cancelling);
    assert!(client.controller.state.queue_paused);
    assert_eq!(client.controller.state.runtime, before);
    assert!(key(&mut client.controller, KeyCode::Esc).is_empty());
    assert!(client.controller.input("/sandbox off".into()).is_empty());

    assert!(client
        .controller
        .message(json!({
            "id":id,
            "error":{"code":-32603,"message":"Sandbox preparation cancelled; settings unchanged"}
        }))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert!(!client.controller.state.sandbox);
    assert_eq!(client.controller.state.runtime, before);
    assert!(client.controller.state.queue_paused);
    assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);

    let actions = client.controller.input("/sandbox on".into());
    let (id, _) = client.register(actions, "_aster/session/configure");
    let mut accepted = before;
    accepted["executionBackend"] = json!("sandbox");
    assert!(client.answer(id, accepted).is_empty());
    assert!(client.controller.state.sandbox);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn sandbox_switching_rejects_legacy_capabilities_and_invalid_commands() {
    let mut client = Client::ready();
    client
        .controller
        .state
        .runtime
        .as_object_mut()
        .unwrap()
        .remove("executionBackends");
    let before = client.controller.state.runtime.clone();
    for command in ["/sandbox on", "/sandbox off", "/sandbox maybe"] {
        assert!(client.controller.input(command.into()).is_empty());
        assert_eq!(client.controller.state.runtime, before);
        assert!(!client.controller.state.sandbox);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Notice {text, ..} if text.contains("update aster-acp"))
    }));
    let actions = client.controller.input("/sandbox".into());
    let (id, _) = client.register(actions, "_aster/session/runtime");
    client.answer(id, before);
    assert!(client.controller.state.choices().is_empty());
    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
}

#[test]
fn sandbox_configuration_is_blocked_by_active_work_and_pending_approvals() {
    for blocker in ["foreground", "background", "approval"] {
        let mut client = Client::ready();
        if blocker == "foreground" {
            client.start_prompt();
        } else {
            publish_tasks(
                &mut client,
                task_snapshot(json!([
                    {"id":"background", "status":"running", "canStop":true}
                ])),
            );
            if blocker == "approval" {
                client
                    .controller
                    .permission_request(&permission("approval-1", "session-a"));
                assert!(!client.controller.state.approvals.is_empty());
            }
        }
        let before = client.controller.state.runtime.clone();
        let phase = client.controller.state.phase;
        assert!(client.controller.input("/sandbox on".into()).is_empty());
        assert_eq!(client.controller.state.runtime, before);
        assert_eq!(client.controller.state.phase, phase);
        assert!(!client.controller.state.sandbox);
    }
}

#[test]
fn older_backends_cannot_silently_ignore_permission_changes() {
    for (permission, index) in [("manual", 1), ("plan", 3)] {
        for picker in [false, true] {
            let mut client = Client::ready();
            client.controller.paste("keep this draft".into());
            let actions = client.controller.input("/permissions".into());
            let (inspection, _) = client.register(actions, "_aster/session/runtime");
            let mut legacy = runtime("model-a", "code", "session");
            legacy.as_object_mut().unwrap().remove("permissionMode");
            client.answer(inspection, legacy.clone());
            assert_eq!(client.controller.state.permission_mode(), "auto");

            let actions = if picker {
                for _ in 0..index {
                    key(&mut client.controller, KeyCode::Down);
                }
                key(&mut client.controller, KeyCode::Enter)
            } else {
                key(&mut client.controller, KeyCode::Esc);
                client
                    .controller
                    .input(format!("/permissions {permission}"))
            };
            assert!(
                actions.is_empty(),
                "{permission}, picker={picker}: {actions:?}"
            );
            assert_eq!(client.controller.state.runtime, legacy);
            assert_eq!(client.controller.state.permission_mode(), "auto");
            assert_eq!(client.controller.state.phase, Phase::Ready);
            assert_ne!(client.controller.state.status, "Session settings updated");
            assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
            assert!(client.controller.state.entries.iter().any(|entry| {
                matches!(entry, Entry::Notice {text, ..} if text.contains("Permissions are unchanged") && text.contains("update aster-acp"))
            }));
        }
    }
}

#[test]
fn legacy_permission_commands_keep_using_supported_fields() {
    for (command, field, value) in [
        ("/permissions ask", "approval", "ask"),
        ("/permissions session", "approval", "session"),
        ("/plan", "mode", "plan"),
    ] {
        let mut client = Client::ready();
        let actions = client.controller.input("/permissions".into());
        let (inspection, _) = client.register(actions, "_aster/session/runtime");
        let mut legacy = runtime("model-a", "code", "session");
        legacy.as_object_mut().unwrap().remove("permissionMode");
        client.answer(inspection, legacy.clone());
        key(&mut client.controller, KeyCode::Esc);

        let actions = client.controller.input(command.into());
        let (configuration, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(params[field], value);
        assert!(params.get("permissionMode").is_none());
        legacy[field] = json!(value);
        if field == "mode" {
            legacy["approval"] = json!("ask");
        }
        assert!(client.answer(configuration, legacy.clone()).is_empty());
        assert_eq!(client.controller.state.runtime, legacy);
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn permissions_picker_displays_legacy_modes_but_only_changes_supported_backends() {
    for (index, permission, mode, approval, label) in [
        (0, "auto", "code", "session", "Auto"),
        (1, "manual", "code", "ask", "Manual"),
        (2, "accept_edits", "code", "accept_edits", "Accept edits"),
        (3, "plan", "plan", "ask", "Plan"),
    ] {
        for explicit_permission in [false, true] {
            let mut client = Client::ready();
            client.controller.paste("keep this draft".into());
            let actions = client.controller.input("/permissions".into());
            let (inspection, _) = client.register(actions, "_aster/session/runtime");
            assert_eq!(client.controller.state.selected, 1);
            assert_eq!(client.controller.state.permission_label(), "Manual");
            let mut status = runtime("model-a", mode, approval);
            if !explicit_permission {
                status.as_object_mut().unwrap().remove("permissionMode");
            }
            client.answer(inspection, status);
            assert_eq!(client.controller.state.selected, index);
            assert_eq!(client.controller.state.permission_mode(), permission);
            assert_eq!(client.controller.state.permission_label(), label);
            let choices = client.controller.state.choices();
            assert_eq!(
                choices
                    .iter()
                    .map(|choice| (choice.label.as_str(), choice.value.as_str()))
                    .collect::<Vec<_>>(),
                [
                    ("Auto", "auto"),
                    ("Manual", "manual"),
                    ("Accept edits", "accept_edits"),
                    ("Plan", "plan"),
                ]
            );

            let actions = key(&mut client.controller, KeyCode::Enter);
            if explicit_permission {
                let (_, params) = client.register(actions, "_aster/session/configure");
                assert_eq!(params["permissionMode"], permission);
                assert!(params.get("approval").is_none());
            } else {
                assert!(actions.is_empty());
                assert_eq!(client.controller.state.phase, Phase::Ready);
            }
            assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
        }
    }
}

#[test]
fn permissions_picker_preserves_a_choice_made_while_runtime_refreshes() {
    let mut client = Client::ready();
    let actions = client.controller.input("/permissions".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");
    assert_eq!(client.controller.state.selected, 1);
    key(&mut client.controller, KeyCode::Down);
    assert_eq!(client.controller.state.selected, 2);
    let mut status = runtime("model-a", "plan", "ask");
    status["permissionMode"] = json!("plan");
    client.answer(inspection, status);
    assert_eq!(client.controller.state.selected, 2);
    assert_eq!(client.controller.state.permission_mode(), "plan");
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (configuration, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["permissionMode"], "accept_edits");
    assert_eq!(client.controller.state.permission_mode(), "plan");
    let mut selected = runtime("model-a", "code", "accept_edits");
    selected["permissionMode"] = json!("accept_edits");
    client.answer(configuration, selected);
    assert_eq!(client.controller.state.mode, "code");
    assert_eq!(client.controller.state.permission_label(), "Accept edits");
}

#[test]
fn rejected_permission_changes_keep_the_current_mode_and_draft() {
    let mut client = Client::ready();
    client.controller.paste("keep this draft".into());
    let actions = client.controller.input("/permissions auto".into());
    let (configuration, _) = client.register(actions, "_aster/session/configure");
    client.controller.message(json!({
        "id": configuration,
        "error": {"code": -32602, "message": "Permission change unavailable"}
    }));
    assert_eq!(client.controller.state.permission_label(), "Manual");
    assert_eq!(client.controller.state.mode, "code");
    assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn an_active_prompt_rejects_runtime_changes_without_losing_the_composer() {
    let mut client = Client::ready();
    client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    client.controller.paste("next instruction".into());

    for command in [
        "/model model-b",
        "/reasoning high",
        "/mode plan",
        "/permissions session",
        "/permissions ask",
    ] {
        assert!(client.controller.input(command.into()).is_empty());
        assert_eq!(client.controller.state.model, "model-a");
        assert_eq!(client.controller.state.mode, "code");
        assert_eq!(client.controller.state.approval_policy, "ask");
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.active_turn_id, turn);
        assert_eq!(
            client.controller.state.editor.lines(),
            &["next instruction"]
        );
    }
}

#[test]
fn permissions_can_change_during_a_prompt_without_replacing_or_finishing_it() {
    for rejected in [false, true] {
        let mut client = Client::ready();
        let prompt = client.start_prompt();
        client.controller.paste("next instruction".into());
        let turn = client.controller.state.active_turn_id.clone();
        let status = client.controller.state.status.clone();
        let started = client.controller.state.started;
        let actions = client
            .controller
            .key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
        let (configuration, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(params["permissionMode"], "accept_edits");
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.status, status);
        assert_eq!(client.controller.state.active_turn_id, turn);
        assert_eq!(client.controller.state.started, started);
        assert_eq!(client.controller.state.permission_mode(), "manual");
        assert!(client.controller.input("/permissions auto".into()).is_empty());

        let actions = if rejected {
            client.controller.message(json!({
                "id": configuration,
                "error": {"code": -32602, "message": "Resolve the pending approval first"}
            }))
        } else {
            client.answer(configuration, runtime("model-a", "code", "accept_edits"))
        };
        assert!(actions.is_empty());
        assert_eq!(client.controller.state.phase, Phase::Busy);
        assert_eq!(client.controller.state.active_turn_id, turn);
        assert_eq!(client.controller.state.status, status);
        assert_eq!(client.controller.state.editor.lines(), &["next instruction"]);
        assert_eq!(
            client.controller.state.permission_mode(),
            if rejected { "manual" } else { "accept_edits" }
        );
        let cancellation = client.controller.cancel();
        assert!(matches!(cancellation.as_slice(), [Action::Notify { method: "session/cancel", .. }]));
        assert!(client.answer(prompt, json!({"stopReason": "cancelled"})).is_empty());
        assert_eq!(client.controller.state.phase, Phase::Ready);
    }
}

#[test]
fn a_late_permission_reply_does_not_complete_the_next_prompt() {
    let mut client = Client::ready();
    let first = client.start_prompt();
    let actions = client.controller.input("/permissions plan".into());
    let (configuration, _) = client.register(actions, "_aster/session/configure");
    assert!(client.answer(first, json!({"stopReason": "end_turn"})).is_empty());
    assert!(client.controller.input("/model model-b".into()).is_empty());
    let second = client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    assert!(client.answer(configuration, runtime("model-a", "plan", "ask")).is_empty());
    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert_eq!(client.controller.state.permission_mode(), "plan");
    assert!(client.answer(second, json!({"stopReason": "end_turn"})).is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn permissions_can_change_while_background_work_continues() {
    let mut client = Client::ready();
    let tasks = task_snapshot(json!([
        {"id": "background", "status": "running", "canStop": true}
    ]));
    publish_tasks(&mut client, tasks.clone());
    let actions = client.controller.input("/permissions plan".into());
    let (configuration, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["permissionMode"], "plan");
    assert!(client.answer(configuration, runtime("model-a", "plan", "ask")).is_empty());
    assert_eq!(client.controller.state.tasks, tasks);
    assert_eq!(client.controller.state.permission_mode(), "plan");
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

fn skill_list() -> Value {
    json!({"skills": [
        {"name":"aster.repo", "description":"Inspect repository files", "active":true, "loaded":true, "kind":"python"},
        {"name":"aster.shell", "description":"Run shell commands", "active":true, "loaded":true, "kind":"python"},
        {"name":"custom.notes", "description":"Write project notes", "active":false, "loaded":true, "kind":"text"}
    ], "total": 3, "truncated": false})
}

#[test]
fn skills_browser_reads_and_searches_during_a_task_without_changing_the_draft() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    client.controller.paste("keep this 中文 draft".into());
    key(&mut client.controller, KeyCode::Left);
    let cursor = client.controller.state.editor.cursor();
    let turn = client.controller.state.active_turn_id.clone();
    let status = client.controller.state.status.clone();
    let actions = client.controller.input("/skills".into());
    let (list, params) = client.register(actions, "_aster/session/skills");
    assert_eq!(params, json!({"sessionId":"session-a"}));
    assert!(client.answer(list, skill_list()).is_empty());
    assert_eq!(client.controller.state.choices().len(), 3);
    client.controller.paste("commands".into());
    assert_eq!(client.controller.state.choices()[0].value, "aster.shell");
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (detail, params) = client.register(actions, "_aster/session/skills");
    assert_eq!(params["name"], "aster.shell");
    assert!(client.answer(detail, json!({"skill":{
        "name":"aster.shell", "text":"Run commands.\nKeeps shell state.",
        "active":true, "loaded":true, "kind":"python", "truncated":false
    }})).is_empty());
    key(&mut client.controller, KeyCode::PageDown);
    assert!(client.controller.state.detail_scroll > 0);
    key(&mut client.controller, KeyCode::Esc);
    assert_eq!(client.controller.state.overlay, Some(Overlay::Skills));
    assert!(client.controller.state.skills.detail.is_none());
    assert_eq!(client.controller.state.query, "commands");
    assert_eq!(client.controller.state.choices()[0].value, "aster.shell");
    key(&mut client.controller, KeyCode::Esc);
    assert_eq!(client.controller.state.overlay, None);
    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert_eq!(client.controller.state.status, status);
    assert_eq!(client.controller.state.editor.lines(), &["keep this 中文 draft"]);
    assert_eq!(client.controller.state.editor.cursor(), cursor);
    assert!(client.answer(prompt, json!({"stopReason":"end_turn"})).is_empty());
}

#[test]
fn skills_browser_ignores_old_replies_after_refresh_back_close_or_conversation_change() {
    let mut client = Client::ready();
    let actions = client.controller.input("/skills".into());
    let (old, _) = client.register(actions, "_aster/session/skills");
    let actions = client.controller.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
    let (current, _) = client.register(actions, "_aster/session/skills");
    client.answer(current, skill_list());
    client.answer(old, json!({"skills":[],"total":0}));
    assert_eq!(client.controller.state.skills.items.len(), 3);
    key(&mut client.controller, KeyCode::Down);
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (detail, _) = client.register(actions, "_aster/session/skills");
    key(&mut client.controller, KeyCode::Esc);
    client.answer(detail, json!({"skill":{"name":"aster.shell","text":"Late text"}}));
    assert!(client.controller.state.skills.detail.is_none());
    assert_eq!(client.controller.state.selected, 1);
    let actions = client.controller.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
    let (refresh, _) = client.register(actions, "_aster/session/skills");
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| { aster_tui::ui::draw(frame, &mut client.controller.state); }).unwrap();
    client.answer(refresh, skill_list());
    assert_eq!(client.controller.state.selected, 1);
    let actions = client.controller.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
    let (closed, _) = client.register(actions, "_aster/session/skills");
    key(&mut client.controller, KeyCode::Esc);
    client.answer(closed, json!({"skills":[],"total":0}));
    assert_eq!(client.controller.state.skills.items.len(), 3);
    let actions = client.controller.input("/skills".into());
    let (foreign, _) = client.register(actions, "_aster/session/skills");
    client.controller.state.conversation_generation += 1;
    client.answer(foreign, json!({"skills":[],"total":0}));
    assert_eq!(client.controller.state.skills.items.len(), 3);
}

#[test]
fn skills_browser_reports_unsupported_and_failed_requests_with_a_retry() {
    let mut client = Client::ready();
    client.controller.paste("draft".into());
    client.controller.state.skills.supported = false;
    assert!(client.controller.input("/skills".into()).is_empty());
    assert!(client.controller.state.skills.error.as_ref().unwrap().contains("Update aster-acp"));
    client.controller.state.skills.supported = true;
    let actions = client.controller.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
    let (failed, _) = client.register(actions, "_aster/session/skills");
    client.controller.message(json!({"id":failed,"error":{"code":-32603,"message":"Read failed"}}));
    assert!(!client.controller.state.skills.loading);
    assert!(client.controller.state.skills.error.as_ref().unwrap().contains("Ctrl+R"));
    let actions = client.controller.key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::CONTROL));
    let (retry, _) = client.register(actions, "_aster/session/skills");
    client.answer(retry, skill_list());
    assert!(client.controller.state.skills.error.is_none());
    assert_eq!(client.controller.state.skills.items.len(), 3);
    assert_eq!(client.controller.state.editor.lines(), &["draft"]);
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn a_runtime_inspection_response_does_not_finish_or_replace_the_active_prompt() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    let started = client.controller.state.started;
    client.controller.paste("keep editing".into());
    let actions = client.controller.input("/model".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");

    assert!(client
        .answer(inspection, runtime("model-a", "code", "ask"))
        .is_empty());

    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert_eq!(client.controller.state.started, started);
    assert_eq!(client.controller.state.editor.lines(), &["keep editing"]);
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn failed_runtime_configuration_retains_the_effective_settings() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model model-b".into());
    let (configuration, _) = client.register(actions, "_aster/session/configure");

    assert!(client
        .controller
        .message(json!({
            "id": configuration,
            "error": {"code": -32602, "message": "Model is unavailable"}
        }))
        .is_empty());

    assert_eq!(client.controller.state.model, "model-a");
    assert_eq!(client.controller.state.mode, "code");
    assert_eq!(client.controller.state.approval_policy, "ask");
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn reasoning_commands_wait_for_backend_confirmation_and_default_clears_selection() {
    let mut client = Client::ready();
    assert!(client
        .controller
        .state
        .commands
        .iter()
        .any(|command| command.name == "/reasoning"));
    let actions = client.controller.input("/reasoning high".into());
    let (id, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["reasoningLevel"], "high");
    assert_eq!(params["sessionId"], "session-a");
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "low");
    let mut selected = runtime("model-a", "code", "ask");
    selected["reasoningLevel"] = json!("high");
    selected["reasoningSelection"] = json!("high");
    assert!(client.answer(id, selected).is_empty());
    assert_eq!(
        client.controller.state.runtime["reasoningSelection"],
        "high"
    );
    assert_eq!(client.controller.state.phase, Phase::Ready);

    let actions = client.controller.input("/reasoning default".into());
    let (id, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params.get("reasoningLevel"), Some(&Value::Null));
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "high");
    assert!(client
        .answer(id, runtime("model-a", "code", "ask"))
        .is_empty());
    assert!(client.controller.state.runtime["reasoningSelection"].is_null());
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "low");
}

#[test]
fn model_picker_arrows_confirm_model_and_effort_in_one_request() {
    let mut client = Client::ready();
    let before = client.controller.state.runtime.clone();
    let actions = client.controller.input("/model".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");
    let mut status = runtime("model-a", "code", "ask");
    status["models"][1]["reasoningLevels"] = json!(["medium", "max"]);
    status["models"][1]["reasoningDefault"] = json!("medium");
    status["models"][1]["reasoningSelection"] = Value::Null;
    client.answer(inspection, status);

    assert!(key(&mut client.controller, KeyCode::Down).is_empty());
    assert!(key(&mut client.controller, KeyCode::Right).is_empty());
    assert!(key(&mut client.controller, KeyCode::Right).is_empty());
    assert_eq!(client.controller.state.selected, 1);
    assert_eq!(client.controller.state.model, "model-a");
    assert_eq!(
        client.controller.state.runtime["reasoningLevel"],
        before["reasoningLevel"]
    );
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (id, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["model"], "model-b");
    assert_eq!(params["reasoningLevel"], "max");
    assert_eq!(client.controller.state.model, "model-a");
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "low");
    let mut accepted = runtime("model-b", "code", "ask");
    accepted["reasoningLevels"] = json!(["medium", "max"]);
    accepted["reasoningSelection"] = json!("max");
    accepted["reasoningLevel"] = json!("max");
    client.answer(id, accepted);
    assert_eq!(client.controller.state.model, "model-b");
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "max");
}

#[test]
fn model_picker_keeps_drafts_per_model_and_discards_them_on_escape() {
    let mut client = Client::ready();
    let before = client.controller.state.runtime.clone();
    let actions = client.controller.input("/model".into());
    let (id, _) = client.register(actions, "_aster/session/runtime");
    client.answer(id, before.clone());
    key(&mut client.controller, KeyCode::Right);
    assert_eq!(
        client.controller.state.model_effort_drafts["model-a"],
        Some("low".into())
    );
    key(&mut client.controller, KeyCode::Down);
    assert!(key(&mut client.controller, KeyCode::Right).is_empty());
    assert!(!client
        .controller
        .state
        .model_effort_drafts
        .contains_key("model-b"));
    key(&mut client.controller, KeyCode::Up);
    key(&mut client.controller, KeyCode::Right);
    assert_eq!(
        client.controller.state.model_effort_drafts["model-a"],
        Some("high".into())
    );
    key(&mut client.controller, KeyCode::Esc);
    assert_eq!(client.controller.state.runtime, before);
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert!(client.controller.state.model_effort_drafts.is_empty());

    let actions = client.controller.input("/model".into());
    client.register(actions, "_aster/session/runtime");
    assert!(client.controller.state.model_effort_drafts.is_empty());
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["model"], "model-a");
    assert!(params.get("reasoningLevel").is_none());
}

#[test]
fn model_picker_does_not_offer_effort_for_unknown_or_unsupported_models() {
    for levels in [Value::Null, json!([])] {
        let mut client = Client::ready();
        let actions = client.controller.input("/model".into());
        let (id, _) = client.register(actions, "_aster/session/runtime");
        let mut status = runtime("model-a", "code", "ask");
        status["models"][1]["reasoningLevels"] = levels;
        client.answer(id, status);
        key(&mut client.controller, KeyCode::Down);
        key(&mut client.controller, KeyCode::Right);
        assert!(client.controller.state.model_effort_drafts.is_empty());
        let actions = key(&mut client.controller, KeyCode::Enter);
        let (_, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(params["model"], "model-b");
        assert!(params.get("reasoningLevel").is_none());
    }
}

#[test]
fn alt_p_opens_the_combined_picker_and_keeps_the_unsent_draft() {
    let mut client = Client::ready();
    client.controller.state.editor.insert_str("keep this draft");
    let actions = client
        .controller
        .key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::ALT));
    client.register(actions, "_aster/session/runtime");
    assert_eq!(client.controller.state.overlay, Some(Overlay::Models));
    assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
    let mut repeat = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    repeat.kind = KeyEventKind::Repeat;
    assert!(client.controller.key(repeat).is_empty());
    key(&mut client.controller, KeyCode::Esc);
    assert_eq!(client.controller.state.editor.lines(), &["keep this draft"]);
}

#[test]
fn effort_alias_opens_at_the_confirmed_level_and_arrows_preview_the_next_value() {
    let mut client = Client::ready();
    assert!(client
        .controller
        .state
        .commands
        .iter()
        .any(|command| command.name == "/effort"));
    let actions = client.controller.input("/effort".into());
    let (id, _) = client.register(actions, "_aster/session/runtime");
    let mut status = runtime("model-a", "code", "ask");
    status["reasoningSelection"] = json!("high");
    status["reasoningLevel"] = json!("high");
    client.answer(id, status);
    assert_eq!(client.controller.state.selected, 2);
    assert!(key(&mut client.controller, KeyCode::Left).is_empty());
    assert_eq!(client.controller.state.selected, 1);
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "high");
    assert!(client.controller.state.query.is_empty());
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (id, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["reasoningLevel"], "low");
    client.answer(id, runtime("model-a", "code", "ask"));
    let actions = client.controller.input("/effort auto".into());
    let (_, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params.get("reasoningLevel"), Some(&Value::Null));
}

#[test]
fn effort_status_reads_the_backend_without_changing_settings_or_work_state() {
    let mut client = Client::ready();
    let before = client.controller.state.runtime.clone();
    let actions = client.controller.input("/effort status".into());
    let (id, params) = client.register(actions, "_aster/session/runtime");
    assert!(params.get("reasoningLevel").is_none());
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert_eq!(client.controller.state.runtime, before);
    client.answer(id, before.clone());
    assert_eq!(client.controller.state.runtime, before);
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Notice { text, error: false } if text.contains("auto (low)") && text.contains("available: low, high"))
    }));
}

#[test]
fn runtime_inspection_keeps_a_model_highlighted_while_the_model_list_reorders() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model".into());
    let (id, _) = client.register(actions, "_aster/session/runtime");
    key(&mut client.controller, KeyCode::Down);
    let mut status = runtime("model-a", "code", "ask");
    status["models"].as_array_mut().unwrap().reverse();
    client.answer(id, status);
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["model"], "model-b");
}

#[test]
fn model_effort_preview_is_rejected_if_the_refreshed_capabilities_remove_it() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model".into());
    let (id, _) = client.register(actions, "_aster/session/runtime");
    key(&mut client.controller, KeyCode::Right);
    key(&mut client.controller, KeyCode::Right);
    assert_eq!(
        client.controller.state.model_effort_drafts["model-a"],
        Some("high".into())
    );
    let mut status = runtime("model-a", "code", "ask");
    status["reasoningLevels"] = json!(["low"]);
    client.answer(id, status);
    assert!(key(&mut client.controller, KeyCode::Enter).is_empty());
    assert_eq!(client.controller.state.overlay, Some(Overlay::Models));
    assert_eq!(client.controller.state.model, "model-a");
    assert_eq!(client.controller.state.runtime["reasoningLevel"], "low");
    assert!(client.controller.state.model_effort_drafts.is_empty());
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Notice { text, .. } if text.contains("Available effort levels changed"))
    }));
}

#[test]
fn reasoning_picker_lists_declared_levels_and_sends_the_selected_value() {
    let mut client = Client::ready();
    let actions = client.controller.input("/reasoning".into());
    let (id, _) = client.register(actions, "_aster/session/runtime");
    assert_eq!(client.controller.state.overlay, Some(Overlay::Reasoning));
    client.answer(id, runtime("model-a", "code", "ask"));
    let choices = client.controller.state.choices();
    assert_eq!(
        choices
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>(),
        ["", "low", "high"]
    );
    assert!(choices[0].detail.contains("Current"));
    assert!(choices[0].label.contains("low"));
    key(&mut client.controller, KeyCode::Down);
    key(&mut client.controller, KeyCode::Down);
    let actions = key(&mut client.controller, KeyCode::Enter);
    let (_, params) = client.register(actions, "_aster/session/configure");
    assert_eq!(params["reasoningLevel"], "high");
}

#[test]
fn unknown_and_unsupported_reasoning_routes_offer_only_model_defaults() {
    for (levels, explanation) in [(Value::Null, "unknown"), (json!([]), "does not support")] {
        let mut client = Client::ready();
        let actions = client.controller.input("/reasoning".into());
        let (id, _) = client.register(actions, "_aster/session/runtime");
        let mut status = runtime("model-a", "code", "ask");
        status["reasoningLevel"] = Value::Null;
        status["reasoningDefault"] = Value::Null;
        status["reasoningLevels"] = levels;
        client.answer(id, status);
        let choices = client.controller.state.choices();
        assert_eq!(choices.len(), 1);
        assert!(choices[0].value.is_empty());
        assert!(choices[0].detail.contains(explanation));
        let actions = key(&mut client.controller, KeyCode::Enter);
        let (_, params) = client.register(actions, "_aster/session/configure");
        assert_eq!(params.get("reasoningLevel"), Some(&Value::Null));
    }
}

#[test]
fn reasoning_configuration_preserves_settings_and_shows_the_backend_rejection_reason() {
    let mut client = Client::ready();
    let before = client.controller.state.runtime.clone();
    let actions = client.controller.input("/reasoning max".into());
    let (id, _) = client.register(actions, "_aster/session/configure");
    let reason = "Reasoning levels are unknown for this route; declare reasoning_levels";
    assert!(client.controller.message(json!({
        "id": id, "error": {"code": -32602, "message": "Invalid params", "data": {"reason": reason}}
    })).is_empty());
    assert_eq!(client.controller.state.runtime, before);
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert!(client
        .controller
        .state
        .entries
        .iter()
        .any(|entry| { matches!(entry, Entry::Notice { text, error: true } if text == reason) }));
}

#[test]
fn older_backends_cannot_silently_ignore_a_reasoning_change() {
    let mut client = Client::ready();
    client
        .controller
        .state
        .runtime
        .as_object_mut()
        .unwrap()
        .remove("reasoningLevels");
    let before = client.controller.state.runtime.clone();
    assert!(client.controller.input("/reasoning high".into()).is_empty());
    assert_eq!(client.controller.state.runtime, before);
    assert_eq!(client.controller.state.phase, Phase::Ready);
    assert!(client.controller.state.entries.iter().any(|entry| {
        matches!(entry, Entry::Notice { text, .. } if text.contains("update aster-acp"))
    }));
}

#[test]
fn runtime_replies_from_before_a_conversation_switch_are_ignored() {
    let mut client = Client::ready();
    let actions = client.controller.input("/model".into());
    let (inspection, _) = client.register(actions, "_aster/session/runtime");
    key(&mut client.controller, KeyCode::Esc);
    let actions = client.controller.resume("session-b".into());
    client.register(actions, "session/load");
    let settings = client.controller.state.runtime.clone();
    let mode = client.controller.state.mode.clone();
    let approval = client.controller.state.approval_policy.clone();

    assert!(client
        .answer(inspection, runtime("model-b", "plan", "session"))
        .is_empty());

    assert_eq!(client.controller.state.runtime, settings);
    assert_eq!(client.controller.state.mode, mode);
    assert_eq!(client.controller.state.approval_policy, approval);
    assert_eq!(client.controller.state.phase, Phase::Busy);
}

fn task_snapshot(jobs: Value) -> Value {
    json!({
        "steps": [{"id": "review", "title": "Review parser", "status": "in_progress", "blockedBy": []}],
        "currentTask": "Review parser", "currentTaskId": "review",
        "waitReason": "Waiting for checks", "jobs": jobs, "truncated": false
    })
}

fn publish_tasks(client: &mut Client, snapshot: Value) {
    client.update(json!({
        "sessionUpdate": "plan",
        "entries": [{"content": "Review parser", "priority": "medium", "status": "in_progress"}],
        "_meta": {"aster": {"tasks": snapshot}}
    }));
}

#[test]
fn stopping_a_selected_background_job_keeps_other_jobs_and_the_prompt_running() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    let turn = client.controller.state.active_turn_id.clone();
    let snapshot = task_snapshot(json!([
        {"id": "check-job", "channel": "checks", "title": "Check parser", "status": "running", "canStop": true},
        {"id": "preview-job", "channel": "preview", "title": "Render preview", "status": "running", "canStop": true}
    ]));
    publish_tasks(&mut client, snapshot.clone());
    assert!(client.controller.state.status.contains("Review parser"));
    assert!(client
        .controller
        .state
        .status
        .contains("Waiting for checks"));
    let actions = client.controller.input("/tasks".into());
    let (inspection, _) = client.register(actions, "_aster/session/tasks");
    assert!(client.answer(inspection, snapshot).is_empty());
    assert_eq!(
        client
            .controller
            .state
            .choices()
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>(),
        ["step:review", "job:check-job", "job:preview-job"]
    );
    key(&mut client.controller, KeyCode::Down);

    let actions = client
        .controller
        .key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
    let (stop, params) = client.register(actions, "_aster/session/stop_task");
    assert_eq!(
        params,
        json!({"sessionId": "session-a", "taskId": "check-job"})
    );
    let after = task_snapshot(json!([
        {"id": "check-job", "channel": "checks", "title": "Check parser", "status": "cancelled", "canStop": false},
        {"id": "preview-job", "channel": "preview", "title": "Render preview", "status": "running", "canStop": true}
    ]));
    assert!(client
        .answer(
            stop,
            json!({"id": "check-job", "stopped": true, "tasks": after})
        )
        .is_empty());

    assert_eq!(
        client.controller.state.tasks["jobs"][0]["status"],
        "cancelled"
    );
    assert_eq!(
        client.controller.state.tasks["jobs"][1]["id"],
        "preview-job"
    );
    assert_eq!(client.controller.state.tasks["jobs"][1]["canStop"], true);
    assert_eq!(client.controller.state.phase, Phase::Busy);
    assert_eq!(client.controller.state.active_turn_id, turn);
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn finished_or_replaced_job_ids_cannot_stop_a_replacement_with_the_same_channel() {
    let mut client = Client::ready();
    client.start_prompt();
    publish_tasks(
        &mut client,
        task_snapshot(json!([
            {"id": "old-job", "channel": "checks", "title": "Old check", "status": "running", "canStop": true}
        ])),
    );
    publish_tasks(
        &mut client,
        task_snapshot(json!([
            {"id": "new-job", "channel": "checks", "title": "New check", "status": "running", "canStop": true},
            {"id": "done-job", "channel": "preview", "title": "Preview", "status": "done", "canStop": false}
        ])),
    );

    for command in ["/stop old-job", "/stop done-job", "/stop checks"] {
        assert!(client.controller.input(command.into()).is_empty());
    }
    assert_eq!(client.controller.state.phase, Phase::Busy);
    let actions = client.controller.input("/stop new-job".into());
    let (_, params) = client.register(actions, "_aster/session/stop_task");
    assert_eq!(params["taskId"], "new-job");
}

#[test]
fn a_running_background_job_can_request_approval_after_the_prompt_finishes() {
    let mut client = Client::ready();
    let prompt = client.start_prompt();
    publish_tasks(
        &mut client,
        task_snapshot(json!([
            {"id": "background", "channel": "checks", "title": "Check parser", "status": "running", "canStop": true}
        ])),
    );
    assert!(client
        .answer(prompt, json!({"stopReason": "end_turn"}))
        .is_empty());

    assert!(client
        .controller
        .permission_request(&permission("background-approval", "session-a"))
        .is_empty());

    assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));
    let actions = key(&mut client.controller, KeyCode::Char('1'));
    assert!(matches!(actions.as_slice(), [Action::Reply { id, result }]
        if id == "background-approval" && result["outcome"]["optionId"] == "allow_once"
    ));
    assert_eq!(client.controller.state.phase, Phase::Ready);
}

#[test]
fn task_progress_does_not_hide_a_pending_execution_decision() {
    let mut client = Client::ready();
    client.start_prompt();
    client
        .controller
        .permission_request(&permission("approval-1", "session-a"));

    publish_tasks(&mut client, task_snapshot(json!([])));

    assert_eq!(client.controller.state.overlay, Some(Overlay::Approval));
    assert_eq!(
        client.controller.state.status,
        "Waiting for execution approval"
    );
    assert_eq!(client.controller.state.approvals[0].id, "approval-1");
    assert_eq!(client.controller.state.choices()[2].value, "reject");
}
