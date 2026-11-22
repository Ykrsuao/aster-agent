// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::attachments::Attachment;

fn ready() -> Controller {
    let mut controller = Controller::new(
        State::new("C:/work".into(), "test".into(), false),
        None,
        false,
    );
    controller.ready("Ready");
    controller.state.session_id = Some("session-a".into());
    controller.state.mcp.supported = true;
    controller.state.commands_received = true;
    controller
}

fn register(controller: &mut Controller, id: u64, actions: Vec<Action>, method: &str) -> Value {
    assert_eq!(actions.len(), 1);
    let Action::Request {
        method: actual,
        params,
        pending,
    } = actions.into_iter().next().unwrap()
    else {
        panic!("expected a request, not a backend restart")
    };
    assert_eq!(actual, method);
    controller.registered(id, pending);
    params
}

fn answer(controller: &mut Controller, id: u64, result: Value) -> Vec<Action> {
    controller.message(json!({"id":id,"result":result}))
}

fn status(state: &str) -> Value {
    json!({"servers":[
        {"name":"local", "source":"project", "transport":"stdio", "state":state,
         "toolCount":if state == "ready" { 2 } else { 0 }, "error":null},
        {"name":"remote", "source":"user", "transport":"http", "state":"disabled",
         "toolCount":0, "error":null}
    ],"warnings":[]})
}

fn key(controller: &mut Controller, code: KeyCode, modifiers: KeyModifiers) -> Vec<Action> {
    controller.key(KeyEvent::new(code, modifiers))
}

#[test]
fn mcp_status_is_read_only_during_a_prompt_and_escape_keeps_input_and_attachments() {
    let mut c = ready();
    c.input("current task".into());
    c.paste("next 中文 draft".into());
    key(&mut c, KeyCode::Left, KeyModifiers::NONE);
    let attachment = Attachment::Text {
        name: "notes.txt".into(),
        uri: "file:///notes.txt".into(),
        text: "Keep attached context".into(),
    };
    c.add_attachment(c.state.conversation_generation, Ok(attachment.clone()));
    let cursor = c.state.editor.cursor();
    let turn = c.state.active_turn_id.clone();
    let before = c.state.status.clone();
    let actions = c.input("/mcp".into());
    let params = register(&mut c, 1, actions, "_aster/session/mcp/status");
    assert_eq!(params, json!({"sessionId":"session-a"}));
    assert!(answer(&mut c, 1, status("ready")).is_empty());
    assert_eq!(c.state.choices().len(), 2);
    c.paste("project".into());
    assert_eq!(c.state.choices()[0].value, "local");
    assert!(key(&mut c, KeyCode::Enter, KeyModifiers::NONE).is_empty());
    assert!(key(&mut c, KeyCode::Esc, KeyModifiers::NONE).is_empty());
    assert_eq!(c.state.phase, Phase::Busy);
    assert_eq!(c.state.active_turn_id, turn);
    assert_eq!(c.state.status, before);
    assert_eq!(c.state.editor.lines(), &["next 中文 draft"]);
    assert_eq!(c.state.editor.cursor(), cursor);
    assert_eq!(c.state.attachments, vec![attachment]);
    assert!(c.state.overlay.is_none());
}

#[test]
fn mcp_retry_rejects_foreground_and_background_work_but_status_still_works() {
    for foreground in [false, true] {
        let mut c = ready();
        if foreground {
            c.input("current task".into());
        } else {
            c.state.tasks = json!({"jobs":[{"id":"worker","status":"running","canStop":true}]});
        }
        let phase = c.state.phase;
        assert!(c.input("/mcp reconnect local".into()).is_empty());
        assert!(c
            .state
            .mcp
            .error
            .as_deref()
            .unwrap()
            .contains("background work"));
        assert_eq!(c.state.phase, phase);
        let actions = key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
        register(&mut c, 2, actions, "_aster/session/mcp/status");
        answer(&mut c, 2, status("failed"));
        assert_eq!(c.state.phase, phase);
        assert!(key(&mut c, KeyCode::Char('k'), KeyModifiers::CONTROL).is_empty());
    }
}

#[test]
fn mcp_retry_targets_one_server_and_finishes_after_the_panel_is_closed() {
    let mut c = ready();
    c.paste("keep this draft".into());
    let actions = c.input("/mcp".into());
    register(&mut c, 1, actions, "_aster/session/mcp/status");
    answer(&mut c, 1, status("failed"));
    let actions = key(&mut c, KeyCode::Char('k'), KeyModifiers::CONTROL);
    let params = register(&mut c, 2, actions, "_aster/session/mcp/reconnect");
    assert_eq!(params, json!({"sessionId":"session-a","name":"local"}));
    assert_eq!(c.state.phase, Phase::Busy);
    assert!(!c.state.can_cancel());
    assert!(key(&mut c, KeyCode::Esc, KeyModifiers::NONE).is_empty());
    assert!(answer(&mut c, 2, status("ready")).is_empty());
    assert_eq!(c.state.phase, Phase::Ready);
    assert_eq!(c.state.session_id.as_deref(), Some("session-a"));
    assert_eq!(c.state.editor.lines(), &["keep this draft"]);
    assert!(c.state.overlay.is_none());
    // Closing the inspector discards its content response, not the operation result.
    assert_eq!(c.state.mcp.servers[0]["state"], "failed");
    let actions = c.input("/mcp".into());
    register(&mut c, 3, actions, "_aster/session/mcp/status");
    answer(&mut c, 3, status("ready"));
    assert_eq!(c.state.mcp.servers[0]["state"], "ready");
}

#[test]
fn mcp_retry_errors_keep_the_session_and_allow_a_later_retry() {
    let mut c = ready();
    c.paste("do not lose me".into());
    let actions = c.input("/mcp reconnect local".into());
    register(&mut c, 1, actions, "_aster/session/mcp/reconnect");
    assert!(c
        .message(json!({"id":1,"error":{
            "code":-32602,"message":"Request unavailable",
            "data":{"reason":"Stop background jobs before retrying.\u{1b}[2J"}
        }}))
        .is_empty());
    assert_eq!(
        c.state.mcp.error.as_deref(),
        Some("Stop background jobs before retrying.")
    );
    assert_eq!(c.state.phase, Phase::Ready);
    assert_eq!(c.state.session_id.as_deref(), Some("session-a"));
    assert!(c.fatal_error.is_none());
    let actions = c.input("/mcp reconnect local".into());
    register(&mut c, 2, actions, "_aster/session/mcp/reconnect");
    answer(&mut c, 2, status("ready"));
    assert_eq!(c.state.mcp.servers[0]["toolCount"], 2);
    assert_eq!(c.state.editor.lines(), &["do not lose me"]);
}

#[test]
fn mcp_status_ignores_old_refresh_closed_view_and_foreign_session_replies() {
    let mut c = ready();
    let actions = c.input("/mcp".into());
    register(&mut c, 1, actions, "_aster/session/mcp/status");
    let actions = key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    register(&mut c, 2, actions, "_aster/session/mcp/status");
    answer(&mut c, 1, status("failed"));
    assert!(c.state.mcp.loading);
    assert!(c.state.mcp.servers.is_empty());
    answer(&mut c, 2, status("ready"));
    let actions = key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    register(&mut c, 3, actions, "_aster/session/mcp/status");
    key(&mut c, KeyCode::Esc, KeyModifiers::NONE);
    answer(&mut c, 3, status("failed"));
    assert_eq!(c.state.mcp.servers[0]["state"], "ready");
    let actions = c.input("/mcp".into());
    register(&mut c, 4, actions, "_aster/session/mcp/status");
    c.state.session_id = Some("session-b".into());
    answer(&mut c, 4, status("failed"));
    assert_eq!(c.state.mcp.servers[0]["state"], "ready");
    c.state.session_id = Some("session-a".into());
    let actions = c.input("/mcp".into());
    register(&mut c, 5, actions, "_aster/session/mcp/status");
    c.reset_conversation();
    answer(&mut c, 5, status("failed"));
    assert!(c.state.mcp.servers.is_empty());
}

#[test]
fn mcp_status_follows_session_restore_and_a_failed_restore_keeps_the_source() {
    for failed in [false, true] {
        let mut c = ready();
        c.state
            .entries
            .push(Entry::Assistant("original answer".into()));
        let actions = c.input("/mcp".into());
        register(&mut c, 1, actions, "_aster/session/mcp/status");
        let actions = c.resume("session-b".into());
        let params = register(&mut c, 2, actions, "session/load");
        assert_eq!(params["mcpServers"], json!([]));
        answer(&mut c, 1, status("failed"));
        assert!(c.state.mcp.servers.is_empty());
        if failed {
            c.message(json!({"id":2,"error":{"code":-32602,"message":"Load failed"}}));
            assert!(c
                .state
                .entries
                .iter()
                .any(|entry| matches!(entry, Entry::Assistant(text) if text == "original answer")));
        } else {
            let actions = answer(&mut c, 2, json!({}));
            register(&mut c, 4, actions, "session/close");
            answer(&mut c, 4, json!({}));
        }
        assert_eq!(c.state.phase, Phase::Ready);
        assert!(c.state.mcp.servers.is_empty());
        let expected = if failed { "session-a" } else { "session-b" };
        let actions = c.input("/mcp".into());
        let params = register(&mut c, 3, actions, "_aster/session/mcp/status");
        assert_eq!(params["sessionId"], expected);
        answer(&mut c, 3, status("ready"));
        assert_eq!(c.state.mcp.servers[0]["state"], "ready");
    }
}

#[test]
fn a_refresh_during_mcp_retry_does_not_leave_a_connecting_snapshot() {
    let mut c = ready();
    let actions = c.input("/mcp reconnect local".into());
    register(&mut c, 1, actions, "_aster/session/mcp/reconnect");
    let actions = key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    register(&mut c, 2, actions, "_aster/session/mcp/status");
    answer(&mut c, 2, status("connecting"));
    let actions = answer(&mut c, 1, status("ready"));
    register(&mut c, 3, actions, "_aster/session/mcp/status");
    assert_eq!(c.state.phase, Phase::Ready);
    answer(&mut c, 3, status("ready"));
    assert_eq!(c.state.mcp.servers[0]["state"], "ready");
}

#[test]
fn mcp_retry_keeps_messages_queued_until_discovery_finishes() {
    let mut c = ready();
    let actions = c.input("/mcp reconnect local".into());
    register(&mut c, 1, actions, "_aster/session/mcp/reconnect");
    key(&mut c, KeyCode::Esc, KeyModifiers::NONE);
    assert!(c.input("next task".into()).is_empty());
    assert_eq!(c.state.queued.len(), 1);
    let actions = answer(&mut c, 1, status("ready"));
    let params = register(&mut c, 2, actions, "session/prompt");
    assert_eq!(params["prompt"][0]["text"], "next task");
    assert!(c.state.queued.is_empty());
    assert_eq!(c.state.phase, Phase::Busy);
}

#[test]
fn mcp_capability_is_negotiated_again_after_backend_reconnection() {
    let mut c = ready();
    c.state.mcp.servers = status("ready")["servers"].as_array().unwrap().clone();
    c.reconnected();
    assert!(!c.state.mcp.supported);
    assert!(c.state.mcp.servers.is_empty());
    c.registered(1, Pending::Initialize);
    let actions = answer(
        &mut c,
        1,
        json!({
            "protocolVersion":1,
            "agentCapabilities":{"loadSession":true,"sessionCapabilities":{"list":{},"close":{}}},
            "_meta":{"aster":{"mcp":true}}
        }),
    );
    assert!(c.state.mcp.supported);
    register(&mut c, 2, actions, "session/list");
    let actions = answer(&mut c, 2, json!({"sessions":[]}));
    let params = register(&mut c, 3, actions, "session/new");
    assert_eq!(params["mcpServers"], json!([]));
    answer(&mut c, 3, json!({"sessionId":"new-session"}));
    let actions = c.input("/mcp".into());
    register(&mut c, 4, actions, "_aster/session/mcp/status");
    c.failure("disconnected");
    answer(&mut c, 4, status("ready"));
    assert!(!c.state.mcp.loading);
    assert!(c.state.mcp.servers.is_empty());
    assert!(c.state.mcp.error.as_deref().unwrap().contains("/reconnect"));
}

#[test]
fn mcp_unavailable_invalid_and_empty_responses_offer_recovery_without_fake_rows() {
    let mut c = ready();
    c.state.mcp.supported = false;
    assert!(c.input("/mcp".into()).is_empty());
    assert!(c
        .state
        .mcp
        .error
        .as_deref()
        .unwrap()
        .contains("Update aster-acp"));
    c.state.mcp.supported = true;
    assert!(c.input("/mcp reconnect".into()).is_empty());
    assert!(c.state.mcp.error.as_deref().unwrap().contains("<name>"));
    let actions = key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    register(&mut c, 1, actions, "_aster/session/mcp/status");
    answer(&mut c, 1, json!({"servers":null}));
    assert!(c
        .state
        .mcp
        .error
        .as_deref()
        .unwrap()
        .contains("Invalid MCP status"));
    let actions = key(&mut c, KeyCode::Char('r'), KeyModifiers::CONTROL);
    register(&mut c, 2, actions, "_aster/session/mcp/status");
    answer(
        &mut c,
        2,
        json!({"servers":[],"warnings":["Project MCP configuration requires trust.\u{1b}[2J"]}),
    );
    assert!(c.state.mcp.error.is_none());
    assert!(c.state.choices().is_empty());
    assert_eq!(
        c.state.mcp.warnings,
        ["Project MCP configuration requires trust."]
    );
    for code in [KeyCode::Char('r'), KeyCode::Char('k')] {
        assert!(c
            .key(KeyEvent::new_with_kind(
                code,
                KeyModifiers::CONTROL,
                KeyEventKind::Repeat
            ))
            .is_empty());
    }
}
