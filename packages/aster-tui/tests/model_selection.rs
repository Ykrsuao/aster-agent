// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aster_tui::{
    controller::{Action, Controller},
    state::{Overlay, Phase, State},
    ui,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};
use serde_json::{json, Value};

fn enter() -> KeyEvent {
    KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
}

fn render(state: &mut State) -> String {
    let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
    terminal
        .draw(|frame| {
            ui::draw(frame, state);
        })
        .unwrap();
    terminal
        .backend()
        .buffer()
        .content
        .chunks(100)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

fn register(controller: &mut Controller, actions: Vec<Action>, id: u64) {
    let Action::Request { pending, .. } = actions.into_iter().next().unwrap() else {
        panic!("expected request");
    };
    controller.registered(id, pending);
}

fn reply(controller: &mut Controller, actions: Vec<Action>, id: u64, result: Value) -> Vec<Action> {
    register(controller, actions, id);
    controller.message(json!({"id":id,"result":result}))
}

fn unselected() -> Controller {
    let mut state = State::new("workspace".into(), "".into(), false);
    state.phase = Phase::Ready;
    state.model_selection.supported = true;
    let mut controller = Controller::new(state, None, false);
    controller.paste("kept conversation draft".into());
    controller
}

#[test]
fn removed_connect_command_does_not_open_a_form_or_send_a_prompt() {
    let mut state = State::new("workspace".into(), "existing".into(), false);
    state.phase = Phase::Ready;
    state.session_id = Some("session".into());
    let mut controller = Controller::new(state, None, false);
    controller.paste("kept conversation draft".into());
    assert!(!controller
        .state
        .commands
        .iter()
        .any(|command| command.name == "/connect"));
    assert!(controller.input("/connect".into()).is_empty());
    assert_eq!(controller.state.overlay, None);
    assert_eq!(controller.state.model, "existing");
    assert_eq!(controller.state.session_id.as_deref(), Some("session"));
    assert_eq!(
        controller.state.editor.lines(),
        &["kept conversation draft"]
    );
    assert!(render(&mut controller.state).contains("Unknown command: /connect"));
}

#[test]
fn live_model_switcher_still_lists_existing_models() {
    let mut state = State::new("workspace".into(), "existing".into(), false);
    state.phase = Phase::Ready;
    state.session_id = Some("session".into());
    state.runtime_supported = true;
    let mut controller = Controller::new(state, None, false);
    let actions = controller.input("/model".into());
    assert!(matches!(
        &actions[0],
        Action::Request {
            method: "_aster/session/runtime",
            ..
        }
    ));
    reply(
        &mut controller,
        actions,
        1,
        json!({
            "model":"existing", "models":[{"id":"existing"},{"id":"other"}]
        }),
    );
    assert_eq!(controller.state.overlay, Some(Overlay::Models));
    assert_eq!(controller.state.choices().len(), 2);
    controller.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    let actions = controller.key(enter());
    assert!(matches!(&actions[0], Action::Request {
        method: "_aster/session/configure", params, ..
    } if params["model"] == "other"));
}

#[test]
fn startup_model_choice_selects_a_saved_model_before_opening_a_session() {
    let mut controller = unselected();
    let actions = controller.input("/model".into());
    assert!(matches!(
        &actions[0],
        Action::Request {
            method: "_aster/models/list",
            ..
        }
    ));
    reply(
        &mut controller,
        actions,
        1,
        json!({"models":[{"id":"official","name":"official"}]}),
    );
    let actions = controller.key(enter());
    assert!(matches!(
        &actions[0],
        Action::Request {
            method: "_aster/models/select",
            ..
        }
    ));
    let actions = reply(&mut controller, actions, 2, json!({"model":"official"}));
    assert_eq!(controller.state.model, "official");
    assert_eq!(
        controller.state.editor.lines(),
        &["kept conversation draft"]
    );
    assert!(matches!(
        &actions[0],
        Action::Request {
            method: "session/list",
            ..
        }
    ));
}

#[test]
fn empty_model_picker_explains_file_configuration() {
    let mut controller = unselected();
    let actions = controller.input("/model".into());
    assert!(render(&mut controller.state).contains("Loading configured models"));
    assert!(reply(&mut controller, actions, 1, json!({"models":[]})).is_empty());
    assert_eq!(controller.state.overlay, Some(Overlay::Models));
    assert!(controller.state.choices().is_empty());
    assert!(controller.key(enter()).is_empty());
    let screen = render(&mut controller.state);
    assert!(screen.contains("No models configured."));
    assert!(screen.contains("llm_config.yaml"));
    assert!(!screen.contains("/connect"));
}

#[test]
fn first_start_opens_only_the_model_picker_even_when_no_models_are_configured() {
    for configured in [false, true] {
        let state = State::new("workspace".into(), "".into(), false);
        let mut controller = Controller::new(state, None, false);
        let initialize = controller.initialize();
        let actions = reply(
            &mut controller,
            vec![initialize],
            1,
            json!({
                "protocolVersion":1,
                "agentCapabilities":{
                    "loadSession":true,
                    "sessionCapabilities":{"list":{},"close":{}}
                },
                "_meta":{"aster":{"modelSelection":true}}
            }),
        );
        assert!(matches!(
            &actions[0],
            Action::Request {
                method: "_aster/models/list",
                ..
            }
        ));
        let models = if configured {
            json!([{"id":"official"}])
        } else {
            json!([])
        };
        assert!(reply(&mut controller, actions, 2, json!({"models":models})).is_empty());
        assert_eq!(controller.state.overlay, Some(Overlay::Models));
    }
}

#[test]
fn explicit_model_before_first_session_preserves_the_draft() {
    let mut controller = unselected();
    let actions = controller.input("/model official".into());
    assert!(matches!(&actions[0], Action::Request {
        method:"_aster/models/select", params, ..
    } if params["model"] == "official"));
    let actions = reply(&mut controller, actions, 1, json!({"model":"official"}));
    assert_eq!(controller.state.model, "official");
    assert_eq!(
        controller.state.editor.lines(),
        &["kept conversation draft"]
    );
    assert!(matches!(
        &actions[0],
        Action::Request {
            method: "session/list",
            ..
        }
    ));
}

#[test]
fn a_late_model_list_cannot_reopen_the_picker_or_override_a_selection() {
    let mut controller = unselected();
    let actions = controller.input("/model".into());
    register(&mut controller, actions, 1);
    controller.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    controller.message(json!({"id":1,"result":{"models":[{"id":"old"}]}}));
    assert_eq!(controller.state.overlay, None);

    let actions = controller.input("/model".into());
    register(&mut controller, actions, 2);
    let actions = controller.input("/model official".into());
    reply(&mut controller, actions, 3, json!({"model":"official"}));
    assert!(controller
        .message(json!({"id":2,"result":{"models":[]}}))
        .is_empty());
    assert_eq!(controller.state.model, "official");
    assert_eq!(controller.state.overlay, None);
    assert_eq!(controller.state.runtime["models"][0]["id"], "old");
}

#[test]
fn selection_failure_keeps_the_draft_without_opening_a_session() {
    let mut controller = unselected();
    let actions = controller.input("/model missing".into());
    register(&mut controller, actions, 1);
    assert!(controller.message(json!({
        "id":1,"error":{"code":-32602,"message":"Invalid params","data":{"reason":"Choose a configured model."}}
    })).is_empty());
    assert!(controller.state.model.is_empty());
    assert!(controller.state.session_id.is_none());
    assert_eq!(
        controller.state.editor.lines(),
        &["kept conversation draft"]
    );
    assert!(render(&mut controller.state).contains("Choose a configured model."));
}
