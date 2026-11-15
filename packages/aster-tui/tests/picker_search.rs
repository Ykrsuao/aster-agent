// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aster_tui::{
    controller::Controller,
    state::{Overlay, Phase, State},
    ui,
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};

#[test]
fn visible_picker_search_fields_accept_typing_and_paste() {
    for overlay in [
        Overlay::Commands,
        Overlay::Sessions,
        Overlay::Drafts,
        Overlay::Tools,
        Overlay::Search,
        Overlay::Queue,
        Overlay::Rewind,
        Overlay::Tree,
        Overlay::Files,
        Overlay::Attachments,
        Overlay::Models,
        Overlay::Modes,
        Overlay::Permissions,
        Overlay::Tasks,
        Overlay::Hud,
        Overlay::Changes,
    ] {
        let mut state = State::new("workspace".into(), "old".into(), false);
        state.phase = Phase::Ready;
        state.conversation_changes = false;
        let mut controller = Controller::new(state, None, false);
        controller.paste("keep this draft".into());
        controller.state.open(overlay);
        let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
        terminal
            .draw(|frame| {
                assert!(
                    ui::draw(frame, &mut controller.state).query.is_some(),
                    "{overlay:?} should show its search field"
                );
            })
            .unwrap();

        for character in "搜A".chars() {
            controller.key(KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE));
        }
        assert_eq!(controller.state.query, "搜A", "typing in {overlay:?}");
        controller.key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(controller.state.query, "搜", "backspace in {overlay:?}");
        controller.paste(" pasted".into());
        assert_eq!(
            controller.state.query, "搜 pasted",
            "pasting in {overlay:?}"
        );
        assert_eq!(controller.state.editor.lines(), &["keep this draft"]);
    }
}

#[test]
fn overlays_without_search_do_not_collect_query_text() {
    for overlay in [
        Overlay::Help,
        Overlay::Diagnostics,
        Overlay::HudInfo,
        Overlay::Changes,
    ] {
        let mut state = State::new("workspace".into(), "old".into(), false);
        state.conversation_changes = true;
        state.open(overlay);
        let mut controller = Controller::new(state, None, false);
        let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
        terminal
            .draw(|frame| {
                assert!(
                    ui::draw(frame, &mut controller.state).query.is_none(),
                    "{overlay:?} should not show a search field"
                );
            })
            .unwrap();
        controller.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        controller.paste("text".into());
        assert!(controller.state.query.is_empty(), "{overlay:?}");
    }
}
