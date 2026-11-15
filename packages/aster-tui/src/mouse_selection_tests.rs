// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Selection regressions exercise the actual renderer and production mouse router.

use super::*;
use aster_tui::state::{Entry, Phase, Tool};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    backend::TestBackend,
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier},
};
use serde_json::json;
use unicode_width::UnicodeWidthStr;

const SIZE: (u16, u16) = (110, 32);

fn controller() -> Controller {
    let mut state = State::new("C:/selection-workspace".into(), "test-model".into(), false);
    state.phase = Phase::Ready;
    state.monochrome = false;
    Controller::new(state, None, false)
}

fn render(controller: &mut Controller, size: (u16, u16)) -> (Buffer, ui::HitAreas) {
    let mut terminal = Terminal::new(TestBackend::new(size.0, size.1)).unwrap();
    let mut hits = ui::HitAreas::default();
    terminal
        .draw(|frame| hits = ui::draw(frame, &mut controller.state))
        .unwrap();
    (terminal.backend().buffer().clone(), hits)
}

fn locate(buffer: &Buffer, area: Rect, needle: &str) -> Position {
    let mut rendered = Vec::new();
    for y in area.y..area.bottom() {
        let mut visible = String::new();
        let mut columns = Vec::new();
        let mut x = area.x;
        while x < area.right() {
            let glyph = buffer[(x, y)].symbol();
            columns.extend(std::iter::repeat_n(x, glyph.len()));
            visible.push_str(glyph);
            x += glyph.width().max(1) as u16;
        }
        if let Some(byte) = visible.find(needle) {
            return Position::new(columns[byte], y);
        }
        rendered.push(visible);
    }
    panic!(
        "visible text {needle:?} in {area:?}: {}",
        rendered.join("\n")
    );
}

fn pointer(
    controller: &mut Controller,
    size: (u16, u16),
    kind: MouseEventKind,
    position: Position,
    modifiers: KeyModifiers,
) -> MouseOutcome {
    let (_, hits) = render(controller, size);
    handle_mouse(
        controller,
        &hits,
        MouseEvent {
            kind,
            column: position.x,
            row: position.y,
            modifiers,
        },
    )
}

fn copy(actions: Vec<Action>) -> String {
    match actions.as_slice() {
        [Action::CopySelection { text, cut: false }] => text.clone(),
        _ => panic!("one automatic copy expected: {actions:?}"),
    }
}

fn drag(
    controller: &mut Controller,
    size: (u16, u16),
    from: Position,
    to: Position,
) -> (String, Buffer) {
    for (kind, position) in [
        (MouseEventKind::Down(MouseButton::Left), from),
        (MouseEventKind::Drag(MouseButton::Left), to),
    ] {
        let outcome = pointer(controller, size, kind, position, KeyModifiers::NONE);
        assert!(outcome.changed);
        assert!(outcome.actions.is_empty());
    }
    let (highlighted, _) = render(controller, size);
    let outcome = pointer(
        controller,
        size,
        MouseEventKind::Up(MouseButton::Left),
        to,
        KeyModifiers::NONE,
    );
    (copy(outcome.actions), highlighted)
}

fn drag_text(controller: &mut Controller, visible: &str) -> (String, Buffer) {
    let (buffer, hits) = render(controller, SIZE);
    let from = locate(&buffer, hits.overlay_area.unwrap_or(buffer.area), visible);
    drag(
        controller,
        SIZE,
        from,
        Position::new(from.x + visible.width() as u16, from.y),
    )
}

fn ctrl_copy(controller: &mut Controller) -> Vec<Action> {
    controller.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
}

#[test]
fn hud_click_opens_context_but_drag_copies_and_survives_transcript_updates() {
    let mut controller = controller();
    controller.state.usage = "context n/a".into();
    controller.state.editor.insert_str("unsent draft");
    let (buffer, _) = render(&mut controller, SIZE);
    let from = locate(&buffer, buffer.area, "context n/a");
    for kind in [
        MouseEventKind::Down(MouseButton::Left),
        MouseEventKind::Up(MouseButton::Left),
    ] {
        assert!(
            pointer(&mut controller, SIZE, kind, from, KeyModifiers::NONE)
                .actions
                .is_empty()
        );
    }
    assert_eq!(controller.state.overlay, Some(Overlay::Context));
    controller.state.overlay = None;
    let (buffer, _) = render(&mut controller, SIZE);
    let from = locate(&buffer, buffer.area, "selection-workspace");
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Down(MouseButton::Left),
        from,
        KeyModifiers::NONE,
    );
    controller
        .state
        .entries
        .push(Entry::Assistant("new streamed output".into()));
    let to = Position::new(from.x + "selection-workspace".width() as u16, from.y);
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Drag(MouseButton::Left),
        to,
        KeyModifiers::NONE,
    );
    let copied = copy(
        pointer(
            &mut controller,
            SIZE,
            MouseEventKind::Up(MouseButton::Left),
            to,
            KeyModifiers::NONE,
        )
        .actions,
    );
    assert_eq!(copied, "selection-workspace");
    assert_eq!(controller.state.overlay, None);
    assert_eq!(controller.state.editor.lines(), &["unsent draft"]);
}

fn tool(expanded: bool) -> Tool {
    Tool {
        id: "read-fixture".into(),
        title: "Read project.rs".into(),
        status: "completed".into(),
        content: "tool\t中文👩‍💻e\u{301}\nsecond output".into(),
        diff: "--- a/project.rs\n+++ b/project.rs\n@@ -1 +1 @@\n-before\n+after".into(),
        expanded,
        started: None,
        elapsed: Some(0.25),
    }
}

#[test]
fn copy_feedback_preserves_selections_through_completion_and_expiry() {
    for (overlay, text) in [
        (None, "selection-workspace"),
        (None, "Selected reply"),
        (None, "keep draft"),
        (Some(Overlay::Help), "Send / queue while working"),
        (Some(Overlay::Diagnostics), "Backend message: 中文👩‍💻"),
    ] {
        let mut controller = controller();
        controller.paste("keep draft".into());
        controller
            .state
            .entries
            .push(Entry::Assistant("Selected reply".into()));
        controller
            .state
            .diagnostics
            .push_back("Backend message: 中文👩‍💻".into());
        if let Some(overlay) = overlay {
            controller.state.open(overlay);
        }
        assert_eq!(drag_text(&mut controller, text).0, text);
        assert!(controller.state.copy_confirmation.is_none());

        let target = CopyTarget::capture(&controller);
        complete_copy(
            &mut controller,
            &mut IoTasks::default(),
            &target,
            false,
            Ok(text.chars().count()),
        );
        let (buffer, hits) = render(&mut controller, SIZE);
        let label = format!("copied {} chars to clipboard", text.chars().count());
        let hint = locate(&buffer, buffer.area, &label);
        assert_eq!(buffer[hint].fg, Color::Rgb(177, 185, 249));
        assert!(!buffer[hint].modifier.contains(Modifier::BOLD));
        assert_eq!(hint.x + label.width() as u16, hits.composer.right());
        assert_eq!(hint.y + 1, hits.composer.y);
        let selected = locate(&buffer, buffer.area, text);
        assert_eq!(buffer[selected].bg, Color::Rgb(56, 82, 116));
        assert_eq!(copy(ctrl_copy(&mut controller)), text);

        controller
            .state
            .copy_confirmation
            .as_mut()
            .unwrap()
            .expires_at = Instant::now() - Duration::from_millis(1);
        controller.tick();
        let (buffer, _) = render(&mut controller, SIZE);
        let selected = locate(&buffer, buffer.area, text);
        assert_eq!(buffer[selected].bg, Color::Rgb(56, 82, 116));
        assert_eq!(copy(ctrl_copy(&mut controller)), text);
        assert!(controller.state.copy_confirmation.is_none());
        assert_eq!(controller.state.editor.lines(), &["keep draft"]);
        assert_eq!(controller.state.entries.len(), 1);
    }
}

#[test]
fn conversation_drag_copies_visible_text_and_keeps_the_draft() {
    let mut controller = controller();
    controller.paste("keep selected draft".into());
    controller.state.editor.select_all();
    controller.state.entries = vec![
        Entry::User("用户：复制这一段 👩‍💻 e\u{301}".into()),
        Entry::Assistant("Reply with **colored** words and `code`.".into()),
    ];
    let original = controller.state.editor.lines().to_vec();
    let text = "复制这一段 👩‍💻 e\u{301}";
    let (selected, highlighted) = drag_text(&mut controller, text);
    assert_eq!(selected, text);
    let from = locate(&highlighted, highlighted.area, text);
    let mut x = from.x;
    while x < from.x + text.width() as u16 {
        assert_eq!(highlighted[(x, from.y)].bg, Color::Rgb(56, 82, 116));
        x += highlighted[(x, from.y)].symbol().width().max(1) as u16;
    }
    assert_eq!(copy(ctrl_copy(&mut controller)), text);
    assert!(controller
        .key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL))
        .is_empty());
    assert_eq!(controller.state.editor.lines(), original);
    assert!(controller.state.exit_confirmation.is_none());
    assert!(!controller.is_exiting());

    let (selected, _) = drag_text(&mut controller, "colored words and code");
    assert_eq!(selected, "colored words and code");
    assert_eq!(copy(ctrl_copy(&mut controller)), selected);
}

#[test]
fn tool_output_and_inspectors_copy_original_tabs_and_visible_diffs() {
    let mut controller = controller();
    controller.state.entries.push(Entry::Tool(tool(true)));
    let visible = "tool    中文👩‍💻e\u{301}";
    let expected = "tool\t中文👩‍💻e\u{301}";
    assert_eq!(drag_text(&mut controller, visible).0, expected);
    assert!(matches!(&controller.state.entries[0], Entry::Tool(tool) if tool.expanded));
    controller.state.open(Overlay::Tools);
    assert_eq!(drag_text(&mut controller, visible).0, expected);
    assert_eq!(drag_text(&mut controller, "-before").0, "-before");
    controller.state.open(Overlay::Changes);
    controller.state.conversation_changes = true;
    assert_eq!(drag_text(&mut controller, "+after").0, "+after");
}

#[test]
fn help_diagnostics_workspace_sidebar_and_filter_text_are_selectable() {
    let mut controller = controller();
    assert_eq!(
        drag_text(&mut controller, "selection-workspace").0,
        "selection-workspace"
    );
    controller
        .state
        .entries
        .push(Entry::Assistant("Conversation content".into()));
    controller.state.show_sidebar = true;
    controller.state.sessions.push(aster_tui::state::Session {
        id: "saved-session".into(),
        title: "Saved conversation".into(),
        updated: "today".into(),
    });
    let (buffer, _) = render(&mut controller, SIZE);
    let from = locate(&buffer, buffer.area, "Saved conversation");
    let to = Position::new(from.x + "Saved conversation".width() as u16, from.y);
    assert_eq!(
        drag(&mut controller, SIZE, from, to).0,
        "Saved conversation"
    );
    assert_eq!(controller.state.phase, Phase::Ready);
    controller.state.open(Overlay::Help);
    assert_eq!(
        drag_text(&mut controller, "Send / queue while working").0,
        "Send / queue while working"
    );
    controller
        .state
        .diagnostics
        .push_back("Backend message: 中文👩‍💻".into());
    controller.state.open(Overlay::Diagnostics);
    assert_eq!(
        drag_text(&mut controller, "Backend message: 中文👩‍💻").0,
        "Backend message: 中文👩‍💻"
    );
    controller.state.open(Overlay::Commands);
    controller.state.query = "context".into();
    assert_eq!(drag_text(&mut controller, "context").0, "context");
}

#[test]
fn composer_release_copies_without_editing_and_single_click_does_not_copy() {
    let mut controller = controller();
    let input = "alpha\t中👩‍💻e\u{301}\nsecond";
    controller.paste(input.into());
    assert_eq!(
        drag_text(&mut controller, "alpha   中👩‍💻e\u{301}").0,
        "alpha\t中👩‍💻e\u{301}"
    );
    assert_eq!(controller.state.editor.lines().join("\n"), input);
    let (buffer, hits) = render(&mut controller, SIZE);
    let position = locate(&buffer, hits.composer_text, "second");
    for kind in [
        MouseEventKind::Down(MouseButton::Left),
        MouseEventKind::Up(MouseButton::Left),
    ] {
        assert!(
            pointer(&mut controller, SIZE, kind, position, KeyModifiers::NONE)
                .actions
                .is_empty()
        );
    }
    assert_eq!(controller.state.editor.lines().join("\n"), input);
}

#[test]
fn reverse_shift_and_partial_grapheme_selection_copy_whole_characters() {
    let mut controller = controller();
    controller
        .state
        .entries
        .push(Entry::Assistant("abcdef a中👩‍💻e\u{301}z".into()));
    let (buffer, _) = render(&mut controller, SIZE);
    let start = locate(&buffer, buffer.area, "abcdef");
    let a = Position::new(start.x + 1, start.y);
    let b = Position::new(start.x + 3, start.y);
    assert_eq!(drag(&mut controller, SIZE, b, a).0, "bc");
    // Reverse selection keeps its original anchor at 'd'.
    let extended = Position::new(start.x + 5, start.y);
    for kind in [
        MouseEventKind::Down(MouseButton::Left),
        MouseEventKind::Up(MouseButton::Left),
    ] {
        let result = pointer(&mut controller, SIZE, kind, extended, KeyModifiers::SHIFT);
        if kind == MouseEventKind::Up(MouseButton::Left) {
            assert_eq!(copy(result.actions), "de");
        } else {
            assert!(result.actions.is_empty());
        }
    }
    let start = locate(&buffer, buffer.area, "a中👩‍💻e\u{301}z");
    assert_eq!(
        drag(
            &mut controller,
            SIZE,
            Position::new(start.x + 4, start.y),
            Position::new(start.x + 5, start.y)
        )
        .0,
        "👩‍💻"
    );
    assert_eq!(
        drag(
            &mut controller,
            SIZE,
            Position::new(start.x + 5, start.y),
            Position::new(start.x + 6, start.y)
        )
        .0,
        "e\u{301}"
    );
}

#[test]
fn wrapped_selection_retains_hard_newlines_and_tabs_across_resize() {
    let text = format!("开头 {}\n结尾", "中👩‍💻e\u{301}\tword ".repeat(7));
    for size in [(28, 42), (48, 32), SIZE] {
        let mut controller = controller();
        controller.state.entries.push(Entry::User(text.clone()));
        controller.state.expanded_entries.insert(0);
        let (buffer, hits) = render(&mut controller, size);
        let from = locate(&buffer, hits.transcript, "开头");
        let last = locate(&buffer, hits.transcript, "结尾");
        let to = Position::new(last.x + "结尾".width() as u16, last.y);
        assert_eq!(drag(&mut controller, size, from, to).0, text);
        render(&mut controller, (82, 18));
        assert_eq!(copy(ctrl_copy(&mut controller)), text);
        controller.state.monochrome = true;
        render(&mut controller, (35, 40));
        assert_eq!(copy(ctrl_copy(&mut controller)), text);
    }
}

#[test]
fn ordinary_click_executes_once_on_release_and_drag_never_toggles_tools() {
    let mut controller = controller();
    controller.state.entries.push(Entry::Tool(tool(false)));
    let (buffer, _) = render(&mut controller, SIZE);
    let from = locate(&buffer, buffer.area, "Read project.rs");
    assert!(pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Down(MouseButton::Left),
        from,
        KeyModifiers::NONE
    )
    .actions
    .is_empty());
    assert!(matches!(&controller.state.entries[0], Entry::Tool(tool) if !tool.expanded));
    assert!(pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Up(MouseButton::Left),
        from,
        KeyModifiers::NONE
    )
    .actions
    .is_empty());
    assert!(matches!(&controller.state.entries[0], Entry::Tool(tool) if tool.expanded));
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Up(MouseButton::Left),
        from,
        KeyModifiers::NONE,
    );
    assert!(matches!(&controller.state.entries[0], Entry::Tool(tool) if tool.expanded));
    controller.toggle_tool("read-fixture");
    assert_eq!(
        drag(
            &mut controller,
            SIZE,
            from,
            Position::new(from.x + 4, from.y)
        )
        .0,
        "Read"
    );
    assert!(matches!(&controller.state.entries[0], Entry::Tool(tool) if !tool.expanded));
    for (kind, position) in [
        (MouseEventKind::Down(MouseButton::Left), from),
        (
            MouseEventKind::Drag(MouseButton::Left),
            Position::new(from.x + 4, from.y),
        ),
        (MouseEventKind::Drag(MouseButton::Left), from),
        (MouseEventKind::Up(MouseButton::Left), from),
    ] {
        assert!(
            pointer(&mut controller, SIZE, kind, position, KeyModifiers::NONE)
                .actions
                .is_empty()
        );
    }
    assert!(matches!(&controller.state.entries[0], Entry::Tool(tool) if !tool.expanded));
}

#[test]
fn picker_drag_does_not_choose_and_click_selects_on_release() {
    let mut controller = controller();
    controller.state.open(Overlay::Commands);
    let (buffer, hits) = render(&mut controller, SIZE);
    let (row, index) = *hits
        .choice_rows
        .iter()
        .find(|(_, index)| *index == 1)
        .unwrap();
    let label = controller.state.choices()[index].label.clone();
    let from = locate(&buffer, row, &label);
    assert_eq!(
        drag(
            &mut controller,
            SIZE,
            from,
            Position::new(from.x + label.width() as u16, from.y)
        )
        .0,
        label
    );
    assert_eq!(controller.state.selected, 0);
    assert!(pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Down(MouseButton::Left),
        from,
        KeyModifiers::NONE
    )
    .actions
    .is_empty());
    assert_eq!(controller.state.selected, 0);
    assert!(pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Up(MouseButton::Left),
        from,
        KeyModifiers::NONE
    )
    .actions
    .is_empty());
    assert_eq!(controller.state.selected, index);
}

#[test]
fn inline_commands_copy_the_visible_label_instead_of_covered_conversation() {
    let mut controller = controller();
    controller.state.entries.push(Entry::Assistant(format!(
        "```text\n{}\n```",
        (0..60)
            .map(|i| format!("covered transcript {i}"))
            .collect::<Vec<_>>()
            .join("\n")
    )));
    controller.paste("/".into());
    let (buffer, hits) = render(&mut controller, SIZE);
    let (row, index) = hits.slash_rows[0];
    let label = controller.state.slash_choices()[index].label.clone();
    let from = locate(&buffer, row, &label);
    assert_eq!(
        drag(
            &mut controller,
            SIZE,
            from,
            Position::new(from.x + label.width() as u16, from.y)
        )
        .0,
        label
    );
    assert_eq!(controller.state.editor.lines(), ["/"]);
    assert!(controller.state.overlay.is_none());
}

#[test]
fn approval_code_and_options_copy_without_approving_execution() {
    use aster_tui::state::{ApprovalOption, ApprovalPreview, ApprovalRequest};
    let mut controller = controller();
    controller.state.approvals.push_back(ApprovalRequest {
        id: json!(77),
        title: "Read selection fixture".into(),
        code: "result\t= '中文'\nprint(result)".into(),
        preview: Some(ApprovalPreview {
            summary: "Read selection fixture".into(),
            details: vec!["Inspect text".into()],
            complete: true,
        }),
        show_code: true,
        options: vec![
            ApprovalOption {
                id: "once".into(),
                name: "Allow once".into(),
                kind: "allow_once".into(),
            },
            ApprovalOption {
                id: "reject".into(),
                name: "Reject".into(),
                kind: "reject_once".into(),
            },
        ],
    });
    controller.state.open(Overlay::Approval);
    assert_eq!(
        drag_text(&mut controller, "result    = '中文'").0,
        "result\t= '中文'"
    );
    assert_eq!(drag_text(&mut controller, "Allow once").0, "Allow once");
    assert_eq!(
        drag_text(&mut controller, "Full Python code").0,
        "Full Python code"
    );
    assert_eq!(controller.state.approvals.len(), 1);
    assert_eq!(controller.state.selected, 0);
    assert_eq!(controller.state.overlay, Some(Overlay::Approval));
}

#[test]
fn attachment_text_preview_copies_original_file_content() {
    let mut controller = controller();
    controller.paste("preserved draft".into());
    controller
        .state
        .attachments
        .push(aster_tui::attachments::Attachment::Text {
            name: "notes.txt".into(),
            uri: "file:///notes.txt".into(),
            text: "file content: 中文\tvalue\nsecond file line".into(),
        });
    controller.state.open(Overlay::Attachments);
    assert_eq!(
        drag_text(&mut controller, "file content: 中文    value").0,
        "file content: 中文\tvalue"
    );
    assert_eq!(controller.state.attachments.len(), 1);
    assert_eq!(controller.state.editor.lines(), ["preserved draft"]);
}

#[test]
fn captured_scroll_and_outside_release_keep_the_selected_source() {
    let mut controller = controller();
    controller.state.entries.push(Entry::Assistant(format!(
        "```text\n{}\n```",
        (0..65)
            .map(|i| format!("line {i:02} text"))
            .collect::<Vec<_>>()
            .join("\n")
    )));
    controller.state.scroll = usize::MAX;
    let (buffer, hits) = render(&mut controller, SIZE);
    let from = locate(&buffer, hits.transcript, "line 00 text");
    let region = hits
        .text_regions
        .iter()
        .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
        .unwrap();
    let initial_top = region.top;
    let initial_max = region.max_top;
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Down(MouseButton::Left),
        from,
        KeyModifiers::NONE,
    );
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::ScrollDown,
        from,
        KeyModifiers::NONE,
    );
    let (_, hits) = render(&mut controller, SIZE);
    let region = hits
        .text_regions
        .iter()
        .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
        .unwrap();
    assert_eq!(region.top, initial_top + 3);
    assert_eq!(controller.state.scroll, initial_max - 3);
    let bottom = Position::new(region.area.right(), region.area.bottom());
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Drag(MouseButton::Left),
        bottom,
        KeyModifiers::NONE,
    );
    let (_, hits) = render(&mut controller, SIZE);
    let region = hits
        .text_regions
        .iter()
        .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
        .unwrap();
    let last = region.top + usize::from(region.area.height) - 1;
    let result = pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Up(MouseButton::Left),
        bottom,
        KeyModifiers::NONE,
    );
    let expected = (0..=last)
        .map(|i| format!("line {i:02} text"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(copy(result.actions), expected);
    render(&mut controller, (70, 18));
    assert_eq!(copy(ctrl_copy(&mut controller)), expected);
}

#[test]
fn streaming_does_not_move_text_under_a_captured_selection() {
    let mut controller = controller();
    let mut text = (0..40)
        .map(|i| format!("old paragraph {i}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    text.push_str("\n\nkept target text");
    controller.state.entries.push(Entry::Assistant(text));
    let (buffer, hits) = render(&mut controller, SIZE);
    assert_eq!(controller.state.scroll, 0);
    let from = locate(&buffer, hits.transcript, "kept target text");
    let initial_top = hits
        .text_regions
        .iter()
        .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
        .unwrap()
        .top;
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Down(MouseButton::Left),
        from,
        KeyModifiers::NONE,
    );
    let Entry::Assistant(text) = &mut controller.state.entries[0] else {
        unreachable!()
    };
    text.push_str("\n\nnew streamed paragraph\n\nanother streamed paragraph");
    let (buffer, hits) = render(&mut controller, SIZE);
    assert_eq!(locate(&buffer, hits.transcript, "kept target text"), from);
    assert_eq!(
        hits.text_regions
            .iter()
            .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
            .unwrap()
            .top,
        initial_top
    );
    let to = Position::new(from.x + "kept target".width() as u16, from.y);
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Drag(MouseButton::Left),
        to,
        KeyModifiers::NONE,
    );
    assert_eq!(
        copy(
            pointer(
                &mut controller,
                SIZE,
                MouseEventKind::Up(MouseButton::Left),
                to,
                KeyModifiers::NONE
            )
            .actions
        ),
        "kept target"
    );
    assert_eq!(copy(ctrl_copy(&mut controller)), "kept target");
}

#[test]
fn ctrl_end_resumes_following_output_after_selecting_conversation_text() {
    for initial_scroll in [0, usize::MAX] {
        let mut controller = controller();
        controller.state.editor.insert_str("keep the draft");
        controller.state.entries.push(Entry::Assistant(
            (0..40)
                .map(|i| format!("old paragraph {i:02}"))
                .collect::<Vec<_>>()
                .join("\n\n"),
        ));
        controller.state.scroll = initial_scroll;
        let (buffer, hits) = render(&mut controller, SIZE);
        let selected = if initial_scroll == 0 {
            "old paragraph 39"
        } else {
            "old paragraph 00"
        };
        let from = locate(&buffer, hits.transcript, selected);
        let to = Position::new(from.x + selected.width() as u16, from.y);
        assert_eq!(drag(&mut controller, SIZE, from, to).0, selected);

        assert!(controller
            .key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL))
            .is_empty());
        let (_, hits) = render(&mut controller, SIZE);
        assert_eq!(controller.state.scroll, 0);
        assert_eq!(
            hits.text_regions
                .iter()
                .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
                .unwrap()
                .area,
            hits.transcript,
            "the scroll indicator must stop covering the latest row"
        );

        for index in 0..2 {
            let latest = format!("latest streamed paragraph {index}");
            let Entry::Assistant(text) = &mut controller.state.entries[0] else {
                unreachable!()
            };
            text.push_str(&format!("\n\n{latest}"));
            let (buffer, hits) = render(&mut controller, SIZE);
            assert_eq!(controller.state.scroll, 0);
            let _visible = locate(&buffer, hits.transcript, &latest);
        }
        assert_eq!(controller.state.editor.lines(), ["keep the draft"]);
    }
}

#[test]
fn changed_conversation_panel_subject_or_query_rejects_stale_selection() {
    for changed in ["conversation", "panel", "prefix", "typing"] {
        let mut controller = controller();
        controller
            .state
            .entries
            .push(Entry::Assistant("old selected text".into()));
        assert_eq!(
            drag_text(&mut controller, "selected text").0,
            "selected text"
        );
        match changed {
            "conversation" => controller.state.conversation_generation += 1,
            "panel" => controller.state.open(Overlay::Help),
            "prefix" => controller.state.entries[0] = Entry::Assistant("new selected text".into()),
            "typing" => {
                controller.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE));
            }
            _ => unreachable!(),
        }
        render(&mut controller, SIZE);
        assert!(
            !ctrl_copy(&mut controller)
                .iter()
                .any(|action| matches!(action, Action::CopySelection { .. })),
            "{changed}"
        );
    }
    for changed in ["query", "subject"] {
        let mut controller = controller();
        controller.state.entries.push(Entry::Tool(tool(false)));
        let mut second = tool(false);
        second.id = "other".into();
        second.title = "Read other.rs".into();
        controller.state.entries.push(Entry::Tool(second));
        controller.state.open(Overlay::Tools);
        assert_eq!(
            drag_text(&mut controller, "second output").0,
            "second output"
        );
        if changed == "query" {
            controller.state.query = "other".into();
        } else {
            controller.state.selected = 1;
        }
        render(&mut controller, SIZE);
        assert!(
            !ctrl_copy(&mut controller)
                .iter()
                .any(|action| matches!(action, Action::CopySelection { .. })),
            "{changed}"
        );
    }
}

#[test]
fn a_new_label_selection_takes_focus_from_the_context_reader() {
    use aster_tui::{
        context::Category,
        context_detail::{Target, View},
    };
    let mut controller = controller();
    controller.paste("unchanged input".into());
    controller.state.open(Overlay::Context);
    let mut view = View::new(
        "request".into(),
        Target::Category(Category::SystemPrompt),
        1,
        0,
    );
    view.complete(&json!({"result": {
        "requestId": "request", "category": "system_prompt", "key": null,
        "source": "model_request", "sections": [{
            "key": "system_prompt", "text": "reader selected body",
            "characters": 20, "truncated": false
        }]
    }}));
    controller.state.context_view = Some(view);
    assert_eq!(
        drag_text(&mut controller, "selected body").0,
        "selected body"
    );
    assert_eq!(copy(ctrl_copy(&mut controller)), "selected body");
    assert_eq!(
        drag_text(&mut controller, "System prompt").0,
        "System prompt"
    );
    assert_eq!(copy(ctrl_copy(&mut controller)), "System prompt");
    assert!(controller
        .state
        .context_view
        .as_ref()
        .unwrap()
        .selected_text()
        .is_none());
    assert_eq!(controller.state.editor.lines(), ["unchanged input"]);
}

#[test]
fn drag_redraws_reuse_the_document_and_literal_box_text_stays_selectable() {
    let mut controller = controller();
    controller.state.entries.push(Entry::Assistant(
        "┌─ literal box text\nordinary text\n└─".into(),
    ));
    let (buffer, first) = render(&mut controller, SIZE);
    let from = locate(&buffer, first.transcript, "┌─ literal box text");
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Down(MouseButton::Left),
        from,
        KeyModifiers::NONE,
    );
    let (_, second) = render(&mut controller, SIZE);
    let document = |hits: &ui::HitAreas| {
        hits.text_regions
            .iter()
            .find(|region| region.id == aster_tui::text_selection::TRANSCRIPT)
            .unwrap()
            .document
            .clone()
    };
    assert!(std::sync::Arc::ptr_eq(
        &document(&first),
        &document(&second)
    ));
    let to = Position::new(from.x + "┌─ literal box text".width() as u16, from.y);
    pointer(
        &mut controller,
        SIZE,
        MouseEventKind::Drag(MouseButton::Left),
        to,
        KeyModifiers::NONE,
    );
    assert_eq!(
        copy(
            pointer(
                &mut controller,
                SIZE,
                MouseEventKind::Up(MouseButton::Left),
                to,
                KeyModifiers::NONE
            )
            .actions
        ),
        "┌─ literal box text"
    );
}
