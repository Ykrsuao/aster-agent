# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise workspace trust choices through real terminal input and rendering."""

import re
import sys
from contextlib import contextmanager
from pathlib import Path

import click
import pytest
from aster_cli._workspace_trust_prompt import prompt_workspace_trust
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.input.typeahead import clear_typeahead, store_typeahead
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import ColorDepth, DummyOutput
from prompt_toolkit.shortcuts.choice_input import ChoiceInput


@pytest.fixture(autouse=True)
def terminal(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


@contextmanager
def press_after_render(monkeypatch, keys, *, output=None):
    original = ChoiceInput._create_application
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=output or DummyOutput()),
    ):

        def create(chooser):
            app = original(chooser)
            sent = False

            def fail_unhandled_input():
                if not app.is_done:
                    app.exit(exception=AssertionError(f"Prompt did not finish after {keys!r}"))

            def rendered(_):
                nonlocal sent
                if not sent and not app.is_done:
                    sent = True
                    app.loop.call_soon(pipe.send_text, keys)
                    app.loop.call_later(5, fail_unhandled_input)

            app.after_render += rendered
            return app

        monkeypatch.setattr(ChoiceInput, "_create_application", create)
        yield pipe


@pytest.mark.parametrize(
    "keys",
    ["\r", "\x1b[B\x1b[A\r", "\x1b[B1\r", "\x1b[By\r"],
    ids=["default", "arrow", "number", "yes"],
)
def test_enter_confirms_the_selected_trust_choice(monkeypatch, tmp_path, keys):
    with press_after_render(monkeypatch, keys):
        assert prompt_workspace_trust(tmp_path) is True


@pytest.mark.parametrize(
    "keys",
    ["2", "n", "q", "\x1b[B\r", "\x1b"],
    ids=["number", "no", "quit", "arrow", "escape"],
)
def test_quit_leaves_without_trusting(monkeypatch, tmp_path, keys):
    with press_after_render(monkeypatch, keys):
        with pytest.raises(click.exceptions.Exit) as caught:
            prompt_workspace_trust(tmp_path)
    assert caught.value.exit_code == 0


@pytest.mark.parametrize("keys", ["\x03", "\x04"], ids=["interrupt", "end-of-input"])
def test_terminal_cancellation_aborts(monkeypatch, tmp_path, keys):
    with press_after_render(monkeypatch, keys):
        with pytest.raises(click.Abort):
            prompt_workspace_trust(tmp_path)


@pytest.mark.parametrize("key", ["1", "y"])
def test_selecting_trust_still_requires_enter(monkeypatch, tmp_path, key):
    original = ChoiceInput._create_application
    pending_frames = []
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):

        def create(chooser):
            app = original(chooser)

            def rendered(_):
                if app.is_done:
                    return
                pending_frames.append(app.renderer.last_rendered_screen)
                if len(pending_frames) == 1:
                    app.loop.call_soon(pipe.send_text, key)
                elif len(pending_frames) == 2:
                    app.loop.call_soon(pipe.send_text, "\x03")

            app.after_render += rendered
            return app

        monkeypatch.setattr(ChoiceInput, "_create_application", create)
        with pytest.raises(click.Abort):
            prompt_workspace_trust(tmp_path)
    assert len(pending_frames) >= 2


@pytest.mark.parametrize("source", ["typeahead", "pipe"])
def test_buffered_enter_cannot_trust_before_the_page_is_visible(monkeypatch, tmp_path, source):
    with press_after_render(monkeypatch, "\x1b") as pipe:
        if source == "typeahead":
            store_typeahead(pipe, [KeyPress(Keys.ControlM, "\r")])
        else:
            pipe.send_text("\r")
        try:
            with pytest.raises(click.exceptions.Exit) as caught:
                prompt_workspace_trust(tmp_path)
            assert caught.value.exit_code == 0
        finally:
            clear_typeahead(pipe)


@pytest.mark.parametrize(("columns", "no_color"), [(40, True), (80, False)])
def test_render_preserves_the_full_trust_target_and_escapes_controls(
    monkeypatch, tmp_path, columns, no_color
):
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    prefix = tmp_path / ("long-workspace-" * 4 + "中文-")
    workspace = Path(str(prefix) + "\u202e-\x1b[31m-end")
    expected_path = str(prefix) + r"\u202e-\x1b[31m-end"
    original = ChoiceInput._create_application
    screens = []
    styles = []
    color_depths = []

    class Output(DummyOutput):
        def get_size(self):
            return Size(rows=50, columns=columns)

        def get_default_color_depth(self):
            return ColorDepth.DEPTH_8_BIT

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=Output()):

        def create(chooser):
            app = original(chooser)

            def rendered(_):
                screen = app.renderer.last_rendered_screen
                if screen is None or app.is_done:
                    return
                screens.append(
                    "\n".join(
                        "".join(row[x].char for x in sorted(row))
                        for _, row in sorted(screen.data_buffer.items())
                    )
                )
                styles.append(
                    {
                        (x, y): app.renderer.style.get_attrs_for_style_str(row[x].style)
                        for y, row in screen.data_buffer.items()
                        for x in range(columns)
                    }
                )
                color_depths.append(app.color_depth)
                if len(screens) == 1:
                    app.loop.call_soon(pipe.send_text, "\x1b[B")
                elif len(screens) == 2:
                    app.loop.call_soon(pipe.send_text, "\x1b[A\r")

            app.after_render += rendered
            return app

        monkeypatch.setattr(ChoiceInput, "_create_application", create)
        assert prompt_workspace_trust(workspace) is True

    assert len(screens) >= 2
    rendered = screens[0]
    assert "Folder access" in rendered
    assert "Trust and continue" in rendered
    assert "Quit" in rendered
    assert rendered.index("Quit") < rendered.index("Enter continue")
    assert "Enter continue   Esc quit" in rendered
    assert not re.search(r"[\u2500-\u257f]", rendered)
    compact = re.sub(r"[\s\u2500-\u257f]", "", rendered)
    assert expected_path.replace(" ", "") in compact
    assert "\x1b" not in rendered and "\u202e" not in rendered

    def location(text, frame=rendered):
        return next(
            (line.index(text), y) for y, line in enumerate(frame.splitlines()) if text in line
        )

    path_start = location(expected_path[:16])
    for position in (
        location("Folder access"),
        path_start,
        location("›"),
        location("Enter continue"),
    ):
        assert position[0] == 2
    assert styles[0][path_start].bold
    content_end = min(columns, 68) - 2
    for index, (selected, unselected) in enumerate(
        [("Trust and continue", "Quit"), ("Quit", "Trust and continue")]
    ):
        selected_y = location(selected, screens[index])[1]
        unselected_y = location(unselected, screens[index])[1]
        assert location("›", screens[index]) == (2, selected_y)
        assert all(styles[index][x, selected_y].reverse for x in range(2, content_end))
        assert not any(styles[index][x, unselected_y].reverse for x in range(columns))
        assert not styles[index][content_end, selected_y].reverse
    if no_color:
        assert all(depth == ColorDepth.DEPTH_1_BIT for depth in color_depths)


@pytest.mark.parametrize(("columns", "rows"), [(24, 10), (20, 8), (80, 7)])
def test_tiny_terminal_keeps_full_choices_in_scrollback(
    monkeypatch, tmp_path, capsys, columns, rows
):
    class Output(DummyOutput):
        def get_size(self):
            return Size(rows=rows, columns=columns)

    with press_after_render(monkeypatch, "\r", output=Output()):
        assert prompt_workspace_trust(tmp_path) is True

    output = capsys.readouterr().out
    assert str(tmp_path) in output
    assert "1. Trust and continue (default)" in output
    assert "2. Quit" in output
    assert "Enter continue" in output
