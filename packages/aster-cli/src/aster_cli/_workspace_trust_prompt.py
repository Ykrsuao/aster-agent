# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First-use workspace trust prompt, before project configuration is loaded."""

import os
import sys
import textwrap
import time
from pathlib import Path

import click


def _terminal_text(value: str) -> str:
    # A folder name must not inject terminal controls or disguise the trust target.
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in value)


def prompt_workspace_trust(workspace: Path) -> bool:
    """Return whether to trust the folder; exit or abort without granting trust."""
    title = "Folder access"
    path = _terminal_text(str(workspace))
    explanation = (
        "Aster can read, edit, and run files here. Project settings and skills can run "
        "code automatically.\n\n"
        "Command approvals still apply. Trust is saved for this folder and its subfolders."
    )
    details = f"{title}\n\n{path}\n\n{explanation}\n"
    if not sys.stdin.isatty():
        click.echo(details)
        if not click.confirm("Trust and continue?", default=False):
            raise click.exceptions.Exit(0)
        return True

    from prompt_toolkit.application import get_app
    from prompt_toolkit.filters import to_filter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.input.typeahead import clear_typeahead
    from prompt_toolkit.key_binding import KeyBindings, KeyPress
    from prompt_toolkit.layout import Dimension, HSplit, VSplit, Window
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.shortcuts.choice_input import ChoiceInput
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Box, Label

    compact = False

    def message():
        if compact:
            return FormattedText([("class:prompt", title)])
        width = max(1, min(64, get_app().output.get_size().columns - 4))
        paragraphs = "\n\n".join(
            textwrap.fill(paragraph, width=width, break_on_hyphens=False)
            for paragraph in explanation.split("\n\n")
        )
        return FormattedText(
            [
                ("class:prompt", title + "\n\n"),
                ("class:path", path + "\n\n"),
                ("", paragraphs + "\n"),
            ]
        )

    class TrustChoiceInput(ChoiceInput):
        def _create_application(self):
            nonlocal compact
            app = super()._create_application()
            menu = app.layout.current_window
            menu.wrap_lines = to_filter(True)
            menu.cursorline = to_filter(True)
            app.layout.container = VSplit(
                [
                    Box(
                        HSplit(
                            [
                                Label(self.message),
                                menu,
                                Label("\nEnter continue   Esc quit", style="class:hint"),
                            ]
                        ),
                        padding=0,
                        padding_left=2,
                        padding_right=2,
                        padding_top=1,
                        width=Dimension(preferred=68, max=68),
                    ),
                    Window(),
                ]
            )
            size = app.output.get_size()
            height = app.layout.container.preferred_height(size.columns, size.rows).preferred
            if height > size.rows:
                # Keep the full trust target readable in scrollback on short terminals.
                click.echo(details)
                compact = True
                for number, (value, label) in enumerate(self.options, 1):
                    suffix = " (default)" if value == self.default else ""
                    click.echo(f"{number}. {label}{suffix}")
                click.echo("1–2 select · Enter continue · Esc quit\n")
            if "NO_COLOR" in os.environ:
                app._color_depth = ColorDepth.DEPTH_1_BIT
            app.pre_run_callables.append(lambda: clear_typeahead(app.input))
            input_cleared = False

            def clear_startup_input(_):
                nonlocal input_cleared
                if input_cleared:
                    return
                input_cleared = True
                deadline = time.monotonic() + 1
                try:
                    while True:
                        app.input.read_keys()
                        if app.input.closed or time.monotonic() >= deadline:
                            app.exit(exception=EOFError())
                            return
                        if os.name == "nt":
                            from prompt_toolkit.eventloop.win32 import wait_for_handles

                            pending = wait_for_handles([app.input.handle], timeout=0) is not None
                        else:
                            import select

                            pending = bool(select.select([app.input.fileno()], [], [], 0)[0])
                        if not pending:
                            return
                except (OSError, ValueError):
                    app.exit(exception=EOFError())

            # Only input arriving after the page is visible can confirm trust.
            app.after_render += clear_startup_input
            return app

    bindings = KeyBindings()

    @bindings.add("escape")
    @bindings.add("2", eager=True)
    @bindings.add("n", eager=True)
    @bindings.add("q", eager=True)
    def exit_prompt(event):
        event.app.exit(result="exit")

    @bindings.add("y")
    def select_trust(event):
        event.key_processor.feed(KeyPress("1"), first=True)

    @bindings.add("c-d")
    def end_input(event):
        event.app.exit(exception=EOFError())

    try:
        result = TrustChoiceInput(
            message=message,
            options=(
                ("trust", "Trust and continue"),
                ("exit", "Quit"),
            ),
            default="trust",
            symbol="›",
            show_frame=False,
            show_numbers=True,
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "prompt": "bold",
                    "path": "bold",
                    "selected-option": "bold",
                    "cursor-line": "fg:default bg:default reverse nounderline",
                    "hint": "" if "NO_COLOR" in os.environ else "ansibrightblack",
                }
            ),
        ).prompt()
    except (EOFError, KeyboardInterrupt):
        raise click.Abort() from None
    if result != "trust":
        raise click.exceptions.Exit(0)
    return True
