# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host review of actions whose progress was not saved before an interruption."""

from __future__ import annotations

import asyncio
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from aster.agentdoc import hidden
from aster.interactive import Done
from aster.runtime.sandbox.errors import (
    CellMemoryError,
    CellTimeoutError,
    SandboxExecutionError,
    WorkerDiedError,
)
from aster.skill import Skill, slash_command
from aster.strategy_validation import InvariantError

if TYPE_CHECKING:
    from aster_cli.sessions.actions import SessionActions


def finish_action(
    journal: SessionActions | None,
    action_id: str | None,
    error: BaseException | None = None,
    *,
    requires_review: bool = False,
) -> None:
    if journal is None or action_id is None:
        return
    interrupted = error.original_error if isinstance(error, SandboxExecutionError) else error
    requires_review = requires_review or isinstance(
        interrupted,
        (
            asyncio.CancelledError,
            GeneratorExit,
            TimeoutError,
            CellTimeoutError,
            CellMemoryError,
            WorkerDiedError,
        ),
    )
    outcome = (
        f"Interrupted: {type(error).__name__}"
        if error is not None
        else "Interrupted: execution did not finish"
        if requires_review
        else "Tool returned"
    )
    try:
        journal.finish(
            action_id,
            outcome=outcome,
            **({"requires_review": True} if requires_review else {}),
        )
    except Exception as storage_error:
        if error is None:
            raise
        # Keep cancellation and the original tool failure intact. The journal
        # retains the unresolved intent and blocks subsequent actions.
        error.add_note(f"Could not save action outcome: {type(storage_error).__name__}")


@contextmanager
def record_action(
    journal: SessionActions | None, kind: str, detail: str, cwd: str
) -> Iterator[str | None]:
    action_id = journal.begin(kind, detail, cwd) if journal is not None else None
    try:
        yield action_id
    except BaseException as error:
        finish_action(journal, action_id, error)
        raise
    else:
        finish_action(journal, action_id)


class RecoveryTools(Skill):
    """Inspect interrupted actions without running or repeating them."""

    def __init__(self) -> None:
        super().__init__()
        self._journal: SessionActions | None = None

    def status(self) -> dict[str, Any]:
        """Show actions that require a human to check their actual effects."""
        pending = self._journal.pending() if self._journal is not None else []
        return {"pending_actions": pending}

    @hidden
    @slash_command(
        "recovery", argument_hint='[<action-id> --note "checked result"]', output_to_agent=False
    )
    def _command(self, args: str = "") -> str:
        """View interrupted actions or confirm a result you have checked."""
        if self._journal is None:
            return "Recovery records are available in a persistent session."
        try:
            tokens = shlex.split(args)
        except ValueError as error:
            return f"Invalid recovery command: {error}"
        if tokens:
            if len(tokens) < 3 or tokens[1] != "--note":
                return 'Usage: /recovery <action-id> --note "what you checked and found"'
            try:
                self._journal.review(tokens[0], " ".join(tokens[2:]))
            except (ValueError, KeyError) as error:
                return f"Could not record the review: {error}"
            return f"Recorded your review of {tokens[0]}. No action was repeated."
        pending = self._journal.pending()
        if not pending:
            return "No interrupted actions need review."
        lines = [
            "These actions may have changed files before progress was saved. "
            "Check their effects before continuing:",
            "",
        ]
        for item in pending:
            lines.append(
                f"- {item['action_id']}: {item['kind']} — {item['detail']} "
                f"(working directory: {item['cwd']}; "
                f"recorded outcome: {item['outcome'] or 'unknown'})"
            )
        lines.extend(["", 'After checking: /recovery <action-id> --note "checked result"'])
        return "\n".join(lines)


def require_recovery(agent: Any, result: Any, call: Any) -> None:
    if call.method_name != "handle" or not isinstance(result, Done):
        return
    journal = getattr(agent, "_action_journal", None)
    if journal is None or not journal.pending():
        return
    if getattr(result, "complete_task", True):
        raise InvariantError(
            "Interrupted actions still need review. Inspect their effects with "
            "self.files.read/list/search, then ask the user to confirm them with /recovery. "
            "Do not repeat the actions or claim the coding task is complete."
        )
    original = result.message or result.explanation
    if hasattr(result, "_original_message") and result._original_message is None:
        result._original_message = original
    result.message = f"{original}\n\nInterrupted actions still need review with /recovery."
