# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-owned side-effect records; outcomes never stand in for saved progress."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from aster.runtime.event_manager import EventManager
from aster_cli.sessions.events import (
    SESSION_ACTION_EVENT_TYPES,
    SessionActionIntent,
    SessionActionOutcome,
    SessionActionReviewed,
    SessionActionsInherited,
    SessionActionsSaved,
)

if TYPE_CHECKING:
    from aster.storage.sqlite import SQLiteStorageManager

_EVENT_TYPES = {event_type.__name__: event_type for event_type in SESSION_ACTION_EVENT_TYPES}
_REVIEW_MESSAGE = (
    "Host actions need review before more side effects. Use /recovery to inspect them, then "
    '/recovery <action_id> --note "what you checked" to record your review.'
)


class PendingSessionActionsError(RuntimeError):
    """Recovered side effects need human reconciliation before executing more."""


@dataclass
class _Action:
    intent: SessionActionIntent
    outcome: str | None = None
    resolved: bool = False
    outcome_event: SessionActionOutcome | None = None


def _read_actions(path: Path) -> dict[str, _Action]:
    """Read only journal rows, failing closed on missing or damaged records."""
    actions: dict[str, _Action] = {}
    seen: set[str] = set()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        placeholders = ", ".join("?" for _ in _EVENT_TYPES)
        rows = connection.execute(
            "SELECT event_id, event_type, data FROM events "
            f"WHERE event_type IN ({placeholders}) ORDER BY insertion_order",
            tuple(_EVENT_TYPES),
        ).fetchall()
    for event_id, event_type, data in rows:
        event = _EVENT_TYPES[event_type].model_validate_json(data)
        if event.id != event_id or event.event_type != event_type or event_id in seen:
            raise ValueError("A host action has conflicting event identities.")
        seen.add(event_id)
        if isinstance(event, SessionActionIntent):
            if event.id in actions:
                raise ValueError("A host action has repeated intent records.")
            actions[event.id] = _Action(event)
        elif isinstance(event, SessionActionsSaved | SessionActionsInherited):
            for action_id in event.action_ids:
                action = actions.get(action_id)
                if action is None:
                    raise ValueError("A host action record has no recorded intent.")
                if isinstance(event, SessionActionsSaved) and action.outcome is None:
                    raise ValueError("Saved progress refers to an unfinished host action.")
                action.resolved = isinstance(event, SessionActionsSaved)
        else:
            action = actions.get(event.action_id)
            if action is None:
                raise ValueError("A host action outcome has no recorded intent.")
            if isinstance(event, SessionActionOutcome):
                if action.outcome is not None and action.outcome != event.outcome:
                    raise ValueError("A host action has conflicting outcomes.")
                action.outcome = event.outcome
                action.outcome_event = event
                action.resolved = False
            elif isinstance(event, SessionActionReviewed):
                action.resolved = True
    return actions


def _same_intent(first: SessionActionIntent, second: SessionActionIntent) -> bool:
    # Tags, event status and tracing context belong to each branch's insertion.
    exclude = {"tag", "status", "metadata"}
    return first.model_dump(exclude=exclude) == second.model_dump(exclude=exclude)


def inherit_actions(*sources: Path, restored: SQLiteStorageManager) -> None:
    """Retain side effects without borrowing later proof for an older snapshot."""
    from aster_cli.sessions.store import SessionCheckpointUnavailableError

    path = restored.path
    if path is None or any(path.resolve() == source.resolve() for source in sources):
        raise ValueError("Host actions must be inherited into a separate saved branch.")
    required: dict[str, _Action] = {}
    needs_review: set[str] = set()
    for source in dict.fromkeys(sources):
        actions = _read_actions(source)
        for action_id, action in actions.items():
            if not action.resolved:
                needs_review.add(action_id)
            previous = required.get(action_id)
            if previous is not None and (
                not _same_intent(previous.intent, action.intent)
                or (
                    previous.outcome is not None
                    and action.outcome is not None
                    and previous.outcome != action.outcome
                )
            ):
                raise SessionCheckpointUnavailableError(
                    "Saved branches disagree about a host action."
                )
            if previous is None or previous.outcome is None:
                required[action_id] = action
    copied = _read_actions(path)
    additions: list[SessionActionIntent | SessionActionOutcome] = []
    for action_id, action in required.items():
        existing = copied.get(action_id)
        if existing is not None and (
            not _same_intent(existing.intent, action.intent)
            or (
                existing.outcome is not None
                and action.outcome is not None
                and existing.outcome != action.outcome
            )
        ):
            raise SessionCheckpointUnavailableError(
                "The selected snapshot disagrees with the host action journal."
            )
        if existing is None:
            additions.append(action.intent)
            needs_review.add(action_id)
        if (existing is None or existing.outcome is None) and action.outcome_event is not None:
            additions.append(action.outcome_event)
            needs_review.add(action_id)

    # Validate identities before committing any additions to the new branch.
    addition_ids: set[str] = set()
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        for event in additions:
            if (
                event.id in addition_ids
                or connection.execute(
                    "SELECT 1 FROM events WHERE event_id = ?", (event.id,)
                ).fetchone()
            ):
                raise SessionCheckpointUnavailableError(
                    "An inherited host action conflicts with a saved event identity."
                )
            addition_ids.add(event.id)
    events = EventManager(backend=restored.event_backend)
    for event_type in SESSION_ACTION_EVENT_TYPES:
        events.register_event_type(event_type)
    for event in additions:
        events.add(event.model_copy(deep=True))
    if needs_review:
        events.add(SessionActionsInherited(action_ids=sorted(needs_review)))


def _bounded(value: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Host action descriptions must be non-empty text.")
    return value[:limit]


class SessionActions:
    """Durable host journal, deliberately separate from model-facing tools.

    Callers provide safe descriptions, never stdin, credentials or raw output.
    Only the host's explicit human review command may call :meth:`review`.
    """

    def __init__(
        self,
        events: EventManager,
        path: Path,
        lock: RLock,
        ensure_open: Callable[[], None],
    ) -> None:
        self._events = events
        self._lock = lock
        self._ensure_open = ensure_open
        self._actions = _read_actions(path)
        self._pending = {
            action_id for action_id, action in self._actions.items() if not action.resolved
        }

    def begin(self, kind: str, detail: str, cwd: str) -> str:
        """Commit intent before the caller performs a host side effect."""
        with self._lock:
            self.require_clear()
            event = SessionActionIntent(
                kind=_bounded(kind, 64),
                detail=_bounded(detail, 1024),
                cwd=_bounded(cwd, 4096),
            )
            self._events.add(event)
            self._actions[event.id] = _Action(event)
            return event.id

    def finish(self, action_id: str, *, outcome: str, requires_review: bool = False) -> None:
        """Record a result without claiming that agent progress was saved."""
        with self._lock:
            self._ensure_open()
            action = self._actions.get(action_id)
            if action is None or action.outcome is not None or action.resolved:
                raise ValueError("The host action is unknown or already finished.")
            event = SessionActionOutcome(action_id=action_id, outcome=_bounded(outcome, 256))
            try:
                self._events.add(event)
            except BaseException:
                self._pending.add(action_id)
                raise
            action.outcome = event.outcome
            action.outcome_event = event
            if requires_review:
                self._pending.add(action_id)

    def pending(self) -> list[dict[str, str | None]]:
        """Return independent, bounded summaries of actions needing review."""
        with self._lock:
            self._ensure_open()
            return [
                {
                    "action_id": action_id,
                    "kind": action.intent.kind,
                    "detail": action.intent.detail,
                    "cwd": action.intent.cwd,
                    "outcome": action.outcome,
                }
                for action_id, action in self._actions.items()
                if action_id in self._pending
            ]

    def require_clear(self) -> None:
        with self._lock:
            self._ensure_open()
            if self._pending:
                raise PendingSessionActionsError(_REVIEW_MESSAGE)

    def review(self, action_id: str, note: str) -> None:
        """Persist an explicit human check; this is never an automatic retry."""
        with self._lock:
            self._ensure_open()
            if action_id not in self._pending:
                raise ValueError("This host action is not awaiting review.")
            self._events.add(SessionActionReviewed(action_id=action_id, note=_bounded(note, 1024)))
            self._actions[action_id].resolved = True
            self._pending.remove(action_id)

    def _save_candidates(self) -> list[str]:
        return [
            action_id
            for action_id, action in self._actions.items()
            if action.outcome is not None and not action.resolved and action_id not in self._pending
        ]

    def _mark_pending(self) -> None:
        self._pending.update(
            action_id for action_id, action in self._actions.items() if not action.resolved
        )

    def _has_records(self) -> bool:
        return bool(self._actions)

    def _saved(self, action_ids: list[str]) -> None:
        if action_ids:
            self._events.add(SessionActionsSaved(action_ids=action_ids))
            for action_id in action_ids:
                self._actions[action_id].resolved = True
        # Work finishing after the snapshot cannot be acknowledged by a later, unrelated answer.
        self._mark_pending()
