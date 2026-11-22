# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-memory StorageManager — no persistence, drop-in for current behavior."""

import copy
from typing import TYPE_CHECKING, Any

from aster.runtime.event_backend import EventBackend, InMemoryBackend

if TYPE_CHECKING:
    from aster.agent import Agent


class InMemoryStorageManager:
    """StorageManager that keeps everything in memory.

    This is the default when no storage is configured. Events live in
    an ``InMemoryBackend`` (same as today), and save/restore are no-ops
    that raise — there's nowhere to persist to.

    This allows us to centralize storage in StorageManager, rather than
    having to special-case "no storage provided".
    """

    def __init__(self) -> None:
        self._event_backend: EventBackend = InMemoryBackend()
        self._task_budget: dict[str, Any] | None = None

    def load_task_budget(self) -> dict[str, Any] | None:
        """In-memory sessions retain accounting only for this process lifetime."""
        return copy.deepcopy(self._task_budget)

    def save_task_budget(self, state: dict[str, Any]) -> None:
        self._task_budget = copy.deepcopy(state)

    @property
    def event_backend(self) -> EventBackend:
        return self._event_backend

    def save_snapshot(self, agent: "Agent") -> str:
        from aster.errors.storage import StorageNotConfiguredError

        raise StorageNotConfiguredError(
            "InMemoryStorageManager does not support persistence. "
            "Pass a persistent StorageManager to enable save/restore."
        )

    def restore_snapshot(self, snapshot_id: str, agent: "Agent") -> None:
        from aster.errors.storage import StorageNotConfiguredError

        raise StorageNotConfiguredError(
            "InMemoryStorageManager does not support persistence. "
            "Pass a persistent StorageManager to enable save/restore."
        )
