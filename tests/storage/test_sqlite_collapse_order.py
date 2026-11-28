# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite keeps active history order after earlier collapses leave position gaps."""

import pytest

from aster.context_blocks import EventStatus, UserEvent
from aster.events import Summary
from aster.runtime.event_backend import InMemoryBackend
from aster.runtime.event_manager import EventManager
from aster.storage import SQLiteStorageManager
from aster.storage.sqlite import SQLiteEventBackend


def test_repeated_collapses_and_recollapse_keep_order(sqlite_conn):
    backends = [InMemoryBackend(), SQLiteEventBackend(sqlite_conn)]
    managers = [EventManager(backend) for backend in backends]
    for manager in managers:
        for i in range(20):
            manager.add(UserEvent(content=f"message {i + 1}"))

    for start, end in (("3", "6"), ("12", "15"), ("17", "18")):
        for manager in managers:
            manager.collapse(start, end, summary_text="original summary")
        assert backends[1].active_tags() == backends[0].active_tags()

    expected = ["1", "2", "3..6", "7", "8", "9", "10", "11", "12..15", "16", "17..18", "19", "20"]
    assert backends[1].active_tags() == expected
    for manager, backend in zip(managers, backends, strict=True):
        assert manager.collapse("12", "15", summary_text="revised summary") == "12..15"
        summary = backend.get("12..15")
        assert isinstance(summary, Summary)
        assert summary.children_tags == ["12", "13", "14", "15"]
        assert summary.summary_text == "revised summary"
        assert backend.active_tags() == expected


def test_collapses_and_tail_append_keep_order_across_reopen(tmp_path):
    path = tmp_path / "history.db"
    with SQLiteStorageManager(path) as storage:
        manager = EventManager(storage.event_backend)
        for i in range(20):
            manager.add(UserEvent(content=f"message {i + 1}"))
        manager.collapse("3", "6", summary_text="first summary")

    with SQLiteStorageManager(path) as storage:
        manager = EventManager(storage.event_backend)
        manager.collapse("12", "15", summary_text="second summary")
        manager.collapse("17", "18", summary_text="third summary")
        assert manager.add(UserEvent(content="tail message")) == "21"
        manager.collapse("19", "21", summary_text="tail summary")

    with SQLiteStorageManager(path) as storage:
        assert storage.event_backend.active_tags() == [
            "1",
            "2",
            "3..6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12..15",
            "16",
            "17..18",
            "19..21",
        ]
        original = storage.event_backend.get("12")
        assert isinstance(original, UserEvent)
        assert original.content == "message 12"
        assert original.status == EventStatus.ARCHIVED
        tail = storage.event_backend.get("19..21")
        assert isinstance(tail, Summary)
        assert tail.children_tags == ["19", "20", "21"]


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, ["6", "1", "4", "5"]),
        (1, ["1", "6", "4", "5"]),
        (2, ["1", "4", "6", "5"]),
        (3, ["1", "4", "5", "6"]),
        (99, ["1", "4", "5", "6"]),
        (-1, ["1", "4", "6", "5"]),
        (-3, ["6", "1", "4", "5"]),
        (-99, ["6", "1", "4", "5"]),
    ],
)
def test_insert_active_tag_uses_list_index_after_removals(sqlite_conn, index, expected):
    backend = SQLiteEventBackend(sqlite_conn)
    for i in range(1, 7):
        backend.store(str(i), UserEvent(content=f"message {i}"))
    for tag in ("2", "3", "6"):
        backend.remove_active_tag(tag)

    backend.insert_active_tag("6", index)

    assert backend.active_tags() == expected


@pytest.mark.parametrize("index", [0, 99, -1])
def test_insert_active_tag_into_empty_history(sqlite_conn, index):
    backend = SQLiteEventBackend(sqlite_conn)
    backend.store("1", UserEvent(content="message"))
    backend.remove_active_tag("1")

    backend.insert_active_tag("1", index)

    assert backend.active_tags() == ["1"]
