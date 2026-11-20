# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input recall can be bootstrapped without resuming or locking old conversations."""

import json
import sqlite3
from datetime import UTC, datetime

import aster_cli.sessions.store as store_module
import pytest
from aster_acp.server import CodingACPAdapter
from aster_cli.sessions import SessionStore

from aster.interactive import AgentMessage
from aster.unifiedllm import FakeLLMClient


async def test_list_opt_in_recovers_only_user_inputs_without_opening_agent(tmp_path):
    store = SessionStore(tmp_path / ".aster" / "sessions")
    first = store.create(working_directory=str(tmp_path))
    first.record_user_message("first input")
    first.events.add(AgentMessage(content="not an input"))
    second = store.create(working_directory=str(tmp_path))
    second.record_user_message("  中文\nsecond input  \n")
    first.record_user_message("latest input")

    # Both writer handles remain open while the picker reads their history.
    def factory():
        raise AssertionError("Reading input history must not create an agent")

    adapter = CodingACPAdapter(factory)
    try:
        ordinary = await adapter.list_sessions(str(tmp_path))
        assert ordinary.field_meta is None
        page = await adapter.list_sessions(str(tmp_path), aster={"includeInputHistory": True})
        assert (page.field_meta or {}).get("aster", {}).get("inputHistory") == [
            "first input",
            "  中文\nsecond input  \n",
            "latest input",
        ]
        assert adapter._client is None
    finally:
        await adapter.close()
        first.close()
        second.close()


async def test_empty_workspace_history_does_not_create_a_session_directory(tmp_path):
    adapter = CodingACPAdapter(FakeLLMClient)
    try:
        page = await adapter.list_sessions(str(tmp_path), aster={"includeInputHistory": True})
        assert (page.field_meta or {}).get("aster", {}).get("inputHistory") == []
        assert not (tmp_path / ".aster").exists()
    finally:
        await adapter.close()


def test_history_bootstrap_keeps_latest_inputs_with_count_and_encoded_byte_bounds(tmp_path):
    import json

    from aster_acp._input_history import recent_inputs

    store = SessionStore(tmp_path / ".aster" / "sessions")
    with store.create() as session:
        for text in ["old input", "x" * 100, "中文", "latest"]:
            session.record_user_message(text)
        assert recent_inputs(store, limit=2) == ["中文", "latest"]
        budget = sum(
            len(json.dumps(text, ensure_ascii=False).encode()) + 1 for text in ["中文", "latest"]
        )
        assert recent_inputs(store, max_bytes=budget) == ["中文", "latest"]


@pytest.mark.parametrize(
    "fields",
    [
        {},
        *(
            {"timestamp": value}
            for value in (
                None,
                "invalid",
                True,
                False,
                float("nan"),
                float("inf"),
                float("-inf"),
                1e100,
                -1e100,
                10**400,
                -(10**400),
                253402300800,
            )
        ),
    ],
)
async def test_imported_timestamps_preserve_sessions_and_input_history(
    tmp_path, monkeypatch, fields
):
    store = SessionStore(tmp_path / ".aster" / "sessions")
    for session_id, text in (("healthy", "historical input"), ("imported", "imported input")):
        with store.create(session_id=session_id, working_directory=str(tmp_path)) as session:
            session.record_user_message(text)
        connection = sqlite3.connect(store.path_for(session_id))
        try:
            with connection:
                rows = connection.execute("SELECT tag, data FROM events").fetchall()
                for tag, data in rows:
                    raw = json.loads(data)
                    raw.pop("timestamp", None)
                    raw.update(fields if session_id == "imported" else {"timestamp": 1700000000})
                    connection.execute(
                        "UPDATE events SET data = ? WHERE tag = ?", (json.dumps(raw), tag)
                    )
        finally:
            connection.close()
    monkeypatch.setattr(store_module.time, "time", lambda: 1800000000.25)
    # WAL readers may create sidecars; saved database bytes and mtimes must stay unchanged.
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in store.root.glob("*.db")
    }

    def factory():
        raise AssertionError("Listing saved sessions must not construct a model or agent")

    adapter = CodingACPAdapter(factory, model_factory=factory)
    try:
        ordinary = await adapter.list_sessions(str(tmp_path))
        assert ordinary.field_meta is None
        page = await adapter.list_sessions(str(tmp_path), aster={"includeInputHistory": True})
        assert {session.session_id for session in ordinary.sessions} == {"healthy", "imported"}
        info = store.get("imported")
        assert info.started_at == info.last_active == store.path_for("imported").stat().st_mtime
        assert {session.session_id for session in page.sessions} == {"healthy", "imported"}
        assert page.next_cursor is None
        for session in page.sessions:
            expected = datetime.fromtimestamp(
                store.get(session.session_id).last_active, UTC
            ).isoformat()
            assert session.updated_at == expected
        turns = store.load_turns("imported")
        assert [(turn.content, turn.timestamp) for turn in turns] == [
            ("imported input", 1800000000.25)
        ]
        assert store.load_turns("healthy")[0].timestamp == 1700000000.0
        assert page.field_meta["aster"]["inputHistory"] == ["historical input", "imported input"]
        assert adapter._client is None
    finally:
        await adapter.close()
    assert {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in store.root.glob("*.db")
    } == before
