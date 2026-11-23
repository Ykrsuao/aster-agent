# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Windows/Linux ACP memory persistence without a network model or user data."""

import asyncio
import json
import sys
from pathlib import Path

import yaml
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import AgentMessageChunk, RequestPermissionResponse


class _Client:
    def __init__(self):
        self.updates = []
        self.received = asyncio.Event()

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)
        self.received.set()

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate({"outcome": {"outcome": "cancelled"}})

    def text(self):
        return "".join(
            update.content.text for update in self.updates if isinstance(update, AgentMessageChunk)
        )

    async def wait_for(self, text):
        async with asyncio.timeout(10):
            while text not in self.text():
                self.received.clear()
                await self.received.wait()


async def test_project_memory_survives_acp_restart_deletes_and_stays_off(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("ASTER_USER_DIR", str(home / "user"))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.delenv("ASTER_LLM_CONFIG", raising=False)
    root = tmp_path / "workspace"
    settings = root / ".aster" / "settings.yaml"
    settings.parent.mkdir(parents=True)
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(settings.parent))
    settings.write_text(
        yaml.safe_dump(
            {
                "coding": {
                    "project_memory": {
                        "enabled": True,
                        "embedding": {"backend": "hashing", "dim": 64},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    fixture = Path(__file__).parent / "fixtures" / "project_memory_agent.py"
    client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(client, sys.executable, str(fixture), "write", cwd=tmp_path) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session = await connection.new_session(str(root))
        await connection.ext_method(
            "aster/session/configure",
            {
                "sessionId": session.session_id,
                "permissionMode": "auto",
            },
        )
        await connection.prompt(session.session_id, [text_block("Remember the project backend.")])
        await client.wait_for("memory-write-verified")
        client.updates.clear()
        await connection.prompt(session.session_id, [text_block("/memory list")])
        await client.wait_for("cobalt backend")
        rows = json.loads(client.text())
        assert len(rows) == 1
        memory_id = rows[0]["id"]
        await connection.close_session(session.session_id)

    client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(client, sys.executable, str(fixture), "read", cwd=tmp_path) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        await connection.load_session(str(root), session.session_id)
        await connection.ext_method(
            "aster/session/configure",
            {
                "sessionId": session.session_id,
                "permissionMode": "auto",
            },
        )
        await connection.prompt(
            session.session_id, [text_block("Recall the backend after restart.")]
        )
        await client.wait_for("memory-read-verified")
        await connection.close_session(session.session_id)
        original_session_id = session.session_id
        session = await connection.new_session(str(root))
        assert session.session_id != original_session_id
        await connection.ext_method(
            "aster/session/configure",
            {"sessionId": session.session_id, "permissionMode": "auto"},
        )
        client.updates.clear()
        await connection.prompt(session.session_id, [text_block("Recall in a different session.")])
        await client.wait_for("memory-read-verified")
        client.updates.clear()
        await connection.prompt(session.session_id, [text_block(f"/memory delete {memory_id}")])
        await client.wait_for("Deleted the project memory")
        client.updates.clear()
        await connection.prompt(session.session_id, [text_block("/memory list")])
        await client.wait_for("No saved project memories")
        await connection.prompt(session.session_id, [text_block("/memory off")])
        await client.wait_for("including new sessions")
        await connection.close_session(session.session_id)

    assert yaml.safe_load(settings.read_text())["coding"]["project_memory"]["enabled"] is False
    client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(client, sys.executable, str(fixture), "off", cwd=tmp_path) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session = await connection.new_session(str(root))
        await connection.ext_method(
            "aster/session/configure",
            {
                "sessionId": session.session_id,
                "permissionMode": "auto",
            },
        )
        await connection.prompt(session.session_id, [text_block("Confirm memory remains off.")])
        await client.wait_for("memory-off-verified")
        await connection.close_session(session.session_id)
        await connection.load_session(str(root), original_session_id)
        await connection.ext_method(
            "aster/session/configure",
            {"sessionId": original_session_id, "permissionMode": "auto"},
        )
        client.updates.clear()
        await connection.prompt(
            original_session_id, [text_block("Confirm an older snapshot cannot re-enable memory.")]
        )
        await client.wait_for("memory-off-verified")
        await connection.close_session(original_session_id)
