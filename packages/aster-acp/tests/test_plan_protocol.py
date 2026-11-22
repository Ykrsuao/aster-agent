# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The actual ACP process can inspect this repository in Plan, including after reopen."""

import asyncio
import json
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import RequestPermissionResponse, ToolCallProgress


class _Client:
    def __init__(self):
        self.updates = []
        self.permissions = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.permissions.append(tool_call)
        return RequestPermissionResponse.model_validate({"outcome": {"outcome": "cancelled"}})


async def test_plan_inspects_the_checkout_over_acp_and_reopens_with_fresh_permissions(
    tmp_path, monkeypatch
):
    for name, directory in (
        ("ASTER_USER_DIR", "user"),
        ("ASTER_PROJECT_DIR", "project"),
        ("ASTER_SESSIONS_DIR", "sessions"),
    ):
        monkeypatch.setenv(name, str(tmp_path / directory))
    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    root = Path(__file__).resolve().parents[3]
    fixture = Path(__file__).parent / "fixtures" / "plan_agent.py"
    source = root / "packages/aster-acp/src/aster_acp/runtime_controls.py"
    original = source.read_bytes()
    sentinel = tmp_path / "forbidden.txt"
    client = _Client()
    async with (
        asyncio.timeout(60),
        spawn_agent_process(client, sys.executable, str(fixture), str(sentinel), cwd=tmp_path) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session = await connection.new_session(str(root))
        for reopened in (False, True):
            if reopened:
                await connection.load_session(str(root), session.session_id)
                status = await connection.ext_method(
                    "aster/session/runtime", {"sessionId": session.session_id}
                )
                assert status["permissionMode"] == "manual"
            status = await connection.ext_method(
                "aster/session/configure",
                {"sessionId": session.session_id, "permissionMode": "plan"},
            )
            assert status["permissionMode"] == "plan"
            client.updates.clear()
            result = await connection.prompt(
                session.session_id,
                [text_block("Inspect this repository and plan; do not change files.")],
            )
            assert result.stop_reason == "end_turn"
            completed = {
                update.tool_call_id: update.status
                for update in client.updates
                if isinstance(update, ToolCallProgress)
                and update.tool_call_id.startswith("plan-wire-")
                and update.status in {"completed", "failed"}
            }
            assert completed == {
                "plan-wire-0": "completed",
                "plan-wire-1": "completed",
                "plan-wire-2": "completed",
                "plan-wire-3": "failed",
            }
            serialized = json.dumps([update.model_dump(mode="json") for update in client.updates])
            assert "Explicit bidirectional controls reorder" in serialized
            assert "Plan mode" in serialized
            assert not client.permissions
            assert not sentinel.exists()
            assert source.read_bytes() == original
            await connection.close_session(session.session_id)
