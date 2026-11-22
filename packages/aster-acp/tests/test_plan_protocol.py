# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The actual ACP process can inspect this repository in Plan, including after reopen."""

import asyncio
import json
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import (
    AgentMessageChunk,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallProgress,
)


class _Client:
    def __init__(self):
        self.updates = []
        self.permissions = []
        self.updates_received = asyncio.Condition()

    async def session_update(self, session_id, update, **kwargs):
        async with self.updates_received:
            self.updates.append(update)
            self.updates_received.notify_all()

    async def wait_for_turn(self, tool_call_ids, reply):
        async with self.updates_received:
            await self.updates_received.wait_for(
                lambda: (
                    tool_call_ids.issubset(
                        update.tool_call_id
                        for update in self.updates
                        if isinstance(update, ToolCallProgress)
                        and update.status in {"completed", "failed"}
                    )
                    and any(
                        isinstance(update, AgentMessageChunk)
                        and isinstance(update.content, TextContentBlock)
                        and update.content.text.strip() == reply
                        and (update.field_meta or {}).get("aster", {}).get("messageState")
                        == "complete"
                        for update in self.updates
                    )
                )
            )

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
        for round_id, reopened in enumerate((False, True)):
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
            prefix = f"plan-wire-{round_id}-"
            expected = {
                f"{prefix}0": "completed",
                f"{prefix}1": "completed",
                f"{prefix}2": "completed",
                f"{prefix}3": "failed",
            }
            result = await connection.prompt(
                session.session_id,
                [text_block("Inspect this repository and plan; do not change files.")],
            )
            assert result.stop_reason == "end_turn"
            # The response and tool statuses can precede output callbacks. Await the
            # unique final reply too, so replayed updates cannot satisfy this turn.
            await client.wait_for_turn(set(expected), f"Plan inspection round {round_id} finished.")
            turn_updates = [
                update
                for update in client.updates
                if isinstance(update, ToolCallProgress) and update.tool_call_id.startswith(prefix)
            ]
            completed = {
                update.tool_call_id: update.status
                for update in turn_updates
                if update.status in {"completed", "failed"}
            }
            assert completed == expected
            serialized = json.dumps([update.model_dump(mode="json") for update in turn_updates])
            assert "Explicit bidirectional controls reorder" in serialized
            assert "Plan mode" in serialized
            assert not client.permissions
            assert not sentinel.exists()
            assert source.read_bytes() == original
            await connection.close_session(session.session_id)
