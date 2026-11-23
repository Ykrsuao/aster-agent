# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real process death between a non-idempotent effect and its checkpoint."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import AgentMessageChunk, RequestPermissionResponse, ToolCallProgress
from aster_cli.sessions import SessionStore


class _Client:
    def __init__(self):
        self.updates = []
        self.received = asyncio.Event()

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)
        self.received.set()

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate({"outcome": {"outcome": "cancelled"}})

    async def wait_for_message(self, text):
        async with asyncio.timeout(10):
            while not any(
                isinstance(update, AgentMessageChunk) and text in update.content.text
                for update in self.updates
            ):
                self.received.clear()
                await self.received.wait()


@pytest.mark.parametrize("phase", ["intent", "outcome"])
@pytest.mark.parametrize("operation", ["shell", "python"])
async def test_process_kill_after_append_requires_review_before_continuing(
    tmp_path, monkeypatch, caplog, phase, operation
):
    caplog.set_level("DEBUG", logger="aster_cli.sessions.store")
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    root = tmp_path / "workspace"
    root.mkdir()
    sessions = root / ".aster" / "sessions"
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(root / ".aster"))
    marker = tmp_path / "effect-ready.json"
    fixture = Path(__file__).parent / "fixtures" / "recovery_agent.py"
    client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(
            client,
            sys.executable,
            str(fixture),
            "crash",
            phase,
            str(marker),
            operation,
            cwd=tmp_path,
        ) as (connection, process),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session = await connection.new_session(str(root))
        await connection.ext_method(
            "aster/session/configure", {"sessionId": session.session_id, "permissionMode": "auto"}
        )
        prompt = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("Append one line.")])
        )
        try:
            async with asyncio.timeout(30):
                while not marker.exists():
                    if prompt.done():
                        await prompt
                        pytest.fail("The process completed before reaching the crash boundary")
                    await asyncio.sleep(0.02)
            action = json.loads(marker.read_text(encoding="utf-8"))["action_id"]
            assert (root / "append.txt").read_text(encoding="utf-8") == "once\n"
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
            await asyncio.gather(prompt, return_exceptions=True)

    with SessionStore(sessions).open(session.session_id) as saved:
        pending = saved.actions.pending()
        expected_kinds = {"python.execute"}
        if operation == "shell":
            expected_kinds.add("shell.run")
        assert {item["kind"] for item in pending} == expected_kinds
        interrupted_action = next(item for item in pending if item["action_id"] == action)
        assert (interrupted_action["outcome"] is not None) == (phase == "outcome")
        action_ids = [item["action_id"] for item in pending]

    restored_client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(
            restored_client,
            sys.executable,
            str(fixture),
            "resume",
            phase,
            str(marker),
            operation,
            cwd=tmp_path,
        ) as (connection, _process),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        await connection.load_session(str(root), session.session_id)
        await connection.ext_method(
            "aster/session/configure", {"sessionId": session.session_id, "permissionMode": "auto"}
        )
        await connection.prompt(session.session_id, [text_block("Continue the interrupted task.")])
        await restored_client.wait_for_message("Confirm the appended line")
        assert any(
            isinstance(update, ToolCallProgress)
            and update.tool_call_id == "resume-2"
            and update.status == "completed"
            for update in restored_client.updates
        )
        assert (root / "append.txt").read_text(encoding="utf-8") == "once\n"
        for action_id in action_ids:
            await connection.prompt(
                session.session_id,
                [
                    text_block(
                        f'/recovery {action_id} --note "Checked append.txt; exactly one line is present"'
                    )
                ],
            )
            await restored_client.wait_for_message(f"Recorded your review of {action_id}")
        await connection.prompt(session.session_id, [text_block("Finish the checked task.")])
        await restored_client.wait_for_message("recovered-without-repeating")
        assert (root / "append.txt").read_text(encoding="utf-8") == "once\n"
        await connection.close_session(session.session_id)

    with SessionStore(sessions).open(session.session_id) as saved:
        assert not saved.actions.pending()
