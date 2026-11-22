# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A saved task cannot lose its acceptance contract when the ACP process restarts."""

import asyncio
import json
import shlex
import sys
from pathlib import Path

import yaml
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import AgentMessageChunk, RequestPermissionResponse, ToolCallProgress


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
        return json.dumps([update.model_dump(mode="json") for update in self.updates])

    async def wait_for_message(self, text):
        # A prompt response can arrive before notification callbacks have run.
        async with asyncio.timeout(10):
            while not any(
                isinstance(update, AgentMessageChunk) and text in update.content.text
                for update in self.updates
            ):
                self.received.clear()
                await self.received.wait()


async def test_pending_verification_survives_plain_answer_and_acp_process_restart(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    root = tmp_path / "workspace"
    root.mkdir()
    settings = root / ".aster"
    settings.mkdir()
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(settings))
    args = [
        Path(sys.executable).as_posix(),
        "-c",
        "from pathlib import Path; assert Path('code.txt').read_text() == 'fixed'",
    ]
    command = (
        "& " + " ".join("'" + value.replace("'", "''") + "'" for value in args)
        if sys.platform == "win32"
        else shlex.join(args)
    )
    (settings / "settings.yaml").write_text(
        yaml.safe_dump(
            {"coding": {"required_checks": {"tests": {"command": command, "files": ["code.txt"]}}}}
        ),
        encoding="utf-8",
    )
    (root / "code.txt").write_text("original", encoding="utf-8")
    fixture = Path(__file__).parent / "fixtures" / "verification_agent.py"
    client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(client, sys.executable, str(fixture), "edit", cwd=tmp_path) as (
            connection,
            _process,
        ),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session = await connection.new_session(str(root))
        await connection.ext_method(
            "aster/session/configure", {"sessionId": session.session_id, "permissionMode": "auto"}
        )
        await connection.prompt(session.session_id, [text_block("Edit code.txt, then ask me.")])
        await client.wait_for_message("Choose the next step")
        assert (root / "code.txt").read_text() == "fixed"
        client.updates.clear()
        await connection.prompt(session.session_id, [text_block("What is the capital of France?")])
        await client.wait_for_message("Paris")
        assert "still needs verification: tests" in client.text()
        await connection.close_session(session.session_id)

    restored_client = _Client()
    async with (
        asyncio.timeout(90),
        spawn_agent_process(
            restored_client, sys.executable, str(fixture), "resume", cwd=tmp_path
        ) as (connection, _process),
    ):
        await connection.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        await connection.load_session(str(root), session.session_id)
        await connection.ext_method(
            "aster/session/configure", {"sessionId": session.session_id, "permissionMode": "auto"}
        )
        restored_client.updates.clear()
        result = await connection.prompt(
            session.session_id, [text_block("Complete the saved task.")]
        )
        assert result.stop_reason == "end_turn"
        await restored_client.wait_for_message("verified-after-reopen")
        assert any(
            isinstance(update, ToolCallProgress)
            and update.tool_call_id == "verification-wire-1"
            and update.status == "completed"
            for update in restored_client.updates
        )
        assert (root / "code.txt").read_text() == "fixed"
        await connection.close_session(session.session_id)
