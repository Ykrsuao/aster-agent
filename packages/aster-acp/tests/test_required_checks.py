# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configured completion checks use real execution and keep ACP permission boundaries."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest
from acp import PROTOCOL_VERSION, text_block
from acp.schema import AgentMessageChunk, RequestPermissionResponse
from aster_acp.server import CodingACPAdapter

from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _Client:
    def __init__(self, choice):
        self.choice = choice
        self.requests = []
        self.messages = []

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.requests.append(tool_call)
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": self.choice}}
        )

    async def session_update(self, session_id, update, **kwargs):
        if isinstance(update, AgentMessageChunk):
            self.messages.append(update.content.text)


def _response(code=None, message="finished"):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="required-check-result",
                name="execute_python" if code is not None else "return_result",
                arguments=json.dumps(
                    {"code": code}
                    if code is not None
                    else {"result": {"message": message, "explanation": "synthetic result"}}
                ),
            )
        ],
    )


@pytest.fixture
async def checked_adapters(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_SETTINGS", "ASTER_MODEL", "ASTER_LLM_CONFIG", "ASTER_RICH_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    config = tmp_path / ".aster"
    config.mkdir()
    (tmp_path / "target.txt").write_text("original", encoding="utf-8")
    script = tmp_path / "check.py"
    script.write_text(
        "from pathlib import Path\n"
        "assert Path('target.txt').read_text(encoding='utf-8').startswith('changed')\n"
        "Path('check-ran.txt').write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    arguments = (Path(sys.executable).as_posix(), script.as_posix())
    command = (
        "& " + " ".join("'" + part.replace("'", "''") + "'" for part in arguments)
        if sys.platform == "win32"
        else shlex.join(arguments)
    )
    (config / "settings.yaml").write_text(
        json.dumps(
            {"coding": {"required_checks": {"unit": {"command": command, "files": ["target.txt"]}}}}
        ),
        encoding="utf-8",
    )
    created = []

    def make(*responses, choice="allow_once"):
        llm = FakeLLMClient(list(responses), strict_exhaustion=True)
        client = _Client(choice)
        adapter = CodingACPAdapter(lambda: llm)
        adapter.on_connect(client)
        created.append(adapter)
        return adapter, llm, client

    yield make
    for adapter in reversed(created):
        async with asyncio.timeout(15):
            await adapter.close()


async def test_acp_rejects_done_after_edit_until_configured_check_passes(
    tmp_path, checked_adapters
):
    adapter, llm, client = checked_adapters(
        _response("await self.shell.write_file('target.txt', 'changed')"),
        _response(message="too early"),
        _response("print(await self.verify('unit'))"),
        _response(),
    )
    async with asyncio.timeout(20):
        await adapter.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        response = await adapter.prompt(session_id, [text_block("Edit and verify the file")])
    assert response.stop_reason == "end_turn"
    assert llm.call_count == 4
    assert "Required checks" in repr(llm.calls[2].messages)
    assert "too early" not in client.messages
    assert "finished" in client.messages
    assert (tmp_path / "check-ran.txt").read_text(encoding="utf-8") == "ran"


@pytest.mark.parametrize("mode,requests", [("manual", 1), ("plan", 0)])
async def test_acp_verify_respects_execution_denial(tmp_path, checked_adapters, mode, requests):
    (tmp_path / "target.txt").write_text("changed", encoding="utf-8")
    adapter, llm, client = checked_adapters(
        _response("print(await self.verify('unit'))"), _response(), choice="reject"
    )
    async with asyncio.timeout(20):
        await adapter.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        await adapter.ext_method(
            "aster/session/configure",
            {
                "sessionId": session_id,
                "mode": "code" if mode == "manual" else "plan",
                "approval": "ask",
            },
        )
        response = await adapter.prompt(session_id, [text_block("Run the configured check")])
    assert response.stop_reason == "end_turn"
    assert len(client.requests) == requests
    assert "not run" in repr(llm.calls[-1].messages).lower()
    assert not (tmp_path / "check-ran.txt").exists()
    session = (await adapter._sessions.get(session_id)).value
    assert session.agent.verification_status()["checks"] == {"unit": "not_run"}


async def test_loaded_acp_session_invalidates_restored_passes_after_new_edits(
    tmp_path, checked_adapters
):
    first, _, _ = checked_adapters(
        _response("await self.shell.write_file('target.txt', 'changed')"),
        _response("print(await self.verify('unit'))"),
        _response(),
    )
    async with asyncio.timeout(25):
        await first.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        session_id = (await first.new_session(str(tmp_path))).session_id
        await first.prompt(session_id, [text_block("Edit and verify the file")])
        session = (await first._sessions.get(session_id)).value
        saved_status = session.agent.verification_status()
        assert saved_status["state"] == "verified"
        assert saved_status["checks"] == {"unit": "passed"}
        await first.close()

        restarted, llm, _ = checked_adapters(
            _response("await self.shell.write_file('target.txt', 'changed again')"),
            _response(message="stale pass"),
            _response("print(await self.verify('unit'))"),
            _response(),
        )
        await restarted.initialize(PROTOCOL_VERSION, aster={"runtimeControls": True})
        await restarted.load_session(str(tmp_path), session_id)
        restored = (await restarted._sessions.get(session_id)).value
        assert not llm.calls
        assert restored.agent.verification_status() == saved_status
        response = await restarted.prompt(
            session_id, [text_block("Make and verify another change")]
        )
    assert response.stop_reason == "end_turn"
    assert llm.call_count == 4
    assert "Required checks" in repr(llm.calls[2].messages)
    status = restored.agent.verification_status()
    assert status["state"] == "verified"
    assert status["checks"] == {"unit": "passed"}
    assert status["task_id"] != saved_status["task_id"]
