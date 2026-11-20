# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider failure preserves saved work for a later prompt or reopened session."""

from __future__ import annotations

import asyncio
import json

import pytest
from acp import RequestError, text_block
from acp.schema import RequestPermissionResponse, ToolCallStart
from aster_acp.server import CodingACPAdapter
from aster_cli.sessions import SessionTurnFinished

from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TIMEOUT = 15


class _Client:
    def __init__(self):
        self.updates = []

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))


def _response(*, source=None):
    value = {"message": "The synthetic check is complete.", "explanation": "synthetic result"}
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="provider-recovery-result",
                name="execute_python" if source is not None else "return_result",
                arguments=json.dumps({"code": source} if source is not None else {"result": value}),
            )
        ],
    )


@pytest.fixture
async def adapters(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_SETTINGS", "ASTER_RICH_URL", "ASTER_PROMPT_TIMING", "ASTER_SESSIONS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    created = []

    def make(*responses):
        llm = FakeLLMClient(list(responses), strict_exhaustion=True)
        client = _Client()
        adapter = CodingACPAdapter(lambda: llm)
        adapter.on_connect(client)
        created.append(adapter)
        return adapter, llm, client

    yield make
    for adapter in reversed(created):
        async with asyncio.timeout(_TIMEOUT):
            await adapter.close()


@pytest.mark.parametrize("provider_failure", ["timeout", "unavailable"])
@pytest.mark.parametrize("restart", [False, True], ids=["same-session", "reload-session"])
async def test_provider_failure_preserves_work_for_continuation(
    tmp_path, adapters, monkeypatch, provider_failure, restart
):
    import litellm

    from aster.errors import GenerationError

    ledger = tmp_path / "applied.txt"
    ledger.write_text("", encoding="utf-8")
    continued = tmp_path / "continued.txt"
    apply = (
        "from pathlib import Path\n"
        f"Path({str(ledger)!r}).write_text("
        f"Path({str(ledger)!r}).read_text(encoding='utf-8') + 'applied\\n', encoding='utf-8')\n"
        "self.vars['stage'] = 'applied'"
    )
    resume = (
        "from pathlib import Path\n"
        "assert self.vars['stage'] == 'applied'\n"
        f"assert Path({str(ledger)!r}).read_text(encoding='utf-8') == 'applied\\n'\n"
        f"Path({str(continued)!r}).write_text('continued\\n', encoding='utf-8')\n"
        "self.vars['stage'] = 'continued'"
    )
    adapter, llm, _ = adapters(_response(source=apply), _response(source=resume), _response())
    original_call = llm.acall
    unavailable = True
    failures = 0
    error_type = (
        litellm.Timeout if provider_failure == "timeout" else litellm.ServiceUnavailableError
    )

    async def call(*args, **kwargs):
        nonlocal failures
        if llm.call_count and unavailable:
            failures += 1
            raise error_type(
                message="synthetic provider failure", model="test-model", llm_provider="openai"
            )
        return await original_call(*args, **kwargs)

    monkeypatch.setattr(llm, "acall", call)
    async with asyncio.timeout(_TIMEOUT):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        with pytest.raises(RequestError, match="synthetic provider failure") as caught:
            await adapter.prompt(session_id, [text_block("Apply one change, then finish")])
        assert caught.value.code == -32603
        assert isinstance(caught.value.__cause__, GenerationError)
        assert isinstance(caught.value.__cause__.__cause__, error_type)
        assert failures > 0
        assert llm.call_count == 1
        assert ledger.read_text(encoding="utf-8") == "applied\n"
        assert not continued.exists()
        session = (await adapter._sessions.get(session_id)).value
        assert session.agent.vars["stage"] == "applied"
        interrupted = [
            event
            for event in session.handle.events.values()
            if isinstance(event, SessionTurnFinished)
        ]
        assert len(interrupted) == 1
        assert interrupted[0].completed_message_id is None

        if restart:
            await adapter.close()
            adapter, llm, client = adapters(_response(source=resume), _response())
            await adapter.load_session(str(tmp_path), session_id)
            assert not llm.calls
            assert ledger.read_text(encoding="utf-8") == "applied\n"
            session = (await adapter._sessions.get(session_id)).value
            assert session.agent.vars["stage"] == "applied"
            assert any(isinstance(update, ToolCallStart) for _, update in client.updates)
        else:
            unavailable = False

        response = await adapter.prompt(session_id, [text_block("Continue the unfinished work")])
        assert response.stop_reason == "end_turn"
        assert session.agent.vars["stage"] == "continued"
        assert ledger.read_text(encoding="utf-8") == "applied\n"
        assert continued.read_text(encoding="utf-8") == "continued\n"
        assert llm.remaining_responses == 0
        completions = [
            event
            for event in session.handle.events.values()
            if isinstance(event, SessionTurnFinished)
        ]
        assert len(completions) == 2
        assert completions[0].completed_message_id is None
        assert completions[1].completed_message_id is not None
