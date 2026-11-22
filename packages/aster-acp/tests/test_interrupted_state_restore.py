# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP reopens the state behind saved actions when the next prompt is interrupted."""

import asyncio
import json
import time

import pytest
from acp import text_block
from acp.schema import RequestPermissionResponse
from aster_acp._foreground_turn import ForegroundTurn
from aster_acp.server import CodingACPAdapter

from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _Client:
    async def session_update(self, session_id, update, **kwargs):
        pass

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate({"outcome": {"outcome": "cancelled"}})


def _response(tool_id):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id=tool_id,
                name="return_result",
                arguments=json.dumps({"result": {"message": "Saved", "explanation": "Saved"}}),
            )
        ],
    )


@pytest.fixture
async def adapters(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_SETTINGS", "ASTER_RICH_URL", "ASTER_PROMPT_TIMING", "ASTER_SESSIONS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    created = []

    def make(*responses):
        llm = FakeLLMClient(list(responses), strict_exhaustion=True)
        adapter = CodingACPAdapter(lambda: llm)
        adapter.on_connect(_Client())
        created.append(adapter)
        return adapter, llm

    yield make
    for adapter in reversed(created):
        async with asyncio.timeout(15):
            await adapter.close()


@pytest.mark.parametrize("later_action", [False, True])
async def test_acp_restores_before_first_tool_and_keeps_later_effect_pending(
    tmp_path, adapters, monkeypatch, later_action
):
    adapter, _ = adapters(_response("completed-turn"))
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    session = (await adapter._sessions.get(session_id)).value
    effect = tmp_path / "append.txt"
    action = session.handle.actions.begin("file.write_file", str(effect), str(tmp_path))
    effect.write_text("once\n", encoding="utf-8")
    session.handle.actions.finish(action, outcome="Tool returned")
    session.agent.vars["completed"] = [action]
    await adapter.prompt(session_id, [text_block("Keep the completed operation")])

    late = None
    # Skip only process-exit finalization. The real foreground entry still
    # captures and invalidates state before accepting this second prompt.
    with monkeypatch.context() as interrupted:
        interrupted.setattr(ForegroundTurn, "_finalize", lambda self: None)
        async with session.foreground.prompt("interrupted", time.monotonic(), "native"):
            session.handle.record_user_message("Continue", turn_id="interrupted")
            if later_action:
                late = session.handle.actions.begin("python.execute", "Python cell", str(tmp_path))
                with effect.open("a", encoding="utf-8") as stream:
                    stream.write("later\n")
                session.handle.actions.finish(late, outcome="Tool returned")
                session.agent.vars["completed"].append(late)
    await adapter.close_session(session_id)

    reopened, llm = adapters(_response("continued-turn"))
    await reopened.load_session(str(tmp_path), session_id)
    restored = (await reopened._sessions.get(session_id)).value
    assert restored.agent.vars["completed"] == [action]
    assert {item["action_id"] for item in restored.handle.actions.pending()} == (
        {late} if late is not None else set()
    )
    assert effect.read_text(encoding="utf-8") == ("once\nlater\n" if later_action else "once\n")
    restored.controls.set_permission_mode("auto")
    refusal = await restored.controls.authorize("open('append.txt', 'a').write('again')")
    assert (refusal is not None) == later_action
    assert not llm.calls
    assert restored.agent._storage is restored.handle.storage
    assert restored.agent.runtime.event_manager is restored.agent.event_manager

    if late is not None:
        await reopened.prompt(
            session_id,
            [text_block(f'/recovery {late} --note "Checked the later line on disk"')],
        )
        assert not restored.handle.actions.pending()
        assert not llm.calls
    restored.agent.vars["continued"] = True
    await reopened.prompt(session_id, [text_block("Continue from the recovered state")])
    assert llm.call_count == 1
    await reopened.close_session(session_id)

    again, again_llm = adapters()
    await again.load_session(str(tmp_path), session_id)
    saved = (await again._sessions.get(session_id)).value
    assert saved.agent.vars["continued"] is True
    assert saved.agent.vars["completed"] == [action]
    assert not saved.handle.actions.pending()
    assert not again_llm.calls


async def test_acp_cancellation_snapshot_keeps_a_partial_python_effect_pending(tmp_path, adapters):
    code = (
        "with open(self.cwd / 'append.txt', 'a', encoding='utf-8') as stream:\n"
        "    stream.write('once\\n')\nawait asyncio.sleep(3600)"
    )
    response = LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="partial-cell", name="execute_python", arguments=json.dumps({"code": code}))
        ],
    )
    adapter, _ = adapters(response)
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    session = (await adapter._sessions.get(session_id)).value
    session.controls.set_permission_mode("auto")
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("Append and wait")]))
    try:
        async with asyncio.timeout(15):
            while not (tmp_path / "append.txt").exists():
                if prompt.done():
                    await prompt
                    pytest.fail("The cell returned before producing its partial effect")
                await asyncio.sleep(0.01)
            await adapter.cancel(session_id)
            assert (await prompt).stop_reason == "cancelled"
    finally:
        if not prompt.done():
            prompt.cancel()
        await asyncio.gather(prompt, return_exceptions=True)
    assert [item["kind"] for item in session.handle.actions.pending()] == ["python.execute"]
    await adapter.close_session(session_id)

    reopened, llm = adapters()
    await reopened.load_session(str(tmp_path), session_id)
    restored = (await reopened._sessions.get(session_id)).value
    assert [item["kind"] for item in restored.handle.actions.pending()] == ["python.execute"]
    assert (tmp_path / "append.txt").read_text(encoding="utf-8") == "once\n"
    assert not llm.calls
