# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foreground completion preserves cleanup ordering and durable reply ownership."""

from __future__ import annotations

import asyncio
import json
from typing import Literal

import pytest
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, RequestPermissionResponse
from aster_acp.server import CodingACPAdapter
from aster_cli.sessions import SessionTurnFinished

from aster.skill import Skill, slash_command
from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TIMEOUT = 15
_REPLY = "The synthetic check is complete."


class _Client:
    def __init__(self):
        self.updates = []
        self.message_received = asyncio.Event()

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))
        if isinstance(update, AgentMessageChunk):
            self.message_received.set()

    def messages(self, session_id):
        return [
            update.content.text
            for current_id, update in self.updates
            if current_id == session_id and isinstance(update, AgentMessageChunk)
        ]


def _response(*, message=_REPLY, source=None, waiting=False):
    value = {"message": message, "explanation": "synthetic result"}
    if waiting:
        value["on"] = ["system_messages"]
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id="foreground-result",
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


async def _tree(adapter, session_id):
    return await adapter.ext_method("aster/session/tree", {"sessionId": session_id})


async def test_cancel_waits_for_background_cleanup_before_finishing_or_reusing_turn(
    tmp_path, adapters
):
    adapter, llm, client = adapters(
        _response(message="Waiting for the check.", waiting=True), _response()
    )
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    session = (await adapter._sessions.get(session_id)).value
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def background():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            session.agent.vars["cleanup"] = "finished"

    session.agent.queue_manager.queue("checks")
    job = session.agent.queue_manager.spawn(background(), channel="checks")
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("Wait for the check")]))
    cancellation = None
    try:
        async with asyncio.timeout(_TIMEOUT):
            await started.wait()
            await client.message_received.wait()
            cancellation = asyncio.create_task(adapter.cancel(session_id))
            await cleaning.wait()
            # Let a premature completion run; cleanup itself remains gated.
            await session.bridge.flush()
            assert not prompt.done()
            assert not cancellation.done()
            assert client.messages(session_id) == ["Waiting for the check."]
            with pytest.raises(RequestError) as busy:
                await adapter.prompt(session_id, [text_block("Too early")])
            assert busy.value.code == -32600

            release.set()
            await cancellation
            assert (await prompt).stop_reason == "cancelled"
            assert job.state == "cancelled"
            assert session.agent.vars["cleanup"] == "finished"
            assert client.messages(session_id) == [
                "Waiting for the check.",
                "Stopped at your request.",
            ]
            stopped = await _tree(adapter, session_id)
            assert all(
                not node["navigable"] for node in stopped["nodes"] if node["role"] == "assistant"
            )

            response = await adapter.prompt(session_id, [text_block("Continue after cleanup")])
            assert response.stop_reason == "end_turn"
            assert client.messages(session_id)[-1] == _REPLY
            assert llm.call_count == 2
    finally:
        release.set()
        if not prompt.done():
            prompt.cancel()
        await asyncio.gather(
            prompt, *([cancellation] if cancellation else []), return_exceptions=True
        )
        await job.cancel()


class _HostCommands(Skill):
    @slash_command("host", output_to_agent=False, argument_hint="<reply|empty|fail>")
    def host(self, mode: Literal["reply", "empty", "fail"]) -> str:
        """Return a host-rendered response without calling the model."""
        if mode == "fail":
            raise RuntimeError("synthetic command failure")
        return "Host reply" if mode == "reply" else ""


@pytest.mark.parametrize("mode", ["reply", "empty", "invalid", "fail"])
async def test_host_slash_completion_keeps_its_own_durable_state(tmp_path, adapters, mode):
    adapter, llm, client = adapters()
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    session = (await adapter._sessions.get(session_id)).value
    session.agent.skills.register("test.host_commands", _HostCommands())
    session.agent.vars["stage"] = "foreground state"
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def background():
        started.set()
        await release.wait()
        session.agent.vars["stage"] = "background state"
        session.agent.context["background_only"] = "arrived after the reply"
        session.agent.message("Background finished")
        finished.set()

    def release_on_reply(event):
        if event.content != "Background finished":
            release.set()

    unsubscribe = session.agent.event_manager.on("AgentMessage", release_on_reply)
    session.agent.queue_manager.queue("checks")
    job = session.agent.queue_manager.spawn(background(), channel="checks")
    try:
        async with asyncio.timeout(_TIMEOUT):
            await started.wait()
            response = await adapter.prompt(session_id, [text_block(f"/host {mode}")])
            assert response.stop_reason == "end_turn"
            assert llm.call_count == 0
            if mode == "empty":
                assert client.messages(session_id) == []
                release.set()
            await finished.wait()
            current = await _tree(adapter, session_id)
            replies = [
                node
                for node in current["nodes"]
                if node["role"] == "assistant" and node["preview"] != "Background finished"
            ]
            assert len(replies) == (0 if mode == "empty" else 1)
            background_node = next(
                node for node in current["nodes"] if node["preview"] == "Background finished"
            )
            assert not background_node["navigable"]
            outcomes = [
                event
                for event in session.handle.events.values()
                if isinstance(event, SessionTurnFinished)
            ]
            assert len(outcomes) == 1
            assert outcomes[0].completed_message_id == (replies[0]["id"] if replies else None)
            if replies:
                expected = {"reply": "Host reply", "invalid": "/host:", "fail": "/host failed:"}
                assert replies[0]["preview"].startswith(expected[mode])
                assert replies[0]["navigable"]

            await adapter.close()
            restarted, restarted_llm, _ = adapters()
            await restarted.load_session(str(tmp_path), session_id)
            assert await _tree(restarted, session_id) == current
            if replies:
                selected = await restarted.ext_method(
                    "aster/session/navigate",
                    {
                        "sessionId": session_id,
                        "nodeId": replies[0]["id"],
                        "expectedCurrentNodeId": current["currentNodeId"],
                    },
                )
                await restarted.load_session(str(tmp_path), selected["sessionId"])
                restored = (await restarted._sessions.get(selected["sessionId"])).value
                assert restored.agent.vars["stage"] == "foreground state"
                assert "background_only" not in restored.agent.context
            assert restarted_llm.call_count == 0
    finally:
        unsubscribe()
        release.set()
        await job.cancel()


def _track_message_subscriptions(monkeypatch, agent):
    original = agent.event_manager.on
    active = set()

    def subscribe(event_type, handler):
        unsubscribe = original(event_type, handler)
        if event_type != "AgentMessage":
            return unsubscribe
        token = object()
        active.add(token)

        def remove():
            unsubscribe()
            active.discard(token)

        return remove

    monkeypatch.setattr(agent.event_manager, "on", subscribe)
    return active


@pytest.mark.parametrize("outcome", ["success", "generation-error", "cancel", "metadata-error"])
async def test_prompt_releases_message_observers_and_completion_callback(
    tmp_path, adapters, monkeypatch, outcome
):
    adapter, llm, client = adapters(
        _response(source=f"self.message({_REPLY!r})"),
        LLMResponse(content="", finish_reason="error")
        if outcome == "generation-error"
        else _response(),
        _response(),
    )
    session_id = (await adapter.new_session(str(tmp_path))).session_id
    session = (await adapter._sessions.get(session_id)).value
    active = _track_message_subscriptions(monkeypatch, session.agent)
    previous_completions = []

    def previous_callback(reply_id):
        previous_completions.append(reply_id)

    session.dispatcher.on_turn_completed = previous_callback
    entered = asyncio.Event()
    original_call = llm.acall
    calls = 0

    async def call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and outcome == "cancel":
            entered.set()
            await asyncio.Event().wait()
        return await original_call(*args, **kwargs)

    monkeypatch.setattr(llm, "acall", call)
    if outcome == "metadata-error":
        original_add = session.handle.events.add
        failed = False

        def add(event, *, record=True):
            nonlocal failed
            if isinstance(event, SessionTurnFinished) and not failed:
                failed = True
                raise OSError("synthetic completion write failure")
            return original_add(event, record=record)

        monkeypatch.setattr(session.handle.events, "add", add)

    prompt = asyncio.create_task(
        adapter.prompt(session_id, [text_block("First request")], aster={"turnId": "first"})
    )
    try:
        async with asyncio.timeout(_TIMEOUT):
            if outcome == "cancel":
                await entered.wait()
                await adapter.cancel(session_id)
                assert (await prompt).stop_reason == "cancelled"
            elif outcome == "generation-error":
                with pytest.raises(RequestError, match="incomplete response"):
                    await prompt
            elif outcome == "metadata-error":
                with pytest.raises(OSError, match="synthetic completion write failure"):
                    await prompt
            else:
                assert (await prompt).stop_reason == "end_turn"
            assert not active
            assert session.dispatcher.on_turn_completed is previous_callback
            assert previous_completions == []
            first = await _tree(adapter, session_id)
            reply = next(node for node in first["nodes"] if node["preview"] == _REPLY)
            assert reply["navigable"] is (outcome == "success")

            response = await adapter.prompt(
                session_id, [text_block("Second request")], aster={"turnId": "second"}
            )
            assert response.stop_reason == "end_turn"
            assert not active
            assert session.dispatcher.on_turn_completed is previous_callback
            assert previous_completions == []
            assert client.messages(session_id).count(_REPLY) == 2
            current = await _tree(adapter, session_id)
            assert current["nodes"][-1]["preview"] == _REPLY
            assert current["nodes"][-1]["navigable"]
            user = next(
                node
                for node in current["nodes"]
                if node["role"] == "user" and node["preview"] == "Second request"
            )
            completion = [
                event
                for event in session.handle.events.values()
                if isinstance(event, SessionTurnFinished)
            ][-1]
            assert completion.user_message_id == user["id"]
            assert completion.completed_message_id == current["currentNodeId"]
    finally:
        if not prompt.done():
            prompt.cancel()
        await asyncio.gather(prompt, return_exceptions=True)
