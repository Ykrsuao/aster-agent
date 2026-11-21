# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permission modes govern real CodeAct cells, file effects and live sessions."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager, suppress

import pytest
from acp import RequestError, text_block
from acp.schema import RequestPermissionResponse
from aster_acp.runtime_controls import RuntimeControls
from aster_acp.server import CodingACPAdapter
from aster_cli.coding import CodingAgent
from aster_cli.coding.activity import FileEdit
from aster_cli.coding.reply_strategy import CodingReplyStrategy

from aster.config import CodeActConfig
from aster.context_blocks.events import ResultStatus
from aster.events import ExecutionResult, PythonOutput
from aster.interactive import Done
from aster.runtime.hooks import InstrumentationHooks, hooks_scope
from aster.runtime.middleware import ExecutePythonContext, ExecutePythonNext
from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall, registry


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_SETTINGS", "ASTER_MODEL", "ASTER_LLM_CONFIG", "ASTER_RICH_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(
        registry,
        "MODELS",
        {"alpha": {"model_name": "test-alpha"}, "beta": {"model_name": "test-beta"}},
    )


class _Client:
    def __init__(self, choice="reject"):
        self.choice = choice
        self.requests = []
        self.updates = []
        self.entered = asyncio.Event()
        self.release = None

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.requests.append(tool_call)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": self.choice}}
        )

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))


def _turn(*codes: str, label="turn") -> list[LLMResponse]:
    calls = [("execute_python", {"code": code}) for code in codes]
    calls.append(("return_result", {"result": {"message": "Finished.", "explanation": "complete"}}))
    return [
        LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=f"permission-{label}-{index}",
                    name=name,
                    arguments=json.dumps(arguments),
                )
            ],
        )
        for index, (name, arguments) in enumerate(calls)
    ]


@asynccontextmanager
async def _agent(tmp_path, responses, *, client=None, **settings):
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    agent = CodingAgent(llm=llm, cwd=tmp_path, skills_dirs=[])
    client = client or _Client()
    controls = RuntimeControls(
        agent,
        client,
        "permissions",
        publish=lambda update: client.updates.append(("permissions", update)),
        model="alpha",
        model_names=lambda: ["alpha", "beta"],
        **settings,
    )
    agent._execution_control = controls
    try:
        async with asyncio.timeout(20):
            yield agent, controls, client, llm
    finally:
        await agent.close()


def _outputs(agent) -> list[PythonOutput]:
    return [
        event
        for event in agent.event_manager.values()
        if isinstance(event, PythonOutput) and event.tool_call_id.startswith("permission-")
    ]


@pytest.mark.parametrize(
    "permission_mode,legacy_mode,approval,written,asks",
    [
        ("auto", "code", "session", True, 0),
        ("manual", "code", "ask", False, 1),
        ("accept_edits", "code", "accept_edits", False, 1),
        ("plan", "plan", "ask", False, 0),
    ],
)
async def test_permission_modes_control_real_python_execution(
    tmp_path, permission_mode, legacy_mode, approval, written, asks
):
    target = tmp_path / "python-effect.txt"
    code = f"from pathlib import Path\nPath({str(target)!r}).write_text('executed')"
    async with _agent(tmp_path, _turn(code)) as (agent, controls, client, llm):
        state = await controls.configure(permission_mode=permission_mode)
        assert (state["permissionMode"], state["mode"], state["approval"]) == (
            permission_mode,
            legacy_mode,
            approval,
        )
        assert client.updates[-1][1].field_meta["aster"]["runtime"] == state

        result = await agent.handle({"user_messages": ["Create the synthetic file if permitted."]})

        assert isinstance(result, Done)
        assert target.exists() is written
        assert len(client.requests) == asks
        if written:
            assert target.read_text() == "executed"
        else:
            assert _outputs(agent)[0].execution_status is ResultStatus.ERROR
            assert "not run" in repr(llm.calls[-1].messages).lower()
        if asks:
            assert client.requests[0].raw_input == {"code": code}


async def test_accept_edits_reads_writes_and_replaces_without_approval(tmp_path):
    codes = (
        "await self.shell.write_file('note.txt', 'before\\n')",
        "await self.shell.replace('note.txt', 'before', new='after')",
        "await self.shell.read(path='note.txt', lines=(1, 1))",
    )
    async with _agent(tmp_path, _turn(*codes)) as (agent, controls, client, llm):
        seen = []
        edits: list[FileEdit] = []
        agent.event_manager.on("FileEdit", edits.append)

        async def observe(ctx: ExecutePythonContext, nxt: ExecutePythonNext):
            result = await nxt(ctx)
            if ctx.code in codes:
                seen.append((ctx.code, result.result.success))
            return result

        agent.event_manager.intercept("execute_python", observe)
        await controls.configure(permission_mode="accept_edits")

        result = await agent.handle({"user_messages": ["Write, edit and read the note."]})

        assert isinstance(result, Done)
        assert not client.requests
        assert (tmp_path / "note.txt").read_text() == "after\n"
        assert seen == [(code, True) for code in codes]
        outputs = _outputs(agent)
        assert len(outputs) == 3
        assert all(output.execution_status is ResultStatus.COMPLETE for output in outputs)
        assert "after" in str(outputs[-1].value)
        assert "after" in repr(llm.calls[-1].messages)
        assert [(edit.operation, edit.new_text) for edit in edits] == [
            ("create", "before\n"),
            ("update", "after"),
        ]


@pytest.mark.parametrize("operation", ["shell", "mixed", "dynamic", "assigned", "multiple"])
async def test_accept_edits_rejects_other_cells_before_any_side_effect(tmp_path, operation):
    target = tmp_path / "note.txt"
    target.write_text("unchanged")
    sentinel = tmp_path / "must-not-exist.txt"
    effect = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('executed')"
    shell_command = (
        "Set-Content -LiteralPath 'must-not-exist.txt' -Value 'executed'"
        if sys.platform == "win32"
        else "printf executed > must-not-exist.txt"
    )
    code = {
        "shell": f"await self.shell.run({shell_command!r})",
        "mixed": "await self.shell.write_file('note.txt', 'changed')\n" + effect,
        "dynamic": f"await self.shell.write_file('note.txt', ({effect}, 'changed')[1])",
        "assigned": "result = await self.shell.write_file('note.txt', 'changed')",
        "multiple": (
            "await self.shell.write_file('note.txt', 'changed')\nawait self.shell.read('note.txt')"
        ),
    }[operation]
    async with _agent(tmp_path, _turn(code)) as (agent, controls, client, llm):
        await controls.configure(permission_mode="accept_edits")

        assert isinstance(await agent.handle({"user_messages": ["Perform the operation."]}), Done)

        assert len(client.requests) == 1
        assert client.requests[0].raw_input == {"code": code}
        assert target.read_text() == "unchanged"
        assert not sentinel.exists()
        assert "rejected" in repr(llm.calls[-1].messages).lower()


async def test_plan_does_not_auto_accept_a_direct_file_edit(tmp_path):
    code = "await self.shell.write_file('planned.txt', 'must not be written')"
    async with _agent(tmp_path, _turn(code)) as (agent, controls, client, llm):
        await controls.configure(permission_mode="accept_edits")
        await controls.configure(permission_mode="plan")

        assert isinstance(await agent.handle({"user_messages": ["Plan this change."]}), Done)

        assert not (tmp_path / "planned.txt").exists()
        assert not client.requests
        assert "plan mode" in repr(llm.calls[-1].messages).lower()


@pytest.mark.parametrize("previous_mode", ["auto", "accept_edits"])
async def test_switching_back_to_manual_requires_approval_for_the_next_edit(
    tmp_path, previous_mode
):
    responses = [
        *_turn("await self.shell.write_file('note.txt', 'first')", label="first"),
        *_turn("await self.shell.write_file('note.txt', 'second')", label="second"),
    ]
    async with _agent(tmp_path, responses) as (agent, controls, client, _):
        await controls.configure(permission_mode=previous_mode)
        await agent.handle({"user_messages": ["Make the first edit."]})
        assert (tmp_path / "note.txt").read_text() == "first"
        assert not client.requests

        await controls.configure(permission_mode="manual")
        await agent.handle({"user_messages": ["Make the second edit only if approved."]})

        assert len(client.requests) == 1
        assert (tmp_path / "note.txt").read_text() == "first"
        assert controls.status()["permissionMode"] == "manual"


async def test_accept_edits_preserves_normal_execution_for_separately_approved_python(tmp_path):
    target = tmp_path / "python.txt"
    python = f"from pathlib import Path\nPath({str(target)!r}).write_text('ordinary Python')"
    codes = (
        "await self.shell.write_file('before.txt', 'accepted edit')",
        python,
        "await self.shell.write_file('after.txt', 'another accepted edit')",
    )
    async with _agent(tmp_path, _turn(*codes), client=_Client("allow_once")) as (
        agent,
        controls,
        client,
        _,
    ):
        await controls.configure(permission_mode="accept_edits")

        assert isinstance(
            await agent.handle({"user_messages": ["Perform these operations."]}), Done
        )

        assert [request.raw_input for request in client.requests] == [{"code": python}]
        assert target.read_text() == "ordinary Python"
        assert (tmp_path / "before.txt").read_text() == "accepted edit"
        assert (tmp_path / "after.txt").read_text() == "another accepted edit"
        assert controls.status()["permissionMode"] == "accept_edits"


async def test_file_errors_reach_the_model_and_do_not_break_later_edits(tmp_path):
    codes = (
        "await self.shell.read('missing.txt')",
        "await self.shell.write_file('recovered.txt', 'created after the failed read')",
    )
    async with _agent(tmp_path, _turn(*codes)) as (agent, controls, client, llm):
        await controls.configure(permission_mode="accept_edits")

        assert isinstance(
            await agent.handle({"user_messages": ["Read, then create the file."]}), Done
        )

        assert not client.requests
        assert "FileNotFoundError" in repr(llm.calls[1].messages)
        assert [output.execution_status for output in _outputs(agent)] == [
            ResultStatus.ERROR,
            ResultStatus.COMPLETE,
        ]
        assert (tmp_path / "recovered.txt").read_text() == "created after the failed read"


async def test_accept_edits_cannot_bypass_execution_middleware(tmp_path):
    code = "await self.shell.write_file('protected.txt', 'must not be written')"
    async with _agent(tmp_path, _turn(code)) as (agent, controls, client, llm):

        async def block_edit(ctx: ExecutePythonContext, nxt: ExecutePythonNext):
            if ctx.code == code:
                ctx.result = ExecutionResult(error=PermissionError("Workspace policy blocked edit"))
                return ctx
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", block_edit)
        await controls.configure(permission_mode="accept_edits")

        assert isinstance(await agent.handle({"user_messages": ["Edit the protected file."]}), Done)

        assert not (tmp_path / "protected.txt").exists()
        assert not client.requests
        assert "Workspace policy blocked edit" in repr(llm.calls[-1].messages)


@pytest.mark.parametrize("permission_mode", ["manual", "accept_edits"])
@pytest.mark.parametrize(
    "initial_timeout,middleware_timeout,written",
    [
        pytest.param(30, 0, False, id="expire"),
        pytest.param(0, 30, True, id="extend"),
        pytest.param(0, None, True, id="remove"),
    ],
)
async def test_file_execution_uses_final_middleware_timeout(
    tmp_path, permission_mode, initial_timeout, middleware_timeout, written
):
    code = "await self.shell.write_file('note.txt', 'written within the final limit')"
    target = tmp_path / "note.txt"
    strategy = CodingReplyStrategy(config=CodeActConfig(cell_timeout=initial_timeout, prefill=None))
    async with _agent(tmp_path, _turn(code), client=_Client("allow_once")) as (
        agent,
        controls,
        client,
        _,
    ):
        edits: list[FileEdit] = []
        agent.event_manager.on("FileEdit", edits.append)

        async def change_timeout(ctx: ExecutePythonContext, nxt: ExecutePythonNext):
            if ctx.code == code:
                ctx.params["timeout"] = middleware_timeout
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", change_timeout)
        await controls.configure(permission_mode=permission_mode)

        result = await agent.handle(
            {"user_messages": ["Write the note within the configured execution limit."]},
            _strategy=strategy,
        )

        assert isinstance(result, Done)
        assert target.exists() is written
        assert [request.raw_input for request in client.requests] == (
            [{"code": code}] if permission_mode == "manual" else []
        )
        (output,) = _outputs(agent)
        if written:
            assert target.read_text() == "written within the final limit"
            assert output.execution_status is ResultStatus.COMPLETE
            assert not output.error
            assert len(edits) == 1
            assert edits[0].new_text == "written within the final limit"
        else:
            assert output.execution_status is ResultStatus.ERROR
            assert "TimeoutError" in output.error
            assert not edits


@pytest.mark.parametrize("legacy", [{"mode": "code"}, {"approval": "session"}])
async def test_conflicting_permission_parameters_do_not_partially_change_the_model(
    tmp_path, legacy
):
    created = []

    def model_factory(name):
        created.append(name)
        return FakeLLMClient()

    async with _agent(tmp_path, [], model_factory=model_factory) as (agent, controls, _, llm):
        before = controls.status()

        with pytest.raises(ValueError):
            await controls.configure(model="beta", permission_mode="auto", **legacy)

        assert controls.status() == before
        assert agent.llm is llm
        assert not created


async def test_legacy_mode_and_approval_settings_remain_compatible(tmp_path):
    async with _agent(tmp_path, []) as (_, controls, _, _):
        assert controls.status()["permissionMode"] == "manual"
        state = await controls.configure(approval="session")
        assert state["permissionMode"] == "auto"
        state = await controls.configure(mode="plan")
        assert (state["permissionMode"], state["approval"]) == ("plan", "ask")
        state = await controls.configure(mode="code")
        assert state["permissionMode"] == "manual"
        state = await controls.configure(approval="accept_edits")
        assert state["permissionMode"] == "accept_edits"
        state = await controls.configure(approval="ask")
        assert state["permissionMode"] == "manual"


async def test_permission_mode_cannot_change_during_a_pending_execution_approval(tmp_path):
    code = "await self.shell.write_file('pending.txt', 'must not be written')"
    client = _Client()
    client.release = asyncio.Event()
    async with _agent(tmp_path, _turn(code), client=client) as (agent, controls, _, _):
        running = asyncio.create_task(agent.handle({"user_messages": ["Request an edit."]}))
        try:
            await client.entered.wait()
            before = controls.status()
            assert before["approvalPending"] is True

            with pytest.raises(ValueError, match="approval"):
                await controls.configure(permission_mode="auto")

            assert controls.status() == before
            assert not (tmp_path / "pending.txt").exists()
            client.release.set()
            assert isinstance(await running, Done)
            assert controls.status()["permissionMode"] == "manual"
            assert controls.status()["approvalPending"] is False
        finally:
            client.release.set()
            if not running.done():
                running.cancel()
            with suppress(asyncio.CancelledError):
                await running


async def test_permission_mode_can_change_while_background_work_is_running(tmp_path):
    async with _agent(tmp_path, []) as (agent, controls, _, _):
        started = asyncio.Event()

        async def background():
            started.set()
            await asyncio.Event().wait()

        agent.queue_manager.queue("checks")
        job = agent.queue_manager.spawn(background(), channel="checks")
        try:
            await started.wait()
            state = await controls.configure(permission_mode="accept_edits")
            assert state["permissionMode"] == "accept_edits"
            assert job.state == "running"
            with pytest.raises(ValueError, match="background"):
                await controls.configure(permission_mode="plan", model="beta")
            assert controls.status() == state
        finally:
            await job.cancel()


@pytest.mark.parametrize("permission_mode", ["manual", "plan"])
async def test_background_generation_uses_the_new_permission_for_its_next_cell(
    tmp_path, monkeypatch, permission_mode
):
    code = "await self.shell.write_file('background.txt', 'must stay unwritten')"
    async with _agent(tmp_path, _turn(code)) as (agent, controls, client, llm):
        await controls.configure(permission_mode="auto")
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = llm.acall
        results = []

        async def acall(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        async def background():
            results.append(await agent.handle({"user_messages": ["Write the file."]}))
            finished.set()

        monkeypatch.setattr(llm, "acall", acall)
        agent.queue_manager.queue("checks")
        job = agent.queue_manager.spawn(background(), channel="checks")
        try:
            await entered.wait()
            await controls.configure(permission_mode=permission_mode)
            assert job.state == "running"
            release.set()
            await finished.wait()
            assert isinstance(results[0], Done)
            assert not (tmp_path / "background.txt").exists()
            assert len(client.requests) == int(permission_mode == "manual")
        finally:
            release.set()
            await job.cancel()


async def test_plan_is_rechecked_after_waiting_for_the_approval_lock(tmp_path):
    async with _agent(tmp_path, []) as (_, controls, client, _):
        await controls._approval_lock.acquire()
        waiting = asyncio.create_task(controls.authorize("print('not run')"))
        try:
            await asyncio.sleep(0)
            assert not waiting.done()
            await controls.configure(permission_mode="plan")
        finally:
            controls._approval_lock.release()
        assert "Plan mode" in await waiting
        assert not client.requests


@asynccontextmanager
async def _adapter(responses, *, model_factory=None, sandbox=False):
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    adapter = CodingACPAdapter(
        lambda: llm, model_factory=model_factory, model="alpha", sandbox=sandbox
    )
    client = _Client()
    adapter.on_connect(client)
    await adapter.initialize(1, aster={"runtimeControls": True})
    try:
        async with asyncio.timeout(120 if sandbox else 20):
            yield adapter, client, llm
    finally:
        await adapter.close()


async def _configure(adapter, session_id, **changes):
    return await adapter.ext_method("aster/session/configure", {"sessionId": session_id, **changes})


@pytest.mark.parametrize("permission_mode", ["auto", "manual", "accept_edits", "plan"])
async def test_rpc_permission_change_applies_to_the_next_cell_without_stopping_a_prompt(
    tmp_path, monkeypatch, permission_mode
):
    code = "await self.shell.write_file('live.txt', 'next cell')"
    async with _adapter(_turn(code)) as (adapter, client, llm):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        await _configure(adapter, session_id, permissionMode="auto")
        entered, release = asyncio.Event(), asyncio.Event()
        original = llm.acall

        async def acall(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(llm, "acall", acall)
        turn = asyncio.create_task(adapter.prompt(session_id, [text_block("Write the file.")]))
        try:
            await entered.wait()
            runtime = await adapter._get_runtime(session_id)
            async with runtime.access() as session:
                active = session.dispatcher._active_task
            state = await _configure(adapter, session_id, permissionMode=permission_mode)
            assert state["permissionMode"] == permission_mode
            assert runtime.busy and not turn.done()
            assert session.dispatcher._active_task is active
            instruction = {
                "auto": "allowed",
                "manual": "requires approval",
                "accept_edits": "Accept edits",
                "plan": "Plan mode",
            }[permission_mode]
            assert instruction in session.controls.instructions()
        finally:
            release.set()
            response = await turn
        assert response.stop_reason == "end_turn"
        assert (tmp_path / "live.txt").exists() == (permission_mode in {"auto", "accept_edits"})
        assert len(client.requests) == int(permission_mode == "manual")


@pytest.mark.parametrize(
    "extra",
    [{"model": "beta"}, {"reasoningLevel": None}, {"mode": "code"}, {"unexpected": True}],
)
async def test_mixed_permission_configuration_does_not_bypass_the_foreground_lock(tmp_path, extra):
    async with _adapter([]) as (adapter, _, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        runtime = await adapter._get_runtime(session_id)
        async with runtime.turn() as session:
            before = session.controls.status()
            with pytest.raises(RequestError):
                await _configure(adapter, session_id, permissionMode="auto", **extra)
            assert session.controls.status() == before


@pytest.mark.parametrize("permission_mode", [None, "", "unknown", [], {}, True])
async def test_invalid_live_permission_value_preserves_the_current_mode(tmp_path, permission_mode):
    async with _adapter([]) as (adapter, _, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        runtime = await adapter._get_runtime(session_id)
        async with runtime.turn() as session:
            before = session.controls.status()
            with pytest.raises(RequestError) as error:
                await _configure(adapter, session_id, permissionMode=permission_mode)
            assert error.value.code == -32602
            assert session.controls.status() == before


async def test_unknown_permission_configuration_fields_are_rejected_while_idle(tmp_path):
    async with _adapter([]) as (adapter, _, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        before = await adapter.ext_method("aster/session/runtime", {"sessionId": session_id})
        with pytest.raises(RequestError) as error:
            await _configure(adapter, session_id, permissionMode="auto", unexpected=True)
        assert error.value.code == -32602
        assert (
            await adapter.ext_method("aster/session/runtime", {"sessionId": session_id}) == before
        )


@pytest.mark.parametrize("fails", [False, True])
async def test_live_permissions_wait_for_an_inflight_model_configuration(tmp_path, fails):
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedModel(FakeLLMClient):
        async def refresh_model_info(self):
            entered.set()
            await release.wait()
            if fails:
                raise ValueError("Model preparation failed")
            return await super().refresh_model_info()

    async with _adapter([], model_factory=lambda _: DelayedModel()) as (adapter, _, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        configuration = asyncio.create_task(
            _configure(adapter, session_id, model="beta", permissionMode="auto")
        )
        try:
            await entered.wait()
            with pytest.raises(RequestError) as error:
                await _configure(adapter, session_id, permissionMode="plan")
            assert "settings" in str(error.value.data).lower()
        finally:
            release.set()
            if fails:
                with pytest.raises(RequestError):
                    await configuration
            else:
                assert (await configuration)["permissionMode"] == "auto"
        assert (await _configure(adapter, session_id, permissionMode="manual"))[
            "permissionMode"
        ] == "manual"


async def test_rpc_permission_change_does_not_approve_a_pending_request(tmp_path):
    async with _adapter(_turn("print('approval required')")) as (adapter, client, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        client.release = asyncio.Event()
        turn = asyncio.create_task(adapter.prompt(session_id, [text_block("Request approval.")]))
        try:
            await client.entered.wait()
            with pytest.raises(RequestError) as error:
                await _configure(adapter, session_id, permissionMode="auto")
            assert "approval" in str(error.value.data).lower()
            assert not turn.done()
        finally:
            client.release.set()
            await turn
        state = await adapter.ext_method("aster/session/runtime", {"sessionId": session_id})
        assert state["permissionMode"] == "manual"


async def test_rpc_permission_mode_enables_file_edits_through_the_normal_prompt(tmp_path):
    code = "await self.shell.write_file('rpc.txt', 'configured over ACP')"
    async with _adapter(_turn(code)) as (adapter, client, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        state = await _configure(adapter, session_id, permissionMode="accept_edits")
        assert (state["permissionMode"], state["mode"], state["approval"]) == (
            "accept_edits",
            "code",
            "accept_edits",
        )

        response = await adapter.prompt(session_id, [text_block("Create the file.")])

        assert response.stop_reason == "end_turn"
        assert (tmp_path / "rpc.txt").read_text() == "configured over ACP"
        assert not client.requests


@pytest.mark.parametrize(
    "changes",
    [
        {"permissionMode": "unknown"},
        {"permissionMode": None},
        {"permissionMode": "auto", "mode": "code"},
        {"permissionMode": "manual", "approval": "ask"},
    ],
)
async def test_invalid_rpc_permission_changes_are_atomic(tmp_path, changes):
    created = []

    def model_factory(name):
        created.append(name)
        return FakeLLMClient()

    async with _adapter([], model_factory=model_factory) as (adapter, _, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        before = await _configure(adapter, session_id, permissionMode="accept_edits")

        with pytest.raises(RequestError) as error:
            await _configure(adapter, session_id, model="beta", **changes)

        assert error.value.code == -32602
        assert (
            await adapter.ext_method("aster/session/runtime", {"sessionId": session_id}) == before
        )
        assert not created


@pytest.mark.parametrize("previous_mode", ["auto", "accept_edits", "plan"])
async def test_reconnecting_a_saved_session_starts_in_manual_mode(tmp_path, previous_mode):
    async with _adapter(_turn()) as (adapter, _, _):
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        await _configure(adapter, session_id, permissionMode=previous_mode)
        await adapter.prompt(session_id, [text_block("Save this conversation.")])
        await adapter.close_session(session_id)

    code = "await self.shell.write_file('reconnected.txt', 'must require fresh approval')"
    async with _adapter(_turn(code)) as (adapter, client, _):
        await adapter.load_session(str(tmp_path), session_id)
        state = await adapter.ext_method("aster/session/runtime", {"sessionId": session_id})
        assert (state["permissionMode"], state["mode"], state["approval"]) == (
            "manual",
            "code",
            "ask",
        )

        response = await adapter.prompt(session_id, [text_block("Make the next edit.")])

        assert response.stop_reason == "end_turn"
        assert len(client.requests) == 1
        assert not (tmp_path / "reconnected.txt").exists()


@pytest.mark.sandbox
async def test_accept_edits_preserves_real_sandbox_for_separately_approved_python(tmp_path):
    python = (
        "from pathlib import Path\n"
        "import sys\n"
        "assert sys.platform == 'linux'\n"
        "Path('sandbox.txt').write_text('executed in the Linux guest')"
    )
    codes = ("await self.shell.write_file('host-edit.txt', 'accepted file edit')", python)
    async with _adapter(_turn(*codes), sandbox=True) as (adapter, client, _):
        client.choice = "allow_once"
        session_id = (await adapter.new_session(str(tmp_path))).session_id
        state = await _configure(adapter, session_id, permissionMode="accept_edits")
        assert state["executionBackend"] == "sandbox"

        response = await adapter.prompt(session_id, [text_block("Edit, then run approved code.")])

        assert response.stop_reason == "end_turn"
        assert [request.raw_input for request in client.requests] == [{"code": python}]
        assert (tmp_path / "host-edit.txt").read_text() == "accepted file edit"
        assert (tmp_path / "sandbox.txt").read_text() == "executed in the Linux guest"


@pytest.mark.parametrize("operation", ["write", "missing-read"])
async def test_accept_edits_reports_result_and_error_to_code_hooks(tmp_path, operation):
    code = (
        "await self.shell.write_file('hooked.txt', 'hook-visible write')"
        if operation == "write"
        else "await self.shell.read('missing.txt')"
    )
    started = []
    completed = []
    token = object()

    class RecordingHooks(InstrumentationHooks):
        def before_code_execution(self, *, code, execution_id, **kwargs):
            if code == source:
                started.append(execution_id)
                return token

        def after_code_execution(self, *, code, execution_id, context, result, exception, **kwargs):
            if code == source:
                completed.append(
                    {
                        "execution_id": execution_id,
                        "context": context,
                        "result": result,
                        "exception": exception,
                    }
                )

    source = code
    async with _agent(tmp_path, _turn(code)) as (agent, controls, client, _):
        await controls.configure(permission_mode="accept_edits")

        with hooks_scope(RecordingHooks(), compose=False):
            assert isinstance(
                await agent.handle({"user_messages": ["Run the file operation with tracing."]}),
                Done,
            )

        assert not client.requests
        assert len(started) == len(completed) == 1
        observed = completed[0]
        assert observed["execution_id"] == started[0]
        assert observed["context"] is token
        assert isinstance(observed["result"], ExecutionResult), observed
        execution = observed["result"]
        if operation == "write":
            assert (tmp_path / "hooked.txt").read_text() == "hook-visible write"
            assert execution.success
            assert execution.has_return
            assert observed["exception"] is None
        else:
            assert not (tmp_path / "missing.txt").exists()
            assert not execution.success
            assert isinstance(observed["exception"], FileNotFoundError)
            assert observed["exception"] is execution.error
