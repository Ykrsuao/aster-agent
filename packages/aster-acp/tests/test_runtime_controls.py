# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution approvals and live model changes exercise actual session state."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import litellm
import pytest
from acp.schema import RequestPermissionResponse
from aster_acp.runtime_controls import RuntimeControls
from aster_cli.coding import CodingAgent

from aster.unifiedllm import (
    AdmissionControl,
    AdmissionControlConfig,
    CompletionClient,
    FakeLLMClient,
    LLMResponse,
    ResponsesClient,
    RetryConfig,
    ToolCall,
)
from aster.unifiedllm.reasoning import ReasoningConfig


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)


class PermissionClient:
    def __init__(self, choices: list[str]) -> None:
        self.choices = list(choices)
        self.requests: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.requests.append({"session": session_id, "tool": tool_call, "options": options})
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        choice = self.choices.pop(0)
        if choice == "error":
            raise ConnectionError("transport closed")
        outcome = (
            {"outcome": "cancelled"}
            if choice == "cancelled"
            else {"outcome": "selected", "optionId": choice}
        )
        return RequestPermissionResponse.model_validate({"outcome": outcome})


def controls_for(agent, choices, **kwargs):
    client = PermissionClient(choices)
    updates = []
    controls = RuntimeControls(
        agent,
        client,
        "s1",
        publish=updates.append,
        model_names=lambda: ["alpha", "beta"],
        **kwargs,
    )
    agent._execution_control = controls
    return controls, client, updates


async def test_allow_once_requests_each_cell_and_session_grant_is_live_only(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, updates = controls_for(agent, ["allow_once", "allow_session"])
    try:
        assert await controls.authorize("print('one')", tool_call_id="one") is None
        assert await controls.authorize("print('two')", tool_call_id="two") is None
        assert await controls.authorize("print('three')") is None
        assert len(client.requests) == 2
        assert controls.status()["approval"] == "session"
        assert client.requests[0]["tool"].raw_input == {"code": "print('one')"}
        assert client.requests[0]["tool"].tool_call_id == "one"
        assert [option.kind for option in client.requests[0]["options"]] == [
            "allow_once",
            "allow_always",
            "reject_once",
        ]
        assert updates[-1].field_meta["aster"]["runtime"]["approvalPending"] is False
        replacement, replacement_client, _ = controls_for(agent, ["reject"])
        assert await replacement.authorize("print('four')") is not None
        assert len(replacement_client.requests) == 1
    finally:
        await agent.close()


async def test_approval_preview_shows_command_and_preserves_complete_python_cell(tmp_path):
    code = (
        "# Inspect the selected workspace.\n"
        'result = await self.shell.run("Get-Location", timeout=20)\n'
        "print(result)"
    )
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert "not run" in (await controls.authorize(code))
        tool = client.requests[0]["tool"]
        preview = tool.field_meta["aster"]["approvalPreview"]
        assert preview["summary"] == "Run shell command"
        assert "Command: Get-Location" in preview["details"]
        assert "Timeout: 20 seconds" in preview["details"]
        assert preview["complete"] is True
        assert tool.title == "Run shell command?"
        assert tool.raw_input == {"code": code}
        assert tool.content[0].content.text == code
        assert tool.field_meta["aster"]["approvalScope"] == "python_execution"
        assert client.requests[0]["options"][1].name == "Allow all code for this session"
        assert controls.approval == "ask"
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "code,summary,details",
    [
        (
            'r = await self.shell.run("cd && ls -a | head -20", timeout=20)\nprint(r)',
            "Run shell command",
            ["Command: cd && ls -a | head -20", "Timeout: 20 seconds"],
        ),
        (
            'await self.shell.run(command="python -", stdin="print(42)")',
            "Run shell command",
            ["Command: python -", "Standard input:\nprint(42)"],
        ),
        (
            'result = await self.shell.read(path="src/app.py", lines=(3, 8))\npprint(result)',
            "Read file",
            ["Read file: src/app.py", "Lines: 3–8"],
        ),
        (
            'await self.shell.write_file("src/app.py", "print(42)\\n")',
            "Write file",
            ["Write file: src/app.py", "Full file content:\nprint(42)\n"],
        ),
        (
            'await self.shell.replace("src/app.py", "before", new="after")',
            "Edit file",
            ["Edit file: src/app.py", "Find:\nbefore", "Replace with:\nafter"],
        ),
        (
            'await self.shell.read("a.py")\nawait self.shell.read("b.py")',
            "Run 2 operations",
            ["Read file: a.py", "Read file: b.py"],
        ),
    ],
)
async def test_approval_preview_represents_literal_operations(tmp_path, code, summary, details):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert await controls.authorize(code) is not None
        preview = client.requests[0]["tool"].field_meta["aster"]["approvalPreview"]
        assert preview == {"summary": summary, "details": details, "complete": True}
        assert client.requests[0]["tool"].raw_input == {"code": code}
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "code",
    [
        "await self.shell.run(command)",
        'await self.shell.run(f"Get-Content {path}")',
        'await self.shell.run("Get-Location", timeout=choose_timeout())',
        'await self.shell.run("Get-Location", **options)',
        'await self.shell.run("Get-Location", unexpected=True)',
        'await self.shell.run("Get-Location")\n__import__("os").system("whoami")',
        'await self.shell.write_file("app.py", generate_content())',
        'await self.shell.replace(match, "new text")',
        'await self.shell.replace("app.py", "old", new=render())',
        'def deferred():\n    self.shell.run("Get-Location")',
        'self.shell = replacement\nawait self.shell.run("Get-Location")',
        'self = await self.shell.read("app.py")',
        'print = await self.shell.read("app.py")',
        'self.shell.run("Get-Location")',
        'result = await self.shell.run("Get-Location")\nprint(result, file=target)',
        'print("Hello!")',
        "await self.shell.run(",
        "pass\n" * 4100,
        f"await self.shell.run({('echo ' + 'x' * 2000)!r})",
        f"await self.shell.write_file('app.py', {('x' * 2000)!r})",
    ],
)
async def test_partial_or_unknown_approval_preview_requires_full_code(tmp_path, code):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert await controls.authorize(code) is not None
        tool = client.requests[0]["tool"]
        preview = tool.field_meta["aster"]["approvalPreview"]
        assert preview["complete"] is False
        assert preview["summary"] == "Run Python code"
        assert tool.raw_input == {"code": code}
        assert controls.approval == "ask"
    finally:
        await agent.close()


async def test_approval_preview_never_executes_code_and_bounds_only_the_preview(tmp_path):
    target = tmp_path / "preview-must-not-execute.txt"
    code = f"from pathlib import Path\nPath({str(target)!r}).write_text('bad')\n" + "\n".join(
        f"await self.shell.run({('x' * 4000 + str(i))!r})" for i in range(10)
    )
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["reject"])
    try:
        assert await controls.authorize(code) is not None
        tool = client.requests[0]["tool"]
        preview = tool.field_meta["aster"]["approvalPreview"]
        assert not target.exists()
        assert preview["complete"] is False
        assert len(preview["details"]) <= 8
        assert sum(map(len, preview["details"])) <= 6000
        assert all(len(detail) <= 1500 for detail in preview["details"])
        assert "full-code review" in preview["details"][-1]
        assert tool.raw_input == {"code": code}
        assert tool.content[0].content.text == code
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "hidden",
    [
        chr(27) + "]",  # an OSC sequence swallows the text after it in a terminal
        chr(27) + "[2K",
        chr(13),  # Python reads a lone carriage return as a line break
        chr(0x9D),
        chr(0x202E),
        chr(8),
        chr(0),
    ],
)
async def test_code_that_cannot_be_shown_exactly_is_refused_without_asking(tmp_path, hidden):
    code = f"print('ok')  # {hidden}import os\nvalue = 1"
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["allow_once"])
    try:
        reason = await controls.authorize(code)
        assert reason and "not run" in reason and "control characters" in reason
        assert client.requests == []
        assert controls.approval == "ask"
    finally:
        await agent.close()


async def test_tabs_and_windows_line_endings_are_still_reviewable(tmp_path):
    code = "if True:" + chr(13) + chr(10) + chr(9) + "print('ok')" + chr(13) + chr(10)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["allow_once"])
    try:
        assert await controls.authorize(code) is None
        assert client.requests[0]["tool"].raw_input == {"code": code}
    finally:
        await agent.close()


@pytest.mark.parametrize("choice", ["reject", "cancelled", "unknown", "error"])
async def test_rejected_or_missing_approval_never_allows_execution(tmp_path, choice):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, _, _ = controls_for(agent, [choice])
    try:
        reason = await controls.authorize("dangerous()")
        assert reason and "not run" in reason
        assert controls.approval == "ask"
    finally:
        await agent.close()


async def test_cancelling_an_approval_unwinds_and_does_not_grant_permission(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, ["allow_session"])
    client.release = asyncio.Event()
    operation = asyncio.create_task(controls.authorize("work()"))
    try:
        await asyncio.wait_for(client.started.wait(), 2)
        assert controls.status()["approvalPending"]
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert not controls.status()["approvalPending"]
        assert controls.approval == "ask"
    finally:
        client.release.set()
        await agent.close()


async def test_plan_mode_disables_code_without_asking_and_revokes_session_grant(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, client, _ = controls_for(agent, [])
    try:
        await controls.configure(approval="session")
        await controls.configure(mode="plan")
        assert "not run" in (await controls.authorize("open('x', 'w')"))
        assert not client.requests
        assert controls.status()["executionEnabled"] is False
        await controls.configure(mode="code")
        assert controls.approval == "ask"
        with pytest.raises(ValueError, match="Code mode"):
            await controls.configure(mode="plan", approval="session")
        assert controls.mode == "code"
    finally:
        await agent.close()


async def test_model_switch_changes_real_agent_and_closes_previous_client(tmp_path):
    class ClosingModel(FakeLLMClient):
        def __init__(self, name):
            super().__init__()
            self.model = name
            self.closed = False

        async def aclose(self):
            self.closed = True
            await super().aclose()

    previous = ClosingModel("alpha")
    candidate = ClosingModel("beta")
    agent = CodingAgent(llm=previous, cwd=tmp_path)
    controls, _, _ = controls_for(agent, [], model_factory=lambda name: candidate)
    try:
        result = await controls.configure(model="beta", mode="plan")
        assert agent.llm is candidate
        assert previous.closed
        assert result["model"] == "beta"
        assert result["mode"] == "plan"
        with pytest.raises(ValueError, match="configured"):
            await controls.configure(model="unknown", mode="code")
        assert agent.llm is candidate
        assert controls.mode == "plan"
    finally:
        await agent.close()


async def test_runtime_change_rejected_while_background_work_is_active(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    controls, _, _ = controls_for(agent, [])
    running = asyncio.Event()

    async def job():
        running.set()
        await asyncio.Event().wait()

    agent.queue_manager.queue("check")
    handle = agent.queue_manager.spawn(job(), channel="check")
    try:
        await asyncio.wait_for(running.wait(), 2)
        with pytest.raises(ValueError, match="background jobs"):
            await controls.configure(mode="plan")
        assert controls.mode == "code"
        await handle.cancel()
        await controls.configure(mode="plan")
        assert controls.mode == "plan"
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "change",
    [{"model": "beta"}, {"mode": "plan"}, {"approval": "session"}, {"reasoning_level": None}],
)
async def test_runtime_change_waits_for_older_job_after_newest_on_same_channel_finishes(
    tmp_path: Path, change: dict[str, str | None]
) -> None:
    closed = asyncio.Event()

    class OriginalModel(FakeLLMClient):
        async def aclose(self) -> None:
            closed.set()
            await super().aclose()

    original = OriginalModel()
    agent = CodingAgent(llm=original, cwd=tmp_path)
    controls, _, _ = controls_for(agent, [], model_factory=lambda name: FakeLLMClient())
    started = asyncio.Event()

    async def long_job() -> None:
        started.set()
        await asyncio.Event().wait()

    async def short_job() -> str:
        return "finished"

    channel = agent.queue_manager.queue("checks")
    first = agent.queue_manager.spawn(long_job(), channel="checks")
    try:
        await asyncio.wait_for(started.wait(), 2)
        newest = agent.queue_manager.spawn(short_job(), channel="checks")
        assert await asyncio.wait_for(channel.get(), 2) == "finished"
        assert newest.state == "done"
        assert first.state == "running"
        before = controls.status()

        with pytest.raises(ValueError, match="background jobs"):
            await controls.configure(**change)
        assert controls.status() == before
        assert agent.llm is original
        assert not closed.is_set()

        await first.cancel()
        configured = await controls.configure(**change)
        assert all(
            configured["reasoningSelection" if key == "reasoning_level" else key] == value
            for key, value in change.items()
        )
    finally:
        await agent.close()


async def test_cancelled_model_cleanup_still_publishes_effective_settings(tmp_path):
    closing = asyncio.Event()

    class PreviousModel(FakeLLMClient):
        async def aclose(self):
            closing.set()
            await asyncio.Event().wait()

    agent = CodingAgent(llm=PreviousModel(), cwd=tmp_path)
    replacement = FakeLLMClient()
    controls, _, updates = controls_for(agent, [], model_factory=lambda name: replacement)
    change = asyncio.create_task(controls.configure(model="beta", mode="plan"))
    try:
        await asyncio.wait_for(closing.wait(), 2)
        assert agent.llm is replacement
        state = updates[-1].field_meta["aster"]["runtime"]
        assert state["model"] == "beta"
        assert state["mode"] == "plan"
        change.cancel()
        with pytest.raises(asyncio.CancelledError):
            await change
        assert controls.status()["model"] == state["model"]
    finally:
        change.cancel()
        await agent.close()


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_reasoning_switch_changes_the_next_http_request_and_automatic_limits(
    tmp_path, monkeypatch, responses, wrapped
):
    bodies = []
    reply_cap = "max_output_tokens" if responses else "max_tokens"
    effort_field = "reasoning" if responses else "reasoning_effort"

    def effort(level):
        return {"effort": level} if responses else level

    async def send(http_client, request, **kwargs):
        bodies.append(json.loads(request.content))
        arguments = json.dumps({"result": {"message": "ok", "explanation": "complete"}})
        payload = (
            {
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {
                        "id": "fc_test",
                        "call_id": "return_test",
                        "type": "function_call",
                        "status": "completed",
                        "name": "return_result",
                        "arguments": arguments,
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            }
            if responses
            else {
                "id": "chat_test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "return_test",
                                    "type": "function",
                                    "function": {"name": "return_result", "arguments": arguments},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        )
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(litellm, "drop_params", False)
    cls = ResponsesClient if responses else CompletionClient
    base = cls(
        "openai/gpt-5",
        api_base="https://gateway.example.com/v1",
        api_key="test",
        context_window=131_072,
        max_tokens=4096,
        reasoning_levels={
            "low": {effort_field: effort("low"), reply_cap: 8192},
            "high": {effort_field: effort("high"), reply_cap: 16_384},
        },
        reasoning_default="low",
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    llm = AdmissionControl(base, AdmissionControlConfig(max_in_flight=1)) if wrapped else base
    agent = CodingAgent(llm=llm, cwd=tmp_path, skills_dirs=[])
    controls, _, updates = controls_for(agent, [])
    try:
        initial = controls.status()
        assert initial["reasoningSelection"] is None
        assert initial["reasoningLevels"] == ["low", "high"]
        assert initial["reasoningDefault"] == "low"
        budgets = [s.config.max_tokens for s in agent._summarizers if s._automatic_context_budget]
        assert budgets
        for level, cap in [("high", 16_384), ("low", 8192), (None, 4096)]:
            configured = await controls.configure(reasoning_level=level)
            assert configured["reasoningSelection"] == level
            assert base.reasoning_level == level
            assert updates[-1].field_meta["aster"]["runtime"] == configured
            assert llm.get_context_limits().reserved_output_tokens == cap
            if level == "high":
                changed = [
                    s.config.max_tokens for s in agent._summarizers if s._automatic_context_budget
                ]
                assert all(new < old for new, old in zip(changed, budgets, strict=True))
                await controls.configure(mode="plan")
                assert base.reasoning_level == "high"
                await controls.configure(mode="code")
            result = await agent.handle({"user_messages": ["hello"]})
            assert result.message == "ok"
            body = bodies[-1]
            assert [
                value
                for key, value in body.items()
                if key in {"max_tokens", "max_completion_tokens", "max_output_tokens"}
            ] == [cap]
            if level is None:
                assert effort_field not in body
            else:
                assert body[effort_field] == effort(level)
        assert len(bodies) == 3
        assert base.config["max_tokens"] == 4096
        assert controls.status()["reasoningLevel"] == "low"
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "levels,expected",
    [
        (None, "unknown"),
        ({}, "not supported"),
        ({"low": {"reasoning_effort": "low"}}, "allowed: low"),
    ],
)
async def test_invalid_reasoning_does_not_change_any_runtime_setting(tmp_path, levels, expected):
    agent = CodingAgent(llm=CompletionClient("openai/test", reasoning_levels=levels), cwd=tmp_path)
    controls, _, updates = controls_for(agent, [])
    try:
        before = controls.status()
        with pytest.raises(ValueError, match=expected):
            await controls.configure(reasoning_level="max", mode="plan")
        assert controls.status() == before
        assert not updates
        await controls.configure(reasoning_level=None)
        assert controls.status()["reasoningSelection"] is None
    finally:
        await agent.close()


async def test_model_and_reasoning_changes_validate_together_before_commit(tmp_path):
    class ClosingModel(FakeLLMClient):
        def __init__(self, name, level):
            super().__init__()
            self.model = name
            self._reasoning_config = ReasoningConfig(levels={level: {"reasoning_effort": level}})
            self.set_reasoning_level(level)
            self.closed = False

        async def aclose(self):
            self.closed = True
            await super().aclose()

    previous = ClosingModel("alpha", "high")
    candidate = ClosingModel("beta", "low")
    agent = CodingAgent(llm=previous, cwd=tmp_path)
    controls, _, updates = controls_for(agent, [], model_factory=lambda name: candidate)
    try:
        before = controls.status()
        with pytest.raises(ValueError, match="allowed: low"):
            await controls.configure(model="beta", mode="plan", reasoning_level="high")
        assert agent.llm is previous
        assert controls.status() == before
        assert candidate.closed and not previous.closed
        assert not updates
    finally:
        await agent.close()


async def test_clearing_reasoning_restores_known_native_request_defaults(tmp_path):
    llm = CompletionClient(
        "openai/test",
        reasoning_effort="medium",
        reasoning_levels={"high": {"reasoning_effort": "high"}},
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    controls, _, _ = controls_for(agent, [])
    try:
        assert controls.status()["reasoningLevel"] == "medium"
        await controls.configure(reasoning_level="high")
        await controls.configure(mode="plan")
        assert controls.status()["reasoningSelection"] == "high"
        reset = await controls.configure(reasoning_level=None)
        assert reset["reasoningLevel"] == "medium"
        assert reset["reasoningSelection"] is None
        assert llm._prepare_call_config({})["reasoning_effort"] == "medium"
    finally:
        await agent.close()


async def test_model_picker_previews_each_alias_without_creating_clients(tmp_path, monkeypatch):
    from aster.unifiedllm import registry

    secret = "private-route-credential"
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(
        registry,
        "MODELS",
        {
            "alpha": {
                "reasoning_levels": {"stale": {"reasoning_effort": "stale"}},
                "reasoning_level": "stale",
            },
            "beta": {
                "api_key": secret,
                "api_base": "https://private.example/v1",
                "reasoning_levels": {
                    "medium": {"reasoning_effort": "medium"},
                    "max": {"reasoning_effort": "max", "max_tokens": 32000},
                },
                "reasoning_default": "medium",
                "reasoning_level": "max",
            },
        },
    )
    created = []

    def factory(name):
        created.append(name)
        raise AssertionError("Inspecting the model picker must not construct a client")

    llm = CompletionClient(
        "alpha",
        reasoning_levels={"low": {"reasoning_effort": "low"}},
        reasoning_level="low",
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    controls, _, _ = controls_for(agent, [], model_factory=factory)
    try:
        status = controls.status()
        models = {model["id"]: model for model in status["models"]}
        assert models["alpha"]["reasoningLevels"] == ["low"]
        assert models["alpha"]["reasoningSelection"] == "low"
        assert models["beta"]["reasoningLevels"] == ["medium", "max"]
        assert models["beta"]["reasoningDefault"] == "medium"
        assert models["beta"]["reasoningSelection"] == "max"
        assert models["beta"]["reasoningLevel"] == "max"
        assert created == []
        serialized = json.dumps(status)
        assert secret not in serialized
        assert "private.example" not in serialized
        assert "reasoning_effort" not in serialized
        assert "max_tokens" not in serialized
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, None),
        ({"reasoning_levels": {}}, []),
        ({"reasoning_default": "high"}, None),
        ({"reasoning_levels": {"high": {"api_key": "private"}}}, None),
    ],
)
def test_model_picker_preserves_unknown_and_unsupported_capabilities(monkeypatch, config, expected):
    from aster_acp.runtime_controls import _configured_reasoning

    from aster.unifiedllm import registry

    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", {"preview": config})
    assert _configured_reasoning("preview")["reasoningLevels"] == expected
    assert _configured_reasoning("missing")["reasoningLevels"] is None


def _response(name, arguments, index):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=f"call-{index}", name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
    )


@pytest.mark.parametrize(
    "mode,choice,exists",
    [("plan", "reject", False), ("code", "reject", False), ("code", "allow_once", True)],
)
async def test_real_codeact_cannot_write_without_execution_approval(tmp_path, mode, choice, exists):
    target = tmp_path / "execution.txt"
    llm = FakeLLMClient(
        [
            _response(
                "execute_python",
                {"code": f"from pathlib import Path\nPath({str(target)!r}).write_text('executed')"},
                1,
            ),
            _response(
                "return_result", {"result": {"message": "Finished.", "explanation": "done"}}, 2
            ),
        ],
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    controls, client, _ = controls_for(agent, [choice])
    try:
        await controls.configure(mode=mode)
        await asyncio.wait_for(agent.handle({"user_messages": ["Perform the requested work"]}), 10)
        assert target.exists() is exists
        assert len(client.requests) == (0 if mode == "plan" else 1)
    finally:
        await agent.close()


async def test_controlled_return_result_does_not_eval_constructor_expressions(tmp_path):
    target = tmp_path / "coercion.txt"
    expression = f"Done(message=str(__import__('pathlib').Path({str(target)!r}).write_text('bad')), explanation='coerced')"
    llm = FakeLLMClient(
        [
            _response("return_result", {"result": expression}, 1),
            _response(
                "return_result", {"result": {"message": "A plan.", "explanation": "planned"}}, 2
            ),
        ],
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    controls, client, _ = controls_for(agent, [])
    try:
        await controls.configure(mode="plan")
        await asyncio.wait_for(agent.handle({"user_messages": ["Plan the work"]}), 10)
        assert not target.exists()
        assert not client.requests
    finally:
        await agent.close()
