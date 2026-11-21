# SPDX-License-Identifier: Apache-2.0
"""Python helper definitions execute once, behind the cell's execution guard."""

import json

import pytest

from aster import Agent, strategy
from aster.config import CodeActConfig
from aster.events import ExecutionResult
from aster.strategies import CodeActStrategy
from aster.strategies.codeact import CodeActSession
from aster.strategies.pure_python import GenerationSession, PurePythonStrategy
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _CellAgent(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=8)))
    async def task(self) -> int: ...


class _PureCellAgent(Agent, llm=FakeLLMClient()):
    @strategy(PurePythonStrategy(max_iterations=8))
    async def task(self) -> int: ...


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, PurePythonStrategy])
async def test_denied_helper_defaults_and_decorators_have_no_side_effects(strategy_type):
    agent = _CellAgent()
    seen = []

    def decorate(fn):
        seen.append("decorator")
        return fn

    async def deny(ctx, nxt):
        ctx.result = ExecutionResult(stdout="", error=PermissionError("denied"))
        return ctx

    agent.event_manager.intercept("execute_python", deny)
    session_type = CodeActSession if strategy_type is CodeActStrategy else GenerationSession
    session = session_type(5, 3, "task", agent.event_manager)
    try:
        result = await strategy_type()._execute_code(
            agent.runtime,
            "@decorate\ndef helper(value=seen.append('default')):\n    return value",
            {"seen": seen, "decorate": decorate},
            session,
            "task",
        )
        assert isinstance(result.error, PermissionError)
        assert seen == []
        assert "helper" not in session.session_locals
    finally:
        await agent.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, PurePythonStrategy])
async def test_invalid_cell_rejected_before_helper_side_effects(strategy_type):
    agent = _CellAgent()
    seen = []

    def decorate(fn):
        seen.append("decorator")
        return fn

    session_type = CodeActSession if strategy_type is CodeActStrategy else GenerationSession
    session = session_type(5, 3, "task", agent.event_manager)
    try:
        result = await strategy_type()._execute_code(
            agent.runtime,
            "@decorate\ndef helper(value=seen.append('default')):\n    return value\n"
            "import subprocess",
            {"seen": seen, "decorate": decorate},
            session,
            "task",
        )
        assert result.error is not None
        assert "subprocess" in str(result.error)
        assert seen == []
    finally:
        await agent.aclose()


@pytest.mark.parametrize("strategy_type", [CodeActStrategy, PurePythonStrategy])
async def test_helpers_use_source_order_and_evaluate_defaults_once(strategy_type):
    agent = _CellAgent()
    seen = []
    session_type = CodeActSession if strategy_type is CodeActStrategy else GenerationSession
    session = session_type(5, 3, "task", agent.event_manager)
    try:
        result = await strategy_type()._execute_code(
            agent.runtime,
            "seen.append('before')\nlimit = 3\n"
            "def helper(value=seen.append('default'), limit=limit):\n    return limit\n"
            "seen.append('after')\nhelper()",
            {"seen": seen},
            session,
            "task",
        )
        assert result.error is None
        assert result.returned_value == 3
        assert seen == ["before", "default", "after"]
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "cells",
    [
        ["x = 1", "def answer():\n    return x", "x = 2", "return answer()"],
        ["x = 1\ndef answer():\n    return x", "x = 2", "return answer()"],
        ["x: int = 1\ndef answer():\n    return x", "x: int = 2", "return answer()"],
        [
            "x = 1\ndef answer():\n    global x\n    x += 1\n    return x",
            "answer()\npass",
            "return answer() - 1",
        ],
    ],
)
@pytest.mark.parametrize("agent_type", [_CellAgent, _PureCellAgent])
async def test_public_generation_helpers_see_current_cell_variables(cells, agent_type):
    llm = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content=code if agent_type is _PureCellAgent else "",
                tool_calls=[
                    ToolCall(
                        id=f"cell-{index}",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ]
                if agent_type is _CellAgent
                else [],
                finish_reason="tool_calls" if agent_type is _CellAgent else "stop",
            )
            for index, code in enumerate(cells)
        ],
        strict_exhaustion=True,
    )
    agent = agent_type(llm=llm)
    try:
        assert await agent.task() == 2
    finally:
        await agent.aclose()
        await llm.aclose()
