# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nested agents and compaction share the provider allowance and stop contract."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import litellm
import pytest

from aster import Agent, strategy
from aster.agents import SummarizationAgent, TokenBudgetSummarizer
from aster.config import CodeActConfig
from aster.config.summarizer_config import TokenBudgetConfig
from aster.events import BeforeTurn
from aster.runtime.middleware import LLMCallContext
from aster.strategies.codeact import CodeActStrategy
from aster.task_budget import TaskBudget, TaskBudgetExceeded, TaskBudgetLimits, task_budget_scope
from aster.unifiedllm import CompletionClient, FakeLLMClient


class BudgetChild(Agent):
    @strategy(CodeActStrategy(config=CodeActConfig(prefill=None)))
    async def number(self) -> int:
        """Return an integer."""
        ...


class BudgetParent(BudgetChild):
    async def workflow(self, child: BudgetChild) -> tuple[int, int]:
        """A deterministic workflow retains one owner while choosing different models."""
        return await self.number(), await child.number()


def response(tool=True):
    message = {"role": "assistant", "content": "compacted history"}
    if tool:
        message["tool_calls"] = [
            {
                "id": "number",
                "type": "function",
                "function": {
                    "name": "return_result",
                    "arguments": json.dumps({"result": 7}),
                },
            }
        ]
    return litellm.ModelResponse(
        model="test-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool else "stop",
                "message": message,
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    )


async def test_parent_other_model_child_and_summary_consume_one_owner(monkeypatch):
    primary, secondary = CompletionClient("test-primary"), CompletionClient("test-secondary")
    stub = AsyncMock(side_effect=[response(), response(), response(False)])
    monkeypatch.setattr(litellm, "acompletion", stub)
    parent, child = BudgetParent(llm=primary), BudgetChild(llm=secondary)
    summarizer = TokenBudgetSummarizer.install(parent, config=TokenBudgetConfig(max_tokens=1000))
    context = LLMCallContext(
        agent=parent,
        runtime=parent.runtime,
        client=primary,
        messages=[{"role": "user", "content": "Summarize this history."}],
        params={},
    )
    owner = TaskBudget(TaskBudgetLimits(calls=3, tokens=100))
    try:
        with task_budget_scope(owner):
            assert await parent.workflow(child) == (7, 7)
            await summarizer._run_fork(context)
            assert summarizer._pending_summary == "compacted history"
            with pytest.raises(TaskBudgetExceeded, match="calls"):
                await summarizer._run_fork(context)
        assert [entry.kwargs["model"] for entry in stub.call_args_list] == [
            "test-primary",
            "test-secondary",
            "test-primary",
        ]
        assert owner.snapshot()["calls"] == 3
        assert owner.snapshot()["input_tokens"] + owner.snapshot()["output_tokens"] == 42
        assert summarizer._failed_forks == 0
    finally:
        await parent.aclose()
        await child.aclose()
        await primary.aclose()
        await secondary.aclose()


@pytest.mark.parametrize("method", ["answer", "with_setup"])
async def test_nested_budget_stop_escapes_codeact_cells_and_prefill(monkeypatch, method):
    provider = AsyncMock()
    monkeypatch.setattr(litellm, "acompletion", provider)
    nested = CompletionClient("test-child")
    outer = FakeLLMClient.with_tool_call("execute_python", {"code": "await self.probe()"})

    class Caller(Agent):
        async def probe(self):
            """Call a different model from within a generated cell."""
            await nested.acall([{"role": "user", "content": "child request"}])

        @strategy(CodeActStrategy(config=CodeActConfig(prefill=None)))
        async def answer(self) -> int:
            """Call probe before completing."""
            ...

        @strategy(CodeActStrategy(config=CodeActConfig(prefill=None)))
        async def with_setup(self) -> int:
            """Call probe as deterministic setup before generation."""
            await self.probe()
            ...

    agent = Caller(llm=outer)
    try:
        with task_budget_scope(TaskBudget(TaskBudgetLimits(calls=0))):
            with pytest.raises(TaskBudgetExceeded, match="calls"):
                await getattr(agent, method)()
        provider.assert_not_awaited()
        assert len(outer.calls) == (1 if method == "answer" else 0)
    finally:
        await agent.aclose()
        await nested.aclose()


async def test_background_summary_keeps_counting_after_the_foreground_turn():
    started, finish = asyncio.Event(), asyncio.Event()
    now = [0.0]
    owner = TaskBudget(TaskBudgetLimits(), monotonic=lambda: now[0])
    parent = BudgetParent(llm=FakeLLMClient())

    class WaitingSummary(SummarizationAgent):
        async def summarize(self, history_markdown: str, target_chars: int) -> str:
            started.set()
            await finish.wait()
            return "summary"

    summarizer = WaitingSummary.install(parent, config=SimpleNamespace(target_chars=100))
    try:
        with task_budget_scope(owner):
            summarizer._schedule_summarization("1", "2")
            await started.wait()
        now[0] = 5.0
        finish.set()
        await summarizer._pending_task
        assert owner.snapshot()["active_seconds"] == 5.0
        assert not owner.busy
    finally:
        await parent.aclose()


@pytest.mark.parametrize("new_task", [False, True])
async def test_background_summary_budget_error_reaches_the_next_turn(monkeypatch, new_task):
    provider = AsyncMock()
    monkeypatch.setattr(litellm, "acompletion", provider)
    client = CompletionClient("test-summary")
    parent = BudgetParent(llm=client)
    summarizer = SummarizationAgent.install(parent, config=SimpleNamespace(target_chars=100))
    owner = TaskBudget(TaskBudgetLimits(calls=0))
    try:
        with task_budget_scope(owner):
            summarizer._schedule_summarization("1", "2")
            task = summarizer._pending_task
            await asyncio.wait({task})
        error = task.exception()
        assert isinstance(error, TaskBudgetExceeded)
        event = BeforeTurn(
            method_name="number", strategy="codeact", generation_id="next", turn_number=1
        )
        if new_task:
            with task_budget_scope(TaskBudget(TaskBudgetLimits(calls=1))):
                parent.event_manager.add(event)
        else:
            with pytest.raises(TaskBudgetExceeded) as raised:
                parent.event_manager.add(event)
            assert raised.value is error
        assert summarizer._pending_task is None
        assert summarizer._pending_summary is None
        provider.assert_not_awaited()
    finally:
        await parent.aclose()
        await client.aclose()
