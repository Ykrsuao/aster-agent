# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host budget stops cross the sandbox broker without becoming cell feedback."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from aster import Agent
from aster.agentdoc import visibility
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.task_budget import (
    TaskBudget,
    TaskBudgetError,
    TaskBudgetExceeded,
    TaskBudgetLimits,
    TaskBudgetStorageError,
    current_task_budget,
    task_budget_scope,
)
from aster.unifiedllm import FakeLLMClient


class _BudgetAgent(Agent):
    def reject(self) -> None:
        assert current_task_budget() is self._owner
        raise self._budget_error

    async def reject_async(self) -> None:
        self.reject()


@pytest.fixture(params=["configuration", "exhausted", "storage"])
def budget_error(request):
    if request.param == "exhausted":
        return TaskBudgetExceeded("calls", "offline-task")
    if request.param == "storage":
        return TaskBudgetStorageError("offline ledger write failed")
    return TaskBudgetError("offline host-only provider restriction")


@pytest_asyncio.fixture
async def executor(tmp_path):
    agent = _BudgetAgent(llm=FakeLLMClient())
    agent._owner = TaskBudget(TaskBudgetLimits(calls=1))
    result = SandboxedExecutor(
        agent,
        SandboxConfig(workspace=str(tmp_path)),
        cell_timeout=1,
    )
    result._guest_allow = []
    try:
        yield result
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["reject", "reject_async"])
async def test_host_dispatch_preserves_budget_error_identity(executor, budget_error, method):
    executor._agent._budget_error = budget_error
    with task_budget_scope(executor._agent._owner):
        with pytest.raises(TaskBudgetError) as raised:
            await executor._dispatch_tool_call(
                {"kind": "call", "path": [method], "args": (), "kwargs": {}}
            )
    assert raised.value is budget_error


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["reject", "reject_async"])
async def test_run_cell_waits_for_disposal_then_propagates_budget_error(
    executor, budget_error, method, monkeypatch
):
    executor._agent._budget_error = budget_error
    executor._vm = object()
    executor._messages = asyncio.Queue()
    executor._messages.put_nowait(
        {
            "type": "tool_call",
            "tool_call_id": 1,
            "kind": "call",
            "path": [method],
            "payload": executor._codec.dumps(((), {})),
        }
    )
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def dispose(*, after_cancel=False):
        cleanup_started.set()
        await allow_cleanup.wait()
        executor._vm = None

    cleanup = AsyncMock(side_effect=dispose)
    send = AsyncMock()
    recover = AsyncMock()
    feedback = Mock(wraps=executor._synth_error)
    monkeypatch.setattr(executor, "_dispose", cleanup)
    monkeypatch.setattr(executor, "_send", send)
    monkeypatch.setattr(executor, "_recover", recover)
    monkeypatch.setattr(executor, "_synth_error", feedback)

    code = f"self.{method}()"
    if method == "reject_async":
        code = f"await {code}"
    with task_budget_scope(executor._agent._owner):
        cell = asyncio.create_task(executor.run_cell(code))
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=5)
            cleanup.assert_awaited_once_with(after_cancel=True)
            assert not cell.done()
            assert executor._vm is not None
            allow_cleanup.set()
            with pytest.raises(TaskBudgetError) as raised:
                await asyncio.wait_for(cell, timeout=5)
            assert raised.value is budget_error
        finally:
            allow_cleanup.set()
            if not cell.done():
                cell.cancel()
            await asyncio.gather(cell, return_exceptions=True)

    assert executor._vm is None
    assert executor._broker_task is None
    send.assert_awaited_once_with({"op": "run", "id": 1, "code": code, "execution_count": 1})
    recover.assert_not_awaited()
    feedback.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_bootstrap_declares_when_provider_calls_must_stay_on_host(
    executor, enabled, monkeypatch
):
    monkeypatch.setattr(visibility, "filter_mro_module_globals", lambda _: {})
    with task_budget_scope(executor._agent._owner if enabled else None):
        bootstrap = executor._bootstrap()
    assert bootstrap["task_budget_host_only"] is enabled
