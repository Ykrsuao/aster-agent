# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task accounting survives interruption and bounds real asynchronous host work."""

import asyncio
import pickle
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aster.llm_types import LLMUsage
from aster.task_budget import (
    TaskBudget,
    TaskBudgetError,
    TaskBudgetExceeded,
    TaskBudgetLimits,
    TaskBudgetStorageError,
    current_task_budget,
    task_budget_activity,
    task_budget_scope,
)
from aster.unifiedllm import FakeLLMClient
from aster.unifiedllm.broker_admission import BrokerAdmissionConfig
from aster.unifiedllm.unifiedllm import _run_async_provider_call, _run_sync_provider_call


def _usage():
    return LLMUsage(
        input_tokens=10,
        output_tokens=4,
        cached_input_tokens=8,
        cache_write_input_tokens=2,
        reasoning_tokens=3,
        cost_usd=0,
    )


@pytest.mark.parametrize(
    "limits",
    [
        {"calls": True},
        {"calls": -1},
        {"calls": 1.5},
        {"tokens": "100"},
        {"tokens": False},
        {"time_seconds": True},
        {"time_seconds": -1},
        {"time_seconds": float("nan")},
        {"time_seconds": float("inf")},
        {"unknown": 1},
    ],
)
def test_limit_declarations_reject_ambiguous_or_unbounded_values(limits):
    with pytest.raises(ValidationError):
        TaskBudgetLimits.model_validate(limits)


def test_tokens_count_input_and_output_once_and_preserve_unknown_cost():
    owner = TaskBudget(TaskBudgetLimits(tokens=100))
    attempt = owner.reserve()
    owner.settle(attempt, _usage())
    owner.settle(attempt, _usage())  # repeated finalization must be idempotent
    assert owner.snapshot()["input_tokens"] + owner.snapshot()["output_tokens"] == 14
    assert owner.snapshot()["cost_known"] is True
    owner.settle(owner.reserve(), LLMUsage(input_tokens=1, output_tokens=1))
    assert owner.snapshot()["cost_known"] is False


def test_active_wall_time_is_union_and_normal_user_wait_is_not_charged():
    now = [100.0]
    owner = TaskBudget(TaskBudgetLimits(time_seconds=50), monotonic=lambda: now[0])
    with task_budget_scope(owner):
        now[0] += 2
        with task_budget_scope(owner):
            now[0] += 3
        now[0] += 4
    assert owner.snapshot()["active_seconds"] == 9
    now[0] += 500  # user was away; no task scope is active
    assert owner.snapshot()["active_seconds"] == 9
    assert current_task_budget() is None


def test_recovery_keeps_limits_charges_orphan_time_and_marks_pending_usage_unknown():
    saved = []
    now = [100.0]
    owner = TaskBudget(
        TaskBudgetLimits(calls=10, tokens=100, time_seconds=50),
        save=saved.append,
        monotonic=lambda: now[0],
        clock=lambda: now[0],
    )
    with task_budget_scope(owner):
        owner.reserve()
        crash_state = saved[-1]
    now[0] += 12
    recovered = TaskBudget(TaskBudgetLimits(calls=1000), state=crash_state, clock=lambda: now[0])
    state = recovered.snapshot()
    assert state["task_id"] == owner.task_id
    assert state["limits"]["calls"] == 10
    assert state["active_seconds"] == 12
    assert state["calls"] == state["unknown_usage"] == 1
    assert not state["pending_attempts"] and not state["cost_known"]
    with pytest.raises(TaskBudgetExceeded, match="unknown_usage"):
        recovered.reserve()


def test_storage_failure_prevents_provider_and_stays_fail_closed():
    fail = [False]
    dispatched = []

    def save(state):
        if fail[0]:
            raise OSError("disk full")

    owner = TaskBudget(TaskBudgetLimits(calls=2), save=save)
    with pytest.raises(TaskBudgetStorageError):
        with task_budget_scope(owner):
            fail[0] = True
            _run_sync_provider_call(lambda: dispatched.append(True))
    fail[0] = False
    with pytest.raises(TaskBudgetStorageError):
        owner.reserve()
    assert not dispatched
    assert owner.snapshot()["calls"] == 0


def test_clock_rollback_during_interrupted_work_stops_instead_of_refunding_time():
    saved = []
    owner = TaskBudget(TaskBudgetLimits(time_seconds=50), save=saved.append, clock=lambda: 100.0)
    with task_budget_scope(owner):
        crash_state = saved[-1]
    recovered = TaskBudget(TaskBudgetLimits(), state=crash_state, clock=lambda: 90.0)
    with pytest.raises(TaskBudgetExceeded, match="unknown_active_time"):
        recovered.reserve()
    assert recovered.snapshot()["calls"] == 0
    assert "unknown" in recovered.snapshot()["recovery_note"]


async def test_task_deadline_cancels_long_host_tool_and_awaits_cleanup():
    cleaned = False
    owner = TaskBudget(TaskBudgetLimits(time_seconds=0.03))

    async def tool():
        nonlocal cleaned
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned = True

    with pytest.raises(TaskBudgetExceeded, match="time"):
        async with task_budget_activity(owner):
            await tool()
    assert cleaned
    assert owner.snapshot()["stopped"] == "time"
    assert owner.snapshot()["active_started_at"] is None


async def test_provider_settles_after_its_caller_is_cancelled():
    started, finish = asyncio.Event(), asyncio.Event()
    owner = TaskBudget(TaskBudgetLimits(tokens=100))

    async def provider():
        started.set()
        await finish.wait()
        return SimpleNamespace(usage={"input_tokens": 10, "output_tokens": 4})

    with task_budget_scope(owner):
        caller = asyncio.create_task(_run_async_provider_call(provider, request_timeout=None))
        await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    assert owner.busy
    assert len(owner.snapshot()["pending_attempts"]) == 1
    finish.set()
    for _ in range(30):
        if not owner.busy:
            break
        await asyncio.sleep(0)
    state = owner.snapshot()
    assert not owner.busy and not state["pending_attempts"]
    assert state["input_tokens"] == 10 and state["output_tokens"] == 4
    assert state["unknown_usage"] == 0


async def test_token_limited_concurrent_calls_wait_for_settlement_without_losing_allowance():
    started, finish = asyncio.Event(), asyncio.Event()
    owner = TaskBudget(TaskBudgetLimits(calls=3, tokens=30))
    calls = 0

    async def provider():
        nonlocal calls
        calls += 1
        started.set()
        await finish.wait()
        return SimpleNamespace(usage={"input_tokens": 10, "output_tokens": 4})

    with task_budget_scope(owner):
        first = asyncio.create_task(_run_async_provider_call(provider, request_timeout=None))
        await started.wait()
        second = asyncio.create_task(_run_async_provider_call(provider, request_timeout=None))
        for _ in range(3):
            await asyncio.sleep(0)
        assert calls == 1 and not second.done()
        finish.set()
        await asyncio.gather(first, second)
    assert calls == owner.snapshot()["calls"] == 2
    assert owner.snapshot()["stopped"] is None


async def test_provider_deadline_marks_cancelled_usage_unknown_and_releases_active_time():
    cleaned = asyncio.Event()
    owner = TaskBudget(TaskBudgetLimits(tokens=100, time_seconds=0.03))

    async def provider():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with pytest.raises(TaskBudgetExceeded):
        with task_budget_scope(owner):
            await _run_async_provider_call(provider, request_timeout=None)
    await asyncio.wait_for(cleaned.wait(), 1)
    for _ in range(10):
        if not owner.busy:
            break
        await asyncio.sleep(0)
    assert owner.snapshot()["calls"] == owner.snapshot()["unknown_usage"] == 1
    assert not owner.busy


async def test_normal_close_settles_shielded_provider_before_storage_is_closed():
    started, cleaned = asyncio.Event(), asyncio.Event()
    saved = []
    owner = TaskBudget(TaskBudgetLimits(tokens=100), save=saved.append)

    async def provider():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with task_budget_scope(owner):
        caller = asyncio.create_task(_run_async_provider_call(provider, request_timeout=None))
        await started.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    await owner.aclose()
    assert cleaned.is_set() and not owner.busy
    assert saved[-1]["active_started_at"] is None
    assert saved[-1]["unknown_usage"] == 1
    assert not saved[-1]["pending_attempts"]
    reopened = TaskBudget(TaskBudgetLimits(), state=saved[-1], clock=lambda: 1e12)
    assert reopened.snapshot()["active_seconds"] == saved[-1]["active_seconds"]


def test_existing_clients_and_broker_configs_cannot_be_exported_under_a_budget():
    client = FakeLLMClient()
    config = BrokerAdmissionConfig("127.0.0.1", 1, "test", "group", 1, None, None)
    # The guard applies at serialization, even when construction preceded the task.
    assert pickle.loads(pickle.dumps(config)) == config
    assert isinstance(pickle.loads(pickle.dumps(client)), FakeLLMClient)
    with task_budget_scope(TaskBudget(TaskBudgetLimits(calls=1))):
        for value in (config, client):
            with pytest.raises(TaskBudgetError, match="independent"):
                pickle.dumps(value)


def test_forked_owner_refuses_dispatch_before_storage(monkeypatch):
    import aster.task_budget as module

    owner = TaskBudget(TaskBudgetLimits(calls=2))
    monkeypatch.setattr(module.os, "getpid", lambda: owner._process_id + 1)
    with pytest.raises(TaskBudgetError, match="forked"):
        owner.reserve()
    assert owner.snapshot()["calls"] == 0


def test_guest_bootstrap_guard_blocks_unowned_provider_entry(monkeypatch):
    import aster.task_budget as module

    monkeypatch.setattr(module, "_guest_host_only", True)
    calls = []
    with pytest.raises(TaskBudgetError, match=r"self\.\*"):
        _run_sync_provider_call(lambda: calls.append(1))
    assert not calls
