# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background jobs share task time and close cleanly when their budget ends."""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest
import pytest_asyncio

from aster.runtime.channels import JobError, QueueManager, StreamEnd
from aster.task_budget import (
    TaskBudget,
    TaskBudgetLimits,
    current_task_budget,
    task_budget_scope,
)


class _EventManager:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(event)


@pytest_asyncio.fixture
async def manager():
    events = _EventManager()
    queues = QueueManager(event_manager=events)
    queues.queue("work")
    try:
        yield queues, events
    finally:
        await queues.shutdown()


@pytest.mark.asyncio
async def test_background_jobs_accumulate_shared_active_time_without_counting_idle(manager):
    queues, _ = manager
    clock = [0.0]
    owner = TaskBudget(TaskBudgetLimits(time_seconds=100), monotonic=lambda: clock[0])
    started = [asyncio.Event() for _ in range(3)]
    release = [asyncio.Event() for _ in range(3)]

    async def work(index):
        assert current_task_budget() is owner
        started[index].set()
        await release[index].wait()
        return index

    with task_budget_scope(owner):
        first = queues.spawn(work(0), channel="work")
        second = queues.spawn(work(1), channel="work")
        await asyncio.wait_for(asyncio.gather(started[0].wait(), started[1].wait()), timeout=2)

    clock[0] = 2.0
    release[0].set()
    await asyncio.wait_for(first._task, timeout=2)
    assert first.state == "done"
    assert owner.busy is True
    assert owner.snapshot()["active_seconds"] == pytest.approx(2)

    clock[0] = 5.0
    release[1].set()
    await asyncio.wait_for(second._task, timeout=2)
    assert second.state == "done"
    assert owner.busy is False
    assert owner.snapshot()["active_seconds"] == pytest.approx(5)

    clock[0] = 15.0
    assert owner.snapshot()["active_seconds"] == pytest.approx(5)
    with task_budget_scope(owner):
        third = queues.spawn(work(2), channel="work")
        await asyncio.wait_for(started[2].wait(), timeout=2)
    clock[0] = 18.0
    release[2].set()
    await asyncio.wait_for(third._task, timeout=2)

    assert third.state == "done"
    assert owner.busy is False
    assert owner.snapshot()["active_seconds"] == pytest.approx(8)
    assert owner.snapshot()["active_started_at"] is None
    assert owner.snapshot()["calls"] == 0


@pytest.mark.asyncio
async def test_background_deadline_waits_for_job_cleanup_before_reporting_failure(manager):
    queues, events = manager
    owner = TaskBudget(TaskBudgetLimits(time_seconds=0.1))
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleaned = asyncio.Event()

    async def work():
        assert current_task_budget() is owner
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleaned.set()

    with task_budget_scope(owner):
        handle = queues.spawn(work(), channel="work")
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert not handle._task.done()
        assert handle.state == "running"
        assert owner.busy is True
        assert not events.events
        allow_cleanup.set()
        await asyncio.wait_for(handle._task, timeout=2)
    finally:
        allow_cleanup.set()

    assert cleaned.is_set()
    assert handle.state == "failed"
    assert owner.busy is False
    assert owner.snapshot()["stopped"] == "time"
    # asyncio may run a deadline up to one monotonic clock tick early.
    clock_resolution = time.get_clock_info("monotonic").resolution
    assert owner.snapshot()["active_seconds"] >= 0.1 - clock_resolution
    assert owner.snapshot()["active_started_at"] is None
    errors = [event for event in events.events if isinstance(event, JobError)]
    assert len(errors) == 1
    assert errors[0].error_type == "TaskBudgetExceeded"
    assert sum(isinstance(event, StreamEnd) for event in events.events) == 1
    channel = queues.get_channel("work")
    assert await asyncio.wait_for(channel.reader.get(), timeout=2) is errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits",
    [TaskBudgetLimits(calls=0), TaskBudgetLimits(time_seconds=0)],
    ids=["calls", "time"],
)
async def test_rejection_before_job_start_closes_the_unawaited_coroutine(manager, limits):
    queues, events = manager
    owner = TaskBudget(limits)
    entered = False

    async def work():
        nonlocal entered
        entered = True
        return "must not run"

    coroutine = work()
    with task_budget_scope(owner):
        handle = queues.spawn(coroutine, channel="work")
    await asyncio.wait_for(handle._task, timeout=2)

    assert not entered
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    assert handle.state == "failed"
    assert owner.busy is False
    assert owner.snapshot()["calls"] == owner.snapshot()["unknown_usage"] == 0
    errors = [event for event in events.events if isinstance(event, JobError)]
    assert len(errors) == 1
    assert errors[0].error_type == "TaskBudgetExceeded"
    assert sum(isinstance(event, StreamEnd) for event in events.events) == 1
