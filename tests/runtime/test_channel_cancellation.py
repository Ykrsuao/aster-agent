# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation must reach callers without losing queued items or live jobs."""

import asyncio

import pytest

from aster.runtime.channels import JobHandle, QueueManager
from aster.runtime.event_manager import EventManager

pytestmark = pytest.mark.timeout(10)


async def _slow_to_cancel(started, cleaning, interruptions=()):
    started.set()
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        cleaning.set()
        for interrupted in interruptions:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                interrupted.set()
        await asyncio.Future()
        raise


@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_handle_collects_target_cancellation_or_cleanup_error(cleanup_error):
    started = asyncio.Event()

    async def job():
        started.set()
        try:
            await asyncio.Future()
        finally:
            if cleanup_error:
                raise RuntimeError("cleanup failed")

    target = asyncio.create_task(job())
    handle = JobHandle("job", target)
    await started.wait()
    await handle.cancel()
    assert target.done()
    assert handle.state == "cancelled"
    await handle.cancel()


@pytest.mark.parametrize(
    ("first_stop", "second_stop"),
    [
        ("cancel", "cancel"),
        ("cancel", "shutdown"),
        ("shutdown", "cancel"),
        ("shutdown", "shutdown"),
        ("cancel", "remove"),
        ("cancel", "queue"),
        ("cancel", "event"),
        ("remove", "cancel"),
    ],
)
async def test_repeated_stop_preserves_job_cleanup(first_stop, second_stop, monkeypatch):
    manager = QueueManager(event_manager=EventManager())
    manager.queue("job")
    started, cleaning, release, cleaned, cancelling = (asyncio.Event() for _ in range(5))

    async def job():
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    handle = manager.spawn(job(), channel="job")
    original_cancel = handle.cancel

    async def cancel():
        cancelling.set()
        await original_cancel()

    monkeypatch.setattr(handle, "cancel", cancel)

    def stop(operation):
        if operation == "cancel":
            return asyncio.create_task(handle.cancel())
        if operation == "shutdown":
            return asyncio.create_task(manager.shutdown())
        if operation == "remove":
            manager.remove_channel("job")
        else:
            getattr(manager, operation)("job", replace=True)
        return None

    waiters = []
    try:
        await started.wait()
        first = stop(first_stop)
        if first is not None:
            waiters.append(first)
        await cleaning.wait()
        cancelling.clear()

        second = stop(second_stop)
        if second is not None:
            waiters.append(second)
            await cancelling.wait()

        assert handle._task.cancelling() == 1
        assert all(not waiter.done() for waiter in waiters)
        assert not cleaned.is_set()
        release.set()
        await asyncio.gather(*waiters)
        assert cleaned.is_set()
        assert handle._task.cancelled()
        assert handle.state == "cancelled"
    finally:
        release.set()
        await asyncio.gather(*waiters, handle._task, return_exceptions=True)
        await manager.shutdown()


@pytest.mark.parametrize("cancel_requests", [1, 2])
async def test_handle_propagates_caller_cancellation_during_cleanup(cancel_requests):
    started, cleaning = asyncio.Event(), asyncio.Event()
    interruptions = [asyncio.Event() for _ in range(cancel_requests - 1)]
    target = asyncio.create_task(_slow_to_cancel(started, cleaning, interruptions))
    handle = JobHandle("job", target)
    await started.wait()
    caller = asyncio.create_task(handle.cancel())
    try:
        await cleaning.wait()
        for interrupted in interruptions:
            caller.cancel()
            await interrupted.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert target.cancelled()
    finally:
        target.cancel()
        caller.cancel()
        await asyncio.gather(target, caller, return_exceptions=True)


@pytest.mark.parametrize("cancel_requests", [1, 2])
async def test_race_restores_winner_when_cancelled_during_loser_cleanup(
    cancel_requests, monkeypatch
):
    manager = QueueManager()
    fast, slow = manager.queue("fast"), manager.queue("slow")
    started, cleaning = asyncio.Event(), asyncio.Event()
    interruptions = [asyncio.Event() for _ in range(cancel_requests - 1)]
    drains = []

    async def slow_drain():
        drains.append(asyncio.current_task())
        await _slow_to_cancel(started, cleaning, interruptions)

    monkeypatch.setattr(slow, "_drain_one", slow_drain)
    delivered = []
    monkeypatch.setattr(fast, "_fire_on_get", delivered.append)
    caller = asyncio.create_task(manager.race())
    try:
        await started.wait()
        fast.put("winner")
        await cleaning.wait()
        for interrupted in interruptions:
            caller.cancel()
            await interrupted.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert delivered == []
        assert not fast.is_empty()
        assert await fast.get() == "winner"
        assert fast.is_empty()
        assert delivered == ["winner"]
        assert all(task.done() for task in drains)
    finally:
        for task in drains:
            task.cancel()
        caller.cancel()
        await asyncio.gather(caller, *drains, return_exceptions=True)


async def test_shutdown_in_cancelled_callers_finally_finishes_all_jobs():
    manager = QueueManager()
    handles = []

    async def parked(started):
        started.set()
        await asyncio.Future()

    for name in ("first", "second"):
        manager.queue(name)
        started = asyncio.Event()
        handles.append(manager.spawn(parked(started), channel=name))
        await started.wait()

    ready, cleaned = asyncio.Event(), asyncio.Event()

    async def caller():
        try:
            ready.set()
            await asyncio.Future()
        finally:
            await manager.shutdown()
            cleaned.set()

    task = asyncio.create_task(caller())
    try:
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert all(handle._task.cancelled() for handle in handles)
        assert manager.job_handles() == []
    finally:
        task.cancel()
        for handle in handles:
            handle._task.cancel()
        await asyncio.gather(task, *(handle._task for handle in handles), return_exceptions=True)


async def test_shutdown_requests_all_cancellations_before_waiting_for_cleanup():
    manager = QueueManager()
    release = asyncio.Event()
    handles, cleaning = [], []

    async def job(started, entered):
        started.set()
        try:
            await asyncio.Future()
        finally:
            entered.set()
            await release.wait()

    for name in ("first", "second"):
        manager.queue(name)
        started, entered = asyncio.Event(), asyncio.Event()
        handles.append(manager.spawn(job(started, entered), channel=name))
        cleaning.append(entered)
        await started.wait()

    shutdown = asyncio.create_task(manager.shutdown())
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in cleaning)), timeout=1)
        assert not shutdown.done()
        assert manager.job_handles() == handles
        release.set()
        await shutdown
        assert manager.job_handles() == []
    finally:
        release.set()
        await asyncio.gather(shutdown, return_exceptions=True)
        for handle in handles:
            handle._task.cancel()
        await asyncio.gather(*(handle._task for handle in handles), return_exceptions=True)


async def test_interrupted_shutdown_propagates_cancellation_and_finishes_jobs():
    manager = QueueManager()
    manager.queue("slow")
    manager.queue("other")
    started, cleaning, other_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def other_job():
        other_started.set()
        await asyncio.Future()

    handles = [
        manager.spawn(_slow_to_cancel(started, cleaning), channel="slow"),
        manager.spawn(other_job(), channel="other"),
    ]
    await asyncio.gather(started.wait(), other_started.wait())
    shutdown = asyncio.create_task(manager.shutdown())
    try:
        await cleaning.wait()
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert all(handle._task.cancelled() for handle in handles)
        assert manager.job_handles() == []
        await manager.shutdown()
    finally:
        shutdown.cancel()
        for handle in handles:
            handle._task.cancel()
        await asyncio.gather(
            shutdown, *(handle._task for handle in handles), return_exceptions=True
        )


async def test_shutdown_interrupted_before_cancel_tasks_run_retains_jobs_for_retry():
    manager = QueueManager()
    manager.queue("job")
    started = asyncio.Event()

    async def job():
        started.set()
        await asyncio.Future()

    handle = manager.spawn(job(), channel="job")
    await started.wait()
    shutdown = asyncio.create_task(manager.shutdown())
    # This callback runs after shutdown starts, before its child cancel tasks.
    asyncio.get_running_loop().call_soon(shutdown.cancel)
    try:
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert not handle._task.done()
        assert manager.job_handles() == [handle]
        await manager.shutdown()
        assert handle._task.cancelled()
        assert manager.job_handles() == []
    finally:
        shutdown.cancel()
        handle._task.cancel()
        await asyncio.gather(shutdown, handle._task, return_exceptions=True)


async def test_shutdown_keeps_jobs_spawned_during_cleanup():
    manager = QueueManager()
    manager.queue("job")
    started, cleaning, release, replacement_started = (asyncio.Event() for _ in range(4))

    async def job():
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()

    async def replacement_job():
        replacement_started.set()
        await asyncio.Future()

    original = manager.spawn(job(), channel="job")
    await started.wait()
    shutdown = asyncio.create_task(manager.shutdown())
    replacement = None
    try:
        await cleaning.wait()
        replacement = manager.spawn(replacement_job(), channel="job")
        await replacement_started.wait()
        release.set()
        await shutdown
        assert original._task.done()
        assert not replacement._task.done()
        assert manager.job_handles() == [replacement]
        await manager.shutdown()
        assert replacement._task.cancelled()
        assert manager.job_handles() == []
    finally:
        release.set()
        original._task.cancel()
        if replacement is not None:
            replacement._task.cancel()
        await asyncio.gather(shutdown, original._task, return_exceptions=True)
        await manager.shutdown()
