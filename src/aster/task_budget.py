# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One task's durable allowance, independent of turns, models and chat snapshots.

Provider attempts reserve calls synchronously before dispatch. Tokens are settled
after responses: this is an observed allowance, not a provider token hard cap.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskBudgetError(RuntimeError):
    """A host control result; retrying a model cannot resolve it."""


class TaskBudgetExceeded(TaskBudgetError):
    def __init__(self, reason: str, task_id: str = "") -> None:
        self.reason = reason
        self.task_id = task_id
        super().__init__(
            f"Task budget stopped ({reason}) for task {task_id}. "
            "Use /task-budget to inspect it; /task-budget new explicitly starts a new allowance."
        )


class TaskBudgetStorageError(TaskBudgetError):
    """Accounting could not be saved, so no further work may be dispatched."""


class TaskBudgetLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    calls: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    time_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("time_seconds", mode="before")
    @classmethod
    def _numeric_time(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("time_seconds must be a finite number")
        return value


class _Ledger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    task_id: str
    limits: TaskBudgetLimits
    calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    unknown_usage: int = Field(default=0, ge=0)
    pending_attempts: tuple[str, ...] = ()
    known_cost_usd: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    cost_known: bool = True
    active_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    active_started_at: float | None = Field(default=None, allow_inf_nan=False)
    stopped: str | None = None
    recovery_note: str | None = None


class TaskBudget:
    """Shared owner for one task, with a synchronous, fail-closed save callback.

    A resumed ledger owns its original limits. Passing new settings never changes
    those limits; only creating a new owner starts another allowance.
    """

    __nosnapshot__ = True

    def __init__(
        self,
        limits: TaskBudgetLimits,
        *,
        state: Mapping[str, Any] | None = None,
        save: Callable[[dict[str, Any]], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._lock = threading.RLock()
        self._process_id = os.getpid()
        self._changed = threading.Condition(self._lock)
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []
        self._providers: set[asyncio.Task[Any]] = set()
        self._save = save
        self._monotonic = monotonic
        self._clock = clock
        self._active_count = 0
        self._active_start = 0.0
        self._storage_error: TaskBudgetStorageError | None = None
        if state is None:
            self._state = _Ledger(task_id=uuid.uuid4().hex, limits=limits)
        else:
            restored = dict(state)
            restored["pending_attempts"] = tuple(restored.get("pending_attempts", ()))
            self._state = _Ledger.model_validate(restored)
            changes: dict[str, Any] = {}
            if self._state.active_started_at is not None:
                interrupted_seconds = self._clock() - self._state.active_started_at
                changes.update(
                    active_seconds=self._state.active_seconds + max(0.0, interrupted_seconds),
                    active_started_at=None,
                    recovery_note="Interrupted active interval charged through recovery.",
                )
                if interrupted_seconds < 0:
                    changes.update(
                        stopped=self._state.stopped or "unknown_active_time",
                        recovery_note="The clock moved backward during interrupted work; active time is unknown.",
                    )
            if self._state.pending_attempts:
                changes.update(
                    unknown_usage=self._state.unknown_usage + len(self._state.pending_attempts),
                    pending_attempts=(),
                    cost_known=False,
                    recovery_note="Interrupted provider attempts have unknown usage and cost.",
                )
                if self._state.limits.tokens is not None:
                    changes["stopped"] = (
                        changes.get("stopped") or self._state.stopped or "unknown_usage"
                    )
            self._state = self._state.model_copy(update=changes)
        self._write(self._state)

    @property
    def task_id(self) -> str:
        return self._state.task_id

    @property
    def limits(self) -> TaskBudgetLimits:
        return self._state.limits

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._active_count or self._state.pending_attempts)

    def _write(self, state: _Ledger) -> None:
        self._check_process()
        if self._storage_error is not None:
            raise self._storage_error
        try:
            if self._save is not None:
                self._save(state.model_dump(mode="json"))
        except Exception as exc:
            self._storage_error = TaskBudgetStorageError(
                f"Task {state.task_id} accounting could not be saved; dispatch is stopped."
            )
            raise self._storage_error from exc
        self._state = state

    def _check_process(self) -> None:
        if os.getpid() != self._process_id:
            raise TaskBudgetError(
                "A forked process cannot use the parent's task budget. "
                "Run model calls through the owning host agent."
            )

    def _elapsed(self) -> float:
        return self._state.active_seconds + (
            max(0.0, self._monotonic() - self._active_start) if self._active_count else 0.0
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = self._state.model_dump(mode="json")
            result["active_seconds"] = self._elapsed()
            result["cost_known"] = self._state.cost_known and not self._state.pending_attempts
            result["storage_failed"] = self._storage_error is not None
            return result

    def stop(self, reason: str) -> None:
        with self._lock:
            self._write(self._state.model_copy(update={"stopped": self._state.stopped or reason}))
            self._notify()

    def check(self) -> None:
        with self._lock:
            self._check_process()
            if self._storage_error is not None:
                raise self._storage_error
            state = self._state
            reason = state.stopped
            if reason is None and state.limits.time_seconds is not None:
                if self._elapsed() >= state.limits.time_seconds:
                    reason = "time"
            if reason is None and state.limits.tokens is not None:
                if state.unknown_usage:
                    reason = "unknown_usage"
                elif state.input_tokens + state.output_tokens >= state.limits.tokens:
                    reason = "tokens"
            if reason is None and state.limits.calls is not None:
                if state.calls >= state.limits.calls:
                    reason = "calls"
            if reason is not None:
                if state.stopped is None:
                    self.stop(reason)
                raise TaskBudgetExceeded(reason, state.task_id)

    def remaining_time(self) -> float | None:
        with self._lock:
            limit = self._state.limits.time_seconds
            return None if limit is None else max(0.0, limit - self._elapsed())

    def track_provider(self, task: asyncio.Task[Any]) -> None:
        with self._lock:
            self._providers.add(task)
        task.add_done_callback(self._forget_provider)

    def _forget_provider(self, task: asyncio.Task[Any]) -> None:
        with self._lock:
            self._providers.discard(task)

    async def aclose(self) -> None:
        """Finish local shielded attempts before the host closes their ledger."""
        with self._lock:
            tasks = tuple(self._providers)
        loop = asyncio.get_running_loop()
        if any(task.get_loop() is not loop for task in tasks):
            raise TaskBudgetError(
                "Close provider work on its owning event loop before closing the task."
            )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._storage_error is not None:
            raise self._storage_error

    def try_reserve(self) -> str | None:
        """Persist a dispatched attempt; None asks the caller to wait outside admission."""
        with self._lock:
            self.check()
            if self._state.limits.tokens is not None and self._state.pending_attempts:
                return None
            attempt = uuid.uuid4().hex
            self._write(
                self._state.model_copy(
                    update={
                        "calls": self._state.calls + 1,
                        "pending_attempts": (*self._state.pending_attempts, attempt),
                    }
                )
            )
            return attempt

    def reserve(self) -> str:
        """Synchronous callers may wait in a worker, but never block an asyncio loop."""
        with self._changed:
            while (attempt := self.try_reserve()) is None:
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    self._changed.wait(timeout=self.remaining_time())
                else:
                    raise TaskBudgetError(
                        "A task provider attempt is still settling; use acall() in an asyncio loop."
                    )
            return attempt

    async def wait_settled(self) -> None:
        with self._lock:
            self.check()
            if not self._state.pending_attempts:
                return
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            item = (loop, future)
            self._waiters.append(item)
        try:
            await future
        finally:
            with self._lock:
                if item in self._waiters:
                    self._waiters.remove(item)

    def _notify(self) -> None:
        self._changed.notify_all()
        for loop, future in self._waiters:
            if not loop.is_closed():
                loop.call_soon_threadsafe(_wake, future)
        self._waiters.clear()

    def abandon_before_dispatch(self, attempt: str) -> None:
        """Refund only work proven never dispatched (e.g. create_task failed)."""
        with self._lock:
            if attempt in self._state.pending_attempts:
                self._write(
                    self._state.model_copy(
                        update={
                            "calls": self._state.calls - 1,
                            "pending_attempts": tuple(
                                item for item in self._state.pending_attempts if item != attempt
                            ),
                        }
                    )
                )
                self._notify()

    def settle(self, attempt: str, usage: Any) -> None:
        """Settle once, retaining explicit unknowns and never adding token subsets twice."""
        with self._lock:
            if attempt not in self._state.pending_attempts:
                return
            known = usage is not None and all(
                getattr(usage, f"{name}_reported", False) is True
                and type(getattr(usage, name, None)) is int
                and getattr(usage, name) >= 0
                for name in ("input_tokens", "output_tokens")
            )
            cost = getattr(usage, "cost_usd", None)
            cost_known = (
                getattr(usage, "cost_known", False) is True
                and type(cost) in (int, float)
                and math.isfinite(cost)
                and cost >= 0
            )
            changes: dict[str, Any] = {
                "pending_attempts": tuple(
                    item for item in self._state.pending_attempts if item != attempt
                ),
                "input_tokens": self._state.input_tokens + (usage.input_tokens if known else 0),
                "output_tokens": self._state.output_tokens + (usage.output_tokens if known else 0),
                "unknown_usage": self._state.unknown_usage + (0 if known else 1),
                "known_cost_usd": self._state.known_cost_usd + (cost if cost_known else 0.0),
                "cost_known": self._state.cost_known and cost_known,
            }
            if not known and self._state.limits.tokens is not None:
                changes["stopped"] = self._state.stopped or "unknown_usage"
            try:
                self._write(self._state.model_copy(update=changes))
            finally:
                self._notify()

    @contextmanager
    def active(self) -> Iterator[None]:
        """Count the union of active intervals, including concurrent child work."""
        with self._lock:
            if not self._active_count:
                self._write(self._state.model_copy(update={"active_started_at": self._clock()}))
                self._active_start = self._monotonic()
            self._active_count += 1
        try:
            yield
        finally:
            with self._lock:
                elapsed = self._elapsed()
                self._active_count -= 1
                if not self._active_count:
                    self._write(
                        self._state.model_copy(
                            update={"active_seconds": elapsed, "active_started_at": None}
                        )
                    )


def _wake(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


_current: ContextVar[TaskBudget | None] = ContextVar("aster_task_budget", default=None)
_guest_host_only = False


def require_host_model_calls() -> None:
    """Trusted sandbox bootstrap disables unaccounted guest provider dispatch."""
    global _guest_host_only
    _guest_host_only = True


def current_task_budget() -> TaskBudget | None:
    if _guest_host_only:
        raise TaskBudgetError(
            "Task budgets do not support guest-created model clients. "
            "Call the owning host agent through self.* so usage can be accounted for."
        )
    owner = _current.get()
    if owner is not None:
        owner._check_process()
    return owner


@contextmanager
def task_budget_scope(budget: TaskBudget | None) -> Iterator[None]:
    """Bind the same owner for models, children, summaries and retries in this task."""
    token = _current.set(budget)
    try:
        if budget is None:
            yield
        else:
            with budget.active():
                yield
    finally:
        _current.reset(token)


@asynccontextmanager
async def task_budget_activity(budget: TaskBudget | None = None, *, check: bool = True):
    """Run active host work with a deadline that cancels and awaits normal cleanup."""
    owner = budget if budget is not None else current_task_budget()
    if owner is not None and check:
        owner.check()
    with task_budget_scope(owner):
        deadline = asyncio.timeout(owner.remaining_time() if owner is not None else None)
        try:
            async with deadline:
                yield
        except TimeoutError:
            if owner is None or not deadline.expired():
                raise
            owner.stop("time")
            raise TaskBudgetExceeded("time", owner.task_id) from None


def restore_task_budget(storage: Any, limits: TaskBudgetLimits | None) -> TaskBudget | None:
    """Load a session ledger independently of its rewindable conversation snapshot."""
    read = getattr(storage, "load_task_budget", None)
    save = getattr(storage, "save_task_budget", None)
    if not callable(read) or not callable(save):
        if limits is None:
            return None
        raise TaskBudgetStorageError("This storage does not support durable task budgets.")
    state = read()
    if state is None and limits is None:
        return None
    return TaskBudget(limits or TaskBudgetLimits(), state=state, save=save)
