# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent task accounting is independent of rewindable conversation state."""

import pytest

from aster import Agent
from aster.llm_types import LLMUsage
from aster.storage import SQLiteStorageManager
from aster.task_budget import TaskBudget, TaskBudgetExceeded, TaskBudgetLimits, restore_task_budget
from aster.unifiedllm import FakeLLMClient


class BudgetAgent(Agent, llm=FakeLLMClient()):
    note: str = "before"


def _spend(owner):
    owner.settle(owner.reserve(), LLMUsage(input_tokens=5, output_tokens=2))


def test_reopen_keeps_task_identity_spend_and_original_limits(tmp_path):
    path = tmp_path / "session.db"
    with SQLiteStorageManager(path) as store:
        owner = restore_task_budget(store, TaskBudgetLimits(calls=2))
        _spend(owner)
        task_id = owner.task_id
    with SQLiteStorageManager(path) as store:
        owner = restore_task_budget(store, TaskBudgetLimits(calls=100))
        assert owner.task_id == task_id
        assert owner.snapshot()["calls"] == 1 and owner.limits.calls == 2
        _spend(owner)
        with pytest.raises(TaskBudgetExceeded, match="calls"):
            owner.reserve()
        assert store.load_task_budget()["stopped"] == "calls"


@pytest.mark.parametrize("source_readonly", [False, True])
@pytest.mark.parametrize("checkpoint", ["old", None])
def test_fork_inherits_latest_spend_and_stops_until_explicit_new(
    tmp_path, source_readonly, checkpoint
):
    path = tmp_path / "source.db"
    with SQLiteStorageManager(path) as source, SQLiteStorageManager(tmp_path / "fork.db") as target:
        agent = BudgetAgent(storage=source)
        owner = restore_task_budget(source, TaskBudgetLimits(calls=5))
        _spend(owner)
        source.save_checkpoint("old", agent)
        source.save_current_snapshot(agent)
        _spend(owner)  # spent after the selected chat checkpoint
        if source_readonly:
            SQLiteStorageManager.fork_checkpoint_from_path(path, checkpoint, target, retain=[])
        else:
            source.fork_checkpoint(checkpoint, target, retain=[])
        copied = restore_task_budget(target, TaskBudgetLimits(calls=100))
        assert copied.task_id == owner.task_id
        assert copied.snapshot()["calls"] == 2 and copied.limits.calls == 5
        with pytest.raises(TaskBudgetExceeded, match="forked_task"):
            copied.reserve()
        new = TaskBudget(TaskBudgetLimits(calls=3), save=target.save_task_budget)
        assert new.task_id != owner.task_id
        _spend(new)
        assert target.load_task_budget()["calls"] == 1
        assert source.load_task_budget()["calls"] == 2


def test_restoring_old_snapshot_does_not_restore_old_allowance(tmp_path):
    with SQLiteStorageManager(tmp_path / "session.db") as store:
        agent = BudgetAgent(storage=store)
        agent.note = "before"
        saved = store.save_snapshot(agent)
        owner = restore_task_budget(store, TaskBudgetLimits(calls=4))
        _spend(owner)
        agent.note = "after"
        store.restore_snapshot(saved, agent)
        assert agent.note == "before"
        assert store.load_task_budget()["calls"] == 1


def test_late_old_ledger_update_cannot_replace_current_task(tmp_path):
    with SQLiteStorageManager(tmp_path / "session.db") as store:
        old = restore_task_budget(store, TaskBudgetLimits(calls=4))
        pending = old.reserve()
        new = TaskBudget(TaskBudgetLimits(calls=2), save=store.save_task_budget)
        old.settle(pending, LLMUsage(input_tokens=1, output_tokens=1))
        assert store.load_task_budget()["task_id"] == new.task_id
        assert store.load_task_budget()["calls"] == 0
