# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only task-budget command; never registered as an agent tool."""

from typing import Any

from aster.slash_dispatch import CoercionError
from aster.task_budget import TaskBudget, TaskBudgetLimits, TaskBudgetStorageError
from aster_cli.coding.settings import load_task_budget


def task_budget_command(agent: Any, action: str = "") -> str:
    if action not in ("", "new"):
        raise CoercionError("Expected no argument or 'new'.", hint="[new]")
    budget = agent._task_budget
    if action == "new":
        if budget is not None and budget.busy:
            raise CoercionError(
                "Task work is still active or a provider attempt is settling; try again after it stops.",
                hint="[new]",
            )
        save = getattr(agent._storage, "save_task_budget", None)
        if not callable(save):
            raise TaskBudgetStorageError("This storage does not support task budgets.")
        budget = TaskBudget(load_task_budget(agent.cwd) or TaskBudgetLimits(), save=save)
        # Other host stops (including unresolved execution recovery) are separate.
        agent._task_budget = budget
    if budget is None:
        return "No task budget configured. Set coding.task_budget or use /task-budget new."
    state = budget.snapshot()
    limits = state["limits"]
    token_count = state["input_tokens"] + state["output_tokens"]
    unknown = state["unknown_usage"] + len(state["pending_attempts"])
    tokens = f"{token_count} known" + (
        f"; {unknown} attempt(s) unknown or unsettled" if unknown else ""
    )
    cost = f"${state['known_cost_usd']:.6f}" if state["cost_known"] else "unknown"
    stopped = state["stopped"] or ("storage failure" if state["storage_failed"] else "active")
    lines = [
        f"Task: {state['task_id']} ({stopped})",
        f"Calls: {state['calls']} / {limits['calls'] if limits['calls'] is not None else 'unlimited'}",
        f"Tokens (input + output): {tokens} / {limits['tokens'] if limits['tokens'] is not None else 'unlimited'}",
        f"Active time: {state['active_seconds']:.1f}s / {limits['time_seconds'] if limits['time_seconds'] is not None else 'unlimited'}",
        f"Cost: {cost}",
    ]
    if state["recovery_note"]:
        lines.append(state["recovery_note"])
    return "\n".join(lines)
