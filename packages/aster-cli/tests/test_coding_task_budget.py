# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coding sessions share one allowance across turns, recovery and model changes."""

import asyncio
import json
import shlex
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import litellm
import pytest
import yaml
from aster_cli.coding.agent import CodingAgent
from aster_cli.coding.settings import load_task_budget
from aster_cli.coding.slash_commands import CodingSlashCommandRegistry
from aster_cli.coding.task_budget import task_budget_command

from aster.agentdoc import doc
from aster.interactive import Done, SummarizationConfig
from aster.skill import Skill, slash_command
from aster.slash_dispatch import CoercionError
from aster.storage import SQLiteStorageManager
from aster.task_budget import TaskBudget, TaskBudgetExceeded, TaskBudgetLimits, task_budget_activity
from aster.tools import ShellTools
from aster.unifiedllm import CompletionClient, FakeLLMClient


class _BudgetSlashSkill(Skill):
    @slash_command("probe", output_to_agent=False)
    async def probe(self) -> str:
        await self._agent.llm.acall([{"role": "user", "content": "skill request"}])
        return "probed"

    @slash_command("task-budget", output_to_agent=False)
    def hijack(self) -> str:
        return "hijacked"


def settings(root, monkeypatch, limits):
    path = root / "settings.yaml"
    path.write_text(yaml.safe_dump({"coding": {"task_budget": limits}}), encoding="utf-8")
    monkeypatch.setenv("ASTER_SETTINGS", str(path))
    monkeypatch.setenv("ASTER_USER_DIR", str(root / "user-config"))
    return path


def reply():
    return litellm.ModelResponse(
        model="test-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "done",
                            "type": "function",
                            "function": {
                                "name": "return_result",
                                "arguments": json.dumps(
                                    {"result": {"message": "finished", "explanation": "done"}}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    )


def agent_for(root, llm, **kwargs):
    return CodingAgent(
        llm=llm,
        cwd=root,
        libs_dir=root / "libs",
        skills_dirs=[],
        summarization=SummarizationConfig(policy="none"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "value", [[], True, {"calls": -1}, {"tokens": "100"}, {"time_seconds": float("nan")}]
)
def test_invalid_budget_settings_fail_instead_of_disabling_limits(tmp_path, monkeypatch, value):
    settings(tmp_path, monkeypatch, value)
    with pytest.raises(ValueError):
        load_task_budget(tmp_path)


async def test_turns_model_switch_and_restore_keep_original_task_allowance(tmp_path, monkeypatch):
    settings(tmp_path, monkeypatch, {"calls": 2, "tokens": 100})
    stub = AsyncMock(side_effect=lambda **_: reply())
    monkeypatch.setattr(litellm, "acompletion", stub)
    client = CompletionClient("test-model")
    other = CompletionClient("other-model")
    path = tmp_path / "session.db"
    try:
        with SQLiteStorageManager(path) as store:
            agent = agent_for(tmp_path, client, storage=store)
            try:
                owner = agent._task_budget
                task_id = owner.task_id
                assert isinstance(await agent.handle({"user_messages": ["first"]}), Done)
                agent.set_llm(other)
                assert isinstance(await agent.handle({"user_messages": ["follow-up"]}), Done)
                assert agent._task_budget is owner
                assert owner.snapshot()["calls"] == 2
                store.save_current_snapshot(agent)
            finally:
                await agent.close()
        settings(tmp_path, monkeypatch, {"calls": 100})
        with SQLiteStorageManager(path) as store:
            resumed = agent_for(tmp_path, other, storage=store)
            try:
                assert store.restore_current_snapshot(resumed)
                resumed.refresh_host_context()
                assert resumed._task_budget.task_id == task_id
                assert resumed._task_budget.limits.calls == 2
                with pytest.raises(TaskBudgetExceeded, match="calls"):
                    await resumed.handle({"user_messages": ["continue after restore"]})
                assert stub.call_count == 2
                registry = CodingSlashCommandRegistry(resumed)
                status = await registry.invoke("task-budget", "")
                assert not status.output_to_agent and "Calls: 2 / 2" in str(status)
                await registry.invoke("task-budget", "new")
                assert resumed._task_budget.task_id != task_id
                assert resumed._task_budget.limits.calls == 100
                assert isinstance(await resumed.handle({"user_messages": ["new task"]}), Done)
                assert stub.call_count == 3
            finally:
                await resumed.close()
    finally:
        await client.aclose()
        await other.aclose()


async def test_new_task_does_not_clear_other_host_stop_and_is_not_an_agent_tool(
    tmp_path, monkeypatch
):
    settings(tmp_path, monkeypatch, {"calls": 2})
    agent = agent_for(tmp_path, FakeLLMClient())
    try:
        other_stop = object()
        agent._execution_control = other_stop
        old_id = agent._task_budget.task_id
        task_budget_command(agent, "new")
        assert agent._task_budget.task_id != old_id
        assert agent._execution_control is other_stop
        assert "task_budget_command" not in doc(agent)
        assert "_task_budget" not in doc(agent)
        async with task_budget_activity(agent._task_budget):
            with pytest.raises(CoercionError, match="still active"):
                task_budget_command(agent, "new")
    finally:
        agent._execution_control = None
        await agent.close()


async def test_python_skill_commands_share_the_budget_and_cannot_replace_its_host_command(
    tmp_path, monkeypatch
):
    settings(tmp_path, monkeypatch, {"calls": 1})
    stub = AsyncMock(side_effect=lambda **_: reply())
    monkeypatch.setattr(litellm, "acompletion", stub)
    agent = agent_for(tmp_path, CompletionClient("test-model"))
    agent.skills.register("test.budget", _BudgetSlashSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        assert (await registry.invoke("probe", "")).text == "probed"
        assert agent._task_budget.snapshot()["calls"] == 1
        with pytest.raises(TaskBudgetExceeded, match="calls"):
            await registry.invoke("probe", "")
        assert stub.call_count == 1
        assert "Calls: 1 / 1" in (await registry.invoke("task-budget", "")).text
        await registry.invoke("task-budget", "new")
        assert (await registry.invoke("probe", "")).text == "probed"
        assert stub.call_count == 2
    finally:
        registry.close()
        await agent.close()


async def test_host_deadline_cleans_up_a_real_long_shell_command(tmp_path):
    shell = ShellTools(cwd=str(tmp_path))
    started, finished = tmp_path / "started", tmp_path / "finished"
    code = (
        "from pathlib import Path; import time; "
        "Path('started').write_text('yes'); time.sleep(2); Path('finished').write_text('late')"
    )
    args = [Path(sys.executable).as_posix(), "-c", code]
    command = (
        "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in args)
        if sys.platform == "win32"
        else shlex.join(args)
    )
    try:
        assert (await shell.run("echo ready")).returncode == 0
        owner = TaskBudget(TaskBudgetLimits(time_seconds=1))
        with pytest.raises(TaskBudgetExceeded, match="time"):
            async with task_budget_activity(owner):
                await shell.run(command, timeout=30)
        assert started.exists(), "the deadline must cover a running tool, not just setup"
        await asyncio.sleep(1.2)
        assert not finished.exists()
        assert (await shell.run("echo usable", timeout=5)).returncode == 0
    finally:
        await shell.close()
