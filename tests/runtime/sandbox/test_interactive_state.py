# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive state writes must reach the host that owns its checkpoints."""

import sys
import types

import pytest

from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.strategies import CodeActStrategy

pytestmark = pytest.mark.sandbox


@pytest.fixture
def state_owner(monkeypatch):
    module = types.ModuleType("aster_sandbox_state_owner")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(
        "from aster.agentdoc import doc\n"
        "from aster.interactive import AgentVars\n"
        "from aster.runtime.context import ContextApi\n"
        "from aster.runtime.context_manager import ContextManager\n"
        "from aster.storage.persistent_vars import PersistentVars\n"
        "from aster.storage.snapshot_vars import SnapshotVars\n"
        "from aster.tools.todo import Todo, TodoManager\n"
        "class Service:\n"
        "    def __init__(self):\n"
        "        self.vars = SnapshotVars()\n"
        "        self.persistent = PersistentVars(self)\n"
        "        self.context_manager = ContextManager()\n"
        "        self.context = ContextApi(self)\n"
        "        self.todo = TodoManager()\n"
        "        self.data = {'items': [1, 2]}\n"
        "    @property\n"
        "    def v(self):\n"
        "        return AgentVars(self)\n"
        "    def host_data(self):\n"
        "        return self.data\n",
        module.__dict__,
    )
    return module.Service()


async def test_snapshot_vars_writes_reach_host_and_survive_guest_restart(tmp_path, state_owner):
    agent = state_owner
    executor = SandboxedExecutor(agent, SandboxConfig(workspace=str(tmp_path)), cell_timeout=10)
    try:
        result = await executor.run_cell(
            "assert not self.vars\n"
            "vars_ref = self.vars\n"
            "self.vars['kept'] = 7\n"
            "self.vars['stage'] = 'alpha'\n"
            "self.vars.update(total=63)\n"
            "self.vars.setdefault('kept', 99)\n"
            "assert self.vars\n"
        )
        assert result.error is None, result.error
        expected = {"kept": 7, "stage": "alpha", "total": 63}
        assert dict(agent.vars) == expected
        agent.vars["stage"] = "host-update"
        result = await executor.run_cell(
            "assert vars_ref['stage'] == 'host-update'\n"
            "vars_ref['stage'] = 'alpha'\n"
            "self.v.extra = 'from-v'\n"
            "self.v.temporary = 'to-delete'\n"
            "del self.v.temporary\n"
            "self.persistent.temporary = 'to-delete'\n"
            "del self.persistent.temporary\n"
            "self.persistent.set('facade', 'from-persistent')\n"
            "dict(self.vars)",
            execution_count=2,
        )
        assert result.error is None, result.error
        expected.update(extra="from-v", facade="from-persistent")
        assert dict(agent.vars) == expected
        assert result.returned_value == expected
        await executor.aclose()

        executor = SandboxedExecutor(agent, SandboxConfig(workspace=str(tmp_path)), cell_timeout=10)
        result = await executor.run_cell(
            "assert self.vars['stage'] == 'alpha'\n"
            "assert len(self.vars) == 5 and 'total' in self.vars\n"
            "self.vars['stage'] = 'beta'\n"
            "del self.vars['extra']\n"
            "self.persistent.clear()\n"
            "assert not self.vars\n"
        )
        assert result.error is None, result.error
        assert dict(agent.vars) == {}
    finally:
        await executor.aclose()


async def test_stateful_skills_write_to_host_while_returned_data_stays_local(tmp_path, state_owner):
    call = types.SimpleNamespace(return_type=object, kwargs={}, method_name="verify")
    builtins = CodeActStrategy()._build_builtins(types.SimpleNamespace(agent=state_owner), call)
    executor = SandboxedExecutor(
        state_owner,
        SandboxConfig(workspace=str(tmp_path)),
        cell_timeout=10,
        framework_builtins={**builtins, "_call": call},
    )
    try:
        result = await executor.run_cell(
            "assert 'class TodoManager:' in doc(self.todo, inline_depth=0)\n"
            "assert 'class ContextApi:' in doc(self.context, inline_depth=0)\n"
            "assert 'title: str' in doc(self.todo.add, inline_depth=0)\n"
            "assert self.todo\n"
            "assert not self.context\n"
            "task = self.todo.add('Verify state', kept='original')\n"
            "task.title = 'local copy'\n"
            "task.vars['local_only'] = 'not persisted'\n"
            "self.todo.set_var(task, 'checkpoint', 'verified')\n"
            "self.todo.comment(task, 'verified via object')\n"
            "dependent = self.todo.add('Follow-up', deps=[task])\n"
            "assert dependent.deps == [task.id]\n"
            "assert self.todo.get(task).title == 'Verify state'\n"
            "assert self.todo.get_var(task, 'local_only') is None\n"
            "self.context['plan'] = 'host plan'\n"
            "self.context.set('detail', 'host detail')\n"
            "assert self.context\n"
            "local_data = self.host_data()\n"
            "local_data['items'].append(3)\n"
            "(task.id, self.todo.get_var(task, 'checkpoint'), local_data)"
        )
        assert result.error is None, result.error
        task_id, checkpoint, local_data = result.returned_value
        assert checkpoint == state_owner.todo.get_var(task_id, "checkpoint") == "verified"
        assert state_owner.todo.get(task_id).title == "Verify state"
        assert dict(state_owner.todo.get(task_id).vars) == {
            "kept": "original",
            "checkpoint": "verified",
        }
        assert [comment.body for comment in state_owner.todo.comments(task_id)] == [
            "verified via object"
        ]
        assert state_owner.context["plan"] == "host plan"
        assert state_owner.context["detail"] == "host detail"
        assert local_data == {"items": [1, 2, 3]}
        assert state_owner.data == {"items": [1, 2]}
    finally:
        await executor.aclose()
