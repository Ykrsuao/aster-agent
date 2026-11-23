# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP exposes budget stops and host commands without spending another model call."""

import json
from unittest.mock import AsyncMock

import litellm
from acp import PROTOCOL_VERSION, text_block
from acp.schema import AgentMessageChunk, AvailableCommandsUpdate, RequestPermissionResponse
from aster_acp.server import CodingACPAdapter

from aster.unifiedllm import CompletionClient


class Client:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )


def response():
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
                            "id": "finished",
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


async def test_acp_budget_stop_is_durable_and_explicit_new_reenables_dispatch(
    tmp_path, monkeypatch
):
    settings = tmp_path / "settings.yaml"
    settings.write_text("coding:\n  task_budget:\n    calls: 1\n", encoding="utf-8")
    monkeypatch.setenv("ASTER_SETTINGS", str(settings))
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    stub = AsyncMock(side_effect=lambda **_: response())
    monkeypatch.setattr(litellm, "acompletion", stub)
    client = Client()
    adapter = CodingACPAdapter(lambda: CompletionClient("test-model"))
    adapter.on_connect(client)
    try:
        await adapter.initialize(PROTOCOL_VERSION)
        created = await adapter.new_session(str(tmp_path))
        first = await adapter.prompt(created.session_id, [text_block("first task")])
        assert first.stop_reason == "end_turn"
        runtime = await adapter._get_runtime(created.session_id)
        owner = runtime.value.agent._task_budget
        assert owner.snapshot()["calls"] == 1
        task_id = owner.task_id
        stopped = await adapter.prompt(created.session_id, [text_block("continue")])
        assert stopped.stop_reason == "end_turn" and stub.call_count == 1
        assert runtime.value.handle.storage.load_task_budget()["stopped"] == "calls"
        assert any(
            isinstance(update, AgentMessageChunk) and "Task budget stopped" in update.content.text
            for update in client.updates
        )
        assert any(
            isinstance(update, AvailableCommandsUpdate)
            and "task-budget" in [command.name for command in update.available_commands]
            for update in client.updates
        )
        await adapter.prompt(created.session_id, [text_block("/task-budget")])
        assert stub.call_count == 1
        assert runtime.value.agent._task_budget.task_id == task_id
        await adapter.prompt(created.session_id, [text_block("/task-budget new")])
        assert runtime.value.agent._task_budget.task_id != task_id
        assert runtime.value.agent._task_budget.snapshot()["calls"] == 0
        await adapter.prompt(created.session_id, [text_block("new task")])
        assert stub.call_count == 2
    finally:
        await adapter.close()
