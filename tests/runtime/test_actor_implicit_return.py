# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host's unwrapped path executes every statement without return capture."""

from __future__ import annotations

import pytest

from aster.agent import Agent
from aster.runtime.actor import ActorRuntime
from aster.unifiedllm import FakeLLMClient


class _RecorderAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[str] = []

    def record(self, tag: str) -> str:
        self.calls.append(tag)
        return f"recorded:{tag}"


@pytest.mark.asyncio
async def test_unwrapped_path_regression():
    """The default (unwrapped) path never rewrote and must stay correct."""
    agent = _RecorderAgent()
    runtime = ActorRuntime(agent)
    result = await runtime.execute_code("self.record('a'); self.record('b')")
    assert result.error is None
    assert agent.calls == ["a", "b"]
    assert not result.has_return
