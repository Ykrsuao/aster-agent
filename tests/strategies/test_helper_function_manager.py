# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers execute as ordinary definitions and remain local to their cell session."""

import pytest

from aster import Agent
from aster.unifiedllm import FakeLLMClient


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_helpers_are_plain_session_callables(asynchronous):
    agent = Agent(llm=FakeLLMClient())
    other = Agent(llm=FakeLLMClient())
    prefix = "async " if asynchronous else ""
    try:
        result = await agent.runtime.execute_code(
            f"{prefix}def helper(self, value):\n    return value * 2", wrap_in_function=True
        )
        assert result.error is None
        helper = result.captured_locals["helper"]
        answer = helper(agent, 21)
        assert (await answer if asynchronous else answer) == 42
        assert not hasattr(agent, "helper")
        assert not hasattr(other, "helper")
        assert "helper" not in Agent.__dict__
    finally:
        await agent.aclose()
        await other.aclose()


async def test_non_callable_decorator_result_is_preserved():
    agent = Agent(llm=FakeLLMClient())
    try:
        result = await agent.runtime.execute_code(
            "@decorate\ndef helper(self):\n    pass",
            builtins={"decorate": lambda fn: None},
            wrap_in_function=True,
        )
        assert result.error is None
        assert result.captured_locals["helper"] is None
        assert "helper" not in result.defined_methods
    finally:
        await agent.aclose()
