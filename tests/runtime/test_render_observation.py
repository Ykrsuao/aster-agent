# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render observers receive the final request, including an overflow-recovery retry."""

from unittest.mock import MagicMock

import pytest

from aster import Agent
from aster.llm_types import AssistantText
from aster.runtime.actor import _current_llm_var, _current_method_var
from aster.runtime.hooks import hooks_scope
from aster.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage


@pytest.mark.parametrize("middleware", [False, True])
async def test_observation_tracks_the_rebuilt_request_and_reduced_reply_cap(
    middleware: bool,
) -> None:
    class ContextWindowExceededError(Exception):
        pass

    llm = FakeLLMClient()
    agent = Agent(llm=llm)
    attempts = []

    async def handle():
        pass

    async def passthrough(ctx, nxt):
        return await nxt(ctx)

    async def call(messages, tools=None, output_model=None, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ContextWindowExceededError("context window exceeded")
        return LLMResponse(parts=(AssistantText(text="answer"),), usage=LLMUsage(input_tokens=100))

    if middleware:
        agent.event_manager.intercept("llm_call", passthrough)
    llm.acall = call
    observer = MagicMock()
    lt = _current_llm_var.set(llm)
    mt = _current_method_var.set(handle)
    try:
        with hooks_scope(observer):
            await agent.runtime.generate(max_tokens=8192)
        observed = [call.kwargs for call in observer.on_messages_built.call_args_list]
        assert len(observed) == len(attempts) == 2
        assert [item["context_stats"].reserved_output_tokens for item in observed] == [8192, 4096]
        assert attempts[-1]["max_tokens"] == 4096
        assert observed[-1]["rendered_messages"]
        assert all("tools" in item and "output_model" in item for item in observed)
    finally:
        _current_llm_var.reset(lt)
        _current_method_var.reset(mt)
        await agent.aclose()
