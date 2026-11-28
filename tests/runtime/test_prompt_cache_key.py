# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default cache routing keys stay bounded at the LLM request boundary."""

import pytest

from aster import Agent, strategy
from aster.strategies import PredictStrategy
from aster.strategies.base import GenerationStrategy
from aster.unifiedllm import FakeLLMClient, LLMResponse


def _response() -> LLMResponse:
    return LLMResponse(
        raw_response=None, content='{"value": "ok"}', tool_calls=[], finish_reason="stop"
    )


async def _passthrough(ctx, nxt):
    return await nxt(ctx)


class _ExplicitCacheKeyStrategy(GenerationStrategy):
    async def execute(self, runtime, call):
        response, _ = await runtime.generate(prompt_cache_key=call.kwargs["cache_key"])
        return response.content


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["document_pipeline_" * 12, "文档处理流程" * 12])
async def test_standalone_default_key_is_bounded_for_long_identities(identity):
    llm = FakeLLMClient(scripted_responses=[_response() for _ in range(3)], strict_exhaustion=True)

    def make_call(suffix):
        async def summarize(text: str) -> str:
            """Summarize the document."""
            ...

        summarize.__qualname__ = f"{identity}.{suffix}"
        return strategy(PredictStrategy(), llm=llm)(summarize)

    call = make_call("summarize")
    assert await call("input") == "ok"
    assert await call("another input") == "ok"

    other_call = make_call("summarize_other")
    assert await other_call("input") == "ok"

    key = llm.calls[0].kwargs["prompt_cache_key"]
    assert 0 < len(key) <= 64
    assert key.isascii()
    assert llm.calls[1].kwargs["prompt_cache_key"] == key
    assert llm.calls[2].kwargs["prompt_cache_key"] != key


@pytest.mark.asyncio
@pytest.mark.parametrize("middleware", [False, True])
async def test_agent_default_key_is_stable_and_distinct_per_instance(middleware):
    llm = FakeLLMClient(scripted_responses=[_response() for _ in range(3)], strict_exhaustion=True)

    class Summarizer(Agent, llm=llm):
        @strategy(PredictStrategy())
        async def summarize(self, text: str) -> str:
            """Summarize the document."""
            ...

    first, second = Summarizer(), Summarizer()
    if middleware:
        first.event_manager.intercept("llm_call", _passthrough)
        second.event_manager.intercept("llm_call", _passthrough)

    assert await first.summarize("input") == "ok"
    assert await first.summarize("another input") == "ok"
    assert await second.summarize("input") == "ok"

    keys = [call.kwargs["prompt_cache_key"] for call in llm.calls]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert all(0 < len(key) <= 64 and key.isascii() for key in keys)


@pytest.mark.asyncio
@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("strategy_name", ["DocumentStrategy" * 12, "文档处理策略" * 12])
async def test_agent_default_key_separates_strategies(middleware, strategy_name):
    llm = FakeLLMClient(scripted_responses=[_response() for _ in range(3)], strict_exhaustion=True)
    other_strategy = type(strategy_name, (PredictStrategy,), {})()

    class Summarizer(Agent, llm=llm):
        @strategy(PredictStrategy())
        async def summarize(self, text: str) -> str:
            """Summarize the document."""
            ...

        @strategy(PredictStrategy())
        async def classify(self, text: str) -> str:
            """Classify the document."""
            ...

        @strategy(other_strategy)
        async def analyze(self, text: str) -> str:
            """Analyze the document."""
            ...

    agent = Summarizer()
    if middleware:
        agent.event_manager.intercept("llm_call", _passthrough)

    assert await agent.summarize("input") == "ok"
    assert await agent.classify("input") == "ok"
    assert await agent.analyze("input") == "ok"

    keys = [call.kwargs["prompt_cache_key"] for call in llm.calls]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert all(0 < len(key) <= 64 and key.isascii() for key in keys)


@pytest.mark.asyncio
async def test_default_key_is_shared_between_request_paths():
    llm = FakeLLMClient(scripted_responses=[_response(), _response()], strict_exhaustion=True)

    class Summarizer(Agent, llm=llm):
        @strategy(PredictStrategy())
        async def summarize(self, text: str) -> str:
            """Summarize the document."""
            ...

    agent = Summarizer()
    assert await agent.summarize("input") == "ok"
    agent.event_manager.intercept("llm_call", _passthrough)
    assert await agent.summarize("input") == "ok"

    assert llm.calls[0].kwargs["prompt_cache_key"] == llm.calls[1].kwargs["prompt_cache_key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("cache_key", ["custom-shard", "custom_" * 20, "自定义分片" * 20, "", None])
async def test_explicit_key_is_preserved(middleware, cache_key):
    llm = FakeLLMClient(scripted_responses=[_response()], strict_exhaustion=True)

    class Summarizer(Agent, llm=llm):
        @strategy(_ExplicitCacheKeyStrategy())
        async def summarize(self, cache_key: str | None) -> str:
            """Summarize the document."""
            ...

    agent = Summarizer()
    if middleware:
        agent.event_manager.intercept("llm_call", _passthrough)

    assert await agent.summarize(cache_key=cache_key) == _response().content
    assert llm.calls[0].kwargs["prompt_cache_key"] == cache_key


@pytest.mark.asyncio
async def test_middleware_cache_key_override_is_preserved():
    llm = FakeLLMClient(scripted_responses=[_response()], strict_exhaustion=True)

    class Summarizer(Agent, llm=llm):
        @strategy(_ExplicitCacheKeyStrategy())
        async def summarize(self, cache_key: str | None) -> str:
            """Summarize the document."""
            ...

    async def override(ctx, nxt):
        ctx.params["prompt_cache_key"] = "middleware-shard"
        return await nxt(ctx)

    agent = Summarizer()
    agent.event_manager.intercept("llm_call", override)

    assert await agent.summarize(cache_key="caller-shard") == _response().content
    assert llm.calls[0].kwargs["prompt_cache_key"] == "middleware-shard"


@pytest.mark.asyncio
@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
async def test_context_recovery_keeps_cache_key(middleware, explicit):
    class ContextWindowExceededError(Exception):
        pass

    class OverflowOnceLLM(FakeLLMClient):
        async def acall(self, *args, **kwargs):
            response = await super().acall(*args, **kwargs)
            if self.call_count == 1:
                raise ContextWindowExceededError("context window exceeded")
            return response

    llm = OverflowOnceLLM(scripted_responses=[_response(), _response()], strict_exhaustion=True)
    llm.config["max_tokens"] = 8000

    class Summarizer(Agent, llm=llm):
        @strategy(_ExplicitCacheKeyStrategy() if explicit else PredictStrategy())
        async def summarize(self, cache_key: str) -> str:
            """Summarize the document."""
            ...

    agent = Summarizer()
    if middleware:
        agent.event_manager.intercept("llm_call", _passthrough)

    result = await agent.summarize(cache_key="caller-shard")
    assert result == (_response().content if explicit else "ok")
    assert len(llm.calls) == 2
    assert llm.calls[1].kwargs["max_tokens"] == 4000
    key = llm.calls[0].kwargs["prompt_cache_key"]
    assert llm.calls[1].kwargs["prompt_cache_key"] == key
    if explicit:
        assert key == "caller-shard"
    else:
        assert 0 < len(key) <= 64 and key.isascii()
