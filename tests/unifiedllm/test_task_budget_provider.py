# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task budgets count actual provider attempts across both public client APIs."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import litellm
import pytest
import pytest_asyncio
from litellm.caching.llm_caching_handler import LLMClientCache

from aster.task_budget import (
    TaskBudget,
    TaskBudgetError,
    TaskBudgetExceeded,
    TaskBudgetLimits,
    task_budget_scope,
)
from aster.unifiedllm import (
    AdmissionControl,
    AdmissionControlConfig,
    CompletionClient,
    ResponsesClient,
    RetryConfig,
)
from aster.unifiedllm.admission import AdmissionQueueFullError, AdmissionTimeoutError
from aster.unifiedllm.streaming import observe_stream
from aster.unifiedllm.unifiedllm import _ClientHttp

MESSAGES = [{"role": "user", "content": "offline budget check"}]
FAST_RETRY = RetryConfig(
    max_retries=2,
    rate_limit_extra_retries=0,
    base_delay=0,
    max_delay=0,
    jitter_factor=0,
    retry_on_empty_content=True,
)


def _response(transport, *, content="ok", with_usage=True):
    if transport == "completion":
        response = litellm.ModelResponse(
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "offline reasoning",
                    },
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )
        if not with_usage:
            response.usage = None
    else:
        response = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            ],
            output_text=content,
            status="completed",
            usage={"input_tokens": 10, "output_tokens": 4} if with_usage else None,
        )
    response._hidden_params = {"response_cost": 0.25} if with_usage else {}
    return response


@pytest.fixture(params=["completion", "responses"])
def transport(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def is_async(request):
    return request.param


@pytest_asyncio.fixture
async def provider(transport, is_async, monkeypatch):
    client_type = CompletionClient if transport == "completion" else ResponsesClient
    client = client_type("test-model", retry_config=FAST_RETRY)
    stub = AsyncMock() if is_async else Mock()
    monkeypatch.setattr(litellm, f"{'a' if is_async else ''}{transport}", stub)
    try:
        yield client, stub
    finally:
        await client.aclose()


async def _call(client, is_async, **kwargs):
    if is_async:
        return await client.acall(MESSAGES, **kwargs)
    return client.call(MESSAGES, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_limit", [2, 3])
async def test_503_retries_consume_each_dispatch(provider, transport, is_async, call_limit):
    client, stub = provider
    stub.side_effect = [
        litellm.ServiceUnavailableError(
            message="offline 503", llm_provider="openai", model="test-model"
        ),
        litellm.ServiceUnavailableError(
            message="offline 503", llm_provider="openai", model="test-model"
        ),
        _response(transport),
    ]
    owner = TaskBudget(TaskBudgetLimits(calls=call_limit))
    with task_budget_scope(owner):
        if call_limit == 2:
            with pytest.raises(TaskBudgetExceeded):
                await _call(client, is_async)
        else:
            assert (await _call(client, is_async)).content == "ok"

    state = owner.snapshot()
    assert stub.call_count == state["calls"] == call_limit
    assert state["unknown_usage"] == 2
    assert state["input_tokens"] == (10 if call_limit == 3 else 0)
    assert state["output_tokens"] == (4 if call_limit == 3 else 0)
    assert state["known_cost_usd"] == pytest.approx(0.25 if call_limit == 3 else 0)
    assert state["cost_known"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["completion"])
async def test_empty_content_attempt_keeps_its_usage(provider, transport, is_async):
    client, stub = provider
    stub.side_effect = [_response(transport, content=""), _response(transport)]
    owner = TaskBudget(TaskBudgetLimits(calls=2, tokens=100))
    with task_budget_scope(owner):
        assert (await _call(client, is_async)).content == "ok"

    state = owner.snapshot()
    assert stub.call_count == state["calls"] == 2
    assert state["input_tokens"] == 20
    assert state["output_tokens"] == 8
    assert state["unknown_usage"] == 0
    assert state["known_cost_usd"] == pytest.approx(0.5)
    assert state["cost_known"] is True


@pytest.mark.asyncio
async def test_missing_usage_stops_the_next_token_limited_dispatch(provider, transport, is_async):
    client, stub = provider
    stub.return_value = _response(transport, with_usage=False)
    owner = TaskBudget(TaskBudgetLimits(calls=5, tokens=100))
    with task_budget_scope(owner):
        assert (await _call(client, is_async)).content == "ok"
        with pytest.raises(TaskBudgetExceeded):
            await _call(client, is_async)

    state = owner.snapshot()
    assert stub.call_count == state["calls"] == 1
    assert state["unknown_usage"] == 1
    assert state["input_tokens"] == state["output_tokens"] == 0
    assert state["cost_known"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True], ids=["async"])
async def test_admission_rejection_does_not_consume_a_call(provider, transport):
    client, stub = provider
    stub.return_value = _response(transport)
    wrapper = AdmissionControl(
        client,
        AdmissionControlConfig(max_in_flight=1, max_pending=0, concurrency_group=str(uuid4())),
    )
    permit = await wrapper.controller.acquire(lambda _: None)
    owner = TaskBudget(TaskBudgetLimits(calls=1))
    try:
        with task_budget_scope(owner):
            with pytest.raises(AdmissionQueueFullError):
                await wrapper.acall(MESSAGES)
            assert stub.call_count == owner.snapshot()["calls"] == 0
            assert owner.snapshot()["unknown_usage"] == 0
            permit.release()
            assert (await wrapper.acall(MESSAGES)).content == "ok"
        assert stub.call_count == owner.snapshot()["calls"] == 1
    finally:
        permit.release()


@pytest.mark.asyncio
async def test_concurrent_calls_share_the_last_dispatch_slot(provider, transport, is_async):
    client, stub = provider
    release = asyncio.Event() if is_async else threading.Event()

    async def async_provider(**kwargs):
        await asyncio.wait_for(release.wait(), timeout=10)
        return _response(transport)

    def sync_provider(**kwargs):
        assert release.wait(timeout=10)
        return _response(transport)

    stub.side_effect = async_provider if is_async else sync_provider

    async def compete():
        if is_async:
            return await client.acall(MESSAGES)
        return await asyncio.to_thread(client.call, MESSAGES)

    owner = TaskBudget(TaskBudgetLimits(calls=1))
    with task_budget_scope(owner):
        tasks = [asyncio.create_task(compete()) for _ in range(2)]
        try:
            done, pending = await asyncio.wait(
                tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            assert len(done) == len(pending) == 1
            assert isinstance(next(iter(done)).exception(), TaskBudgetExceeded)
            assert stub.call_count == owner.snapshot()["calls"] == 1
        finally:
            release.set()
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5
            )

    assert sum(isinstance(result, TaskBudgetExceeded) for result in results) == 1
    assert sum(getattr(result, "content", None) == "ok" for result in results) == 1
    assert owner.snapshot()["input_tokens"] == 10
    assert owner.snapshot()["unknown_usage"] == 0


@pytest.mark.asyncio
async def test_budget_rejects_hidden_sdk_retries_before_dispatch(provider, is_async):
    client, stub = provider
    owner = TaskBudget(TaskBudgetLimits(calls=5))
    with task_budget_scope(owner):
        with pytest.raises(TaskBudgetError, match="max_retries=0"):
            await _call(client, is_async, max_retries=1)

    assert stub.call_count == owner.snapshot()["calls"] == 0
    assert owner.snapshot()["unknown_usage"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False], ids=["sync"])
async def test_sync_time_budget_is_rejected_before_dispatch(provider, transport, is_async):
    client, stub = provider
    stub.return_value = _response(transport)
    owner = TaskBudget(TaskBudgetLimits(calls=5, time_seconds=1))
    with task_budget_scope(owner):
        with pytest.raises(TaskBudgetError, match=r"use acall\(\) instead of call\(\)"):
            await _call(client, is_async)
    assert stub.call_count == owner.snapshot()["calls"] == 0
    assert owner.snapshot()["unknown_usage"] == 0

    assert (await _call(client, is_async)).content == "ok"
    assert stub.call_count == 1
    assert owner.snapshot()["calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True], ids=["async"])
@pytest.mark.parametrize("cause", ["timeout", "cancel"])
async def test_abandoned_admission_wait_does_not_consume_a_call(
    provider, transport, cause, monkeypatch
):
    client, stub = provider
    stub.return_value = _response(transport)
    wrapper = AdmissionControl(
        client,
        AdmissionControlConfig(
            max_in_flight=1,
            max_pending=1,
            queue_timeout=0.02 if cause == "timeout" else None,
            concurrency_group=str(uuid4()),
        ),
    )
    permit = await wrapper.controller.acquire(lambda _: None)
    acquiring = asyncio.Event()
    real_acquire = wrapper.controller.acquire

    async def acquire(observer):
        acquiring.set()
        return await real_acquire(observer)

    monkeypatch.setattr(wrapper.controller, "acquire", acquire)
    owner = TaskBudget(TaskBudgetLimits(calls=1))
    pending = None
    try:
        with task_budget_scope(owner):
            pending = asyncio.create_task(wrapper.acall(MESSAGES))
            await asyncio.wait_for(acquiring.wait(), timeout=5)
            assert not pending.done()
            assert stub.call_count == owner.snapshot()["calls"] == 0
            if cause == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(AdmissionTimeoutError):
                    await asyncio.wait_for(pending, timeout=5)
            assert stub.call_count == owner.snapshot()["calls"] == 0
            assert owner.snapshot()["unknown_usage"] == 0
            permit.release()
            assert (await wrapper.acall(MESSAGES)).content == "ok"
        assert stub.call_count == owner.snapshot()["calls"] == 1
    finally:
        permit.release()
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_interrupted_stream_keeps_unknown_usage_and_blocks_the_next_dispatch(monkeypatch):
    requests = []
    chunk = {
        "id": "chatcmpl-budget-interruption",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "partial answer"}}],
    }

    def respond(request):
        requests.append(json.loads(request.content))

        class InterruptedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                raise httpx.ReadError("offline stream interrupted", request=request)

        return httpx.Response(
            200,
            stream=InterruptedStream(),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        "openai/gpt-4o-mini",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        stream=True,
    )
    owner = TaskBudget(TaskBudgetLimits(calls=5, tokens=100))
    seen = []
    try:
        with task_budget_scope(owner), observe_stream(lambda kind, value: seen.append(kind)):
            with pytest.raises(Exception, match="offline stream interrupted"):
                await client.acall(MESSAGES)
            assert seen.count("chat") == 1
            with pytest.raises(TaskBudgetExceeded):
                await client.acall(MESSAGES)
    finally:
        await client.aclose()

    state = owner.snapshot()
    assert len(requests) == state["calls"] == 1
    assert requests[0]["stream"] is True
    assert state["unknown_usage"] == 1
    assert state["input_tokens"] == state["output_tokens"] == 0
    assert state["known_cost_usd"] == 0
    assert state["cost_known"] is False
