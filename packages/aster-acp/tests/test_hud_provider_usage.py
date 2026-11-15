# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real SDK normalization must preserve provider facts all the way to the HUD."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
import litellm
import pytest
from aster_acp._hud import HudMetrics
from litellm.caching.llm_caching_handler import LLMClientCache

from aster.llm_types import CacheBoundary, LLMResponse, LLMUsage
from aster.unifiedllm import CompletionClient, RetryConfig
from aster.unifiedllm.unifiedllm import _ClientHttp


async def sdk_response(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    stream: bool,
    is_async: bool,
    *,
    detail: int | None = None,
    headers: dict[str, str] | None = None,
    logger_fn: Callable[[dict[str, Any]], None] | None = None,
) -> LLMResponse:
    requests = []
    model = "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-5"
    usage: dict[str, Any]
    if provider == "openai":
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        if detail is not None:
            usage.update(
                prompt_tokens_details={"cached_tokens": detail},
                cache_creation_input_tokens=detail,
                completion_tokens_details={"reasoning_tokens": detail},
            )
    else:
        usage = {"input_tokens": 100, "output_tokens": 20}
        if detail is not None:
            usage.update(
                cache_read_input_tokens=detail,
                cache_creation_input_tokens=detail,
                output_tokens_details={"thinking_tokens": detail},
            )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if provider == "openai":
            base = {"id": "chatcmpl-hud", "created": 1, "model": model}
            body = {
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
            base["object"] = "chat.completion.chunk"
            events = [
                {**base, "choices": [{"index": 0, "delta": {"content": "answer"}}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**base, "choices": [], "usage": usage},
            ]
            sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            sse += "data: [DONE]\n\n"
        else:
            body = {
                "id": "msg-hud",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": usage,
            }
            events = [
                {
                    "type": "message_start",
                    "message": {
                        **body,
                        "content": [],
                        "stop_reason": None,
                        "usage": {**usage, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "answer"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        key: value
                        for key, value in usage.items()
                        if key in {"output_tokens", "output_tokens_details"}
                    },
                },
                {"type": "message_stop"},
            ]
            sse = "".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            )
        if stream:
            return httpx.Response(
                200, text=sse, headers={**(headers or {}), "content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=body, headers=headers)

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": transport, "trust_env": False}),
    )
    config: dict[str, Any] = {}
    if stream:
        config["stream_options"] = {"include_usage": True}
    if logger_fn is not None:
        config["logger_fn"] = logger_fn
    client = CompletionClient(
        f"{provider}/{model}",
        api_base="https://endpoint.example/v1",
        api_key="synthetic",
        retry_config=RetryConfig(max_retries=0),
        stream=stream,
        **config,
    )
    try:
        messages: list[dict[str, Any] | LLMResponse | CacheBoundary] = [
            {"role": "user", "content": "Return an answer."}
        ]
        response = await client.acall(messages) if is_async else client.call(messages)
    finally:
        await client.aclose()
    assert len(requests) == 1
    assert response.content == "answer"
    return response


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("detail", [None, 0, 7])
async def test_sdk_missing_counters_stay_unknown_and_reported_zero_stays_zero(
    monkeypatch, provider, stream, is_async, detail
):
    response = await sdk_response(monkeypatch, provider, stream, is_async, detail=detail)
    assert response.usage is not None
    restored = LLMUsage.model_validate_json(response.usage.model_dump_json())
    metrics = HudMetrics()
    metrics.response(response, foreground=True)
    snapshot = metrics.snapshot(None)
    for field, label in [
        ("cached_input_tokens", "cacheRead"),
        ("cache_write_input_tokens", "cacheWrite"),
        ("reasoning_tokens", "reasoning"),
    ]:
        assert getattr(restored, f"{field}_reported") is (detail is not None), field
        assert snapshot["latest"]["tokens"][label] == detail, label
        assert snapshot["tokens"][label] == detail, label


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
async def test_sdk_quota_headers_keep_all_windows_and_reset_times(
    monkeypatch, provider, stream, is_async
):
    now = 1_800_000_000.0
    reset = "2027-01-15T12:00:00+00:00"
    monkeypatch.setattr("aster_acp._hud.time", lambda: now)
    headers = (
        {
            "authorization": "synthetic-secret",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "75",
            "x-ratelimit-reset-requests": "1m30s",
        }
        if provider == "openai"
        else {
            "authorization": "synthetic-secret",
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-requests-remaining": "75",
            "anthropic-ratelimit-requests-reset": reset,
            "anthropic-ratelimit-tokens-limit": "2000",
            "anthropic-ratelimit-tokens-remaining": "500",
            "anthropic-ratelimit-tokens-reset": reset,
            "anthropic-ratelimit-input-tokens-limit": "1000",
            "anthropic-ratelimit-input-tokens-remaining": "400",
            "anthropic-ratelimit-input-tokens-reset": reset,
            "anthropic-ratelimit-output-tokens-limit": "200",
            "anthropic-ratelimit-output-tokens-remaining": "150",
            "anthropic-ratelimit-output-tokens-reset": reset,
        }
    )
    response = await sdk_response(monkeypatch, provider, stream, is_async, headers=headers)
    metrics = HudMetrics()
    metrics.response(response, foreground=True)
    snapshot = metrics.snapshot(None)
    expected = (
        [{"label": "requests", "percent": 25, "resetsAt": now + 90}]
        if provider == "openai"
        else [
            {
                "label": "requests",
                "percent": 25,
                "resetsAt": datetime.fromisoformat(reset).timestamp(),
            },
            {
                "label": "tokens",
                "percent": 75,
                "resetsAt": datetime.fromisoformat(reset).timestamp(),
            },
            {
                "label": "input tokens",
                "percent": 60,
                "resetsAt": datetime.fromisoformat(reset).timestamp(),
            },
            {
                "label": "output tokens",
                "percent": 25,
                "resetsAt": datetime.fromisoformat(reset).timestamp(),
            },
        ]
    )
    assert snapshot["latest"]["limits"] == expected
    assert "synthetic-secret" not in json.dumps(snapshot)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_usage_observation_preserves_the_callers_logger(monkeypatch, provider, is_async):
    events = []
    response = await sdk_response(
        monkeypatch,
        provider,
        False,
        is_async,
        logger_fn=lambda details: events.append(details["log_event_type"]),
    )
    assert "pre_api_call" in events
    assert "post_api_call" in events
    assert response.usage is not None
    assert response.usage.reasoning_tokens_reported is False
