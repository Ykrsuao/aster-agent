# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HUD accounting crosses real response and ACP seams without provider credentials."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import TypedDict

import pytest
from acp.schema import UsageUpdate
from aster_acp._hud import HudMetrics
from aster_acp.event_bridge import ACPEventBridge
from aster_cli.coding import CodingAgent

from aster.events import LLMCallEnd, LLMCallStart, LLMResponse
from aster.llm_types import LLMUsage
from aster.unifiedllm import FakeLLMClient


class LifecycleArgs(TypedDict):
    method_name: str
    strategy: str
    generation_id: str
    turn_number: int


def lifecycle(generation: str) -> LifecycleArgs:
    return {
        "method_name": "handle",
        "strategy": "CodeAct",
        "generation_id": generation,
        "turn_number": 1,
    }


def response(generation: str = "foreground", **kwargs) -> LLMResponse:
    return LLMResponse(generation_id=generation, model_name=generation, **kwargs)


def priced(**overrides) -> LLMUsage:
    return LLMUsage(
        **{
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 50,
            "cache_write_input_tokens": 0,
            "reasoning_tokens": 5,
            "cost_usd": 0.1,
            **overrides,
        }
    )


def test_interleaved_calls_keep_foreground_speed_and_include_background_accounting(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("aster_acp._hud.perf_counter", lambda: clock[0])
    metrics = HudMetrics()
    metrics.start(LLMCallStart(**lifecycle("foreground")))
    clock[0] = 100.5
    metrics.start(LLMCallStart(**lifecycle("summary")))
    clock[0] = 102.0
    metrics.end(LLMCallEnd(**lifecycle("foreground")))
    metrics.response(response(usage=priced()), foreground=True)
    clock[0] = 105.0
    metrics.end(LLMCallEnd(**lifecycle("summary")))
    metrics.response(response("summary", usage=priced(cost_usd=0.2)), foreground=False)

    snapshot = metrics.snapshot(None)
    assert snapshot["calls"] == 2
    assert snapshot["tokens"] == {
        "input": 200,
        "output": 40,
        "cacheRead": 100,
        "cacheWrite": 0,
        "reasoning": 10,
    }
    assert snapshot["latest"]["model"] == "foreground"
    assert snapshot["latest"]["duration"] == 2.0
    assert snapshot["latest"]["speed"] == 10.0
    assert metrics.cost == pytest.approx(0.3)
    assert next(iter(snapshot["days"].values()))["cost"] == pytest.approx(0.3)


def test_missing_counters_prices_and_local_estimates_stay_unavailable():
    metrics = HudMetrics()
    metrics.response(response(usage=priced(cost_usd=0)), foreground=True)
    metrics.response(
        response(usage=LLMUsage(input_tokens=120, input_tokens_reported=False)), foreground=True
    )
    snapshot = metrics.snapshot(None)
    assert all(value is None for value in snapshot["tokens"].values())
    assert all(value is None for value in snapshot["latest"]["tokens"].values())
    assert snapshot["latest"]["speed"] is None
    assert metrics.cost is None
    assert next(iter(snapshot["days"].values()))["complete"] is False


def test_restored_session_totals_do_not_rebill_historical_calls():
    older = response(usage=priced(cost_usd=5), timestamp=datetime(2026, 1, 1))
    metrics = HudMetrics([older])
    assert metrics.cost == 5
    assert not metrics.snapshot(None)["days"]
    assert not metrics.snapshot(None)["latest"]
    metrics.response(response(usage=priced(cost_usd=2)), foreground=True)
    snapshot = metrics.snapshot(None)
    assert metrics.cost == 7
    assert snapshot["calls"] == 2
    assert snapshot["restoredCalls"] == 1
    assert snapshot["startedAt"] == older.timestamp.timestamp()
    assert snapshot["tokens"]["input"] == 200
    assert next(iter(snapshot["days"].values())) == {
        "cost": 2,
        "complete": True,
        "calls": 1,
        "priced_calls": 1,
    }


def test_known_costs_survive_unknown_calls_as_explicitly_partial_totals():
    metrics = HudMetrics()
    metrics.response(response(usage=priced(cost_usd=0.25)), foreground=True)
    metrics.response(response(usage=LLMUsage(input_tokens=120)), foreground=True)
    assert metrics.cost is None
    assert metrics.display_cost == 0.25
    snapshot = metrics.snapshot(None)
    assert snapshot["costComplete"] is False
    assert snapshot["costSubtotal"] == 0.25
    assert next(iter(snapshot["days"].values()))["cost"] == 0.25
    assert next(iter(snapshot["days"].values()))["priced_calls"] == 1


def test_cost_buckets_follow_local_calendar_days(monkeypatch):
    midnight = datetime(2026, 11, 16)
    clock = [(midnight - timedelta(seconds=1)).timestamp()]
    monkeypatch.setattr("aster_acp._hud.time", lambda: clock[0])
    metrics = HudMetrics()
    metrics.response(response(usage=priced(cost_usd=1)), foreground=True)
    clock[0] = midnight.timestamp()
    metrics.response(response(usage=priced(cost_usd=2)), foreground=True)
    days = metrics.snapshot(None)["days"]
    assert days["2026-11-15"]["cost"] == 1
    assert days["2026-11-16"]["cost"] == 2


@pytest.mark.parametrize("reset, offset", [("1m30s", 90), ("250ms", 0.25), ("120", 120)])
def test_provider_headers_are_allowlisted_and_reset_durations_are_real(monkeypatch, reset, offset):
    now = 1_800_000_000.0
    monkeypatch.setattr("aster_acp._hud.time", lambda: now)
    raw = SimpleNamespace(
        _hidden_params={
            "additional_headers": {
                "X-RateLimit-Limit-Requests": "100",
                "X-RateLimit-Remaining-Requests": "75",
                "X-RateLimit-Reset-Requests": reset,
                "Authorization": "secret-token",
            }
        },
        prompt_cache={"expires_at": now + 300},
        api_key="secret-key",
    )
    metrics = HudMetrics()
    metrics.response(response(usage=priced(), raw_response=raw), foreground=True)
    latest = metrics.snapshot(None)["latest"]
    assert latest["limits"] == [{"label": "requests", "percent": 25, "resetsAt": now + offset}]
    assert latest["cacheExpiresAt"] == now + 300
    assert "secret" not in json.dumps(metrics.snapshot(None))


def test_anthropic_reset_timestamps_and_missing_cache_expiry():
    raw = {
        "headers": {
            "anthropic-ratelimit-input-tokens-limit": "1000",
            "anthropic-ratelimit-input-tokens-remaining": "400",
            "anthropic-ratelimit-input-tokens-reset": "2026-11-15T12:00:00+00:00",
            "x-ratelimit-limit-requests": "0",
            "x-ratelimit-remaining-requests": "2",
        }
    }
    metrics = HudMetrics()
    metrics.response(response(usage=priced(), raw_response=raw), foreground=True)
    latest = metrics.snapshot(None)["latest"]
    assert latest["limits"] == [
        {
            "label": "input tokens",
            "percent": 60,
            "resetsAt": datetime.fromisoformat("2026-11-15T12:00:00+00:00").timestamp(),
        }
    ]
    assert "cacheExpiresAt" not in latest


def test_environment_distinguishes_skills_mcp_and_only_exports_display_configuration():
    agent = SimpleNamespace(
        skills=SimpleNamespace(
            activated=lambda: ["coding", "mcp.docs"],
            discovered=lambda: ["coding", "review", "mcp.docs"],
        ),
        llm=SimpleNamespace(
            reasoning_level="high", config={"api_key": "secret", "custom_llm_provider": "openai"}
        ),
    )
    snapshot = HudMetrics().snapshot(agent)
    assert snapshot["environment"] == {
        "skills": ["coding"],
        "availableSkills": 2,
        "mcpServers": ["docs"],
        "provider": "openai",
        "effort": "high",
    }
    assert "secret" not in json.dumps(snapshot)


async def test_bridge_publishes_failed_calls_and_reconstructed_cost_before_next_success(tmp_path):
    updates = []

    class Client:
        async def session_update(self, session_id, update, **kwargs):
            updates.append(update)

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path)
    agent.event_manager.add(response(usage=priced(cost_usd=3)))
    bridge = ACPEventBridge(agent, Client(), "session")  # type: ignore[arg-type]
    try:
        agent.event_manager.add(LLMCallStart(**lifecycle("failed")), record=False)
        agent.event_manager.add(LLMCallEnd(**lifecycle("failed"), success=False), record=False)
        await bridge.flush()
        update = next(update for update in updates if isinstance(update, UsageUpdate))
        assert update.cost is not None
        assert update.cost.amount == 3
        assert update.field_meta is not None
        meta = update.field_meta["aster"]
        assert meta["usagePending"] is True
        assert meta["hud"]["failedCalls"] == 1
        assert meta["hud"]["restoredCalls"] == 1
        assert not meta["hud"]["days"]

        agent.event_manager.add(response("helper", usage=priced(cost_usd=1)))
        await bridge.flush()
        final = [update for update in updates if isinstance(update, UsageUpdate)][-1]
        assert final.cost is not None
        assert final.cost.amount == 4
        assert final.field_meta is not None
        assert final.field_meta["aster"]["usagePending"] is True
        assert final.field_meta["aster"]["hud"]["latest"] == {}
    finally:
        await bridge.close()
        await agent.close()
