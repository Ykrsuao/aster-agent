# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gateway records must reach persisted responses and HUD accounting by exact ID."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from acp.schema import UsageUpdate
from aster_acp._gateway_usage import Account, GatewayUsage, connection
from aster_acp.event_bridge import ACPEventBridge
from aster_cli.coding import CodingAgent

from aster.events import LLMResponse
from aster.llm_types import LLMUsage
from aster.unifiedllm import CompletionClient, FakeLLMClient
from aster.unifiedllm.registry import client_from_config


class RecordingClient:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)


def gateway_row(**overrides: Any) -> dict[str, Any]:
    return {
        "rid": "reply-1",
        "response_id": "reply-1",
        "model": "deepseek-v4.1-flash",
        "provider": "workbuddy",
        "providerAccount": "account-1",
        "status": 200,
        "in": 1084,
        "out": 141,
        "cache_read": 5504,
        "cache_write": 0,
        "reasoning": 14,
        "cost": 0.000263712,
        "priced": True,
        **overrides,
    }


def gateway_quotas() -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "data": [
            {
                "provider": "workbuddy",
                "user": "other-account",
                "readAt": now.isoformat(),
                "windows": [{"name": "Credits", "used": 99}],
            },
            {
                "provider": "workbuddy",
                "user": "account-1",
                "readAt": now.isoformat(),
                "windows": [
                    {
                        "name": "Credits",
                        "used": 0,
                        "limit": 500,
                        "unit": "credits",
                        "resetsAt": (now + timedelta(days=1)).isoformat(),
                    },
                    {"name": "Extra credits", "used": 25, "limit": 2000, "unit": "credits"},
                ],
            },
        ]
    }


async def test_gateway_record_reaches_response_history_and_acp_hud(tmp_path, monkeypatch):
    requests: list[httpx.Request] = []
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.host == "gateway.example"
        assert request.headers["authorization"] == "Bearer synthetic-key"
        if request.url.path == "/v1/magpie/usage/requests":
            # Identical counts must never make an unrelated response match.
            return httpx.Response(
                200,
                json={
                    "rows": [gateway_row(rid="other", response_id="other", cost=99), gateway_row()]
                },
            )
        assert request.url.path == "/v1/magpie/quotas"
        return httpx.Response(200, json=gateway_quotas())

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, "transport": httpx.MockTransport(respond)})

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(Done(explanation='done'))"}
    )
    llm.usage_source = "magpie"
    llm.model = "openai/auto/deepseek-v4.1-flash"
    llm.config.update(api_base="https://gateway.example/v1", api_key="synthetic-key")
    original_call = llm.acall

    async def reply(*args: Any, **kwargs: Any) -> LLMResponse:
        result = await original_call(*args, **kwargs)
        result.raw_response = SimpleNamespace(id="reply-1", model="deepseek-v4.1-flash")
        result.usage = LLMUsage(
            input_tokens=6588,
            output_tokens=141,
            cached_input_tokens=0,
            cache_write_input_tokens=0,
            reasoning_tokens=14,
        )
        return result

    monkeypatch.setattr(llm, "acall", reply)
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "session-1")  # type: ignore[arg-type]
    try:
        await agent.handle({"user_messages": ["hello"]})
        await bridge.flush()
        update = [item for item in client.updates if isinstance(item, UsageUpdate)][-1]
        assert update.cost is not None, "Known Magpie request cost still appears as n/a"
        assert update.cost.amount == pytest.approx(0.000263712)
        payload = update.model_dump(mode="json", by_alias=True, exclude_none=True)
        hud = payload["_meta"]["aster"]["hud"]
        assert hud["tokens"]["cacheRead"] == 5504
        assert hud["latest"]["limits"][1]["percent"] == 25
        assert hud["latest"]["limitsUpdatedAt"] > 0
        recorded = [
            event for event in agent.event_manager.all_events() if isinstance(event, LLMResponse)
        ]
        assert recorded[-1].usage is not None and recorded[-1].usage.cost_known
        assert recorded[-1].usage.cached_input_tokens == 5504
        restored = LLMResponse.model_validate_json(recorded[-1].model_dump_json())
        assert restored.usage is not None and restored.usage.cost_known
        assert "synthetic-key" not in json.dumps(payload)
        assert "account-1" not in json.dumps(payload)
        assert len(requests) == 2

        async def refreshed(account: Account) -> dict[str, Any]:
            assert account.user == "account-1"
            return {
                "limits": [{"label": "Credits", "percent": 45}],
                "limitsUpdatedAt": hud["latest"]["limitsUpdatedAt"] + 30,
            }

        monkeypatch.setattr(bridge._gateway, "refresh", refreshed)
        await bridge._refresh_gateway_usage()
        await bridge.flush()
        refreshed_update = [item for item in client.updates if isinstance(item, UsageUpdate)][-1]
        refreshed_hud = refreshed_update.model_dump(mode="json", by_alias=True)["_meta"]["aster"][
            "hud"
        ]
        assert refreshed_hud["calls"] == hud["calls"]
        assert refreshed_hud["latest"]["respondedAt"] == hud["latest"]["respondedAt"]
        assert refreshed_hud["latest"]["limits"][0]["percent"] == 45
        assert refreshed_update.cost == update.cost
    finally:
        await bridge.close()
        await agent.close()


@pytest.fixture
def gateway_api(monkeypatch):
    state: dict[str, Any] = {
        "rows": [gateway_row()],
        "quota": gateway_quotas(),
        "request_status": 200,
        "quota_status": 200,
        "requests": [],
    }

    def respond(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        assert request.method == "GET"
        assert request.url.host == "gateway.example"
        if request.url.path.endswith("/usage/requests"):
            return httpx.Response(state["request_status"], json={"rows": state["rows"]})
        assert request.url.path.endswith("/quotas")
        return httpx.Response(state["quota_status"], json=state["quota"])

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, "transport": httpx.MockTransport(respond)})

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    return state


def configured_client() -> SimpleNamespace:
    return SimpleNamespace(
        usage_source="magpie",
        model="openai/auto/deepseek-v4.1-flash",
        config={"api_base": "https://gateway.example/v1", "api_key": "synthetic-key"},
    )


def unpriced_response(**usage: Any) -> LLMResponse:
    return LLMResponse(
        raw_response=SimpleNamespace(id="reply-1", model="deepseek-v4.1-flash"),
        usage=LLMUsage(input_tokens=6588, output_tokens=141, **usage),
    )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        LLMUsage(
            input_tokens=20,
            input_tokens_reported=False,
            output_tokens=10,
            output_tokens_reported=False,
            total_tokens=30,
        ),
    ],
)
async def test_gateway_usage_replaces_missing_or_estimated_total_tokens(gateway_api, usage):
    response = LLMResponse(
        raw_response=SimpleNamespace(id="reply-1", model="deepseek-v4.1-flash"), usage=usage
    )
    source = connection(configured_client())
    assert source is not None
    gateway = GatewayUsage()
    try:
        assert await gateway.enrich(response, source) is not None
        assert response.usage is not None
        assert response.usage.input_tokens == 6588
        assert response.usage.output_tokens == 141
        assert response.usage.total_tokens == 6729
        restored = LLMResponse.model_validate_json(response.model_dump_json())
        assert restored.usage is not None
        assert restored.usage.total_tokens == 6729
    finally:
        await gateway.close()


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [gateway_row(rid="other", response_id="other")],
        [gateway_row(), gateway_row()],
        [gateway_row(model="another-model")],
        [gateway_row(**{"in": 1085})],
        [gateway_row(out=140)],
        [gateway_row(status=500)],
        [gateway_row(cache_unknown=2)],
        [gateway_row(cache_read=True)],
    ],
)
async def test_unmatched_or_ambiguous_rows_cannot_supply_usage(gateway_api, rows):
    gateway_api["rows"] = rows
    response = unpriced_response()
    original = response.usage
    source = connection(configured_client())
    assert source is not None
    gateway = GatewayUsage()
    try:
        assert await gateway.enrich(response, source) is None
        assert response.usage is original
        assert len(gateway_api["requests"]) == 1
    finally:
        await gateway.close()


@pytest.mark.parametrize("cost", [None, True, -1, "unknown", "Infinity"])
async def test_invalid_gateway_prices_do_not_claim_a_free_call(gateway_api, cost):
    gateway_api["rows"] = [gateway_row(cost=cost)]
    response = unpriced_response()
    source = connection(configured_client())
    assert source is not None
    gateway = GatewayUsage()
    try:
        await gateway.enrich(response, source)
        assert response.usage is not None and not response.usage.cost_known
    finally:
        await gateway.close()


@pytest.mark.parametrize("provider_cost", [0.0, 0.25])
async def test_explicit_provider_cost_takes_precedence_over_gateway_estimates(
    gateway_api, provider_cost
):
    response = unpriced_response(cost_usd=provider_cost)
    source = connection(configured_client())
    assert source is not None
    gateway = GatewayUsage()
    try:
        await gateway.enrich(response, source)
        assert response.usage is not None
        assert response.usage.cost_usd == provider_cost
        assert response.usage.cost_known
    finally:
        await gateway.close()


@pytest.mark.parametrize("status", [401, 403, 500])
async def test_unavailable_gateway_preserves_the_model_response(gateway_api, status):
    gateway_api["request_status"] = status
    response = unpriced_response()
    original = response.usage
    source = connection(configured_client())
    assert source is not None
    gateway = GatewayUsage()
    try:
        assert await gateway.enrich(response, source) is None
        assert response.usage is original
    finally:
        await gateway.close()


async def test_quota_failure_retains_the_known_cost_and_account_for_later_refresh(gateway_api):
    gateway_api["quota_status"] = 503
    source = connection(configured_client())
    assert source is not None
    gateway = GatewayUsage()
    response = unpriced_response()
    try:
        observed = await gateway.enrich(response, source)
        assert observed is not None and not observed.facts
        assert response.usage is not None and response.usage.cost_known
        gateway_api["quota_status"] = 200
        refreshed = await gateway.refresh(observed.account)
        assert refreshed is not None and refreshed["limits"][1]["percent"] == 25
    finally:
        await gateway.close()


async def test_quota_cache_retains_the_provider_observation_time_and_exact_account(gateway_api):
    source = connection(configured_client())
    assert source is not None
    old = "2026-11-01T00:00:00Z"
    gateway_api["quota"]["data"][1]["readAt"] = old
    gateway = GatewayUsage()
    try:
        account = Account(source, "workbuddy", "account-1")
        first = await gateway.refresh(account)
        second = await gateway.refresh(account)
        assert first == second
        assert first is not None
        assert first["limitsUpdatedAt"] == datetime.fromisoformat(old).timestamp()
        assert len(gateway_api["requests"]) == 1
        assert await gateway.refresh(Account(source, "workbuddy", "absent-account")) is None
        assert await gateway.refresh(Account(source, "other-provider", "account-1")) is None
    finally:
        await gateway.close()


@pytest.mark.parametrize(
    "override",
    [
        {"api_base": "https://other.example/v1"},
        {"base_url": "https://other.example/v1"},
        {"api_key": "other-key"},
        {"model": "another-model"},
        {"extra_headers": {"x-tenant": "another-tenant"}},
        {"client": object()},
    ],
)
def test_metadata_queries_do_not_inherit_opt_in_across_request_scope_changes(override):
    client = configured_client()
    assert connection(client) is not None
    assert connection(client, override) is None


def test_standard_and_implicitly_authenticated_clients_never_trigger_gateway_queries():
    client = configured_client()
    client.usage_source = "response"
    assert connection(client) is None
    client.usage_source = "magpie"
    client.config.pop("api_key")
    assert connection(client) is None
    client.config.update(api_key="synthetic-key", client=object())
    assert connection(client) is None


def test_registry_usage_source_is_local_metadata_and_resets_on_route_changes(monkeypatch):
    monkeypatch.setenv("ASTER_TEST_GATEWAY_KEY", "synthetic-key")
    config = {
        "model_name": "openai/example",
        "api_base": "https://gateway.example/v1",
        "api_key_env": "ASTER_TEST_GATEWAY_KEY",
        "usage_source": "magpie",
        "media_inputs": [],
    }
    client = client_from_config("test", config)
    changed = client_from_config("test", config, api_base="https://other.example/v1")
    assert isinstance(client, CompletionClient)
    assert isinstance(changed, CompletionClient)
    try:
        assert client.usage_source == "magpie"
        assert "usage_source" not in client.config
        assert changed.usage_source == "response"
        with pytest.raises(ValueError, match="not in a model request"):
            client._prepare_call_config({"usage_source": "magpie"})
    finally:
        client.close()
        changed.close()
