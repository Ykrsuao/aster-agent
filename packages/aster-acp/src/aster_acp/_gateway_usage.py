# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in gateway accounting, correlated to an individual completed response."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

import httpx

from aster.llm_types import LLMResponse, LLMUsage
from aster_acp._hud import _epoch, _get, _number

_MAX_BYTES = 2 * 1024 * 1024
_TIMEOUT = 2.0
_REFRESH_SECONDS = 30.0


@dataclass(frozen=True)
class Connection:
    base: str
    headers: tuple[tuple[str, str], ...] = field(repr=False)


def connection(client: Any, overrides: Mapping[str, Any] | None = None) -> Connection | None:
    """Use only an explicitly selected gateway and this request's credential scope."""
    if getattr(client, "usage_source", None) != "magpie":
        return None
    config = getattr(client, "config", {})
    if not isinstance(config, Mapping) or config.get("client") is not None:
        return None
    overrides = overrides or {}
    base = config.get("api_base") or config.get("base_url")
    if not isinstance(base, str):
        return None
    base = base.rstrip("/")
    try:
        parsed = urlsplit(base)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return None
    for name in ("api_base", "base_url"):
        override = overrides.get(name)
        if override is not None and str(override).strip() and str(override).rstrip("/") != base:
            return None
    for name in ("api_key", "custom_llm_provider", "client", "extra_headers"):
        if name in overrides and overrides[name] != config.get(name):
            return None
    if "model" in overrides and overrides["model"] != getattr(client, "model", None):
        return None
    headers = config.get("extra_headers") or {}
    if not isinstance(headers, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
    ):
        return None
    headers = {key.lower(): value for key, value in headers.items()}
    if "authorization" not in headers:
        key = config.get("api_key")
        if not isinstance(key, str) or not key:
            return None
        headers["authorization"] = f"Bearer {key}"
    return Connection(base, tuple(sorted(headers.items())))


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value < 1 << 64 else None


def _record_usage(response: LLMResponse, rows: Any) -> Mapping[str, Any] | None:
    response_id = _get(response.raw_response, "id")
    if not isinstance(response_id, str) or not response_id or not isinstance(rows, list):
        return None
    matches = [
        row
        for row in rows[:100]
        if isinstance(row, Mapping)
        and response_id in (row.get("rid"), row.get("response_id"))
        and type(row.get("status")) is int
        and 200 <= row["status"] < 300
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    model = _get(response.raw_response, "model")
    if (
        isinstance(model, str)
        and model
        and model
        not in (
            row.get("model"),
            row.get("req"),
            row.get("served"),
        )
    ):
        return None
    counts = {name: _count(row.get(name)) for name in ("in", "out", "cache_read", "cache_write")}
    if any(value is None for value in counts.values()) or row.get("cache_unknown", 0) != 0:
        return None
    # Magpie records uncached input separately from cache reads/writes.
    total_input = sum(counts[name] or 0 for name in ("in", "cache_read", "cache_write"))
    if total_input >= 1 << 64:
        return None
    usage = response.usage
    if usage is not None and (
        (usage.input_tokens_reported and usage.input_tokens != total_input)
        or (usage.output_tokens_reported and usage.output_tokens != counts["out"])
    ):
        return None
    values: dict[str, Any] = {
        "input_tokens": total_input,
        "input_tokens_reported": True,
        "output_tokens": counts["out"],
        "output_tokens_reported": True,
        "total_tokens": total_input + (counts["out"] or 0),
        "cached_input_tokens": counts["cache_read"],
        "cached_input_tokens_reported": True,
        "cache_write_input_tokens": counts["cache_write"],
        "cache_write_input_tokens_reported": True,
    }
    if (reasoning := _count(row.get("reasoning"))) is not None:
        values.update(reasoning_tokens=reasoning, reasoning_tokens_reported=True)
    if (
        (usage is None or not usage.cost_known)
        and row.get("priced") is True
        and (cost := _number(row.get("cost"))) is not None
    ):
        values.update(cost_usd=cost, cost_known=True)
    response.usage = (usage or LLMUsage()).model_copy(update=values)
    return row


@dataclass(frozen=True)
class Account:
    connection: Connection
    provider: str
    user: str = field(repr=False)


def _limits(data: Any, account: Account) -> dict[str, Any] | None:
    entries = data.get("data") if isinstance(data, Mapping) else None
    if not isinstance(entries, list):
        return None
    matches = [
        item
        for item in entries[:256]
        if isinstance(item, Mapping)
        and item.get("provider") == account.provider
        and item.get("user") == account.user
    ]
    if len(matches) != 1:
        return None
    item = matches[0]
    observed = _epoch(item.get("readAt"))
    windows = item.get("windows")
    if observed is None or not isinstance(windows, list):
        return None
    limits = []
    for window in windows[:8]:
        if not isinstance(window, Mapping):
            continue
        name = window.get("name")
        percent = _number(window.get("used"))
        if not isinstance(name, str) or not name or percent is None or percent > 100:
            continue
        reset = _epoch(window.get("resetsAt"))
        if window.get("resetsAt") is not None and reset is None:
            continue
        limits.append(
            {"label": f"{account.provider} {name[:96]}", "percent": percent, "resetsAt": reset}
        )
    return {"limits": limits, "limitsUpdatedAt": observed}


@dataclass
class Observation:
    account: Account
    facts: dict[str, Any]


class GatewayUsage:
    """Bounded GETs; no inference, ambient credentials, redirects, or aggregate billing."""

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self._quotas: dict[Connection, tuple[float, Any]] = {}
        self._closed = False

    async def _read(
        self, source: Connection, path: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        if self._closed:
            return None
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=_TIMEOUT, follow_redirects=False, trust_env=False
            )
        async with self._http.stream(
            "GET", source.base + path, headers=dict(source.headers), params=params
        ) as response:
            if response.status_code != 200:
                return None
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > _MAX_BYTES:
                    return None
                body.extend(chunk)
        return json.loads(body)

    async def refresh(self, account: Account) -> dict[str, Any] | None:
        try:
            async with asyncio.timeout(_TIMEOUT):
                cached = self._quotas.get(account.connection)
                if cached is None or monotonic() - cached[0] >= _REFRESH_SECONDS:
                    data = await self._read(account.connection, "/magpie/quotas")
                    if data is None:
                        return None
                    self._quotas[account.connection] = (monotonic(), data)
                    while len(self._quotas) > 4:
                        self._quotas.pop(next(iter(self._quotas)))
                else:
                    data = cached[1]
                return _limits(data, account)
        except (httpx.HTTPError, ValueError, TimeoutError):
            return None

    async def enrich(self, response: LLMResponse, source: Connection) -> Observation | None:
        if not _get(response.raw_response, "id"):
            return None
        observation = None
        try:
            async with asyncio.timeout(_TIMEOUT):
                data = await self._read(source, "/magpie/usage/requests", {"limit": 100})
                if not isinstance(data, Mapping):
                    return None
                row = _record_usage(response, data.get("rows"))
                if row is None:
                    return None
                provider, user = row.get("provider"), row.get("providerAccount")
                if not isinstance(provider, str) or not isinstance(user, str) or not user:
                    return None
                account = Account(source, provider[:64], user)
                observation = Observation(account, {})
                facts = await self.refresh(account)
                observation.facts = facts or {}
                return observation
        except (httpx.HTTPError, ValueError, TimeoutError):
            return observation

    async def close(self) -> None:
        self._closed = True
        if self._http is not None:
            await self._http.aclose()
        self._quotas.clear()
