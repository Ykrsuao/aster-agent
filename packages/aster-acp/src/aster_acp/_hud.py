# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider telemetry for the native HUD; observing a response performs no IO."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from aster.events import LLMCallEnd, LLMCallStart, LLMResponse


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value) if math.isfinite(value) and value >= 0 else None
        except OverflowError:
            return None
    return None


def _get(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _epoch(value: Any) -> float | None:
    if (number := _number(value)) is not None:
        return number
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    return None


def _reset(value: Any, now: float) -> float | None:
    if (epoch := _epoch(value)) is not None:
        return epoch if epoch >= 1_000_000_000 else now + epoch
    if not isinstance(value, str):
        return None
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if _number(number) is not None:
        return number if number >= 1_000_000_000 else now + number
    # OpenAI reset headers use durations such as 1m30s or 250ms.
    if re.fullmatch(r"(?:\d+(?:\.\d+)?(?:ms|[smhd]))+", value):
        seconds = sum(
            float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
            for amount, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|[smhd])", value)
        )
        return now + seconds if math.isfinite(seconds) else None
    return None


def _provider_facts(response: Any, now: float) -> dict[str, Any]:
    """Copy only allowlisted quota/cache facts, never arbitrary response headers."""
    facts: dict[str, Any] = {}
    hidden = _get(response, "_hidden_params")
    headers = _get(response, "headers") or _get(hidden, "additional_headers")
    if isinstance(headers, Mapping):
        safe = {str(key).lower(): value for key, value in headers.items()}
        # LiteLLM prefixes headers it does not promote to its common quota fields.
        # Prefer an original header when both spellings are present.
        for key, value in list(safe.items()):
            if key.startswith("llm_provider-"):
                safe.setdefault(key.removeprefix("llm_provider-"), value)
        # Native Anthropic fields and LiteLLM's common aliases describe the same
        # windows. Keep one copy, preferring native fields which retain resets.
        for kind in ("requests", "tokens"):
            for part in ("limit", "remaining", "reset"):
                native = f"anthropic-ratelimit-{kind}-{part}"
                if native in safe:
                    safe[f"x-ratelimit-{part}-{kind}"] = safe[native]
        limits = []
        for kind, names in (
            (
                "requests",
                (
                    "x-ratelimit-limit-requests",
                    "x-ratelimit-remaining-requests",
                    "x-ratelimit-reset-requests",
                ),
            ),
            (
                "tokens",
                (
                    "x-ratelimit-limit-tokens",
                    "x-ratelimit-remaining-tokens",
                    "x-ratelimit-reset-tokens",
                ),
            ),
            (
                "input tokens",
                (
                    "anthropic-ratelimit-input-tokens-limit",
                    "anthropic-ratelimit-input-tokens-remaining",
                    "anthropic-ratelimit-input-tokens-reset",
                ),
            ),
            (
                "output tokens",
                (
                    "anthropic-ratelimit-output-tokens-limit",
                    "anthropic-ratelimit-output-tokens-remaining",
                    "anthropic-ratelimit-output-tokens-reset",
                ),
            ),
        ):
            try:
                limit = float(safe[names[0]])
                remaining = float(safe[names[1]])
            except (KeyError, ValueError, TypeError, OverflowError):
                continue
            if _number(limit) is not None and limit > 0 and _number(remaining) is not None:
                limits.append(
                    {
                        "label": kind,
                        "percent": 100 * (1 - min(remaining / limit, 1)),
                        "resetsAt": _reset(safe.get(names[2]), now),
                    }
                )
        if limits:
            facts["limits"] = limits
    cache = _get(response, "prompt_cache")
    if isinstance(cache, Mapping) and (expiry := _epoch(cache.get("expires_at"))) is not None:
        facts["cacheExpiresAt"] = expiry
    return facts


class HudMetrics:
    """Monotonic per-bridge totals, with independent foreground and accounting scopes."""

    def __init__(self, history: Iterable[LLMResponse] = ()) -> None:
        self.stream_id = uuid4().hex
        self.started_at = time()
        self.sequence = 0
        self._starts: dict[tuple[str, int], float] = {}
        self._durations: dict[str, float] = {}
        self._totals: dict[str, int] = dict.fromkeys(
            ("input", "output", "cacheRead", "cacheWrite", "reasoning"), 0
        )
        self._complete: dict[str, bool] = dict.fromkeys(self._totals, True)
        self._days: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, Any] = {}
        self.model_calls = 0
        self.failed_calls = 0
        self.cost: float | None = 0.0
        self._known_cost: float | None = 0.0
        self._priced_calls = 0
        for event in history:
            self._add_usage(event)
            self.started_at = min(self.started_at, event.timestamp.timestamp())
        self.restored_calls = self.model_calls

    def start(self, event: LLMCallStart) -> None:
        self._starts[(event.generation_id, event.turn_number)] = perf_counter()
        while len(self._starts) > 128:
            self._starts.pop(next(iter(self._starts)))

    def end(self, event: LLMCallEnd) -> None:
        started = self._starts.pop((event.generation_id, event.turn_number), None)
        if not event.success:
            self.failed_calls += 1
            self.sequence += 1
        elif started is not None:
            self._durations[event.generation_id] = max(0.0, perf_counter() - started)
        while len(self._durations) > 128:
            self._durations.pop(next(iter(self._durations)))

    def _add_usage(self, event: LLMResponse) -> dict[str, int | None]:
        self.model_calls += 1
        usage = event.usage
        names = {
            "input": ("input_tokens", "input_tokens_reported"),
            "output": ("output_tokens", "output_tokens_reported"),
            "cacheRead": ("cached_input_tokens", "cached_input_tokens_reported"),
            "cacheWrite": ("cache_write_input_tokens", "cache_write_input_tokens_reported"),
            "reasoning": ("reasoning_tokens", "reasoning_tokens_reported"),
        }
        tokens: dict[str, int | None] = {}
        for key, (field, reported) in names.items():
            value = getattr(usage, field, None)
            if (
                usage is not None
                and getattr(usage, reported, False)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < 1 << 64
            ):
                tokens[key] = value
                self._totals[key] += value
                if self._totals[key] >= 1 << 64:
                    self._complete[key] = False
            else:
                tokens[key] = None
                self._complete[key] = False
        cost = _number(usage.cost_usd) if usage is not None and usage.cost_known else None
        if cost is not None:
            self._priced_calls += 1
            if self._known_cost is not None:
                self._known_cost = _number(self._known_cost + cost)
        if self.cost is not None and cost is not None:
            self.cost = _number(self.cost + cost)
        else:
            self.cost = None
        return tokens

    @property
    def display_cost(self) -> float | None:
        """Keep a known subtotal without presenting incomplete accounting as a total."""
        if self.cost is not None:
            return self.cost
        return self._known_cost if self._priced_calls else None

    def response(
        self, event: LLMResponse, *, foreground: bool, facts: dict[str, Any] | None = None
    ) -> None:
        self.sequence += 1
        tokens = self._add_usage(event)
        usage = event.usage
        now = time()
        day = datetime.fromtimestamp(now).date().isoformat()
        daily = self._days.setdefault(
            day, {"cost": 0.0, "complete": True, "calls": 0, "priced_calls": 0}
        )
        daily["calls"] += 1
        cost = _number(usage.cost_usd) if usage is not None and usage.cost_known else None
        if cost is None:
            daily["complete"] = False
        else:
            daily["cost"] += cost
            daily["priced_calls"] += 1
            if not math.isfinite(daily["cost"]):
                daily["cost"] = 0.0
                daily["complete"] = False
                daily["priced_calls"] = 0
        duration = self._durations.pop(event.generation_id, None)
        if foreground:
            self.latest = {
                "model": event.model_name,
                "respondedAt": now,
                "tokens": tokens,
                "duration": duration,
                "speed": tokens["output"] / duration
                if tokens["output"] is not None and duration is not None and duration >= 0.5
                else None,
                **_provider_facts(event.raw_response, now),
                **(facts or {}),
            }

    def snapshot(self, agent: Any) -> dict[str, Any]:
        skills = getattr(agent, "skills", None)
        active = skills.activated() if skills is not None else []
        available = skills.discovered() if skills is not None else []
        mcp = [name[4:] for name in active if name.startswith("mcp.")]
        active = [name for name in active if not name.startswith("mcp.")]
        available = [name for name in available if not name.startswith("mcp.")]
        llm = getattr(agent, "llm", None)
        config = getattr(llm, "config", {})
        provider = config.get("custom_llm_provider") if isinstance(config, Mapping) else None
        effort = getattr(llm, "reasoning_level", None) or getattr(llm, "reasoning_default", None)
        if effort is None and isinstance(config, Mapping):
            effort = config.get("reasoning_effort") or _get(config.get("reasoning"), "effort")
        return {
            "streamId": self.stream_id,
            "sequence": self.sequence,
            "startedAt": self.started_at,
            "calls": self.model_calls,
            "restoredCalls": self.restored_calls,
            "failedCalls": self.failed_calls,
            "costComplete": self.cost is not None,
            "costSubtotal": self.display_cost,
            "tokens": {
                key: value if self._complete[key] else None for key, value in self._totals.items()
            },
            "days": {day: dict(values) for day, values in self._days.items()},
            "latest": dict(self.latest),
            "environment": {
                "skills": active[:100],
                "availableSkills": len(available),
                "mcpServers": mcp[:100],
                "provider": provider[:128] if isinstance(provider, str) else "",
                "effort": effort[:64] if isinstance(effort, str) else "",
            },
        }
