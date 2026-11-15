# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline provider discovery shared by Connect and the terminal workspace.

The installed LiteLLM registry owns the provider names. Its bundled onboarding
and endpoint metadata enrich the list; missing metadata never hides a provider.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def _metadata() -> tuple[dict[str, dict], dict[str, dict]]:
    def read(relative: str, fallback: Any) -> Any:
        try:
            return json.loads(files("litellm").joinpath(relative).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return fallback

    onboarding = read("proxy/public_endpoints/provider_create_fields.json", [])
    endpoints = read("provider_endpoints_support_backup.json", {})
    return (
        {
            entry["litellm_provider"]: entry
            for entry in onboarding
            if isinstance(entry, dict) and isinstance(entry.get("litellm_provider"), str)
        }
        if isinstance(onboarding, list)
        else {},
        endpoints.get("providers", {}) if isinstance(endpoints, dict) else {},
    )


def providers() -> list[dict[str, Any]]:
    """List every installed provider, including services without chat endpoints."""
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    from .connect import PROVIDERS

    onboarding, endpoints = _metadata()
    names = set(litellm.provider_list) | set(JSONProviderRegistry.list_providers())
    rows = []
    for name in sorted(names):
        info = onboarding.get(name, {})
        capabilities = endpoints.get(name, {}).get("endpoints", {})
        chat = (
            any(capabilities.get(key) for key in ("chat_completions", "responses", "messages"))
            if capabilities
            else None
        )
        label = info.get("provider_display_name") or name.replace("_", " ").title()
        if name == "deepseek":
            label = "DeepSeek (official)"
        rows.append(
            {
                "id": name,
                "name": label,
                "chat": chat,
                "description": "Chat models"
                if chat
                else "Other services (no chat endpoint)"
                if chat is False
                else "Provider settings; choose an exact chat model ID",
            }
        )
    # Retain Aster's named gateway presets. `custom` is already a LiteLLM
    # provider, so use a distinct ID for an arbitrary OpenAI-compatible URL.
    for name, preset in PROVIDERS.items():
        if name not in names:
            rows.append({"id": name, "name": preset.label, "chat": True, "description": "Preset"})
    rows.append(
        {
            "id": "custom-endpoint",
            "name": "Custom endpoint (OpenAI compatible)",
            "chat": True,
            "description": "Your own server or gateway URL",
        }
    )
    return sorted(rows, key=lambda item: (item["name"].casefold(), item["id"]))


def _is_secret_field(field: dict[str, Any]) -> bool:
    """Treat credential names as sensitive even when the SDK uses a plain text control."""
    key = str(field.get("key", "")).casefold()
    return (
        field.get("field_type") in {"password", "upload"}
        or key.rsplit("_", 1)[-1]
        in {"key", "token", "secret", "password", "credential", "credentials"}
        or "private_key" in key
        or "private key" in str(field.get("label", "")).casefold()
    )


def provider_details(name: str) -> dict[str, Any]:
    """Return safe form metadata, native routing and local model suggestions."""
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    from .connect import PROVIDERS

    row = next((item for item in providers() if item["id"] == name), None)
    if row is None:
        raise ValueError("Choose a provider from the installed LiteLLM provider list.")
    onboarding, _ = _metadata()
    info = onboarding.get(name, {})
    native = name
    base = ""
    key_env = ""
    style = "chat"
    if name in PROVIDERS:
        preset = PROVIDERS[name]
        base, key_env, style = preset.api_base, preset.api_key_env, preset.api_style
        if name in {"nvidia", "hub", "google"}:
            native = "openai"
    if name == "custom-endpoint":
        native = "openai"
    if name == "deepseek":
        base, key_env = "https://api.deepseek.com", "DEEPSEEK_API_KEY"
    simple = JSONProviderRegistry.get(name)
    if simple is not None:
        base, key_env = simple.base_url, simple.api_key_env

    fields: list[dict[str, Any]] = []
    seen = set()
    for raw in info.get("credential_fields", []):
        key = raw.get("key")
        if not isinstance(key, str) or not key.isidentifier() or key in seen:
            continue
        seen.add(key)
        secret = _is_secret_field(raw)
        fields.append(
            {
                "id": key,
                "label": raw.get("label") or key,
                "secret": secret,
                "kind": raw.get("field_type", "text"),
                "required": bool(raw.get("required")) and not secret,
                "default": "" if secret else str(raw.get("default_value") or ""),
                "hint": "Paste a value, use env:VARIABLE, or leave blank for existing credentials."
                if secret
                else raw.get("tooltip") or raw.get("placeholder") or "Optional",
            }
        )
    if "api_base" not in seen:
        fields.insert(
            0,
            {
                "id": "api_base",
                "label": "API address",
                "secret": False,
                "required": name == "custom-endpoint",
                "default": base,
                "hint": "Leave blank to use the provider's default address.",
            },
        )
    elif base:
        next(field for field in fields if field["id"] == "api_base")["default"] = base
    if "api_key" not in seen and not info:
        fields.append(
            {
                "id": "api_key",
                "label": "API key",
                "secret": True,
                "required": False,
                "default": "",
                "hint": "Paste a key, use env:VARIABLE, or leave blank for existing credentials.",
            }
        )
    models = {
        model.removeprefix(f"{name}/")
        for model in set(litellm.models_by_provider.get(name, []))
        if litellm.model_cost.get(model, litellm.model_cost.get(f"{name}/{model}", {})).get("mode")
        in {None, "chat", "responses", "completion"}
    }
    labels = {}
    preferred = {}
    hint = (
        "Type any exact model ID. Suggestions come from the installed LiteLLM catalog, "
        "not a live account listing."
    )
    if name == "deepseek":
        # Official API aliases, not release names. Do not freeze the underlying
        # Flash version here: https://api-docs.deepseek.com/quick_start/pricing
        labels = {
            "deepseek-flash": "DeepSeek Flash (current version)",
            "deepseek-v4-pro": "DeepSeek V4 Pro",
            "deepseek-v4-flash": "DeepSeek Flash (legacy alias)",
            "deepseek-v4-flash-vision-exp": "DeepSeek Flash (legacy alias)",
        }
        preferred = {"deepseek-flash": 0, "deepseek-v4-pro": 1}
        hint = (
            "deepseek-flash is the current Flash model's stable API ID. "
            "Choose a suggestion or type any exact model ID."
        )
    return {
        **row,
        "routeProvider": native,
        "apiStyle": style,
        "keyEnv": key_env,
        "fields": fields,
        # These are suggestions from the installed package, not a live account listing.
        "models": sorted(models, key=lambda model: (preferred.get(model, 2), model)),
        "modelLabels": {model: label for model, label in labels.items() if model in models},
        "modelsSource": "Local LiteLLM",
        "modelHint": hint,
    }
