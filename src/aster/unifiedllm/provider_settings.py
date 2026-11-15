# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native provider connection fields, with credentials referenced by environment name."""

from __future__ import annotations

import os
import re
from typing import Any

_RESERVED = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "stream",
    "api_key",
    "api_base",
    "base_url",
    "custom_llm_provider",
    "client",
    "client_type",
    "response_format",
    "output_model",
    "provider_params",
    "provider_credentials",
    "max_tokens",
    "max_output_tokens",
}


def validate_provider_settings(config: dict[str, Any]) -> None:
    params = config.get("provider_params", {})
    credentials = config.get("provider_credentials", [])
    if not isinstance(params, dict) or not isinstance(credentials, list):
        raise ValueError("provider_params must be a mapping and provider_credentials a list")
    names = set()
    for key in params:
        if not isinstance(key, str) or not key.isidentifier() or key in _RESERVED:
            raise ValueError("Provider fields cannot replace model routing or request controls")
        names.add(key)
    for item in credentials:
        if not isinstance(item, dict) or set(item) != {"param", "env"}:
            raise ValueError("Provider credentials must contain only param and env names")
        key, env = item["param"], item["env"]
        if (
            not isinstance(key, str)
            or not key.isidentifier()
            or key in _RESERVED
            or key in names
            or not isinstance(env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env)
        ):
            raise ValueError("Invalid or duplicate provider credential reference")
        names.add(key)


def resolve_provider_settings(config: dict[str, Any]) -> dict[str, Any]:
    validate_provider_settings(config)
    params = dict(config.get("provider_params", {}))
    for item in config.get("provider_credentials", []):
        value = os.environ.get(item["env"])
        if not value:
            raise ValueError(f"Provider credential variable {item['env']} is not set")
        params[item["param"]] = value
    return params
