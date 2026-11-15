# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model setup for human frontends: offline forms and a reviewed, local save.

No inference or model-list requests run here. Credentials never enter model
entries, returned results, or the conversation; only environment names do.
"""

from __future__ import annotations

import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from .providers import provider_details


def _entries() -> dict[str, dict[str, Any]]:
    from aster.llm_config import llm_config_chain
    from aster.unifiedllm.registry import load_model_entries

    return load_model_entries(*llm_config_chain())


def _target(path: Path | None) -> Path:
    from aster.paths import get_user_dir

    return (path or get_user_dir("llm_config.yaml")).resolve()


def saved_models() -> list[dict[str, str]]:
    """Return explicitly configured chat aliases for first-run model selection."""
    from aster.llm_config import bundled_config_paths, llm_config_chain
    from aster.unifiedllm.registry import load_model_entries

    bundled = {path.resolve() for path in bundled_config_paths()}
    entries = load_model_entries(
        *(path for path in llm_config_chain() if path.resolve() not in bundled)
    )
    return [
        {"id": name, "name": name}
        for name, entry in sorted(entries.items())
        if entry.get("client_type", "completion") in {"completion", "responses"}
    ]


def _field(
    name: str,
    label: str,
    default: str = "",
    *,
    required: bool = False,
    advanced: bool = False,
    hint: str = "",
) -> dict[str, Any]:
    return {
        "id": name,
        "label": label,
        "default": default,
        "secret": False,
        "required": required,
        "advanced": advanced,
        "hint": hint,
    }


def form(provider: str, *, path: Path | None = None) -> dict[str, Any]:
    """Describe an add-model form without returning stored credential values."""
    from aster.secrets import load_secrets_into_env

    load_secrets_into_env()
    details = provider_details(provider)
    if details["chat"] is False:
        return {**details, "fields": [], "target": str(_target(path))}
    entries = _entries()
    alias = provider
    suffix = 2
    while alias in entries:
        alias = f"{provider}-{suffix}"
        suffix += 1
    fields = [
        _field(
            "model",
            "Model ID",
            required=True,
            hint=details["modelHint"],
        ),
        _field("alias", "Name in Aster", alias, required=True, hint="A new name for /model."),
        *details["fields"],
        _field(
            "context_window",
            "Context window",
            advanced=True,
            hint="Optional token limit; leave blank if unknown.",
        ),
        _field(
            "max_tokens",
            "Reply token limit",
            "4096",
            required=True,
            advanced=True,
            hint="Maximum tokens for each reply. Must be smaller than the context window.",
        ),
        _field(
            "additional_drop_params",
            "Parameters to omit",
            advanced=True,
            hint="Optional comma-separated names, for example response_format.",
        ),
    ]
    key_field = next((item for item in fields if item["id"] == "api_key"), None)
    if key_field:
        base = next((item["default"] for item in fields if item["id"] == "api_base"), "")
        keys = []
        for entry in entries.values():
            if (
                str(entry.get("model_name", "")).startswith(details["routeProvider"] + "/")
                and str(entry.get("api_base", "")).rstrip("/").removesuffix("/v1")
                == base.rstrip("/").removesuffix("/v1")
                and isinstance(entry.get("api_key_env"), str)
            ):
                keys.append(entry["api_key_env"])
        keys.append(details["keyEnv"])
        for name in keys:
            if name and os.environ.get(name):
                key_field["default"] = "env:" + name
                key_field["hint"] = f"Using {name}. Leave this reference or paste a new key."
                break
    return {**details, "fields": fields, "target": str(_target(path))}


def prepare(provider: str, values: dict[str, str]) -> tuple[str, dict, dict[str, str]]:
    """Validate explicit form values and separate new secrets from model settings."""
    from . import configure_entry, normalize_endpoint

    details = provider_details(provider)
    if details["chat"] is False:
        raise ValueError(
            "This provider does not advertise a chat endpoint for Aster conversations."
        )
    fields = {item["id"]: item for item in details["fields"]}
    allowed = {*fields, "model", "alias", "context_window", "max_tokens", "additional_drop_params"}
    if (
        not isinstance(values, dict)
        or set(values) - allowed
        or any(not isinstance(value, str) or len(value) > 64 * 1024 for value in values.values())
    ):
        raise ValueError("Invalid model setup fields")
    alias = values.get("alias", "").strip()
    model = values.get("model", "").strip()
    if not re.fullmatch(r"[\w][\w.\-/]{0,127}", alias):
        raise ValueError("Use a model name of 1–128 letters, numbers, dots, dashes or underscores.")
    if not model or len(model) > 512 or any(char.isspace() or ord(char) < 32 for char in model):
        raise ValueError("Enter the provider's exact model ID.")
    native = details["routeProvider"]
    model = model.removeprefix(native + "/")
    if not model:
        raise ValueError("Enter the provider's exact model ID.")
    style = details["apiStyle"]
    entry: dict[str, Any] = {
        "model_name": native + "/" + model,
        "client_type": "responses" if style == "responses" else "completion",
        "api_style": style,
        "transport": "litellm",
        "media_inputs": [],
    }
    secrets: dict[str, str] = {}
    params = {}
    credentials = []
    for name, field in fields.items():
        value = values.get(name, "").strip()
        if not value:
            if field["required"]:
                raise ValueError(f"{field['label']} is required.")
            continue
        if name == "api_base":
            entry["api_base"] = normalize_endpoint(value)
            if style == "anthropic":
                entry["api_base"] = entry["api_base"].removesuffix("/v1")
            continue
        if field["secret"] or value.startswith("env:"):
            if value.startswith("env:"):
                env = value[4:]
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
                    raise ValueError(f"Use env:VARIABLE for {field['label']}.")
                if not os.environ.get(env):
                    raise ValueError(f"Credential variable {env} is not set.")
            else:
                # Credential names are independent of the user-facing Unicode alias.
                prefix = re.sub(r"\W", "_", f"ASTER_{provider}_{name}", flags=re.ASCII).upper()
                env = prefix
                suffix = 2
                while env in os.environ or env in secrets:
                    env = f"{prefix}_{suffix}"
                    suffix += 1
                secrets[env] = value
            if name == "api_key":
                entry["api_key_env"] = env
            else:
                credentials.append({"param": name, "env": env})
        else:
            if field.get("kind") in {"boolean", "bool"}:
                if value.lower() not in {"true", "false"}:
                    raise ValueError(f"{field['label']} must be true or false.")
                params[name] = value.lower() == "true"
            else:
                params[name] = value
    if params:
        entry["provider_params"] = params
    if credentials:
        entry["provider_credentials"] = credentials
    if not values.get("max_tokens", "").strip():
        raise ValueError("The reply token limit is required.")
    for name in ("context_window", "max_tokens"):
        raw = values.get(name, "").strip()
        if raw:
            if not raw.isascii() or not raw.isdecimal() or int(raw) < 1 or int(raw) > 2**31 - 1:
                raise ValueError(f"{name} must be a positive whole number.")
            entry[name] = int(raw)
    if entry.get("context_window") and entry.get("max_tokens", 0) >= entry["context_window"]:
        raise ValueError("The reply limit must be smaller than the context window.")
    dropped = [
        name.strip() for name in values.get("additional_drop_params", "").split(",") if name.strip()
    ]
    if dropped:
        if any(not name.isidentifier() for name in dropped):
            raise ValueError("Parameter names must contain only letters, numbers and underscores.")
        entry["additional_drop_params"] = list(dict.fromkeys(dropped))
    entry = configure_entry(entry)
    entry.pop("provenance", None)
    return alias, entry, secrets


def save(provider: str, values: dict[str, str], *, path: Path | None = None) -> dict[str, Any]:
    """Save one new alias after the frontend's explicit Save action; no model calls."""
    from aster.secrets import load_secrets_into_env, new_secret_envs
    from aster.unifiedllm.registry import reload_registry

    from . import ALIAS_MISSING, registry_write_lock, write

    load_secrets_into_env()
    alias, entry, secrets = prepare(provider, values)
    target = _target(path)
    if alias in _entries():
        raise ValueError("That model name already exists. Choose a new name to keep its settings.")

    def committed() -> bool:
        try:
            document = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            return document.get("models", {}).get(alias) == entry
        except FileNotFoundError:
            return False

    transaction = (
        new_secret_envs(
            target.with_name("secrets.yaml"),
            secrets,
            model_may_be_saved=committed,
            rollback_guard=lambda: registry_write_lock(target),
        )
        if secrets
        else nullcontext()
    )
    with transaction:
        write(entry, target, alias=alias, expected_alias=ALIAS_MISSING)
    # Refresh credentials and aliases for the current CLI process.
    result = {"alias": alias, "path": str(target), "tested": False}
    try:
        load_secrets_into_env()
        reload_registry()
    except (ValueError, OSError, yaml.YAMLError):
        result["warning"] = (
            "The model was saved, but reloading settings failed. Check the configuration files before selecting it."
        )
    return result
