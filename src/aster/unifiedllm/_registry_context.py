# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry context metadata: declaration scope, client provenance and live lookup.

Factory clients retain their own declaration, even when it has no window. Direct
clients consult the current registry and require agreement among matching entries.
Credentials are compared in memory only; lookup does not log them or make requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .route_identity import same_context_route


@dataclass(frozen=True, repr=False)
class RegistryContext:
    """An alias's declaration and the effective constructor route it belongs to.

    A record with no window still marks a factory-origin client. Only the absence
    of a record permits live lookup across other aliases. Route values stay out of
    the representation because they can contain credentials.
    """

    applies: bool
    _model: str
    _config: dict[str, Any]
    _client_type: str
    _window: int | None

    @classmethod
    def from_config(
        cls,
        model: str,
        config: Mapping[str, Any],
        params: Mapping[str, Any],
        overrides: Mapping[str, Any],
        *,
        client_type: str,
    ) -> RegistryContext:
        """Capture applicability and the route passed to the concrete constructor.

        ``params`` contains resolved defaults, before ``overrides``. An explicit
        api_key changes an api_key_env declaration's scope even if its bytes match.
        That declaration-only provenance is not added to the actual client route.
        """
        effective = {**params, **overrides}
        # LLM constructors never receive the registry's decision-only endpoint.
        declared = {
            **{key: value for key, value in config.items() if key != "endpoint"},
            "api_key": params.get("api_key"),
        }
        scope = {
            **effective,
            "api_key_env": config.get("api_key_env") if "api_key" not in overrides else None,
        }
        applies = same_context_route(
            model,
            declared,
            config.get("client_type", "completion"),
            effective.get("model", model),
            scope,
            client_type,
        )
        return cls(
            applies=applies,
            _model=effective.get("model", model),
            _config=effective,
            _client_type=client_type,
            _window=config.get("context_window") if applies else None,
        )


def resolve_context_window(
    model: str,
    config: Mapping[str, Any],
    client_type: str,
    origin: RegistryContext | None,
) -> int | None:
    """Return applicable registry metadata, or None so the caller can fall back."""
    if origin is not None:
        if same_context_route(
            model, config, client_type, origin._model, origin._config, origin._client_type
        ):
            return origin._window
        return None

    # Registry loading, locking and credential resolution belong to this internal
    # registry implementation, never to the LLM client's metadata property.
    from .registry import _context_registry_snapshot, _lookup_api_key_env

    entries = _context_registry_snapshot()
    client_key = config.get("api_key")
    own_route = {
        key: value for key, value in config.items() if key not in ("api_key", "api_key_env")
    }
    windows: set[int] = set()
    for alias, entry in entries.items():
        entry_client_type = entry.get("client_type", "completion")
        if client_key is not None:
            # Use the factory's validated names, synonyms and empty-value rules,
            # silently: this scan can visit unrelated or malformed declarations.
            entry_key = entry.get("api_key")
            if entry_key is None:
                entry_key, _ = _lookup_api_key_env(entry)
            if entry_key != client_key:
                continue
        skipped = ("api_key", "api_key_env") + (
            () if entry_client_type == "decision" else ("endpoint",)
        )
        route = {key: value for key, value in entry.items() if key not in skipped}
        if same_context_route(
            model,
            own_route,
            client_type,
            entry.get("model_name", alias),
            route,
            entry_client_type,
        ):
            window = entry.get("context_window")
            # Direct clients require every matching alias to declare the same
            # positive integer. A missing/invalid declaration makes it ambiguous.
            if type(window) is not int or window <= 0:
                return None
            windows.add(window)
            if len(windows) > 1:
                return None
    return windows.pop() if windows else None
