# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive native-provider configuration for the standalone ``aster connect`` CLI."""

from __future__ import annotations

import click

from . import _connect_prompts as prompts
from . import _connect_view as view


def run(state) -> None:
    from aster.unifiedllm.connect import configuration

    provider = state.provider.removeprefix("litellm:")
    description = configuration.form(provider, path=state.path)
    if description["chat"] is False:
        raise click.UsageError(
            "This LiteLLM provider supplies other services, not chat models for Aster."
        )
    if state.yes and not state.no_probe:
        raise click.UsageError("Native provider setup with --yes requires --no-probe.")
    unsupported = (
        state.test_media_inputs is not None
        or state.media_inputs is not None
        or state.levels_file
        or state.reasoning_template
        or state.structured_output_mode_set
        or state.cache_enabled is not None
        or state.api_style
        and state.api_style != description["apiStyle"]
    )
    if unsupported:
        raise click.UsageError(
            "This native provider form saves connection settings. Configure advanced model "
            "settings in llm_config.yaml; these options cannot be silently ignored."
        )
    view.note("Native LiteLLM provider · save connection settings without model calls.")
    prefill = {
        "model": state.model,
        "alias": state.alias,
        "api_base": state.endpoint,
        "api_key": ("env:" + state.api_key_env if state.api_key_env else "")
        if state.api_key_env is not None
        else None,
        "context_window": state.context_window,
        "max_tokens": state.reply_tokens,
    }
    values = {}
    for field in description["fields"]:
        name = field["id"]
        supplied = prefill.get(name)
        if supplied is not None:
            values[name] = str(supplied)
            continue
        if state.yes:
            values[name] = field["default"]
            continue
        view.note(field["hint"])
        values[name] = prompts.prompt(
            field["label"],
            default=field["default"],
            hide_input=field["secret"],
            suggestions=description["models"] if name == "model" else (),
            labels=description["modelLabels"] if name == "model" else None,
            searchable=name == "model",
            open_menu=name == "model",
        )
    alias, entry, _ = configuration.prepare(provider, values)
    view.fields(
        [("Name", alias), ("Model", entry["model_name"]), ("Save to", description["target"])]
        + [
            (field["label"], "[configured]" if field["secret"] else values[field["id"]])
            for field in description["fields"]
            if values[field["id"]] and field["id"] not in {"alias", "model"}
        ]
    )
    if not state.yes and not click.confirm(
        "Save this model and any new credentials?", default=True
    ):
        return
    result = configuration.save(provider, values, path=state.path)
    if result.get("warning"):
        view.note(result["warning"], err=True)
        return
    view.note(
        f"Saved {result['alias']}. Select it with /model in Aster. Model calls were not tested."
    )
