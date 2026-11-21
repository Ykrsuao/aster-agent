# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for the ASTER ACP agent."""

import asyncio
import os
from typing import TYPE_CHECKING

import click

from aster_acp._startup import mark, phase
from aster_acp._stdio import diagnostic_stdout

if TYPE_CHECKING:
    from aster.unifiedllm import UnifiedLLM


@click.command()
@click.option(
    "--model",
    envvar="ASTER_MODEL",
    default=None,
    help="LiteLLM model name or configured ASTER model alias. Or set ASTER_MODEL.",
)
@click.option(
    "--select-model", is_flag=True, help="Choose a configured model before starting a session."
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured ASTER LLM client type.",
)
@click.option(
    "--sandbox/--no-sandbox",
    envvar="ASTER_SANDBOX",
    default=False,
    help="Execute CodeAct Python in a microsandbox microVM.",
)
def command(model: str | None, client_type: str | None, sandbox: bool, select_model: bool) -> None:
    """Serve the ASTER coding agent over ACP on standard input/output."""
    if model is None and not select_model:
        raise click.UsageError("Missing option '--model'.")
    # Imports and configuration may print provider diagnostics before serve()
    # has created the protocol connection. Keep those off the JSON-RPC pipe too.
    with diagnostic_stdout():
        with phase("framework_import"):
            from aster.secrets import load_secrets_into_env
        with phase("llm_import"):
            from aster.unifiedllm import get_llm_client
        with phase("acp_import"):
            from aster_acp.server import serve

        with phase("configuration"):
            # Opening a folder must not be enough to use the files inside it.
            from aster import workspace_trust

            workspace_trust.enforce()
            load_secrets_into_env()
            # Session startup must not wait for the SDK's remote pricing catalogue.
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    async def prepare_model() -> None:
        from aster_acp._provider_code import activate_provider_code_cache

        with phase("provider_code_cache"):
            await asyncio.to_thread(activate_provider_code_cache)

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

    def model_factory(selected_model: str) -> "UnifiedLLM":
        overrides = (
            {"api_key": nvidia_api_key}
            if nvidia_api_key and selected_model.startswith("nvidia_nim/")
            else {}
        )
        return get_llm_client(selected_model, client_type=client_type, **overrides)

    def llm_factory() -> "UnifiedLLM":
        if model is None:
            raise ValueError("Choose a model in Aster before starting a conversation.")
        return model_factory(model)

    operation = serve(
        llm_factory,
        model_factory=model_factory,
        model=model,
        sandbox=sandbox,
        prepare_model=prepare_model,
    )
    mark("server_start")
    asyncio.run(operation)


def main() -> None:
    command()
