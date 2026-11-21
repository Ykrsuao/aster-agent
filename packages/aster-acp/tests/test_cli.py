# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model selection for the ACP entry point."""

import os
import shutil
import subprocess
from unittest.mock import AsyncMock, Mock

import click.testing
import pytest
from aster_acp.cli import command


@pytest.fixture
def stubbed_serve(monkeypatch):
    """Capture the llm_factory the command builds instead of serving."""
    captured = {}

    def fake_serve(llm_factory, *, model_factory=None, model=None, prepare_model=None, **options):
        captured["llm_factory"] = llm_factory
        captured["model_factory"] = model_factory
        captured["model"] = model
        captured["options"] = options
        return "coroutine-placeholder"

    monkeypatch.setattr("aster_acp.server.serve", fake_serve)
    monkeypatch.setattr("aster_acp.cli.asyncio.run", lambda coro: coro)
    monkeypatch.setattr("aster.secrets.load_secrets_into_env", lambda *a, **k: None)
    return captured


def test_model_is_required(monkeypatch):
    monkeypatch.delenv("ASTER_MODEL", raising=False)

    result = click.testing.CliRunner().invoke(command, [])

    # No default model: the caller has to choose one.
    assert result.exit_code == 2
    assert "--model" in result.output


def test_model_selection_mode_serves_before_a_model_is_selected(monkeypatch, stubbed_serve):
    monkeypatch.delenv("ASTER_MODEL", raising=False)
    result = click.testing.CliRunner().invoke(command, ["--select-model"])
    assert result.exit_code == 0, result.output
    assert stubbed_serve["model"] is None
    with pytest.raises(ValueError, match="Choose a model"):
        stubbed_serve["llm_factory"]()


def test_model_is_read_from_the_environment(monkeypatch, stubbed_serve):
    monkeypatch.setenv("ASTER_MODEL", "openai/gpt-4o-mini")
    requested = {}
    monkeypatch.setattr(
        "aster.unifiedllm.get_llm_client",
        lambda name, **kwargs: requested.setdefault("name", name),
    )

    result = click.testing.CliRunner().invoke(command, [])

    assert result.exit_code == 0, result.output
    stubbed_serve["llm_factory"]()
    assert requested["name"] == "openai/gpt-4o-mini"
    assert stubbed_serve["model"] == "openai/gpt-4o-mini"


def test_explicit_flag_overrides_the_environment(monkeypatch, stubbed_serve):
    monkeypatch.setenv("ASTER_MODEL", "openai/gpt-4o-mini")
    requested = {}
    monkeypatch.setattr(
        "aster.unifiedllm.get_llm_client",
        lambda name, **kwargs: requested.setdefault("name", name),
    )

    result = click.testing.CliRunner().invoke(command, ["--model", "anthropic/claude-sonnet-4-5"])

    assert result.exit_code == 0, result.output
    stubbed_serve["llm_factory"]()
    assert requested["name"] == "anthropic/claude-sonnet-4-5"
    assert stubbed_serve["model"] == "anthropic/claude-sonnet-4-5"


def test_sandbox_flag_reaches_server(monkeypatch, stubbed_serve):
    monkeypatch.delenv("ASTER_SANDBOX", raising=False)
    result = click.testing.CliRunner().invoke(command, ["--model", "fake-model", "--sandbox"])
    assert result.exit_code == 0, result.output
    assert stubbed_serve["options"] == {"sandbox": True}


def test_no_sandbox_overrides_environment(monkeypatch, stubbed_serve):
    monkeypatch.setenv("ASTER_SANDBOX", "true")
    result = click.testing.CliRunner().invoke(command, ["--model", "fake-model", "--no-sandbox"])
    assert result.exit_code == 0, result.output
    assert stubbed_serve["options"] == {"sandbox": False}


@pytest.mark.parametrize("client_type", [None, "completion", "responses"])
@pytest.mark.parametrize("selected_model", ["saved-model", "nvidia_nim/test-model"])
def test_selected_model_preserves_cli_client_overrides(
    monkeypatch: pytest.MonkeyPatch, client_type: str | None, selected_model: str
) -> None:
    from aster_acp.server import CodingACPAdapter

    monkeypatch.delenv("ASTER_MODEL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr("aster.secrets.load_secrets_into_env", lambda: None)
    monkeypatch.setattr(
        "aster.unifiedllm.connect.configuration.saved_models",
        lambda: [{"id": selected_model, "name": selected_model}],
    )
    get_llm_client = Mock()
    monkeypatch.setattr("aster.unifiedllm.get_llm_client", get_llm_client)
    monkeypatch.setattr("aster_acp.server.protocol_streams", AsyncMock(return_value=(None, None)))

    async def select_model(adapter: CodingACPAdapter, **kwargs: object) -> None:
        await adapter.ext_method("aster/models/select", {"model": selected_model})
        get_llm_client.assert_not_called()
        adapter._llm_factory()

    monkeypatch.setattr("aster_acp.server.run_agent", select_model)
    arguments = ["--select-model", "--no-sandbox"]
    if client_type is not None:
        arguments.extend(["--client-type", client_type])
    result = click.testing.CliRunner().invoke(command, arguments)

    assert result.exit_code == 0, result.output
    overrides = {"api_key": "nvapi-test"} if selected_model.startswith("nvidia_nim/") else {}
    get_llm_client.assert_called_once_with(selected_model, client_type=client_type, **overrides)


def test_provider_preparation_follows_handshake_and_survives_cancel(tmp_path, monkeypatch):
    import asyncio
    import threading

    from aster.unifiedllm import FakeLLMClient

    started = threading.Event()
    release = threading.Event()
    preparation_threads = []
    model_threads = []
    clients = []

    def prepare():
        preparation_threads.append(threading.get_ident())
        started.set()
        assert release.wait(3), "Provider preparation blocked the protocol handshake"

    def model(*args, **kwargs):
        assert release.is_set()
        model_threads.append(threading.get_ident())
        client = FakeLLMClient()
        clients.append(client)
        return client

    async def exercise(adapter, **kwargs):
        event_loop_thread = threading.get_ident()
        adapter.on_connect(Mock(session_update=AsyncMock()))
        assert not started.is_set()
        await adapter.initialize(1)
        cancelled = asyncio.create_task(adapter.new_session(str(tmp_path)))
        assert await asyncio.to_thread(started.wait, 1)
        try:
            # Even while disk preparation blocks, protocol requests remain responsive.
            await asyncio.wait_for(adapter.initialize(1), timeout=0.5)
            concurrent = asyncio.create_task(adapter.new_session(str(tmp_path)))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            assert not clients
            release.set()
            created = await asyncio.wait_for(concurrent, timeout=3)
            assert preparation_threads and preparation_threads != [event_loop_thread]
            assert model_threads == [event_loop_thread]
            await adapter.close_session(created.session_id)
            restored = await adapter.load_session(str(tmp_path), created.session_id)
            assert restored.modes is not None
            assert len(preparation_threads) == 1
            assert model_threads == [event_loop_thread, event_loop_thread]
        finally:
            release.set()
            await adapter.close()

    monkeypatch.setattr("aster_acp._provider_code.activate_provider_code_cache", prepare)
    monkeypatch.setattr("aster.secrets.load_secrets_into_env", lambda: None)
    monkeypatch.setattr("aster.unifiedllm.get_llm_client", model)
    monkeypatch.setattr("aster_acp.server.protocol_streams", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr("aster_acp.server.run_agent", exercise)
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / ".aster"))
    result = click.testing.CliRunner().invoke(command, ["--model", "test-model"])
    assert result.exit_code == 0, (result.output, result.exception)


async def test_provider_preparation_failure_is_reported_before_creating_a_model(tmp_path):
    from aster_acp.server import CodingACPAdapter

    prepare = AsyncMock(side_effect=OSError("cache preparation failed"))
    factory = Mock()
    adapter = CodingACPAdapter(factory, prepare_model=prepare)
    try:
        await adapter.initialize(1)
        for _ in range(2):
            with pytest.raises(OSError, match="cache preparation failed"):
                await adapter.new_session(str(tmp_path))
        prepare.assert_awaited_once()
        factory.assert_not_called()
    finally:
        await adapter.close()


async def test_close_releases_preparation_after_session_cancellation(tmp_path):
    import asyncio

    from aster_acp.server import CodingACPAdapter

    started = asyncio.Event()
    released = asyncio.Event()

    async def prepare():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    adapter = CodingACPAdapter(Mock(), prepare_model=prepare)
    request = asyncio.create_task(adapter.new_session(str(tmp_path)))
    await asyncio.wait_for(started.wait(), timeout=1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert not released.is_set()
    await asyncio.wait_for(adapter.close(), timeout=1)
    assert released.is_set()
    assert adapter._model_preparation.done()


def _console_script() -> str:
    """Locate the installed ``aster-acp`` console script.

    Deliberately fails rather than skips. Every other test in this package
    imports ``aster_acp``, so the package is always installed when these run; a
    missing script means the ``[project.scripts]`` entry is broken, which is
    exactly the breakage this test exists to catch.
    """
    path = shutil.which("aster-acp")
    assert path is not None, "aster-acp console script is not installed — check [project.scripts]"
    return path


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("ASTER_MODEL", None)
    return env


def test_console_script_is_installed_and_runnable():
    # Covers the [project.scripts] -> aster_acp.cli:main binding, which the
    # in-process CliRunner tests and the fake_agent fixture both bypass.
    result = subprocess.run(
        [_console_script(), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )

    assert result.returncode == 0, result.stderr
    assert "Serve the ASTER coding agent over ACP" in result.stdout


def test_console_script_requires_a_model():
    # Also proves main() reaches the click command: without the wiring this
    # would not produce a usage error.
    result = subprocess.run(
        [_console_script()],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )

    assert result.returncode == 2
    assert "--model" in result.stderr
