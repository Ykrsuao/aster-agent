# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise admission checks at the worker's actual client call boundaries."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from scripts import terminal_acceptance as acceptance


class RecordingClient:
    """An offline transport recording requests that get past worker admission."""

    model = "openai/acceptance-test"
    reasoning_level = None

    def __init__(self):
        self.requests = []
        self.active = 0
        self.peak = 0
        self.closed = 0
        self.fail_next = False
        self.async_body = None
        self.sync_body = None
        self._lock = threading.Lock()

    def _prepare_call_config(self, overrides):
        return {
            "model": self.model,
            "api_base": "https://model.example.test/v1",
            "temperature": 0,
            "max_tokens": 4096,
            **overrides,
        }

    def _enter(self, kwargs):
        with self._lock:
            self.requests.append(kwargs)
            self.active += 1
            self.peak = max(self.peak, self.active)

    def _leave(self):
        with self._lock:
            self.active -= 1

    def _response(self):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("offline transport failure")
        return SimpleNamespace(usage=None)

    async def acall(self, **kwargs):
        self._enter(kwargs)
        try:
            if self.async_body is not None:
                await self.async_body()
            return self._response()
        finally:
            self._leave()

    def call(self, **kwargs):
        self._enter(kwargs)
        try:
            if self.sync_body is not None:
                self.sync_body()
            return self._response()
        finally:
            self._leave()

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def run_worker(tmp_path, monkeypatch):
    from aster.interactive import Done

    monkeypatch.chdir(acceptance.ROOT)
    for name in ("ASTER_PROJECT_DIR", "ASTER_USER_DIR", "ASTER_SETTINGS"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setattr("aster.unifiedllm.registry.load_model_entries", lambda *a: {"measured": {}})
    monkeypatch.setattr("aster.llm_config.llm_config_chain", lambda: [])
    monkeypatch.setattr("aster.secrets.load_secrets_into_env", lambda: [])
    monkeypatch.setattr("aster.tracing.enable_tracing", lambda **kw: None)

    async def run(client, scenario, **options):
        class Agent:
            def __init__(self, *, llm, **kwargs):
                self.llm = llm

            async def handle(self, message):
                await scenario(self.llm)
                return Done(explanation="Offline call boundary checks completed")

            async def close(self):
                await self.llm.aclose()

        monkeypatch.setattr("aster_cli.coding.agent.CodingAgent", Agent)
        monkeypatch.setattr("aster.unifiedllm.get_llm_client", lambda *a, **kw: client)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        meter = tmp_path / "usage.json"
        await acceptance.worker(
            {
                "workspace": str(workspace),
                "meter": str(meter),
                "model": "measured",
                "prompt": "Exercise offline call boundaries",
                "max_calls": 8,
                "max_output_tokens": 4096,
                "expected_model_fingerprint": acceptance.fingerprint_model(
                    acceptance.model_identity(client)
                ),
                "serial_calls": True,
                **options,
            }
        )
        return json.loads(meter.read_text(encoding="utf-8"))

    return run


async def test_unapproved_initial_identity_never_starts_the_agent(run_worker):
    client = RecordingClient()

    async def unexpected(_):
        pytest.fail("An unapproved client must not start the agent")

    report = await run_worker(client, unexpected, expected_model_fingerprint="0" * 64)
    assert report["status"] == "worker_error"
    assert report["error"] == "ValueError"
    assert report["attempted_calls"] == 0
    assert client.requests == []
    assert client.closed == 1


@pytest.mark.parametrize("approved_calls", [0, 1])
@pytest.mark.parametrize("override", [{"temperature": 0.5}, {"model": "openai/other-model"}])
async def test_request_overrides_are_rejected_before_dispatch(run_worker, approved_calls, override):
    client = RecordingClient()

    async def scenario(llm):
        for _ in range(approved_calls):
            await llm.acall()
        await llm.acall(**override)

    report = await run_worker(client, scenario)
    assert report["status"] == "worker_error"
    assert report["error"] == "ValueError"
    assert report["attempted_calls"] == approved_calls
    assert len(client.requests) == approved_calls
    assert client.closed == 1


async def test_live_requirement_rejects_a_fake_client_before_any_request(run_worker):
    from aster.unifiedllm import FakeLLMClient

    client = FakeLLMClient([], strict_exhaustion=True)

    async def unexpected(_):
        pytest.fail("A FakeLLMClient cannot enter live acceptance")

    report = await run_worker(client, unexpected, require_live_model=True)
    assert report["measurement_kind"] == "offline-scripted"
    assert report["status"] == "worker_error"
    assert report["attempted_calls"] == 0
    assert report["completed_calls"] == 0


async def test_async_requests_share_one_transport_slot(run_worker):
    client = RecordingClient()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        entered.set()
        await release.wait()

    client.async_body = hold

    async def scenario(llm):
        first = asyncio.create_task(llm.acall())
        await entered.wait()
        second = asyncio.create_task(llm.acall())
        await asyncio.sleep(0)
        assert len(client.requests) == 1
        release.set()
        await asyncio.gather(first, second)

    report = await asyncio.wait_for(run_worker(client, scenario), timeout=5)
    assert report["status"] == "completed"
    assert report["attempted_calls"] == report["completed_calls"] == 2
    assert client.peak == 1
    assert client.active == 0


@pytest.mark.parametrize("cancel_count", [1, 3])
async def test_cancelled_caller_cannot_release_a_still_running_provider_slot(
    run_worker, cancel_count
):
    from aster.unifiedllm.unifiedllm import _run_async_provider_call

    client = RecordingClient()
    entered = asyncio.Event()
    release = asyncio.Event()
    providers = set()

    async def shielded_call(**kwargs):
        async def provider():
            client._enter(kwargs)
            current = asyncio.current_task()
            providers.add(current)
            entered.set()
            try:
                await release.wait()
                return SimpleNamespace(usage=None)
            finally:
                client._leave()
                providers.discard(current)

        return await _run_async_provider_call(provider, request_timeout=None)

    client.acall = shielded_call

    async def scenario(llm):
        first = asyncio.create_task(llm.acall())
        await entered.wait()
        for _ in range(cancel_count):
            first.cancel()
            # Give cancellation a chance to escape the client's shielded provider.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        second = asyncio.create_task(llm.acall())
        try:
            for _ in range(5):
                await asyncio.sleep(0)
            assert client.peak == 1
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            if providers:
                await asyncio.gather(*providers, return_exceptions=True)

    report = await asyncio.wait_for(run_worker(client, scenario), timeout=5)
    assert client.peak == 1
    assert report["status"] == "completed"
    assert client.active == 0


async def test_sync_call_during_async_request_fails_without_blocking_the_loop(run_worker):
    client = RecordingClient()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        entered.set()
        await release.wait()

    client.async_body = hold

    async def scenario(llm):
        request = asyncio.create_task(llm.acall())
        await entered.wait()
        try:
            llm.call()
        finally:
            release.set()
            await request

    report = await asyncio.wait_for(run_worker(client, scenario), timeout=5)
    assert report["status"] == "concurrent_call_rejected"
    assert report["attempted_calls"] == report["completed_calls"] == 1
    assert len(client.requests) == 1
    assert client.peak == 1
    assert client.active == 0


async def test_async_call_during_sync_request_rejects_and_releases_the_gate(run_worker):
    client = RecordingClient()
    entered = threading.Event()
    release = threading.Event()

    def hold():
        entered.set()
        assert release.wait(timeout=5), "The test must release the transport thread"

    client.sync_body = hold

    async def scenario(llm):
        request = asyncio.create_task(asyncio.to_thread(llm.call))
        await asyncio.to_thread(entered.wait, 5)
        try:
            await llm.acall()
        finally:
            release.set()
            await request

    report = await asyncio.wait_for(run_worker(client, scenario), timeout=5)
    assert report["status"] == "concurrent_call_rejected"
    assert report["attempted_calls"] == report["completed_calls"] == 1
    assert len(client.requests) == 1
    assert client.peak == 1
    assert client.active == 0


@pytest.mark.parametrize("method", ["call", "acall"])
async def test_transport_error_releases_the_shared_slot(run_worker, method):
    client = RecordingClient()
    client.fail_next = True

    async def invoke(llm):
        if method == "acall":
            return await llm.acall()
        return llm.call()

    async def scenario(llm):
        with pytest.raises(RuntimeError, match="offline transport failure"):
            await invoke(llm)
        await llm.acall()

    report = await asyncio.wait_for(run_worker(client, scenario), timeout=5)
    assert report["status"] == "completed"
    assert report["attempted_calls"] == 2
    assert report["completed_calls"] == 1
    assert len(client.requests) == 2
    assert client.peak == 1
    assert client.active == 0


async def test_ninth_request_cannot_exceed_the_predeclared_call_budget(run_worker):
    client = RecordingClient()

    async def scenario(llm):
        for _ in range(9):
            await llm.acall()

    report = await run_worker(client, scenario)
    assert report["status"] == "call_limit"
    assert report["attempted_calls"] == report["completed_calls"] == 8
    assert len(client.requests) == 8
    assert client.active == 0


@pytest.mark.parametrize("digest", ["not-a-digest", "A" * 64, "a" * 63, "a" * 65])
def test_cli_rejects_malformed_approval_before_dispatch(tmp_path, monkeypatch, digest):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid approved fingerprint must not dispatch a worker")

    monkeypatch.setattr(acceptance, "run_repeats", unexpected)
    with pytest.raises(SystemExit) as error:
        acceptance.main(
            [
                "--model",
                "measured",
                "--output",
                str(tmp_path / "output"),
                "--expected-model-fingerprint",
                digest,
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()
