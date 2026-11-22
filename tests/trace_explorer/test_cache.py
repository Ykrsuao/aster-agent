# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TraceExplorer server-side caching."""

import asyncio
import sqlite3
import threading
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_otlp_spans():
    """Minimal OTLP spans fixture."""
    return [
        {
            "traceId": "trace001",
            "spanId": "aabbccdd11223344",
            "name": "TestAgent.handle",
            "kind": 1,
            "startTimeUnixNano": "1000000000",
            "endTimeUnixNano": "2000000000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "agent.name", "value": {"stringValue": "TestAgent"}},
                {"key": "agent.method", "value": {"stringValue": "handle"}},
                {"key": "agent.call_id", "value": {"stringValue": "call_001"}},
            ],
            "status": {"code": 1},
            "events": [],
            "_resource": {},
        }
    ]


@pytest.fixture
def app():
    from fastapi import FastAPI

    from aster.viewer.explorer_routes import router

    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the explorer cache before each test."""
    from aster.viewer.explorer_routes import clear_explorer_cache

    clear_explorer_cache()
    yield
    clear_explorer_cache()


@pytest.fixture
def mock_store():
    with patch("aster.viewer.explorer_routes.otlp_store") as m:
        m.session_exists.return_value = True
        m.get_session_spans.return_value = _make_otlp_spans()
        yield m


@pytest.fixture
def real_store(tmp_path, monkeypatch):
    from aster.viewer import otlp_store

    path = tmp_path / "traces.db"
    monkeypatch.setattr(otlp_store, "DB_PATH", path)
    otlp_store.init_db()
    try:
        yield otlp_store
    finally:
        otlp_store.close_db(path)


def _payload(spans, session_id="sess-1"):
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [{"key": "session.id", "value": {"stringValue": session_id}}]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


class TestExplorerCache:
    """Tests for LRU caching of TraceExplorer instances."""

    @pytest.mark.asyncio
    async def test_repeated_calls_reuse_cached_explorer(self, app, mock_store):
        """Same session_id should only build TraceExplorer once."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/api/explorer/overview", params={"session_id": "sess-1"})
            await client.get("/api/explorer/errors", params={"session_id": "sess-1"})

        # get_session_spans should only be called once (cached on second call)
        assert mock_store.get_session_spans.call_count == 1

    @pytest.mark.asyncio
    async def test_different_sessions_build_separate_explorers(self, app, mock_store):
        """Different session_ids should each build their own explorer."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/api/explorer/overview", params={"session_id": "sess-1"})
            await client.get("/api/explorer/overview", params={"session_id": "sess-2"})

        assert mock_store.get_session_spans.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_has_max_size(self, app, mock_store):
        """Cache should evict oldest entries when full."""
        from aster.viewer.explorer_routes import _explorer_cache

        # Fill cache beyond max size, verify it doesn't grow unbounded
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for i in range(20):
                await client.get("/api/explorer/overview", params={"session_id": f"sess-{i}"})

        assert len(_explorer_cache) <= 16  # default max size

    @pytest.mark.asyncio
    async def test_cache_clear(self, app, mock_store):
        """Cache should be clearable."""
        from aster.viewer.explorer_routes import clear_explorer_cache

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/api/explorer/overview", params={"session_id": "sess-1"})

        clear_explorer_cache()

        from aster.viewer.explorer_routes import _explorer_cache

        assert len(_explorer_cache) == 0


@pytest.mark.asyncio
async def test_live_errors_include_spans_ingested_after_first_read(app, real_store):
    real_store.ingest(_payload(_make_otlp_spans()))
    params = {"session_id": "sess-1"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        before = await client.get("/api/explorer/errors", params=params)
        assert before.status_code == 200
        assert before.json()["result"] == "No errors found in trace."

        failure = _make_otlp_spans()[0]
        failure["spanId"] = "1122334455667788"
        failure["name"] = "TestAgent.failed"
        failure["attributes"][-1]["value"]["stringValue"] = "call_002"
        failure["status"] = {"code": 2, "message": "NEW_ERROR_AFTER_FIRST_READ"}
        real_store.ingest(_payload([failure]))

        after = await client.get("/api/explorer/errors", params=params)
        assert after.status_code == 200
        assert "Found 1 error(s):" in after.json()["result"]


@pytest.mark.asyncio
async def test_refresh_discards_an_unchanged_explorer_snapshot(app, real_store):
    from aster.viewer.main import refresh_all

    app.add_api_route("/api/refresh", refresh_all, methods=["POST"])
    real_store.ingest(_payload(_make_otlp_spans()))
    with patch.object(real_store, "get_session_spans", wraps=real_store.get_session_spans) as load:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            params = {"session_id": "sess-1"}
            assert (await client.get("/api/explorer/overview", params=params)).status_code == 200
            assert (await client.get("/api/explorer/overview", params=params)).status_code == 200
            assert load.call_count == 1
            assert (await client.post("/api/refresh")).status_code == 200
            assert (await client.get("/api/explorer/overview", params=params)).status_code == 200
            assert load.call_count == 2


@pytest.mark.asyncio
async def test_in_place_span_update_from_another_connection_is_visible(app, real_store):
    real_store.ingest(_payload(_make_otlp_spans()))
    params = {"session_id": "sess-1"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        before = await client.get("/api/explorer/errors", params=params)
        assert before.json()["result"] == "No errors found in trace."
        with sqlite3.connect(real_store.DB_PATH) as writer:
            writer.execute(
                "UPDATE spans SET status_code = 2 WHERE span_id = ?",
                ("aabbccdd11223344",),
            )
        after = await client.get("/api/explorer/errors", params=params)
        assert after.status_code == 200
        assert "Found 1 error(s):" in after.json()["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("deletion", ["single", "all", "batch"])
async def test_deleted_trace_is_not_served_from_cache(app, real_store, deletion):
    from aster.viewer.trace_routes import router

    app.include_router(router)
    payload = _payload(_make_otlp_spans())
    payload["resourceSpans"][0]["resource"]["attributes"].append(
        {"key": "batch_id", "value": {"stringValue": "batch-1"}}
    )
    real_store.ingest(payload)
    params = {"session_id": "sess-1"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/explorer/overview", params=params)).status_code == 200
        if deletion == "single":
            deleted = await client.delete("/api/traces/sess-1")
        else:
            query = {"confirm": True} if deletion == "all" else {"batch_id": "batch-1"}
            deleted = await client.delete("/api/traces", params=query)
        assert deleted.status_code in (200, 204)
        assert (await client.get("/api/explorer/overview", params=params)).status_code == 404


def _journal_spans():
    agent = _make_otlp_spans()[0]
    generation = {
        **agent,
        "spanId": "gen001",
        "parentSpanId": agent["spanId"],
        "name": "generation",
        "attributes": [
            *agent["attributes"],
            {"key": "generation.id", "value": {"stringValue": "gen001abcdef"}},
        ],
    }
    llm = {
        **agent,
        "spanId": "llm001",
        "parentSpanId": generation["spanId"],
        "name": "acompletion",
        "attributes": [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}],
    }
    return [agent, generation, llm]


@pytest.mark.asyncio
@pytest.mark.parametrize("update", ["call", "block"])
async def test_journal_updates_refresh_cached_message_content(app, real_store, update):
    real_store.ingest(_payload(_journal_spans()))
    output = {"role": "assistant", "content": "ORIGINAL_ANSWER"}
    if update == "block":
        output = {"role": "assistant", "parts": [{"block_hash": "late-block"}]}
    call = {
        "call_id": "call-1",
        "session_id": "sess-1",
        "span_id": "llm001",
        "input_skeleton": [{"role": "user", "content": "Question"}],
        "output_messages": [output],
    }
    real_store.ingest_journal_call(call)
    params = {"session_id": "sess-1", "target_session_id": "aabbcc"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        before = await client.get("/api/explorer/session", params=params)
        assert before.status_code == 200
        expected = "ORIGINAL_ANSWER" if update == "call" else "<missing block: late-block>"
        assert expected in before.json()["result"]
        if update == "call":
            call["output_messages"] = [{"role": "assistant", "content": "UPDATED_ANSWER"}]
            real_store.ingest_journal_call(call)
        else:
            real_store.ingest_journal_blocks(
                "sess-1", [{"hash": "late-block", "content": "UPDATED_ANSWER"}]
            )
        after = await client.get("/api/explorer/session", params=params)
        assert after.status_code == 200
        assert "UPDATED_ANSWER" in after.json()["result"]
        assert expected not in after.json()["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["append", "delete"])
async def test_write_during_build_cannot_cache_an_old_snapshot(app, real_store, mutation):
    from aster.trace_explorer.explorer import TraceExplorer

    real_store.ingest(_payload(_make_otlp_spans()))
    loaded = threading.Event()
    resume = threading.Event()
    build = TraceExplorer.from_otlp_spans

    def blocked_build(*args, **kwargs):
        loaded.set()
        assert resume.wait(5), "test did not release the paused explorer build"
        return build(*args, **kwargs)

    params = {"session_id": "sess-1"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch.object(TraceExplorer, "from_otlp_spans", side_effect=blocked_build):
            request = asyncio.create_task(client.get("/api/explorer/errors", params=params))
            try:
                assert await asyncio.to_thread(loaded.wait, 5)
                if mutation == "append":
                    failure = _make_otlp_spans()[0]
                    failure["spanId"] = "1122334455667788"
                    failure["attributes"][-1]["value"]["stringValue"] = "call_002"
                    failure["status"] = {"code": 2}
                    real_store.ingest(_payload([failure]))
                else:
                    real_store.delete_session("sess-1")
            finally:
                resume.set()
                await request
        latest = await client.get("/api/explorer/errors", params=params)
        if mutation == "append":
            assert latest.status_code == 200
            assert "Found 1 error(s):" in latest.json()["result"]
        else:
            assert latest.status_code == 404


def test_reopened_database_cannot_reuse_another_databases_snapshot(
    app, real_store, tmp_path, monkeypatch
):
    async def errors():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/api/explorer/errors", params={"session_id": "sess-1"})

    real_store.ingest(_payload(_make_otlp_spans()))
    before = asyncio.run(errors())
    assert before.json()["result"] == "No errors found in trace."
    # Stop the first request's worker pool before closing its database connections.
    real_store.close_db(real_store.DB_PATH)
    replacement = tmp_path / "replacement.db"
    monkeypatch.setattr(real_store, "DB_PATH", replacement)
    real_store.init_db()
    try:
        failure = _make_otlp_spans()[0]
        failure["status"] = {"code": 2}
        real_store.ingest(_payload([failure]))
        after = asyncio.run(errors())
        assert after.status_code == 200
        assert "Found 1 error(s):" in after.json()["result"]
    finally:
        real_store.close_db(replacement)
