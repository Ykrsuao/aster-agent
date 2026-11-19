# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless scoring waits for committed traces and releases its own database."""

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from aster.viewer import otlp_store
from eval_pipeline.headless_backend import HeadlessOtlpBackend, _make_headless_app


@pytest.mark.asyncio
async def test_sync_waits_for_the_batch_before_its_marker(monkeypatch):
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    committed = []

    def write(batch):
        if batch == [b"first"]:
            first_started.set()
            assert release_first.wait(5)
        else:
            second_started.set()
            assert release_second.wait(5)
        committed.extend(batch)

    monkeypatch.setattr(otlp_store, "init_db", lambda: None)
    monkeypatch.setattr(otlp_store, "ingest_batch_write_bytes", write)
    app = _make_headless_app()
    sync_endpoint = next(
        route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/sync"
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            try:
                assert (await client.post("/v1/traces", content=b"first")).status_code == 200
                assert await asyncio.to_thread(first_started.wait, 5)
                assert (await client.post("/v1/traces", content=b"second")).status_code == 200
                sync = asyncio.create_task(sync_endpoint())
                await asyncio.sleep(0)
                release_first.set()
                assert await asyncio.to_thread(second_started.wait, 5)
                await asyncio.sleep(0)
                assert not sync.done(), "sync returned before its preceding batch committed"
                release_second.set()
                assert json.loads((await asyncio.wait_for(sync, 5)).body) == {"synced": True}
                assert committed == [b"first", b"second"]
            finally:
                release_first.set()
                release_second.set()


def test_stop_closes_only_the_headless_database(tmp_path, monkeypatch):
    previous_path = otlp_store.DB_PATH
    previous_db = otlp_store._db
    live_path = tmp_path / "live-viewer.db"
    monkeypatch.setattr(otlp_store, "DB_PATH", live_path)
    otlp_store.init_db()
    live_db = otlp_store._get_db()
    backend = HeadlessOtlpBackend()
    try:
        endpoint = backend.start()
        temporary = Path(backend._tmpdir.name)
        schema_db = otlp_store._db
        with httpx.Client(base_url=endpoint) as client:
            assert client.post("/v1/traces", json={"resourceSpans": []}).status_code == 200
            assert client.post("/v1/sync").status_code == 200
            assert client.get("/api/traces").status_code == 200
        owned = [
            connection
            for connection, path in otlp_store._connections.items()
            if path == (temporary / "eval_traces.db").resolve()
        ]
        assert len(owned) >= 3, "exercise schema, writer and request-thread connections"
        backend.stop()
        assert not temporary.exists()
        for connection in owned:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        assert schema_db in owned
        assert otlp_store.DB_PATH == live_path
        assert live_db.execute("SELECT 1").fetchone()[0] == 1
        assert otlp_store._get_db() is live_db
        backend.stop()
    finally:
        backend.stop()
        otlp_store.close_db(live_path)
        otlp_store.DB_PATH = previous_path
        otlp_store._db = previous_db


def test_stop_preserves_database_while_server_is_alive(tmp_path):
    backend = HeadlessOtlpBackend()
    backend._server = SimpleNamespace(should_exit=False)
    backend._thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
    cleaned = []
    backend._tmpdir = SimpleNamespace(name=str(tmp_path), cleanup=lambda: cleaned.append(True))
    with pytest.raises(RuntimeError, match="database was preserved"):
        backend.stop()
    assert backend._server.should_exit
    assert not cleaned
