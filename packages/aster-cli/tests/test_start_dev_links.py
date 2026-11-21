# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup links must authorize the browser on an address the viewer binds."""

import logging
import sys
from types import ModuleType
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
import uvicorn
from aster_cli.commands import start_dev
from click.testing import CliRunner


@pytest.fixture
def viewer_start(monkeypatch, tmp_path):
    viewer_module = ModuleType("aster.viewer.main")
    viewer_module.app = object()
    monkeypatch.setitem(sys.modules, "aster.viewer.main", viewer_module)
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setenv("ASTER_TRACE_DB", str(tmp_path / "traces.db"))
    monkeypatch.setenv("ASTER_VIEWER_PUBLIC_HOST", "viewer.example")
    access_log = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(access_log, "filters", list(access_log.filters))
    return run


@pytest.mark.parametrize(
    ("bind_host", "url_host"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
        ("0.0.0.0", "127.0.0.1"),
        ("::", "::1"),
        ("localhost", "localhost"),
        ("192.0.2.10", "192.0.2.10"),
    ],
)
def test_token_bootstrap_matches_the_listener(viewer_start, monkeypatch, bind_host, url_host):
    token = "synthetic-token+/&# =?"
    monkeypatch.setenv("ASTER_VIEWER_AUTH_TOKEN", token)

    result = CliRunner().invoke(start_dev.command, ["--host", bind_host, "--port", "5002"])

    assert result.exit_code == 0, result.output
    local = next(
        line.strip()[len("URL:") :].strip()
        for line in result.output.splitlines()
        if line.strip().startswith("URL:")
    )
    parsed = urlsplit(local)
    assert parsed.hostname == url_host
    assert parsed.port == 5002
    assert parse_qs(parsed.query) == {"token": [token]}
    assert "  Share: http://viewer.example:5002/?token=" in result.output
    assert viewer_start.call_args.kwargs["host"] == bind_host


def test_default_listener_has_a_local_bootstrap_link(viewer_start, monkeypatch):
    monkeypatch.setenv("ASTER_VIEWER_AUTH_TOKEN", "synthetic-viewer-token")
    result = CliRunner().invoke(start_dev.command)

    assert result.exit_code == 0, result.output
    assert "URL:  http://127.0.0.1:5001/?token=synthetic-viewer-token" in result.output
    assert viewer_start.call_args.kwargs["host"] == "127.0.0.1"


def test_no_token_keeps_an_ordinary_local_url(viewer_start, monkeypatch):
    monkeypatch.delenv("ASTER_VIEWER_AUTH_TOKEN", raising=False)
    result = CliRunner().invoke(start_dev.command, ["--host", "::1"])

    assert result.exit_code == 0, result.output
    assert "URL:  http://[::1]:5001" in result.output
    assert "?token=" not in result.output
    assert "Share:" not in result.output


def test_ipv6_public_host_is_bracketed(viewer_start, monkeypatch):
    monkeypatch.setenv("ASTER_VIEWER_AUTH_TOKEN", "synthetic-viewer-token")
    monkeypatch.setenv("ASTER_VIEWER_PUBLIC_HOST", "2001:db8::1")
    result = CliRunner().invoke(start_dev.command, ["--host", "::"])

    assert result.exit_code == 0, result.output
    assert "Share: http://[2001:db8::1]:5001/?token=synthetic-viewer-token" in result.output
