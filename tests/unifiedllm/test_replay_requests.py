# SPDX-License-Identifier: Apache-2.0
import json
import sys

import httpx
import pytest

from aster.unifiedllm.replay_requests import main, replay_from_file, replay_request


def _entry(url):
    return {
        "request": {
            "url": url,
            "method": "POST",
            "headers": {"authorization": "***REDACTED***"},
            "body": {"model": "test"},
        },
        "response": {"status_code": 500},
    }


@pytest.fixture
def requests(monkeypatch):
    sent = []

    async def send(_self, request):
        sent.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    return sent


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/chat",
        "https://api.openai.com.example.invalid/chat",
        "https://example.invalid/api.openai.com/chat",
        "http://api.openai.com/chat",
        "https://api.openai.com:8443/chat",
    ],
)
async def test_replay_does_not_guess_credentials_for_other_origins(url, requests):
    success, result = await replay_request(_entry(url), verbose=False)
    assert not success
    assert "credential" in result["error"]
    assert requests == []


@pytest.mark.parametrize("header", ["authorization", "Authorization"])
async def test_replay_restores_known_https_origin_key(requests, header):
    entry = _entry("https://api.openai.com/v1/chat/completions")
    entry["request"]["headers"] = {header: "***REDACTED***"}
    success, _ = await replay_request(entry, verbose=False)
    assert success
    assert requests[0].headers["authorization"] == "Bearer synthetic-openai-key"


async def test_explicit_credential_stays_bound_to_origin(tmp_path, requests, monkeypatch):
    monkeypatch.setenv("ASTER_TEST_REPLAY_KEY", "synthetic-custom-key")
    entries = [_entry("https://gateway.example/chat"), _entry("https://other.example/chat")]
    path = tmp_path / "errors.jsonl"
    path.write_text("\n".join(map(json.dumps, entries)), encoding="utf-8")
    result = await replay_from_file(
        path,
        delay_seconds=0,
        verbose=False,
        credentials={"https://gateway.example": "ASTER_TEST_REPLAY_KEY"},
    )
    assert result["succeeded"] == 1 and result["failed"] == 1
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer synthetic-custom-key"


async def test_replay_errors_do_not_disclose_request_credentials(requests, monkeypatch, capsys):
    async def fail(_self, request):
        raise httpx.ConnectError("failed " + str(request.url), request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)
    success, result = await replay_request(
        _entry("https://api.openai.com/chat?api_key=synthetic-query-key")
    )
    assert not success
    assert "ConnectError" in result["error"]
    assert "synthetic-query-key" not in capsys.readouterr().out + str(result)


async def test_replay_cli_accepts_origin_bound_gateway_credential(tmp_path, requests, monkeypatch):
    path = tmp_path / "errors.jsonl"
    path.write_text(json.dumps(_entry("https://gateway.example/chat")), encoding="utf-8")
    monkeypatch.setenv("ASTER_TEST_REPLAY_KEY", "synthetic-custom-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_requests",
            str(path),
            "--quiet",
            "--credential",
            "https://gateway.example:443/=ASTER_TEST_REPLAY_KEY",
        ],
    )
    await main()
    assert requests[0].headers["authorization"] == "Bearer synthetic-custom-key"
