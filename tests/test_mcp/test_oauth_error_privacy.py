# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OAuth errors must not echo credentials from untrusted response bodies."""

import asyncio
import io
import logging
import traceback
from types import SimpleNamespace
from urllib.parse import urlencode

import httpx
import pytest

from aster.mcp import oauth

CLIENT_SECRET = "private-client-secret-marker"
AUTH_CODE = "private-authorization-code-marker"
REFRESH_TOKEN = "private-refresh-token-marker"
MARKERS = (CLIENT_SECRET, AUTH_CODE, REFRESH_TOKEN)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase", ["client_credentials", "authorization_code", "registration", "refresh"]
)
@pytest.mark.parametrize(
    "failure", ["http_status", "invalid_json", "missing_field", "non_object", "transport"]
)
async def test_oauth_error_responses_do_not_echo_credentials(monkeypatch, caplog, phase, failure):
    caplog.set_level(logging.INFO, logger=oauth.__name__)
    requests = []

    def transport(request):
        requests.append(request)
        echo = " ".join(MARKERS)
        if failure == "transport":
            raise httpx.RemoteProtocolError(echo, request=request)
        if failure == "http_status":
            return httpx.Response(
                400,
                json={"error": "invalid_grant", "error_description": echo},
                extensions={"reason_phrase": echo.encode()},
            )
        if failure == "invalid_json":
            return httpx.Response(200, text=echo)
        if failure == "non_object":
            return httpx.Response(200, json=list(MARKERS))
        return httpx.Response(200, json={"error_description": echo, "client_secret": CLIENT_SECRET})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        oauth.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(transport), **kwargs),
    )
    config = oauth.OAuthConfig(
        authorization_endpoint="https://auth.example/authorize",
        token_endpoint="https://auth.example/token",
        client_id=None if phase == "registration" else "synthetic-client-id",
        client_secret=CLIENT_SECRET,
        registration_endpoint="https://auth.example/register",
        redirect_uri="http://localhost:0/callback",
    )
    handler = oauth.OAuthHandler(config)
    handler._code_verifier = "synthetic-pkce-verifier"
    error_text = ""
    if phase == "refresh":
        token = await oauth._refresh_access_token(
            config.token_endpoint, config.client_id, REFRESH_TOKEN, CLIENT_SECRET
        )
        assert token is None
    else:
        with pytest.raises(RuntimeError) as caught:
            if phase == "client_credentials":
                await handler.client_credentials_token()
            elif phase == "authorization_code":
                await handler.exchange_code_for_token(AUTH_CODE)
            else:
                await handler._register_dynamic_client(config.redirect_uri)
        error_text = "".join(traceback.format_exception(caught.value))
        if failure != "transport":
            assert f"HTTP {400 if failure == 'http_status' else 200}" in str(caught.value)
    assert len(requests) == 1
    for marker in MARKERS:
        assert marker not in error_text
        assert marker not in caplog.text


@pytest.mark.asyncio
async def test_callback_error_does_not_echo_server_supplied_text(monkeypatch):
    body = io.BytesIO()
    captured = {}
    config = oauth.OAuthConfig(
        authorization_endpoint="https://auth.example/authorize",
        token_endpoint="https://auth.example/token",
        client_id="synthetic-client-id",
        redirect_uri="http://localhost:0/callback",
        timeout=1,
    )
    handler = oauth.OAuthHandler(config)

    class Server:
        server_address = ("localhost", 12345)

        def __init__(self, address, request_handler):
            captured["request_handler"] = request_handler

        def server_close(self):
            pass

    class Worker:
        ident = None

        def __init__(self, **kwargs):
            pass

        def start(self):
            query = urlencode({"state": handler._authorization_state, "error": AUTH_CODE})
            request = SimpleNamespace(
                path=f"/callback?{query}",
                wfile=body,
                send_response=lambda status: None,
                send_header=lambda name, value: None,
                end_headers=lambda: None,
            )
            asyncio.get_running_loop().call_soon(captured["request_handler"].do_GET, request)

    monkeypatch.setattr(oauth, "HTTPServer", Server)
    monkeypatch.setattr(oauth, "Thread", Worker)

    with pytest.raises(RuntimeError, match="authorization server rejected the request") as caught:
        await handler._capture_code_via_local_server(open_browser=False)

    assert AUTH_CODE not in str(caught.value)
    assert AUTH_CODE.encode() not in body.getvalue()
