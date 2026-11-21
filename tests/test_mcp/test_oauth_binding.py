# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached OAuth credentials must never follow changed discovery metadata."""

import json

import httpx
import pytest

from aster.mcp import oauth

SERVER_URL = "https://resource.example/mcp"
AUTH_SERVER = "https://auth.example"
TOKEN_ENDPOINT = f"{AUTH_SERVER}/token"


@pytest.fixture
def oauth_server(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path))
    metadata_url = "https://resource.example/.well-known/oauth-protected-resource/mcp"
    state = {
        "authorization_server": AUTH_SERVER,
        "token_endpoint": TOKEN_ENDPOINT,
        "resource": SERVER_URL,
        "grant_types_supported": ["authorization_code", "client_credentials"],
    }
    requests = []

    def transport(request):
        requests.append(request)
        url = str(request.url)
        if request.method == "POST" and url == SERVER_URL:
            return httpx.Response(
                401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'}
            )
        if url == metadata_url:
            return httpx.Response(
                200,
                json={
                    "resource": state["resource"],
                    "authorization_servers": [state["authorization_server"]],
                },
            )
        if url.endswith("/.well-known/oauth-authorization-server"):
            return httpx.Response(
                200,
                json={
                    "issuer": state.get("issuer", state["authorization_server"]),
                    "authorization_endpoint": f"{state['authorization_server']}/authorize",
                    "token_endpoint": state["token_endpoint"],
                    "grant_types_supported": state["grant_types_supported"],
                },
            )
        if request.method == "POST" and url == state["token_endpoint"]:
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-access-marker",
                    "refresh_token": "rotated-refresh-marker",
                    "expires_in": 3600,
                },
            )
        return httpx.Response(404)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        oauth.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(transport), **kwargs),
    )
    return state, requests


def _cached_token(*, expired=True):
    return oauth.OAuthToken(
        access_token="old-access-marker",
        refresh_token="old-refresh-marker",
        client_id="old-client-marker",
        client_secret="old-secret-marker",
        expires_in=3600,
        obtained_at=oauth.time.time() - (7200 if expired else 0),
        authorization_server=AUTH_SERVER,
        token_endpoint=TOKEN_ENDPOINT,
        resource=SERVER_URL,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize(
    "changed",
    ["authorization_server", "token_endpoint", "resource", "client_id", "client_secret", "legacy"],
)
async def test_changed_or_missing_binding_never_sends_existing_credentials(
    oauth_server, monkeypatch, changed, expired
):
    state, requests = oauth_server
    cached = _cached_token(expired=expired)
    oauth._save_cached_token(SERVER_URL, cached)
    client_config = {"client_id": cached.client_id, "client_secret": cached.client_secret}
    if changed in {"authorization_server", "token_endpoint", "resource"}:
        state[changed] = "https://changed.example/endpoint"
    elif changed in client_config:
        client_config[changed] = "changed-client-marker"
    else:
        path = oauth._token_cache_path()
        data = json.loads(path.read_text())
        for key in ("authorization_server", "token_endpoint", "resource"):
            data[SERVER_URL].pop(key)
        path.write_text(json.dumps(data))

    async def unexpected_auth(*args, **kwargs):
        pytest.fail("A changed binding must not fall through to registration or authorization")

    monkeypatch.setattr(oauth.OAuthHandler, "complete_flow", unexpected_auth)
    monkeypatch.setattr(oauth.OAuthHandler, "_register_dynamic_client", unexpected_auth)

    with pytest.raises(RuntimeError, match="OAuth credential binding is missing or changed") as err:
        await oauth.handle_mcp_oauth(SERVER_URL, **client_config, open_browser=False)

    assert requests
    for marker in (cached.access_token, cached.refresh_token, cached.client_secret):
        assert marker not in str(err.value)
        assert all(marker.encode() not in request.content for request in requests)
    assert not any(request.url == state["token_endpoint"] for request in requests)
    assert oauth._load_cached_token(SERVER_URL).access_token == cached.access_token


@pytest.mark.asyncio
async def test_same_binding_refreshes_and_persists_rotated_credentials(oauth_server):
    _, requests = oauth_server
    oauth._save_cached_token(SERVER_URL, _cached_token())

    token = await oauth.handle_mcp_oauth(SERVER_URL, open_browser=False)

    assert token.access_token == "fresh-access-marker"
    assert token.refresh_token == "rotated-refresh-marker"
    assert token.authorization_server == AUTH_SERVER
    assert token.token_endpoint == TOKEN_ENDPOINT
    assert token.resource == SERVER_URL
    assert oauth._load_cached_token(SERVER_URL) == token
    exchanges = [request for request in requests if request.url == TOKEN_ENDPOINT]
    assert len(exchanges) == 1
    assert dict(httpx.QueryParams(exchanges[0].content.decode())) == {
        "grant_type": "refresh_token",
        "client_id": "old-client-marker",
        "client_secret": "old-secret-marker",
        "refresh_token": "old-refresh-marker",
        "resource": SERVER_URL,
    }


@pytest.mark.asyncio
async def test_client_credentials_flow_persists_authorization_binding(oauth_server):
    token = await oauth.handle_mcp_oauth(
        SERVER_URL, client_id="new-client-marker", client_secret="new-secret-marker"
    )

    assert token.authorization_server == AUTH_SERVER
    assert token.token_endpoint == TOKEN_ENDPOINT
    assert token.resource == SERVER_URL
    assert oauth._load_cached_token(SERVER_URL) == token


@pytest.mark.asyncio
async def test_reauthorization_migrates_only_the_cleared_server(oauth_server):
    _, requests = oauth_server
    legacy = _cached_token()
    legacy.authorization_server = None
    legacy.token_endpoint = None
    oauth._save_cached_token(SERVER_URL, legacy)
    other_url = "https://another-resource.example/mcp"
    other_token = _cached_token(expired=False)
    oauth._save_cached_token(other_url, other_token)

    # The documented migration removes only the verified server's cache entry.
    path = oauth._token_cache_path()
    data = json.loads(path.read_text())
    data.pop(SERVER_URL, None)
    path.write_text(json.dumps(data, indent=2))

    token = await oauth.handle_mcp_oauth(
        SERVER_URL, client_id="new-client-marker", client_secret="new-secret-marker"
    )

    assert token.authorization_server == AUTH_SERVER
    assert token.token_endpoint == TOKEN_ENDPOINT
    assert oauth._load_cached_token(SERVER_URL) == token
    assert oauth._load_cached_token(other_url) == other_token
    assert all(b"old-refresh-marker" not in request.content for request in requests)
    assert all(b"old-secret-marker" not in request.content for request in requests)


@pytest.mark.asyncio
async def test_interactive_flow_persists_authorization_binding(oauth_server, monkeypatch):
    state, _ = oauth_server
    state["grant_types_supported"] = ["authorization_code"]

    async def complete_flow(self, open_browser=True):
        return oauth.OAuthToken(access_token="new-interactive-marker", client_id="new-client")

    monkeypatch.setattr(oauth.OAuthHandler, "complete_flow", complete_flow)
    token = await oauth.handle_mcp_oauth(SERVER_URL, open_browser=False)

    assert token.authorization_server == AUTH_SERVER
    assert token.token_endpoint == TOKEN_ENDPOINT
    assert token.resource == SERVER_URL
    assert oauth._load_cached_token(SERVER_URL) == token


@pytest.mark.asyncio
async def test_declared_issuer_must_match_selected_authorization_server(oauth_server):
    state, requests = oauth_server
    state["issuer"] = "https://different-issuer.example"
    oauth._save_cached_token(SERVER_URL, _cached_token())

    with pytest.raises(ValueError, match="mismatched authorization server issuer"):
        await oauth.handle_mcp_oauth(SERVER_URL, open_browser=False)

    assert not any(request.url == TOKEN_ENDPOINT for request in requests)


@pytest.mark.asyncio
async def test_matching_issuer_preserves_a_trailing_slash(oauth_server):
    state, _ = oauth_server
    state["authorization_server"] = AUTH_SERVER + "/"
    cached = _cached_token()
    cached.authorization_server = AUTH_SERVER + "/"
    oauth._save_cached_token(SERVER_URL, cached)

    token = await oauth.handle_mcp_oauth(SERVER_URL, open_browser=False)

    assert token.authorization_server == AUTH_SERVER + "/"
    assert token.access_token == "fresh-access-marker"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 42, ""])
async def test_invalid_authorization_server_is_rejected(oauth_server, value):
    state, requests = oauth_server
    state["authorization_server"] = value
    oauth._save_cached_token(SERVER_URL, _cached_token())

    with pytest.raises(ValueError, match="invalid authorization_servers"):
        await oauth.handle_mcp_oauth(SERVER_URL, open_browser=False)

    assert not any(request.url == TOKEN_ENDPOINT for request in requests)


@pytest.mark.asyncio
async def test_authorization_server_metadata_ignores_non_objects():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=["not-metadata"]))
    ) as client:
        assert await oauth._fetch_authorization_server_metadata(client, AUTH_SERVER) == {}
