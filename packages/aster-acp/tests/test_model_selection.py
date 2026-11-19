# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sys

import pytest
from acp import RequestError
from aster_acp.server import CodingACPAdapter


def unused_factory():
    pytest.fail("Listing or selecting saved models must not create an LLM client")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["providers", "form", "save"])
async def test_model_configuration_endpoints_are_unavailable(method):
    adapter = CodingACPAdapter(unused_factory)
    with pytest.raises(RequestError) as error:
        await adapter.ext_method(f"aster/models/{method}", {})
    assert error.value.code == -32601


@pytest.mark.asyncio
async def test_model_list_only_reads_saved_chat_models(tmp_path, monkeypatch):
    from aster import llm_config

    path = tmp_path / "llm_config.yaml"
    content = (
        "models:\n"
        "  deepseek-flash:\n"
        "    model_name: deepseek/deepseek-flash\n"
        "    api_key_env: PRIVATE_KEY_REFERENCE\n"
        "  removed: null\n"
        "  decision-only:\n"
        "    client_type: decision\n"
    )
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [path])
    monkeypatch.setattr(llm_config, "bundled_config_paths", lambda: [])

    adapter = CodingACPAdapter(unused_factory)
    result = await adapter.ext_method("aster/models/list", {})
    assert result == {"models": [{"id": "deepseek-flash", "name": "deepseek-flash"}]}
    assert path.read_text(encoding="utf-8") == content
    with pytest.raises(RequestError) as error:
        await adapter.ext_method("aster/models/select", {"model": "removed"})
    assert error.value.code == -32602
    assert await adapter.ext_method("aster/models/select", {"model": "deepseek-flash"}) == {
        "model": "deepseek-flash"
    }
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_stdio_remembers_an_existing_model_without_rewriting_its_connection(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    model_path = user / "llm_config.yaml"
    model_path.write_text(
        "models:\n  stdio-test:\n"
        "    model_name: openai/gpt-4o-mini\n"
        "    api_base: http://127.0.0.1:1\n"
        "    api_key_env: TEST_MODEL_KEY\n"
        "    max_tokens: 4096\n",
        encoding="utf-8",
    )
    secrets_path = user / "secrets.yaml"
    secrets_path.write_text("TEST_MODEL_KEY: synthetic-stdio-key\n", encoding="utf-8")
    before = (model_path.read_bytes(), secrets_path.read_bytes())
    environment = dict(os.environ)
    for name in ("ASTER_MODEL", "ASTER_LLM_CONFIG", "ASTER_SECRETS", "ASTER_SETTINGS"):
        environment.pop(name, None)
    environment.update(
        ASTER_USER_DIR=str(user),
        ASTER_PROJECT_DIR=str(tmp_path / "project"),
        ASTER_SESSIONS_DIR=str(tmp_path / "sessions"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONIOENCODING="utf-8",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-m",
        "aster_acp",
        "--select-model",
        "--no-sandbox",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=environment,
        cwd=workspace,
    )

    async def call(number, method, params):
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            (
                json.dumps({"jsonrpc": "2.0", "id": number, "method": method, "params": params})
                + "\n"
            ).encode()
        )
        await process.stdin.drain()
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
            assert line, "ACP closed before sending a response"
            response = json.loads(line)
            if response.get("id") == number:
                assert "error" not in response, response.get("error")
                return response["result"]

    try:
        initialized = await call(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        assert initialized["_meta"]["aster"]["modelSelection"] is True
        assert "modelConfiguration" not in initialized["_meta"]["aster"]
        assert await call(2, "_aster/models/list", {}) == {
            "models": [{"id": "stdio-test", "name": "stdio-test"}]
        }
        selected = await call(3, "_aster/models/select", {"model": "stdio-test"})
        assert selected["model"] == "stdio-test"
        settings = tmp_path / "project" / "settings.yaml"
        assert not settings.exists()
        session = await call(4, "session/new", {"cwd": str(workspace), "mcpServers": []})
        assert session["sessionId"]
        await call(5, "session/close", {"sessionId": session["sessionId"]})
        import yaml

        assert yaml.safe_load(settings.read_text(encoding="utf-8"))["tui"] == {
            "default_model": "stdio-test"
        }
        assert (model_path.read_bytes(), secrets_path.read_bytes()) == before
    finally:
        if process.returncode is None:
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=10)
