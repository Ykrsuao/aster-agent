# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permission changes preserve paths exposed by a real Linux microVM."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from acp import text_block
from acp.schema import RequestPermissionResponse
from aster_acp.server import CodingACPAdapter

from aster.events import ExecutionResult
from aster.runtime.hooks import InstrumentationHooks, hooks_scope
from aster.runtime.sandbox.config import FileRule, SandboxConfig
from aster.skill_registry import SkillRegistry
from aster.tools.shell_tools import Match, ShellTools
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall, registry

pytestmark = pytest.mark.sandbox


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user-config"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project-config"))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_SETTINGS", "ASTER_MODEL", "ASTER_LLM_CONFIG", "ASTER_RICH_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {"alpha": {"model_name": "test-alpha"}})


@pytest.fixture
def guarded_host_paths(tmp_path, monkeypatch):
    """Block a wrong guest-to-host target before any host file content is accessed."""
    root = tmp_path.resolve()
    blocked = []
    original_resolve = ShellTools._resolve_path

    def resolve(shell, path):
        resolved = (Path(shell.cwd) / path).resolve()
        if not resolved.is_relative_to(root):
            blocked.append(str(resolved))
            raise AssertionError(f"Host file operation escaped the test directory: {resolved}")
        return original_resolve(shell, path)

    # read/write_file/replace and their activity wrappers all resolve the target
    # here before opening it. This guard checks the result; it never remaps it.
    monkeypatch.setattr(ShellTools, "_resolve_path", resolve)
    return blocked


class _Client:
    def __init__(self):
        self.requests = []
        self.updates = []

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        self.requests.append(tool_call)
        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


def _turn(*codes: str, label: str) -> list[LLMResponse]:
    calls = [("execute_python", {"code": code}) for code in codes]
    calls.append(("return_result", {"result": {"message": "Finished.", "explanation": "complete"}}))
    return [
        LLMResponse(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id=f"permission-path-{label}-{index}", name=name, arguments=json.dumps(args)
                )
            ],
        )
        for index, (name, args) in enumerate(calls)
    ]


@asynccontextmanager
async def _sandbox_session(workspace, responses, *, config=None):
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    adapter = CodingACPAdapter(
        lambda: llm, model="alpha", sandbox=True, sandbox_config=config or SandboxConfig()
    )
    client = _Client()
    adapter.on_connect(client)
    await adapter.initialize(1, aster={"runtimeControls": True})
    try:
        async with asyncio.timeout(120):
            session_id = (await adapter.new_session(str(workspace))).session_id
            yield adapter, session_id, client
    finally:
        await adapter.close()


async def _configure(adapter, session_id, permission):
    return await adapter.ext_method(
        "aster/session/configure", {"sessionId": session_id, "permissionMode": permission}
    )


def _assert_completed(client, tool_call_id):
    results = [
        update
        for update in client.updates
        if getattr(update, "tool_call_id", None) == tool_call_id
        and getattr(update, "status", None) in {"completed", "failed"}
    ]
    assert len(results) == 1, f"Missing completed ACP result for {tool_call_id}"
    assert results[0].status == "completed", results[0]


@pytest.mark.parametrize("location", ["workspace", "allowed", "relative"])
async def test_accept_edits_uses_the_same_guest_paths_as_manual(
    tmp_path, guarded_host_paths, location
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    guest_workspace = "/workspace" if sys.platform == "win32" else workspace.as_posix()
    guest_allowed = "/allowed/0" if sys.platform == "win32" else allowed.as_posix()
    config = SandboxConfig(allow=(FileRule(path=str(allowed), access="read_write"),))
    guest_path, host_path = {
        "workspace": (guest_workspace + "/shared.txt", workspace / "shared.txt"),
        "allowed": (guest_allowed + "/shared.txt", allowed / "shared.txt"),
        "relative": ("nested/shared.txt", workspace / "nested" / "shared.txt"),
    }[location]
    manual = (
        "from pathlib import Path\nimport sys\n"
        "assert sys.platform == 'linux'\n"
        f"assert Path.cwd().as_posix() == {guest_workspace!r}\n"
        f"await self.shell.write_file({guest_path!r}, 'manual baseline')\n"
        "Path('guest-proof.txt').write_text(sys.platform)"
    )
    # Path-looking content must remain data. Only the target is translated.
    original = "/workspace/content-only.txt"
    replacement = "/allowed/0/replacement-only.txt"
    accepted = (
        f"await self.shell.write_file(path={guest_path!r}, content={original!r})",
        f"await self.shell.read(path={guest_path!r})",
        f"await self.shell.replace(target={guest_path!r}, old_or_new={original!r}, new={replacement!r})",
        f"await self.shell.read({guest_path!r})",
    )
    read_ids = ("permission-path-accepted-1", "permission-path-accepted-3")
    read_results = {}

    class RecordReads(InstrumentationHooks):
        def after_code_execution(self, *, tool_call_id, result, **kwargs):
            if tool_call_id in read_ids:
                read_results[tool_call_id] = result

    responses = [*_turn(manual, label="manual"), *_turn(*accepted, label="accepted")]
    async with _sandbox_session(workspace, responses, config=config) as (
        adapter,
        session_id,
        client,
    ):
        state = await _configure(adapter, session_id, "manual")
        assert state["executionBackend"] == "sandbox"
        response = await adapter.prompt(session_id, [text_block("Write the approved baseline.")])
        assert response.stop_reason == "end_turn"
        assert (workspace / "guest-proof.txt").read_text() == "linux"
        assert host_path.read_text() == "manual baseline"
        assert not guarded_host_paths

        await _configure(adapter, session_id, "accept_edits")
        with hooks_scope(RecordReads(), compose=False):
            response = await adapter.prompt(
                session_id,
                [text_block("Write, read and replace the same file without more approval.")],
            )

        assert response.stop_reason == "end_turn"
        assert not guarded_host_paths, f"Guest paths reached host roots: {guarded_host_paths}"
        assert host_path.read_text() == replacement
        for read_id, expected in zip(read_ids, (original, replacement), strict=True):
            _assert_completed(client, read_id)
            result = read_results[read_id]
            assert isinstance(result, ExecutionResult)
            assert result.success, result.error
            assert isinstance(result.returned_value, Match)
            assert result.returned_value.text == expected
        assert [request.raw_input for request in client.requests] == [{"code": manual}]
        assert not (workspace / "content-only.txt").exists()
        assert not (allowed / "replacement-only.txt").exists()
