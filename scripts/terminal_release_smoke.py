# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check an installed terminal, then preserve a real session across reinstall.

Run with the installed environment's ``python -I`` from outside the checkout.
``seed`` creates disposable state; after installing again, ``verify`` reopens it
and completes another scripted turn. No provider credentials or network are used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def installed_terminal() -> dict:
    prefix = Path(sys.prefix).resolve()
    assert sys.prefix != sys.base_prefix, "Use the installed tool's isolated Python environment"
    versions = {}
    for name in ("aster", "aster-cli", "aster-acp", "aster-memory", "aster-bench"):
        module = importlib.import_module(name.replace("-", "_"))
        assert Path(module.__file__).resolve().is_relative_to(prefix), name
        versions[name] = importlib.metadata.version(name)
    assert len(set(versions.values())) == 1, versions
    import aster_cli
    from aster_cli._tui_launcher import find_tui_binary

    binary = Path(aster_cli.__file__).resolve().parent / "bin" / "aster.exe"
    assert find_tui_binary() == binary, "Launcher did not choose the installed bundled terminal"
    assert binary.is_file()
    for command in ([str(binary), "--help"], [sys.executable, "-I", "-m", "aster_cli", "--help"]):
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", timeout=30, check=True
        )
        assert "--help" in result.stdout
    return {
        "versions": versions,
        "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "bundled_terminal": True,
    }


def responses(code: str, message: str):
    from aster.unifiedllm import LLMResponse, ToolCall

    return [
        LLMResponse(
            raw_response=None,
            content="",
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id=f"smoke-{index}", name=name, arguments=json.dumps(arguments))],
        )
        for index, (name, arguments) in enumerate(
            [
                ("execute_python", {"code": code}),
                ("return_result", {"result": {"message": message, "explanation": "verified"}}),
            ]
        )
    ]


class Client:
    def __init__(self):
        self.updates = []

    async def request_permission(self, session_id, tool_call, options, **kwargs):
        from acp.schema import RequestPermissionResponse

        return RequestPermissionResponse.model_validate(
            {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        )

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)


async def session_check(root: Path, phase: str, package: dict) -> dict:
    from acp import text_block
    from acp.schema import ToolCallStart
    from aster_acp.server import CodingACPAdapter

    from aster.unifiedllm import FakeLLMClient

    state_path = root / "smoke-state.json"
    workspace = root / "workspace"
    result_file = workspace / "result.txt"
    if phase == "seed":
        assert not state_path.exists() and not workspace.exists(), (
            "Seed requires a fresh state directory"
        )
        workspace.mkdir()
        llm = FakeLLMClient(
            responses("await self.shell.write_file('result.txt', 'original\\n')", "created"),
            strict_exhaustion=True,
        )
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for relative, expected in state["preserved_files"].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
        llm = FakeLLMClient(
            responses("await self.shell.write_file('continued.txt', 'continued\\n')", "continued"),
            strict_exhaustion=True,
        )

    client = Client()
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(client)
    try:
        if phase == "seed":
            session = await adapter.new_session(str(workspace))
            await adapter.prompt(session.session_id, [text_block("Create result.txt")])
            assert result_file.read_text(encoding="utf-8") == "original\n"
            session_id = session.session_id
        else:
            session_id = state["session_id"]
            await adapter.load_session(str(workspace), session_id)
            assert not llm.calls, "Reopening called the model"
            assert any(isinstance(update, ToolCallStart) for update in client.updates)
            assert result_file.read_text(encoding="utf-8") == "user changes\n", "Old edit replayed"
            await adapter.prompt(session_id, [text_block("Create continued.txt, keep result.txt")])
            assert (workspace / "continued.txt").read_text(encoding="utf-8") == "continued\n"
            assert result_file.read_text(encoding="utf-8") == "user changes\n"
            assert llm.calls, "Continuation never reached the model"
    finally:
        await adapter.close()

    if phase == "verify":
        return {
            **package,
            "seed_versions": state["package"]["versions"],
            "session_reopened_without_execution": True,
            "continued_after_install": True,
            "saved_input_bytes_preserved": True,
            "cross_version": state["package"]["versions"] != package["versions"],
        }

    result_file.write_text("user changes\n", encoding="utf-8")
    draft = workspace / ".aster" / "tui-drafts" / f"s-{session_id.encode().hex()}.json"
    draft.parent.mkdir(parents=True)
    draft.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "revision": 1,
                "session_id": session_id,
                "snapshot": {
                    "composer": {"text": "unsent draft 中文", "attachments": []},
                    "cursor": [0, 0],
                    "queued": [{"text": "queued input", "attachments": []}],
                    "in_flight": None,
                    "queue_paused": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = root / "user" / "settings.yaml"
    settings.write_text("# preserve user configuration\ntui: {}\n", encoding="utf-8")
    state = {
        "package": package,
        "session_id": session_id,
        "preserved_files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (result_file, draft, settings)
        },
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {**package, "session_seeded": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seed", "verify"))
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    root = args.state.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "user").mkdir(exist_ok=True)
    for name in list(os.environ):
        if name.startswith("ASTER_"):
            os.environ.pop(name)
    os.environ.update(
        ASTER_USER_DIR=str(root / "user"),
        ASTER_PROJECT_DIR=str(root / "workspace" / ".aster"),
        ASTER_TRUST_ALL_WORKSPACES="1",
        ASTER_MODEL_METADATA_REFRESH="False",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    package = installed_terminal()
    # User-installed skills are outside this package-installation check.
    with patch("aster_acp.server.load_coding_skills_dirs", return_value=[]):
        report = asyncio.run(asyncio.wait_for(session_check(root, args.phase, package), timeout=60))
    (root / f"{args.phase}-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
