# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion depends on fresh command outcomes, not a model's evidence text."""

import asyncio
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from aster_cli.coding.agent import CodingAgent
from aster_cli.coding.reply_strategy import CodingReplyStrategy
from aster_cli.coding.verification import (
    RequiredCheck,
    Verification,
    begin_verification,
    require_verification,
)

from aster.config import CodeActConfig
from aster.interactive import Done, NeedInput, SummarizationConfig, Waiting
from aster.runtime.event_manager import EventManager
from aster.skill import Skill
from aster.strategy_validation import InvariantError
from aster.tools import ShellTools
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def python_command(code):
    args = (Path(sys.executable).as_posix(), "-c", code)
    if sys.platform == "win32":
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in args)
    return shlex.join(args)


def verification(root, code="pass", timeout=10):
    check = RequiredCheck(
        command=python_command(code), files=("code.txt", "new.txt"), timeout=timeout
    )
    result = Verification(root, {"tests": check})
    result.begin("turn")
    return result


def response(name, args):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[ToolCall(id="check", name=name, arguments=json.dumps(args))],
    )


def configured_agent(root, responses, checks=True):
    settings = root / ".aster/settings.yaml"
    settings.parent.mkdir(exist_ok=True)
    settings.write_text(
        yaml.safe_dump(
            {
                "coding": {
                    "required_checks": {
                        "tests": {
                            "command": python_command(
                                "from pathlib import Path; assert Path('code.txt').read_text() == 'fixed'"
                            ),
                            "files": ["code.txt"],
                        }
                    }
                    if checks
                    else {}
                }
            }
        ),
        encoding="utf-8",
    )
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    return CodingAgent(
        llm=llm,
        cwd=root,
        libs_dir=root / ".aster/libs",
        skills_dirs=[],
        summarization=SummarizationConfig(policy="none"),
    ), llm


async def test_done_rejects_fake_evidence_until_real_check_passes(tmp_path):
    agent, llm = configured_agent(
        tmp_path,
        [
            response(
                "execute_python", {"code": "await self.shell.write_file('code.txt', 'fixed')"}
            ),
            response(
                "return_result",
                {
                    "result": {
                        "message": "unverified",
                        "explanation": "done",
                        "evidence": ["all checks passed"],
                    }
                },
            ),
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "verified", "explanation": "done"}}),
        ],
    )
    try:
        result = await agent.handle({"user_messages": ["Fix the file and verify it"]})
        assert isinstance(result, Done) and result.message == "verified"
        assert len(llm.calls) == 4
        assert (tmp_path / "code.txt").read_text() == "fixed"
    finally:
        await agent.close()


@pytest.mark.parametrize(
    "code,timeout", [("raise SystemExit(7)", 10), ("import time; time.sleep(5)", 0.05)]
)
async def test_failed_or_timed_out_check_blocks_completion(tmp_path, code, timeout):
    checks = verification(tmp_path, code, timeout)
    result = await checks.run("tests", EventManager())
    assert result.returncode != 0 or result.timed_out
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()


@pytest.mark.parametrize("change", ["edit", "create", "delete"])
async def test_pass_becomes_stale_after_file_changes(tmp_path, change):
    target = tmp_path / "code.txt"
    target.write_text("initial")
    checks = verification(tmp_path)
    assert (await checks.run("tests", EventManager())).returncode == 0
    checks.require_passes()
    if change == "edit":
        target.write_text("changed")
    elif change == "create":
        (tmp_path / "new.txt").write_text("new")
    else:
        target.unlink()
    with pytest.raises(InvariantError, match="stale"):
        checks.require_passes()


async def test_check_cannot_pass_by_modifying_covered_files(tmp_path):
    checks = verification(
        tmp_path, "from pathlib import Path; Path('code.txt').write_text('changed')"
    )
    assert (await checks.run("tests", EventManager())).returncode == 0
    with pytest.raises(InvariantError):
        checks.require_passes()


async def test_verification_uses_fresh_shell_state(tmp_path):
    (tmp_path / "child").mkdir()
    ordinary = ShellTools(cwd=str(tmp_path))
    if sys.platform == "win32":
        poison = "function aster_check_probe { 'fake' }; $env:ASTER_CHECK_POISON='yes'; Set-Location child"
        isolated = "if (Test-Path Function:\\aster_check_probe) { throw 'leaked function' }; "
    else:
        poison = "aster_check_probe() { echo fake; }; export ASTER_CHECK_POISON=yes; cd child"
        isolated = "if command -v aster_check_probe; then exit 9; fi; "
    try:
        assert (await ordinary.run(poison)).returncode == 0
        code = f"import os; from pathlib import Path; assert 'ASTER_CHECK_POISON' not in os.environ; assert Path.cwd() == Path({str(tmp_path)!r})"
        checks = Verification(
            tmp_path,
            {"tests": RequiredCheck(command=isolated + python_command(code), files=("code.txt",))},
        )
        checks.begin("turn")
        result = await checks.run("tests", EventManager())
        assert result.returncode == 0, result.stderr
        checks.require_passes()
    finally:
        await ordinary.close()


async def test_cancelled_check_clears_previous_pass(tmp_path, monkeypatch):
    checks = verification(tmp_path)
    await checks.run("tests", EventManager())
    started = asyncio.Event()

    async def interrupted(*args, **kwargs):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr("aster_cli.coding.verification.ActivityShellTools.run", interrupted)
    task = asyncio.create_task(checks.run("tests", EventManager()))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(InvariantError):
        checks.require_passes()


@pytest.mark.parametrize("checks", [False, True])
async def test_ordinary_answers_do_not_execute_checks(tmp_path, checks, monkeypatch):
    agent, llm = configured_agent(
        tmp_path,
        [response("return_result", {"result": {"message": "answer", "explanation": "done"}})],
        checks,
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("Plain answers must not run commands")

    monkeypatch.setattr("aster_cli.coding.verification.ActivityShellTools.run", forbidden)
    try:
        with pytest.raises(ValueError, match="verify"):
            agent.skills.register("mcp.verify", Skill(content="replacement skill"))
        assert (await agent.handle({"user_messages": ["Explain the project"]})).message == "answer"
        assert len(llm.calls) == 1
    finally:
        await agent.close()


def test_input_and_background_results_do_not_require_completion_checks():
    class Pending:
        def require_passes(self):
            raise AssertionError("Not a completed task")

    agent = SimpleNamespace(_verification=Pending())
    for result in (
        NeedInput(question="Choose a target"),
        Waiting(message="Running", explanation="wait", on=["job"]),
    ):
        require_verification(agent, result, SimpleNamespace(method_name="handle"))


async def test_new_turn_does_not_reuse_previous_pass(tmp_path):
    checks = verification(tmp_path)
    await checks.run("tests", EventManager())
    checks.begin("next-turn")
    (tmp_path / "code.txt").write_text("changed after restart")
    with pytest.raises(InvariantError):
        checks.require_passes()


def test_check_file_cannot_be_a_directory(tmp_path):
    (tmp_path / "code.txt").mkdir()
    with pytest.raises(ValueError):
        verification(tmp_path)


def test_strategy_rebuild_preserves_unhashable_conditions_without_duplicate_gate():
    @dataclass
    class Condition:
        name: str

        def __call__(self, *args):
            pass

    first, second = Condition("first"), Condition("second")
    config = CodeActConfig(
        preconditions=[first, begin_verification, second],
        postconditions=[first, require_verification, second],
    )
    rebuilt = CodingReplyStrategy(config=CodingReplyStrategy(config=config).config).config
    assert list(rebuilt.preconditions) == [first, begin_verification, second]
    assert list(rebuilt.postconditions) == [first, require_verification, second]
