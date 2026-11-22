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
    CodingDone,
    RequiredCheck,
    Verification,
    VerificationState,
    begin_verification,
    require_verification,
)

from aster.config import CodeActConfig
from aster.interactive import Done, NeedInput, SummarizationConfig, Waiting
from aster.runtime.event_manager import EventManager
from aster.skill import Skill
from aster.storage.sqlite import SQLiteStorageManager
from aster.strategy_validation import InvariantError
from aster.tools import ShellTools
from aster.tools.shell_tools import ShellResult
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


def configured_agent(root, responses, checks=True, **options):
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
        **options,
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


@pytest.fixture
async def controlled_check_runs(monkeypatch):
    started = [asyncio.Event(), asyncio.Event()]
    results = [asyncio.get_running_loop().create_future() for _ in started]
    pending = iter(zip(started, results, strict=True))
    tasks = []

    async def controlled_run(*args, **kwargs):
        entered, result = next(pending)
        entered.set()
        return await result

    monkeypatch.setattr("aster_cli.coding.verification.ActivityShellTools.run", controlled_run)

    async def start(checks):
        index = len(tasks)
        task = asyncio.create_task(checks.run("tests", EventManager()))
        tasks.append(task)
        await asyncio.wait_for(started[index].wait(), timeout=2)
        return task, results[index]

    try:
        yield start
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
async def test_older_success_cannot_replace_newer_failed_or_cancelled_check(
    tmp_path, controlled_check_runs, outcome
):
    checks = verification(tmp_path)
    older, older_result = await controlled_check_runs(checks)
    newer, newer_result = await controlled_check_runs(checks)

    if outcome == "failed":
        newer_result.set_result(ShellResult("", returncode=7))
        assert (await newer).returncode == 7
    else:
        newer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await newer
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()

    # The older success arrives only after the newer attempt has finished.
    assert not older.done()
    older_result.set_result(ShellResult("", returncode=0))
    assert (await older).returncode == 0
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()


@pytest.mark.parametrize("older_returncode", [0, 7])
async def test_newer_success_survives_older_late_result(
    tmp_path, controlled_check_runs, older_returncode
):
    checks = verification(tmp_path)
    older, older_result = await controlled_check_runs(checks)
    newer, newer_result = await controlled_check_runs(checks)

    newer_result.set_result(ShellResult("", returncode=0))
    assert (await newer).returncode == 0
    checks.require_passes()

    assert not older.done()
    older_result.set_result(ShellResult("", returncode=older_returncode))
    assert (await older).returncode == older_returncode
    checks.require_passes()


async def test_previous_turn_pending_check_cannot_satisfy_current_turn(
    tmp_path, controlled_check_runs
):
    checks = verification(tmp_path)
    older, older_result = await controlled_check_runs(checks)
    checks.begin("next-turn")

    # Explicitly request verification: an unedited ordinary turn needs no checks.
    current, current_result = await controlled_check_runs(checks)
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()

    older_result.set_result(ShellResult("", returncode=0))
    assert (await older).returncode == 0
    assert not current.done()
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()

    current_result.set_result(ShellResult("", returncode=0))
    assert (await current).returncode == 0
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
        def observe(self):
            pass

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


def test_unverified_changes_remain_pending_in_the_next_turn(tmp_path):
    (tmp_path / "code.txt").write_text("original")
    checks = verification(tmp_path)
    (tmp_path / "code.txt").write_text("unfinished change")
    checks.begin("next-turn")
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()


@pytest.fixture
def passing_check(monkeypatch):
    async def run(*args, **kwargs):
        return ShellResult("passed", returncode=0)

    monkeypatch.setattr("aster_cli.coding.verification.ActivityShellTools.run", run)


async def test_partial_check_evidence_survives_turns_and_serialization(tmp_path, passing_check):
    check = RequiredCheck(command="trusted-check", files=("code.txt",))
    checks = Verification(tmp_path, {"lint": check, "tests": check})
    checks.begin("edit")
    (tmp_path / "code.txt").write_text("changed")
    await checks.run("lint", EventManager())
    task_id = checks.status()["task_id"]
    checks.begin("continue")
    assert checks.status()["task_id"] == task_id
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()

    saved = VerificationState.model_validate_json(checks.state.model_dump_json())
    restored = Verification(tmp_path, checks.checks)
    restored.configure(checks.checks, state=saved)
    restored.begin("restored")
    assert restored.status()["task_id"] == task_id
    assert restored.status()["checks"] == {"lint": "passed", "tests": "not_run"}
    await restored.run("tests", EventManager())
    restored.require_passes()
    assert restored.status()["state"] == "verified"


@pytest.mark.parametrize("change", ["edit", "delete", "create", "command", "timeout", "files"])
async def test_restored_evidence_is_invalidated_by_files_or_check_contract(
    tmp_path, passing_check, change
):
    (tmp_path / "code.txt").write_text("original")
    checks = verification(tmp_path)
    await checks.run("tests", EventManager())
    checks.require_passes()
    saved = VerificationState.model_validate_json(checks.state.model_dump_json())
    declarations = dict(checks.checks)
    if change == "edit":
        (tmp_path / "code.txt").write_text("changed")
    elif change == "delete":
        (tmp_path / "code.txt").unlink()
    elif change == "create":
        (tmp_path / "new.txt").write_text("created")
    else:
        update = {
            "command": {"command": "different-check"},
            "timeout": {"timeout": 12},
            "files": {"files": ("code.txt", "new.txt", "extra.txt")},
        }[change]
        declarations["tests"] = declarations["tests"].model_copy(update=update)
    restored = Verification(tmp_path, declarations)
    restored.configure(declarations, state=saved)
    restored.begin("reopen")
    assert restored.status()["state"] == "pending"
    with pytest.raises(InvariantError, match="tests"):
        restored.require_passes()


async def test_workspace_relocation_invalidates_old_evidence(tmp_path, passing_check):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    checks = verification(first)
    await checks.run("tests", EventManager())
    checks.require_passes()
    saved = VerificationState.model_validate_json(checks.state.model_dump_json())
    moved = Verification(second, checks.checks)
    moved.configure(checks.checks, state=saved)
    moved.begin("moved")
    with pytest.raises(InvariantError, match="tests"):
        moved.require_passes()
    assert checks.status()["state"] == "verified"


async def test_changed_contract_discards_in_flight_success(tmp_path, controlled_check_runs):
    checks = verification(tmp_path)
    task, result = await controlled_check_runs(checks)
    replacement = checks.checks["tests"].model_copy(update={"command": "different-check"})
    checks.configure({"tests": replacement})
    result.set_result(ShellResult("", returncode=0))
    await task
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()


async def test_removed_checks_are_disabled_without_reusing_old_passes(tmp_path, passing_check):
    checks = verification(tmp_path)
    await checks.run("tests", EventManager())
    checks.require_passes()
    original = dict(checks.checks)
    checks.configure({})
    checks.begin("disabled")
    checks.require_passes()
    assert checks.status()["state"] == "disabled"
    assert not checks.state.passed
    checks.configure(original)
    checks.begin("reenabled")
    with pytest.raises(InvariantError, match="tests"):
        checks.require_passes()


async def test_interrupted_task_cannot_complete_in_a_later_turn_without_checks(tmp_path):
    agent, llm = configured_agent(
        tmp_path,
        [
            response(
                "execute_python", {"code": "await self.shell.write_file('code.txt', 'fixed')"}
            ),
            response("return_result", {"result": {"question": "Choose the next step"}}),
            response("return_result", {"result": {"message": "premature", "explanation": "done"}}),
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "verified", "explanation": "done"}}),
        ],
    )
    try:
        assert isinstance(await agent.handle({"user_messages": ["Edit the file"]}), NeedInput)
        assert agent.verification_status()["state"] == "pending"
        result = await agent.handle({"user_messages": ["Finish the previous task"]})
        assert result.message == "verified"
        assert len(llm.calls) == 5
        assert agent.verification_status()["state"] == "verified"
    finally:
        await agent.close()


async def test_unrelated_reply_preserves_pending_task_without_running_commands(
    tmp_path, monkeypatch
):
    agent, llm = configured_agent(
        tmp_path,
        [
            response(
                "execute_python", {"code": "await self.shell.write_file('code.txt', 'fixed')"}
            ),
            response("return_result", {"result": {"question": "Choose the next step"}}),
            response(
                "return_result",
                {"result": {"message": "Paris", "explanation": "Answered", "complete_task": False}},
            ),
        ],
    )
    try:
        await agent.handle({"user_messages": ["Edit the file"]})
        task_id = agent.verification_status()["task_id"]

        async def forbidden(*args, **kwargs):
            raise AssertionError("The unrelated answer must not execute checks")

        monkeypatch.setattr("aster_cli.coding.verification.ActivityShellTools.run", forbidden)
        result = await agent.handle({"user_messages": ["What is the capital of France?"]})
        assert result.message.startswith("Paris")
        assert "still needs verification: tests" in result.message
        assert len(llm.calls) == 3
        assert agent.verification_status()["state"] == "pending"
        assert agent.verification_status()["task_id"] == task_id
        with pytest.raises(InvariantError, match="tests"):
            agent._verification.require_passes()
    finally:
        await agent.close()


async def test_reply_flag_cannot_bypass_new_edits(tmp_path):
    agent, llm = configured_agent(
        tmp_path,
        [
            response(
                "execute_python", {"code": "await self.shell.write_file('code.txt', 'fixed')"}
            ),
            response(
                "return_result",
                {"result": {"message": "skip", "explanation": "done", "complete_task": False}},
            ),
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "verified", "explanation": "done"}}),
        ],
    )
    try:
        result = await agent.handle({"user_messages": ["Edit and verify"]})
        assert result.message == "verified"
        assert len(llm.calls) == 4
    finally:
        await agent.close()


@pytest.mark.parametrize("complete_task", [True, False])
async def test_explicit_python_return_cannot_bypass_verification(tmp_path, complete_task):
    agent, llm = configured_agent(
        tmp_path,
        [
            response(
                "execute_python",
                {
                    "code": "await self.shell.write_file('code.txt', 'fixed')\n"
                    f"return Done(message=str(42), explanation='done', complete_task={complete_task})"
                },
            ),
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "verified", "explanation": "done"}}),
        ],
    )
    try:
        result = await agent.handle({"user_messages": ["Edit and verify the file"]})
        assert result.message == "verified"
        assert len(llm.calls) == 3
        assert agent.verification_status()["state"] == "verified"
    finally:
        await agent.close()


async def test_session_snapshot_preserves_pending_task_and_rebinds_host_verification(tmp_path):
    database = tmp_path / "conversation.db"
    storage = SQLiteStorageManager(database)
    agent, _ = configured_agent(
        tmp_path,
        [
            response(
                "execute_python", {"code": "await self.shell.write_file('code.txt', 'fixed')"}
            ),
            response("return_result", {"result": {"question": "Choose the next step"}}),
        ],
        storage=storage,
    )
    try:
        await agent.handle({"user_messages": ["Edit the file"]})
        task_id = agent.verification_status()["task_id"]
        storage.save_current_snapshot(agent)
    finally:
        await agent.close()
        storage.close()

    restored_storage = SQLiteStorageManager(database)
    restored, llm = configured_agent(
        tmp_path,
        [
            response("return_result", {"result": {"message": "premature", "explanation": "done"}}),
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "verified", "explanation": "done"}}),
        ],
        storage=restored_storage,
    )
    try:
        assert restored_storage.restore_current_snapshot(restored)
        restored.refresh_host_context()
        assert restored.verification_status()["state"] == "pending"
        assert restored.verification_status()["task_id"] == task_id
        result = await restored.handle({"user_messages": ["Finish the saved task"]})
        assert result.message == "verified"
        assert len(llm.calls) == 3
        assert restored.verification_status()["state"] == "verified"
    finally:
        await restored.close()
        restored_storage.close()


def test_coding_result_accepts_existing_done_values():
    result = CodingDone.model_validate(Done(message="answer", explanation="done"))
    assert result.complete_task is True


async def test_host_detects_config_changes_after_a_check_before_accepting_done(
    tmp_path, monkeypatch
):
    agent, llm = configured_agent(
        tmp_path,
        [
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "stale", "explanation": "done"}}),
            response("execute_python", {"code": "await self.verify('tests')"}),
            response("return_result", {"result": {"message": "fresh", "explanation": "done"}}),
        ],
    )
    calls = []

    async def update_configuration(command, **kwargs):
        calls.append(command)
        settings = tmp_path / ".aster/settings.yaml"
        config = yaml.safe_load(settings.read_text())
        config["coding"]["required_checks"]["tests"]["timeout"] = 120
        settings.write_text(yaml.safe_dump(config))
        return ShellResult("passed", returncode=0)

    monkeypatch.setattr(
        "aster_cli.coding.verification.ActivityShellTools.run",
        lambda self, command, **kwargs: update_configuration(command, **kwargs),
    )
    try:
        result = await agent.handle({"user_messages": ["Run the configured check"]})
        assert result.message == "fresh"
        assert len(calls) == 2 and len(llm.calls) == 4
    finally:
        await agent.close()


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
