# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shell must preserve unresolved effects rather than repeat them on resume."""

import asyncio
import json
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aster_cli.coding import CodingAgent
from aster_cli.coding.activity import ActivityShellTools
from aster_cli.coding.recovery import record_action
from aster_cli.coding.reply_strategy import CodingReplyStrategy
from aster_cli.coding.slash_commands import CodingSlashCommandRegistry
from aster_cli.sessions import SessionStore
from aster_cli.sessions.actions import PendingSessionActionsError
from aster_cli.sessions.events import SessionActionOutcome

from aster import Agent
from aster.agentdoc import doc
from aster.config import CodeActConfig
from aster.events import ExecutionResult
from aster.interactive import NeedInput, SummarizationConfig
from aster.runtime.event_manager import EventManager
from aster.runtime.sandbox.errors import CellTimeoutError
from aster.strategies.codeact import CodeActStrategy
from aster.tools._results import StreamDone, StreamEvent
from aster.tools.shell_tools import ShellResult, ShellTools
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _command():
    args = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('target.txt').write_text('changed')",
    ]
    if sys.platform == "win32":
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in args)
    return shlex.join(args)


@pytest.mark.parametrize("operation", ["run", "run_stream", "write_file", "replace"])
async def test_restored_action_blocks_shell_effects_but_allows_file_inspection(tmp_path, operation):
    target = tmp_path / "target.txt"
    target.write_text("original")
    store = SessionStore(tmp_path / "sessions")
    with store.create(working_directory=str(tmp_path)) as first:
        session_id = first.id
        action = first.actions.begin("shell.run", "Host shell command", str(tmp_path))
    with store.open(session_id) as restored:
        shell = ActivityShellTools(ShellTools(cwd=str(tmp_path)), EventManager(), restored.actions)
        try:
            assert (await shell.read("target.txt")).text == "original"
            with pytest.raises(PendingSessionActionsError):
                if operation == "run":
                    await shell.run(_command())
                elif operation == "run_stream":
                    async for _ in shell.run_stream(_command()):
                        pass
                elif operation == "write_file":
                    await shell.write_file("target.txt", "changed")
                else:
                    await shell.replace("target.txt", "original", "changed")
            assert target.read_text() == "original"
            assert restored.actions.pending()[0]["action_id"] == action
        finally:
            await shell.close()


@pytest.mark.parametrize("operation", ["run", "run_stream", "write_file", "replace"])
async def test_failed_intent_write_prevents_shell_effect(tmp_path, monkeypatch, operation):
    target = tmp_path / "target.txt"
    target.write_text("original")
    with SessionStore(tmp_path / "sessions").create(working_directory=str(tmp_path)) as handle:
        shell = ActivityShellTools(ShellTools(cwd=str(tmp_path)), EventManager(), handle.actions)

        def unavailable(*args, **kwargs):
            raise OSError("journal is unavailable")

        monkeypatch.setattr(handle.actions, "begin", unavailable)
        try:
            with pytest.raises(OSError, match="journal is unavailable"):
                if operation == "run":
                    await shell.run(_command())
                elif operation == "run_stream":
                    async for _ in shell.run_stream(_command()):
                        pass
                elif operation == "write_file":
                    await shell.write_file("target.txt", "changed")
                else:
                    await shell.replace("target.txt", "original", "changed")
            assert target.read_text() == "original"
        finally:
            await shell.close()


async def test_shell_outcomes_remain_pending_without_a_saved_snapshot(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with store.create(working_directory=str(tmp_path)) as handle:
        session_id = handle.id
        shell = ActivityShellTools(ShellTools(cwd=str(tmp_path)), EventManager(), handle.actions)
        try:
            await shell.write_file("target.txt", "original")
            await shell.replace("target.txt", "original", "changed")
            assert not handle.actions.pending()
        finally:
            await shell.close()
    with store.open(session_id) as restored:
        pending = restored.actions.pending()
        assert [item["kind"] for item in pending] == ["shell.write_file", "shell.replace"]
        assert all(item["outcome"] == "Tool returned" for item in pending)
        assert (tmp_path / "target.txt").read_text() == "changed"


async def test_host_review_command_is_hidden_and_spends_no_model_call(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with store.create(working_directory=str(tmp_path)) as first:
        session_id = first.id
        action = first.actions.begin("file.write_file", "target.txt", str(tmp_path))
    with store.open(session_id) as restored:
        llm = FakeLLMClient([])
        agent = CodingAgent(llm=llm, cwd=tmp_path, storage=restored.storage)
        agent.bind_action_journal(restored.actions)
        registry = CodingSlashCommandRegistry(agent)
        try:
            visible = doc(agent.recovery)
            assert "status" in visible
            assert "_command" not in visible
            assert "def review" not in visible
            result = await registry.invoke("recovery", "")
            assert action in str(result)
            assert not result.output_to_agent
            await registry.invoke("recovery", f'{action} --note ""')
            assert restored.actions.pending()
            result = await registry.invoke(
                "recovery", f'{action} --note "Checked target.txt; the content is present"'
            )
            assert "Recorded your review" in str(result)
            assert not restored.actions.pending()
            assert not llm.calls
        finally:
            registry.close()
            await agent.close()


class _RecoveryAgent(Agent, llm=FakeLLMClient()):
    pass


class _PartialShell:
    """Produce an actual partial file effect without launching a subprocess."""

    def __init__(self, root, *, timed_out=False):
        self.cwd = root
        self.timed_out = timed_out
        self.started = asyncio.Event()

    def _append(self):
        with (self.cwd / "append.txt").open("a", encoding="utf-8") as stream:
            stream.write("once\n")
        self.started.set()

    async def run(self, command, *, stdin=None, timeout=None):
        self._append()
        if not self.timed_out:
            await asyncio.Event().wait()
        return ShellResult(stdout="partial", returncode=124, timed_out=True)

    async def run_stream(self, command, *, stdin=None, timeout=None):
        self._append()
        yield StreamEvent(kind="stdout", text="partial")
        if not self.timed_out:
            await asyncio.Event().wait()
        yield StreamDone(kind="done", returncode=124, timed_out=True)


@pytest.mark.parametrize("operation", ["run", "run_stream"])
@pytest.mark.parametrize("timed_out", [False, True])
async def test_cancelled_or_timed_out_shell_effect_stays_pending_after_snapshot(
    tmp_path, operation, timed_out
):
    store = SessionStore(tmp_path / "sessions")
    with store.create(session_id="partial-shell") as handle:
        backend = _PartialShell(tmp_path, timed_out=timed_out)
        shell = ActivityShellTools(backend, EventManager(), handle.actions)

        async def run():
            if operation == "run":
                return await shell.run("append and wait")
            return [item async for item in shell.run_stream("append and wait")]

        task = asyncio.create_task(run())
        await backend.started.wait()
        if timed_out:
            result = await task
            assert (result[-1] if isinstance(result, list) else result).timed_out
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        pending = handle.actions.pending()
        assert len(pending) == 1
        with pytest.raises(PendingSessionActionsError):
            await shell.run("must not repeat")
        handle.save_current_snapshot(_RecoveryAgent(storage=handle.storage))
        assert handle.actions.pending() == pending

    with store.open("partial-shell") as resumed:
        assert resumed.actions.pending() == pending
        resumed.actions.review(pending[0]["action_id"], "Checked exactly one appended line")
        assert not resumed.actions.pending()
    assert (tmp_path / "append.txt").read_text(encoding="utf-8") == "once\n"


async def test_closing_a_partial_shell_stream_requires_review(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with store.create(session_id="closed-stream") as handle:
        shell = ActivityShellTools(_PartialShell(tmp_path), EventManager(), handle.actions)
        stream = shell.run_stream("append and wait")
        assert (await anext(stream)).text == "partial"
        await stream.aclose()
        handle.save_current_snapshot(_RecoveryAgent(storage=handle.storage))
        assert len(handle.actions.pending()) == 1
    with store.open("closed-stream") as resumed:
        assert len(resumed.actions.pending()) == 1
    assert (tmp_path / "append.txt").read_text(encoding="utf-8") == "once\n"


@pytest.mark.parametrize("error_type", [asyncio.CancelledError, GeneratorExit, TimeoutError])
def test_record_action_preserves_the_original_interruption(tmp_path, error_type):
    store = SessionStore(tmp_path / "sessions")
    with store.create(session_id="interrupted-call") as handle:
        error = error_type("interrupted after a partial effect")
        with pytest.raises(error_type) as raised:
            with record_action(handle.actions, "file.write_file", "append.txt", str(tmp_path)):
                (tmp_path / "append.txt").write_text("once\n", encoding="utf-8")
                raise error
        assert raised.value is error
        handle.save_current_snapshot(_RecoveryAgent(storage=handle.storage))
        assert len(handle.actions.pending()) == 1


def test_outcome_storage_failure_preserves_cancellation_and_requires_review(tmp_path, monkeypatch):
    with SessionStore(tmp_path / "sessions").create(session_id="failed-outcome") as handle:
        original_add = handle.events.add

        def fail_outcome(event):
            if isinstance(event, SessionActionOutcome):
                raise OSError("outcome storage unavailable")
            return original_add(event)

        monkeypatch.setattr(handle.events, "add", fail_outcome)
        cancelled = asyncio.CancelledError("original cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            with record_action(handle.actions, "shell.run", "Host shell command", str(tmp_path)):
                raise cancelled
        assert raised.value is cancelled
        assert any("Could not save action outcome" in note for note in cancelled.__notes__)
        handle.save_current_snapshot(_RecoveryAgent(storage=handle.storage))
        assert len(handle.actions.pending()) == 1


@pytest.mark.parametrize(
    ("error_type", "cancelled", "requires_review"),
    [
        (None, False, False),
        (SyntaxError, False, False),
        (ValueError, False, False),
        (TimeoutError, False, True),
        (CellTimeoutError, False, True),
        (None, True, True),
    ],
)
async def test_python_result_only_keeps_interrupted_execution_pending(
    tmp_path, monkeypatch, error_type, cancelled, requires_review
):
    result = ExecutionResult(
        error=error_type("test execution result") if error_type else None, cancelled=cancelled
    )
    monkeypatch.setattr(CodeActStrategy, "_execute_code", AsyncMock(return_value=result))
    store = SessionStore(tmp_path / "sessions")
    with store.create(session_id="python-result") as handle:
        runtime = SimpleNamespace(
            agent=SimpleNamespace(cwd=tmp_path, _action_journal=handle.actions)
        )
        returned = await CodingReplyStrategy()._execute_code(runtime, "pass", {}, None, "handle")
        assert returned is result
        handle.save_current_snapshot(_RecoveryAgent(storage=handle.storage))
        assert bool(handle.actions.pending()) is requires_review
    with store.open("python-result") as resumed:
        assert bool(resumed.actions.pending()) is requires_review


async def test_python_cancellation_keeps_the_original_exception_and_pending_action(
    tmp_path, monkeypatch
):
    cancelled = asyncio.CancelledError("cancelled cell")
    monkeypatch.setattr(CodeActStrategy, "_execute_code", AsyncMock(side_effect=cancelled))
    with SessionStore(tmp_path / "sessions").create(session_id="python-cancel") as handle:
        runtime = SimpleNamespace(
            agent=SimpleNamespace(cwd=tmp_path, _action_journal=handle.actions)
        )
        with pytest.raises(asyncio.CancelledError) as raised:
            await CodingReplyStrategy()._execute_code(runtime, "pass", {}, None, "handle")
        assert raised.value is cancelled
        handle.save_current_snapshot(_RecoveryAgent(storage=handle.storage))
        assert len(handle.actions.pending()) == 1


async def test_native_cell_timeout_after_append_survives_normal_snapshot(tmp_path):
    calls = [
        (
            "execute_python",
            {
                "code": "with open(self.cwd / 'append.txt', 'a', encoding='utf-8') as stream:\n"
                "    stream.write('once\\n')\nawait asyncio.sleep(10)"
            },
        ),
        ("return_result", {"result": {"question": "Confirm the partial append with /recovery."}}),
    ]
    llm = FakeLLMClient(
        [
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id=f"timeout-{index}", name=name, arguments=json.dumps(args))],
            )
            for index, (name, args) in enumerate(calls)
        ],
        strict_exhaustion=True,
    )
    store = SessionStore(tmp_path / "sessions")
    with store.create(session_id="native-timeout") as handle:
        agent = CodingAgent(
            llm=llm,
            cwd=tmp_path,
            storage=handle.storage,
            libs_dir=tmp_path / "libs",
            skills_dirs=[],
            summarization=SummarizationConfig(policy="none"),
        )
        agent.bind_action_journal(handle.actions)
        try:
            result = await agent.handle(
                {"user_messages": ["Append one line"]},
                _strategy=CodingReplyStrategy(config=CodeActConfig(cell_timeout=0.1)),
            )
            assert isinstance(result, NeedInput)
            handle.save_current_snapshot(agent)
            assert [item["kind"] for item in handle.actions.pending()] == ["python.execute"]
        finally:
            await agent.close()
    with store.open("native-timeout") as resumed:
        assert [item["kind"] for item in resumed.actions.pending()] == ["python.execute"]
    assert (tmp_path / "append.txt").read_text(encoding="utf-8") == "once\n"
