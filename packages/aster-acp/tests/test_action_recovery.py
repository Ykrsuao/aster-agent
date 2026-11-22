# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared file edits must journal intent before even creating an empty file."""

import pytest
from aster_acp._file_edits import file_edit_executor
from aster_acp.runtime_controls import RuntimeControls
from aster_cli.coding import CodingAgent
from aster_cli.sessions import SessionStore
from aster_cli.sessions.actions import PendingSessionActionsError

from aster.runtime.event_manager import EventManager
from aster.unifiedllm import FakeLLMClient


async def test_failed_journal_does_not_create_a_file(tmp_path, monkeypatch):
    with SessionStore(tmp_path / "sessions").create(working_directory=str(tmp_path)) as handle:
        code = "await self.shell.write_file('created.txt', 'content')"
        executor = file_edit_executor(
            code,
            cwd=tmp_path,
            event_manager=EventManager(),
            max_file_bytes=4096,
            timeout=2,
            journal=handle.actions,
        )
        assert executor is not None

        def unavailable(*args, **kwargs):
            raise OSError("journal is unavailable")

        monkeypatch.setattr(handle.actions, "begin", unavailable)
        result = await executor.run_cell(code)
        assert isinstance(result.error, OSError)
        assert not (tmp_path / "created.txt").exists()


@pytest.mark.parametrize("mode", ["auto", "manual", "accept_edits", "plan"])
async def test_recovery_allows_bounded_reads_and_refuses_execution_in_every_mode(tmp_path, mode):
    store = SessionStore(tmp_path / "sessions")
    (tmp_path / "target.txt").write_text("already appended")
    with store.create(working_directory=str(tmp_path)) as first:
        session_id = first.id
        action = first.actions.begin("shell.run", "Host shell command", str(tmp_path))
    with store.open(session_id) as restored:
        agent = CodingAgent(llm=FakeLLMClient([]), cwd=tmp_path, storage=restored.storage)
        agent.bind_action_journal(restored.actions)
        controls = RuntimeControls(agent, object(), restored.id, publish=lambda _: None)
        controls.set_permission_mode(mode)
        try:
            refusal = await controls.authorize("open('target.txt', 'a').write('again')")
            assert "human review" in refusal
            code = "await self.files.read('target.txt')"
            executor = controls.file_edit_executor(code)
            assert executor is not None
            result = await executor.run_cell(code)
            assert result.error is None
            assert "already appended" in str(result.returned_value)
            assert (
                controls.file_edit_executor("await self.shell.write_file('target.txt', 'again')")
                is None
            )
            assert (tmp_path / "target.txt").read_text() == "already appended"
            assert restored.actions.pending()[0]["action_id"] == action
        finally:
            await agent.close()


async def test_prepared_edit_rechecks_recovery_at_execution(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    (tmp_path / "target.txt").write_text("original")
    with store.create(working_directory=str(tmp_path)) as handle:
        code = "await self.shell.write_file('target.txt', 'changed')"
        executor = file_edit_executor(
            code,
            cwd=tmp_path,
            event_manager=EventManager(),
            max_file_bytes=4096,
            timeout=2,
            journal=handle.actions,
        )
        assert executor is not None
        action = handle.actions.begin("shell.run", "Host shell command", str(tmp_path))
        session_id = handle.id
    with store.open(session_id) as restored:
        from dataclasses import replace

        executor = replace(executor, _journal=restored.actions)
        result = await executor.run_cell(code)
        assert isinstance(result.error, PendingSessionActionsError)
        assert (tmp_path / "target.txt").read_text() == "original"
        assert restored.actions.pending()[0]["action_id"] == action
