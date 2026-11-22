# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An interrupted prompt must not lose state behind already saved actions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest
from aster_cli.sessions import SessionCheckpointUnavailableError, SessionStore
from aster_cli.sessions.events import SessionSnapshotInvalidated

from aster import Agent
from aster.errors.storage import SerializationError
from aster.interactive import AgentMessage
from aster.unifiedllm import FakeLLMClient


class RestoreAgent(Agent, llm=FakeLLMClient()):
    pass


def _agent(handle):
    agent = RestoreAgent(storage=handle.storage)
    agent.completed_operations = []
    return agent


def _completed_turn(handle, agent):
    handle.checkpoint(agent)
    handle.record_user_message("Append one line")
    action_id = handle.actions.begin("file.write_file", "append.txt", "/workspace")
    handle.actions.finish(action_id, outcome="Tool returned")
    agent.completed_operations.append(action_id)
    reply = AgentMessage(content="Appended once")
    handle.events.add(reply)
    handle.checkpoint_completed(agent, completed_message_id=reply.id)
    handle.save_current_snapshot(agent)
    return action_id


@pytest.mark.parametrize("record_prompt", [False, True])
@pytest.mark.parametrize("later_outcome", ["no action", None, "Tool returned"])
def test_interrupted_prompt_restores_pre_turn_state_and_preserves_later_actions(
    tmp_path, record_prompt, later_outcome
):
    store = SessionStore(tmp_path)
    with store.create(session_id="interrupted") as handle:
        agent = _agent(handle)
        saved = _completed_turn(handle, agent)
        checkpoint = handle.checkpoint(agent)
        handle.invalidate_current_snapshot(checkpoint_id=checkpoint)
        if record_prompt:
            handle.record_user_message("Continue")
        later = None
        if later_outcome != "no action":
            later = handle.actions.begin("python.execute", "Python cell", "/workspace")
            if later_outcome is not None:
                handle.actions.finish(later, outcome=later_outcome)
            agent.completed_operations.append(later)

    with store.open("interrupted") as resumed:
        fresh = _agent(resumed)
        assert resumed.restore_current_snapshot(fresh)
        assert fresh.completed_operations == [saved]
        assert {item["action_id"] for item in resumed.actions.pending()} == (
            {later} if later is not None else set()
        )
        assert [turn.content for turn in resumed.turns() if turn.role == "user"] == (
            ["Append one line", "Continue"] if record_prompt else ["Append one line"]
        )
        # Reading the fallback never makes its later uncertain work saved.
        assert not resumed.storage.restore_current_snapshot(_agent(resumed))


def test_current_snapshot_takes_precedence_over_the_previous_restore_boundary(tmp_path):
    with SessionStore(tmp_path).create(session_id="current") as handle:
        agent = _agent(handle)
        first = _completed_turn(handle, agent)
        handle.invalidate_current_snapshot(checkpoint_id=handle.checkpoint(agent))
        second = _completed_turn(handle, agent)
        fresh = _agent(handle)
        assert handle.restore_current_snapshot(fresh)
        assert fresh.completed_operations == [first, second]
        assert not handle.actions.pending()


def test_failed_restore_binding_does_not_invalidate_saved_state(tmp_path, monkeypatch):
    with SessionStore(tmp_path).create(session_id="binding-failure") as handle:
        agent = _agent(handle)
        action = _completed_turn(handle, agent)
        checkpoint = handle.checkpoint(agent)
        original_add = handle.events.add

        def fail_binding(event):
            if isinstance(event, SessionSnapshotInvalidated):
                raise OSError("restore binding unavailable")
            return original_add(event)

        monkeypatch.setattr(handle.events, "add", fail_binding)
        with pytest.raises(OSError, match="binding unavailable"):
            handle.invalidate_current_snapshot(checkpoint_id=checkpoint)
        fresh = _agent(handle)
        assert handle.storage.restore_current_snapshot(fresh)
        assert fresh.completed_operations == [action]


def test_failed_pre_turn_checkpoint_cannot_reuse_an_older_point(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create(session_id="failed-checkpoint") as handle:
        agent = _agent(handle)
        _completed_turn(handle, agent)

        def unavailable(*args, **kwargs):
            raise SerializationError("cannot capture current task")

        monkeypatch.setattr(handle.storage, "save_checkpoint", unavailable)
        with pytest.raises(SerializationError):
            handle.checkpoint(agent)
        handle.invalidate_current_snapshot(checkpoint_id=None)
        handle.record_user_message("Continue")
    with store.open("failed-checkpoint") as resumed:
        with pytest.raises(SessionCheckpointUnavailableError, match="task state"):
            resumed.restore_current_snapshot(_agent(resumed))


@pytest.mark.parametrize("damage", ["missing", "snapshot", "event", "foreign", "binding"])
def test_unusable_or_foreign_restore_points_fail_closed(tmp_path, damage):
    store = SessionStore(tmp_path)
    with store.create(session_id="damaged") as handle:
        agent = _agent(handle)
        _completed_turn(handle, agent)
        checkpoint = handle.checkpoint(agent)
        handle.invalidate_current_snapshot(checkpoint_id=checkpoint)
        handle.record_user_message("Continue")
    with closing(sqlite3.connect(store.path_for("damaged"))) as connection, connection:
        if damage == "missing":
            connection.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint,))
        elif damage == "snapshot":
            connection.execute(
                "UPDATE checkpoints SET snapshot = ? WHERE checkpoint_id = ?", ("{}", checkpoint)
            )
        elif damage == "event":
            connection.execute("DELETE FROM checkpoint_events")
        else:
            (payload,) = connection.execute(
                "SELECT data FROM events WHERE event_type = 'SessionSnapshotInvalidated'"
            ).fetchone()
            raw = json.loads(payload)
            if damage == "foreign":
                raw["session_id"] = "another-branch"
            else:
                raw = {}
            connection.execute(
                "UPDATE events SET data = ? WHERE event_type = 'SessionSnapshotInvalidated'",
                (json.dumps(raw),),
            )
    with store.open("damaged") as resumed:
        fresh = _agent(resumed)
        with pytest.raises(SessionCheckpointUnavailableError, match="task state"):
            resumed.restore_current_snapshot(fresh)
        assert fresh.completed_operations == []


def test_saved_action_proof_must_belong_to_the_selected_state(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="missing-newer-state") as handle:
        agent = _agent(handle)
        _completed_turn(handle, agent)
        handle.invalidate_current_snapshot(checkpoint_id=handle.checkpoint(agent))
        _completed_turn(handle, agent)
        # Losing a later current snapshot cannot justify borrowing its old proof.
        handle.storage.invalidate_current_snapshot()
    with store.open("missing-newer-state") as resumed:
        with pytest.raises(SessionCheckpointUnavailableError, match="task state"):
            resumed.restore_current_snapshot(_agent(resumed))


def test_later_human_review_does_not_claim_progress_in_an_earlier_restore_point(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="later-review") as handle:
        agent = _agent(handle)
        saved = _completed_turn(handle, agent)
        handle.invalidate_current_snapshot(checkpoint_id=handle.checkpoint(agent))
        later = handle.actions.begin("python.execute", "Python cell", "/workspace")
    with store.open("later-review") as handle:
        handle.actions.review(later, "Checked the effect before saving a later state")
    with store.open("later-review") as resumed:
        fresh = _agent(resumed)
        assert resumed.restore_current_snapshot(fresh)
        assert fresh.completed_operations == [saved]
        assert {item["action_id"] for item in resumed.actions.pending()} == {later}


def test_missing_current_with_saved_actions_and_no_restore_binding_fails_closed(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="unbound") as handle:
        _completed_turn(handle, _agent(handle))
        handle.storage.invalidate_current_snapshot()
    with store.open("unbound") as resumed:
        with pytest.raises(SessionCheckpointUnavailableError, match="task state"):
            resumed.restore_current_snapshot(_agent(resumed))


def test_legacy_history_without_host_actions_does_not_require_a_new_restore_binding(tmp_path):
    with SessionStore(tmp_path).create(session_id="legacy") as handle:
        handle.record_user_message("An old conversation without a task snapshot")
        assert not handle.restore_current_snapshot(_agent(handle))
