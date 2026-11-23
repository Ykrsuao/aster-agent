# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side effects cannot be replayed merely because their progress was lost."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest
from aster_cli.sessions import SessionCheckpointUnavailableError, SessionStore
from aster_cli.sessions.actions import PendingSessionActionsError
from aster_cli.sessions.events import (
    SessionActionIntent,
    SessionActionOutcome,
    SessionActionReviewed,
    SessionActionsInherited,
    SessionActionsSaved,
)

from aster import Agent
from aster.context_blocks.roles import Role
from aster.interactive import AgentMessage
from aster.runtime.event_manager import EventManager
from aster.unifiedllm import FakeLLMClient


class ActionAgent(Agent, llm=FakeLLMClient()):
    pass


def _agent(handle):
    agent = ActionAgent(storage=handle.storage)
    if not handle.storage.restore_current_snapshot(agent):
        agent.completed_operations = []
    return agent


def _begin(handle):
    return handle.actions.begin("file_edit", "report.txt", "/workspace")


def _pending_ids(handle):
    return {item["action_id"] for item in handle.actions.pending()}


def _records(handle, kind):
    return [event for event in handle.events.all_events() if isinstance(event, kind)]


def _turn(handle, agent, prompt):
    handle.checkpoint(agent)
    _, user = handle.record_user_message(prompt)
    reply = AgentMessage(content="done")
    handle.events.add(reply)
    handle.save_current_snapshot(agent)
    handle.checkpoint_completed(agent, completed_message_id=reply.id)
    return user.id, reply.id


def _journal_rows(handle):
    with closing(
        sqlite3.connect(handle.path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        return connection.execute("SELECT * FROM events ORDER BY insertion_order").fetchall()


def _navigate(handle, navigation, user_id, reply_id):
    if navigation == "rewind":
        return handle.rewind(0)[0]
    return handle.navigate_tree(user_id, reply_id)[0]


def test_intent_storage_failure_prevents_the_callers_side_effect(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    changed = tmp_path / "changed.txt"
    with store.create(session_id="failed-intent") as handle:
        real_add = handle.events.add

        def fail_intent(event):
            if isinstance(event, SessionActionIntent):
                raise sqlite3.OperationalError("disk full")
            return real_add(event)

        monkeypatch.setattr(handle.events, "add", fail_intent)
        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            _begin(handle)
            changed.write_text("side effect", encoding="utf-8")
        assert not changed.exists()
        assert handle.actions.pending() == []
        assert _records(handle, SessionActionIntent) == []


@pytest.mark.parametrize("outcome", [None, "tool returned", "OSError", "CancelledError"])
def test_restart_requires_review_even_when_an_outcome_was_recorded(tmp_path, outcome):
    store = SessionStore(tmp_path)
    with store.create(session_id="interrupted") as handle:
        action_id = _begin(handle)
        if outcome is not None:
            handle.actions.finish(action_id, outcome=outcome)
        handle.actions.require_clear()
        assert handle.actions.pending() == []

    with store.open("interrupted") as resumed:
        assert resumed.actions.pending() == [
            {
                "action_id": action_id,
                "kind": "file_edit",
                "detail": "report.txt",
                "cwd": "/workspace",
                "outcome": outcome,
            }
        ]
        with pytest.raises(PendingSessionActionsError, match="/recovery"):
            resumed.actions.require_clear()
        with pytest.raises(PendingSessionActionsError):
            _begin(resumed)


def test_successful_snapshot_acknowledges_only_finished_local_actions(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="saved") as handle:
        agent = _agent(handle)
        finished = _begin(handle)
        handle.actions.finish(finished, outcome="tool returned")
        agent.completed_operations.append(finished)
        running = _begin(handle)
        handle.save_current_snapshot(agent)
        assert _records(handle, SessionActionsSaved)[-1].action_ids == [finished]

    with store.open("saved") as resumed:
        assert _pending_ids(resumed) == {running}
        assert _agent(resumed).completed_operations == [finished]


@pytest.mark.parametrize("requires_review", [False, True])
def test_explicit_uncertain_result_is_not_acknowledged_by_a_snapshot(tmp_path, requires_review):
    store = SessionStore(tmp_path)
    with store.create(session_id="uncertain-result") as handle:
        action_id = _begin(handle)
        # The flag controls review; outcome text is only a human-readable fact.
        handle.actions.finish(action_id, outcome="Result received", requires_review=requires_review)
        assert _pending_ids(handle) == ({action_id} if requires_review else set())
        handle.save_current_snapshot(_agent(handle))
        assert bool(_records(handle, SessionActionsSaved)) is not requires_review
    with store.open("uncertain-result") as resumed:
        assert _pending_ids(resumed) == ({action_id} if requires_review else set())
        if requires_review:
            resumed.actions.review(action_id, "Checked the partial effect")
            resumed.save_current_snapshot(_agent(resumed))
    with store.open("uncertain-result") as resumed:
        assert not resumed.actions.pending()


@pytest.mark.parametrize("failed_stage", ["snapshot", "proof"])
def test_save_or_proof_failure_keeps_actions_pending(tmp_path, monkeypatch, failed_stage):
    store = SessionStore(tmp_path)
    with store.create(session_id="failed-save") as handle:
        agent = _agent(handle)
        action_id = _begin(handle)
        handle.actions.finish(action_id, outcome="tool returned")

        if failed_stage == "snapshot":

            def fail_save(_agent):
                raise OSError("snapshot write failed")

            monkeypatch.setattr(handle.storage, "save_current_snapshot", fail_save)
        else:
            real_add = handle.events.add

            def fail_proof(event):
                if isinstance(event, SessionActionsSaved):
                    raise OSError("proof write failed")
                return real_add(event)

            monkeypatch.setattr(handle.events, "add", fail_proof)

        with pytest.raises(OSError, match="write failed"):
            handle.save_current_snapshot(agent)
        assert _pending_ids(handle) == {action_id}
        assert _records(handle, SessionActionsSaved) == []

    with store.open("failed-save") as resumed:
        assert _pending_ids(resumed) == {action_id}


def test_outcome_storage_failure_blocks_further_side_effects(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create(session_id="failed-outcome") as handle:
        action_id = _begin(handle)
        real_add = handle.events.add

        def fail_outcome(event):
            if isinstance(event, SessionActionOutcome):
                raise OSError("outcome write failed")
            return real_add(event)

        monkeypatch.setattr(handle.events, "add", fail_outcome)
        with pytest.raises(OSError, match="outcome write failed"):
            handle.actions.finish(action_id, outcome="tool returned")
        with pytest.raises(PendingSessionActionsError):
            _begin(handle)
        handle.save_current_snapshot(_agent(handle))

    with store.open("failed-outcome") as resumed:
        assert _pending_ids(resumed) == {action_id}
        assert resumed.actions.pending()[0]["outcome"] is None


def test_late_task_completion_is_not_claimed_by_an_earlier_snapshot(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create(session_id="late") as handle:
        agent = _agent(handle)
        action_id = _begin(handle)
        real_save = handle.storage.save_current_snapshot

        def finish_after_snapshot(agent):
            real_save(agent)
            handle.actions.finish(action_id, outcome="tool returned")

        monkeypatch.setattr(handle.storage, "save_current_snapshot", finish_after_snapshot)
        handle.save_current_snapshot(agent)
        assert _records(handle, SessionActionsSaved) == []
        assert _pending_ids(handle) == {action_id}
        monkeypatch.setattr(handle.storage, "save_current_snapshot", real_save)
        _turn(handle, agent, "Explain the earlier result")
        assert _pending_ids(handle) == {action_id}

    with store.open("late") as resumed:
        assert _pending_ids(resumed) == {action_id}


@pytest.mark.parametrize("outcome", [None, "tool returned"])
def test_recovered_pending_survives_answers_and_snapshot_saves(tmp_path, outcome):
    store = SessionStore(tmp_path)
    with store.create(session_id="recovered") as handle:
        action_id = _begin(handle)
        if outcome:
            handle.actions.finish(action_id, outcome=outcome)

    with store.open("recovered") as resumed:
        agent = _agent(resumed)
        _turn(resumed, agent, "What happened?")
        resumed.storage.save_current_snapshot(agent)
        assert _pending_ids(resumed) == {action_id}

    with store.open("recovered") as resumed_again:
        assert _pending_ids(resumed_again) == {action_id}


def test_turn_finished_is_not_evidence_that_a_checkpoint_was_saved(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create(session_id="turn-marker") as handle:
        agent = _agent(handle)
        handle.checkpoint(agent)
        handle.record_user_message("write a report")
        action_id = _begin(handle)
        handle.actions.finish(action_id, outcome="tool returned")
        reply = AgentMessage(content="done")
        handle.events.add(reply)

        def fail_checkpoint(_key, _agent):
            raise OSError("checkpoint write failed")

        monkeypatch.setattr(handle.storage, "save_checkpoint", fail_checkpoint)
        with pytest.raises(OSError, match="checkpoint write failed"):
            handle.checkpoint_completed(agent, completed_message_id=reply.id)
        assert any(
            event.event_type == "SessionTurnFinished" for event in handle.events.all_events()
        )

    with store.open("turn-marker") as resumed:
        assert _pending_ids(resumed) == {action_id}


def test_explicit_review_is_durable_and_failed_review_does_not_clear(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    with store.create(session_id="reviewed") as handle:
        action_id = _begin(handle)

    with store.open("reviewed") as resumed:
        real_add = resumed.events.add

        def fail_review(_event):
            raise OSError("review write failed")

        monkeypatch.setattr(resumed.events, "add", fail_review)
        with pytest.raises(OSError, match="review write failed"):
            resumed.actions.review(action_id, "Checked the report on disk")
        assert _pending_ids(resumed) == {action_id}
        monkeypatch.setattr(resumed.events, "add", real_add)
        resumed.actions.review(action_id, "Checked the report on disk")
        resumed.actions.require_clear()
        assert _records(resumed, SessionActionReviewed)[-1].note == "Checked the report on disk"
        with pytest.raises(ValueError, match="not awaiting review"):
            resumed.actions.review(action_id, "Again")

    with store.open("reviewed") as resumed_again:
        assert resumed_again.actions.pending() == []


def test_journals_are_isolated_and_do_not_expose_mutable_records(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="one") as first:
        action_id = _begin(first)
    with store.open("one") as first, store.create(session_id="two") as second:
        assert second.actions.pending() == []
        with pytest.raises(ValueError, match="not awaiting review"):
            second.actions.review(action_id, "Different session")
        returned = first.actions.pending()
        returned[0]["outcome"] = "forged"
        returned.clear()
        assert first.actions.pending()[0]["outcome"] is None
        assert _pending_ids(first) == {action_id}


def test_descriptions_are_bounded_metadata_not_public_transcript(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="bounded") as handle:
        action_id = handle.actions.begin("k" * 100, "d" * 2000, "c" * 5000)
        handle.actions.finish(action_id, outcome="o" * 500)
        intent = _records(handle, SessionActionIntent)[0]
        assert (len(intent.kind), len(intent.detail), len(intent.cwd)) == (64, 1024, 4096)
        assert len(_records(handle, SessionActionOutcome)[0].outcome) == 256
        assert handle.transcript() == []
        assert all(event._role == Role.METADATA for event in handle.events.all_events())


@pytest.mark.parametrize("navigation", ["rewind", "tree"])
@pytest.mark.parametrize("outcome", [None, "tool returned"])
def test_navigation_inherits_unreviewed_or_unsaved_source_actions(tmp_path, navigation, outcome):
    store = SessionStore(tmp_path)
    changed = tmp_path / "report.txt"
    with store.create(session_id="source") as handle:
        agent = _agent(handle)
        user_id, reply_id = _turn(handle, agent, "first")
        action_id = _begin(handle)
        changed.write_text("written once", encoding="utf-8")
        if outcome is not None:
            handle.actions.finish(action_id, outcome=outcome)
        before = _journal_rows(handle)
        branch_id = _navigate(handle, navigation, user_id, reply_id)
        assert _journal_rows(handle) == before

    with store.open(branch_id) as branch:
        assert _pending_ids(branch) == {action_id}
        assert branch.actions.pending()[0]["outcome"] == outcome
        assert _records(branch, SessionActionsInherited)[-1].action_ids == [action_id]
        with pytest.raises(PendingSessionActionsError, match="/recovery"):
            _begin(branch)
        branch.save_current_snapshot(_agent(branch))
        assert _pending_ids(branch) == {action_id}
    with store.open(branch_id) as branch:
        assert _pending_ids(branch) == {action_id}
        branch.actions.review(action_id, "Checked the existing file against this restored state")
        branch.actions.require_clear()
    with store.open("source") as source, store.open(branch_id) as branch:
        assert _pending_ids(source) == {action_id}
        assert branch.actions.pending() == []
        assert _journal_rows(source) == before
    assert changed.read_text(encoding="utf-8") == "written once"


@pytest.mark.parametrize("navigation", ["rewind", "tree"])
@pytest.mark.parametrize("resolution", ["saved", "reviewed"])
def test_older_checkpoint_requires_a_new_review_of_later_actions(tmp_path, navigation, resolution):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as handle:
        agent = _agent(handle)
        user_id, reply_id = _turn(handle, agent, "first")
        action_id = _begin(handle)
        handle.actions.finish(action_id, outcome="tool returned")
        outcome_id = _records(handle, SessionActionOutcome)[0].id
        agent.completed_operations.append(action_id)
        if resolution == "saved":
            handle.save_current_snapshot(agent)
    with store.open("source") as source:
        if resolution == "reviewed":
            source.actions.review(action_id, "Checked the action in the original state")
        assert source.actions.pending() == []
        before = _journal_rows(source)
        branch_id = _navigate(source, navigation, user_id, reply_id)
        assert _journal_rows(source) == before
    with store.open(branch_id) as branch:
        assert _agent(branch).completed_operations == []
        assert _pending_ids(branch) == {action_id}
        assert _records(branch, SessionActionsSaved) == []
        assert _records(branch, SessionActionReviewed) == []
        assert _records(branch, SessionActionOutcome)[0].id == outcome_id
        branch.actions.review(action_id, "Checked again after restoring the earlier state")
        branch.actions.require_clear()
    with store.open("source") as source:
        assert source.actions.pending() == []
        assert _journal_rows(source) == before


def test_navigation_inherits_unreviewed_actions_from_both_branches(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as source:
        agent = _agent(source)
        user_id, reply_id = _turn(source, agent, "first")
        branch_id, _ = source.navigate_tree(user_id, reply_id)
        source_action = _begin(source)
        source.actions.finish(source_action, outcome="tool returned")
        original_rows = _journal_rows(source)
    with store.open(branch_id) as branch:
        branch_action = _begin(branch)
        branch.actions.finish(branch_action, outcome="tool returned")
        branch_rows = _journal_rows(branch)
        restored_id, _ = branch.navigate_tree(reply_id, None)
        assert _journal_rows(branch) == branch_rows
    with store.open(restored_id) as restored:
        assert _pending_ids(restored) == {source_action, branch_action}
        assert [turn.content for turn in restored.turns()] == ["first", "done"]
    with store.open("source") as source, store.open(branch_id) as branch:
        assert _journal_rows(source) == original_rows
        assert _journal_rows(branch) == branch_rows
        assert _pending_ids(source) == {source_action}
        assert _pending_ids(branch) == {branch_action}


@pytest.mark.parametrize("outcome_known_at_fork", [False, True])
def test_pending_cannot_be_cleared_by_navigating_to_a_previously_resolved_branch(
    tmp_path, outcome_known_at_fork
):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as source:
        agent = _agent(source)
        first_user, first_reply = _turn(source, agent, "first")
        action_id = _begin(source)
        if outcome_known_at_fork:
            source.actions.finish(action_id, outcome="tool returned")
        branch_id, _ = source.navigate_tree(first_user, first_reply)
        if not outcome_known_at_fork:
            source.actions.finish(action_id, outcome="tool returned")
        agent.completed_operations.append(action_id)
        _, last_reply = _turn(source, agent, "second")
        original_rows = _journal_rows(source)
    with store.open(branch_id) as branch:
        assert _pending_ids(branch) == {action_id}
        restored_id, _ = branch.navigate_tree(last_reply, None)
    with store.open(restored_id) as restored:
        assert _agent(restored).completed_operations == [action_id]
        assert _pending_ids(restored) == {action_id}
        assert restored.actions.pending()[0]["outcome"] == "tool returned"
        restored.save_current_snapshot(_agent(restored))
        assert _pending_ids(restored) == {action_id}
        restored.actions.review(action_id, "Checked the action in this branch")
        again_id, _ = restored.rewind(0)
    with store.open(again_id) as again:
        assert _pending_ids(again) == {action_id}
    with store.open("source") as source:
        assert _journal_rows(source) == original_rows
        assert source.actions.pending() == []


@pytest.mark.parametrize("damage", ["intent", "outcome", "payload", "identity"])
def test_conflicting_or_damaged_branch_journals_fail_closed(tmp_path, damage):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as source:
        user_id, reply_id = _turn(source, _agent(source), "first")
        action_id = _begin(source)
        source.actions.finish(action_id, outcome="tool returned")
        branch_id, _ = source.navigate_tree(user_id, reply_id)
        original_rows = _journal_rows(source)
    with closing(sqlite3.connect(store.path_for(branch_id))) as connection:
        event_type = "SessionActionOutcome" if damage == "outcome" else "SessionActionIntent"
        (payload,) = connection.execute(
            "SELECT data FROM events WHERE event_type = ?", (event_type,)
        ).fetchone()
        raw = json.loads(payload)
        if damage == "intent":
            raw["detail"] = "different-file.txt"
        elif damage == "outcome":
            raw["outcome"] = "OSError"
        elif damage == "identity":
            raw["id"] = "conflicting-identity"
        else:
            raw = {}
        with connection:
            connection.execute(
                "UPDATE events SET data = ? WHERE event_type = ?", (json.dumps(raw), event_type)
            )
    with store.open(branch_id) as branch:
        damaged_rows = _journal_rows(branch)
        with pytest.raises(SessionCheckpointUnavailableError):
            branch.navigate_tree(reply_id, None)
        assert _journal_rows(branch) == damaged_rows
    with store.open("source") as source:
        assert _journal_rows(source) == original_rows
    assert {path.stem for path in tmp_path.glob("*.db")} == {"source", branch_id}


@pytest.mark.parametrize("conflicting_event_type", ["SessionActionIntent", "SessionActionOutcome"])
def test_cross_branch_event_identity_collision_fails_before_inheritance(
    tmp_path, conflicting_event_type
):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as source:
        user_id, reply_id = _turn(source, _agent(source), "first")
        branch_id, _ = source.navigate_tree(user_id, reply_id)
        action_id = _begin(source)
        source.actions.finish(action_id, outcome="tool returned")
        colliding_id = _records(source, SessionActionOutcome)[0].id
        source_rows = _journal_rows(source)
    with store.open(branch_id) as branch:
        action_id = _begin(branch)
        if conflicting_event_type == "SessionActionOutcome":
            branch.actions.finish(action_id, outcome="tool returned")
    with closing(sqlite3.connect(store.path_for(branch_id))) as connection:
        (payload,) = connection.execute(
            "SELECT data FROM events WHERE event_type = ?", (conflicting_event_type,)
        ).fetchone()
        raw = json.loads(payload)
        raw["id"] = colliding_id
        with connection:
            connection.execute(
                "UPDATE events SET event_id = ?, data = ? WHERE event_type = ?",
                (colliding_id, json.dumps(raw), conflicting_event_type),
            )
    with store.open(branch_id) as branch:
        branch_rows = _journal_rows(branch)
        with pytest.raises(SessionCheckpointUnavailableError):
            branch.navigate_tree(reply_id, None)
        assert _journal_rows(branch) == branch_rows
    with store.open("source") as source:
        assert _journal_rows(source) == source_rows
    assert {path.stem for path in tmp_path.glob("*.db")} == {"source", branch_id}


@pytest.mark.parametrize("navigation", ["rewind", "tree"])
@pytest.mark.parametrize(
    "failed_event", [SessionActionIntent, SessionActionOutcome, SessionActionsInherited]
)
def test_failed_inheritance_never_delivers_a_partial_branch(
    tmp_path, monkeypatch, navigation, failed_event
):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as source:
        user_id, reply_id = _turn(source, _agent(source), "first")
        action_id = _begin(source)
        source.actions.finish(action_id, outcome="tool returned")
        before = _journal_rows(source)
        real_add = EventManager.add

        def fail_inheritance(manager, event, **kwargs):
            if isinstance(event, failed_event):
                raise OSError("inheritance write failed")
            return real_add(manager, event, **kwargs)

        monkeypatch.setattr(EventManager, "add", fail_inheritance)
        with pytest.raises((OSError, SessionCheckpointUnavailableError)):
            _navigate(source, navigation, user_id, reply_id)
        assert _journal_rows(source) == before
    assert {path.stem for path in tmp_path.glob("*.db")} == {"source"}


@pytest.mark.parametrize("navigation", ["rewind", "tree"])
def test_navigation_preserves_actions_when_the_selected_checkpoint_includes_them(
    tmp_path, navigation
):
    store = SessionStore(tmp_path)
    with store.create(session_id="source") as source:
        agent = _agent(source)
        action_id = _begin(source)
        source.actions.finish(action_id, outcome="tool returned")
        agent.completed_operations.append(action_id)
        source.save_current_snapshot(agent)
        user_id, reply_id = _turn(source, agent, "first")
        branch_id = _navigate(source, navigation, user_id, reply_id)

    with store.open(branch_id) as branch:
        assert branch.actions.pending() == []
        assert _agent(branch).completed_operations == [action_id]
        assert [event.id for event in _records(branch, SessionActionIntent)] == [action_id]


def test_provisional_cleanup_and_listing_do_not_hide_side_effect_records(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="prepared", provisional=True) as prepared:
        action_id = _begin(prepared)
        assert [item.id for item in store.list()] == [prepared.id]
    with store.open("prepared") as resumed:
        assert _pending_ids(resumed) == {action_id}


def test_unreadable_journal_does_not_turn_into_an_empty_pending_list(tmp_path):
    store = SessionStore(tmp_path)
    with store.create(session_id="damaged") as handle:
        _begin(handle)
    with sqlite3.connect(store.path_for("damaged")) as connection:
        connection.execute(
            "UPDATE events SET data = ? WHERE event_type = ?", ("{}", "SessionActionIntent")
        )
    with store.open("damaged") as resumed:
        with pytest.raises(ValueError):
            resumed.actions.require_clear()
