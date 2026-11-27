# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentSnapshot edge cases.

Covers:
- Line 101: ``_agentdoc_``-prefixed fields are skipped in from_agent()
- Line 105: callable attributes are skipped in from_agent() removed dynamic method restoration from AgentSnapshot — methods live on
the class and are never reattached via a snapshot. Tests for the old
``snap.methods`` / ``restore()`` method-warning paths have been removed.
"""

import pytest

from aster import Agent, Context
from aster.context_blocks import DynamicContext
from aster.context_blocks.exceptions import ProtectedBlockError
from aster.storage.snapshot import SNAPSHOT_VERSION, AgentSnapshot
from aster.unifiedllm import FakeLLMClient


class _SimpleAgent(Agent, llm=FakeLLMClient()):
    value: int = 0


class TestAgentSnapshotFromAgent:
    """Tests for AgentSnapshot.from_agent() field filtering."""

    def test_agentdoc_prefix_fields_are_skipped(self):
        """from_agent() skips attributes whose name starts with '_agentdoc_' (line 101)."""
        agent = _SimpleAgent()
        agent.__dict__["_agentdoc_hidden"] = "should be excluded"
        agent.value = 5

        snap = AgentSnapshot.from_agent(agent)

        assert "_agentdoc_hidden" not in snap.attributes
        assert snap.attributes["value"] == 5

    def test_callable_attributes_are_skipped(self):
        """from_agent() skips callable attributes set on the instance (line 105).: direct ``agent.my_fn = lambda: None`` is now blocked by the
        Agent guard; route the callable straight into ``__dict__`` so we still
        exercise the from_agent() filter.
        """
        agent = _SimpleAgent()
        agent.__dict__["my_fn"] = lambda: None
        agent.value = 7

        snap = AgentSnapshot.from_agent(agent)

        assert "my_fn" not in snap.attributes
        assert snap.attributes["value"] == 7


class _UnsupportedThing:
    """A plain class with no serialization support."""

    pass


class TestFromAgentSkipsUnserializableAttributes:
    """A non-serializable agent attribute must be skipped (and warned), not abort the snapshot."""

    def test_unserializable_attribute_is_skipped_not_raised(self):
        """from_agent() skips an unserializable attribute and still captures the rest."""
        agent = _SimpleAgent()
        agent.bad_attr = _UnsupportedThing()  # type: ignore[attr-defined]
        agent.value = 11

        snap = AgentSnapshot.from_agent(agent)

        assert "bad_attr" not in snap.attributes
        assert snap.attributes["value"] == 11

    def test_unserializable_attribute_emits_warning(self, caplog):
        """Skipping an unserializable attribute logs a warning naming the attribute."""
        import logging

        agent = _SimpleAgent()
        agent.bad_attr = _UnsupportedThing()  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING):
            AgentSnapshot.from_agent(agent)

        assert any("bad_attr" in rec.message for rec in caplog.records)


class TestSnapshotRoundTripWithUnserializableAttr:
    """An unserializable attribute must not prevent durable state (vars) from round-tripping."""

    def test_vars_survive_when_sibling_attr_is_unserializable(self):
        """Snapshot/restore preserves user vars even if another attr can't serialize."""
        from aster.storage.json_snapshot import snapshot_from_dict, snapshot_to_dict

        agent = _SimpleAgent()
        agent.value = 7
        agent.__dict__["_live_only"] = _UnsupportedThing()

        data = snapshot_to_dict(AgentSnapshot.from_agent(agent))
        assert "_live_only" not in data["attributes"]
        assert data["attributes"]["value"] == 7

        restored = _SimpleAgent()
        snapshot_from_dict(data).restore(restored)
        assert restored.value == 7


class TestDynamicContextPlacement:
    @pytest.mark.parametrize("prefix", [False, True])
    @pytest.mark.parametrize("disabled", [False, True])
    def test_expression_placement_round_trip(self, prefix, disabled):
        agent = _SimpleAgent()
        agent.context["status"] = Context(expr="self.value", prefix=prefix)
        if disabled:
            agent.context_manager.disable("status")

        snapshot = AgentSnapshot.from_agent(agent).model_dump_json()
        restored = _SimpleAgent()
        AgentSnapshot.model_validate_json(snapshot).restore(restored)

        raw = dict(restored.context_manager._raw_items())["status"]
        assert isinstance(raw, DynamicContext)
        assert raw.expr == "self.value"
        assert restored.context_manager.is_static("status") is prefix
        assert restored.context_manager.is_disabled("status") is disabled

    def test_legacy_expression_defaults_to_suffix(self):
        snapshot = AgentSnapshot.model_validate(
            {
                "version": SNAPSHOT_VERSION,
                "context": [{"key": "status", "type": "dynamic", "expr": "self.value"}],
            }
        )
        restored = _SimpleAgent()
        snapshot.restore(restored)

        assert not restored.context_manager.is_static("status")
        assert dict(restored.context_manager._raw_items())["status"].expr == "self.value"

    @pytest.mark.parametrize("prefix", [False, True])
    def test_external_protected_expression_keeps_protection(self, prefix):
        snapshot = AgentSnapshot.model_validate(
            {
                "version": SNAPSHOT_VERSION,
                "context": [
                    {"key": "state", "type": "dynamic", "expr": "self.value", "prefix": prefix}
                ],
                "disabled_context": ["state"],
            }
        )
        restored = _SimpleAgent()
        assert "state" in restored.context_manager.protected_keys
        snapshot.restore(restored)

        assert restored.context_manager.is_static("state") is prefix
        assert restored.context_manager.is_disabled("state")
        assert dict(restored.context_manager._raw_items())["state"].expr == "self.value"
        with pytest.raises(ProtectedBlockError):
            restored.context_manager.set_dynamic("state", "None")
        assert all(block.key != "state" for block in AgentSnapshot.from_agent(restored).context)


class TestOptionalContextSnapshot:
    @pytest.mark.parametrize("prefix", [False, True])
    @pytest.mark.parametrize("expression", [False, True])
    @pytest.mark.parametrize("disabled", [False, True])
    def test_optional_block_round_trip(self, prefix, expression, disabled):
        agent = _SimpleAgent()
        content = {"expr": "self.value"} if expression else {"value": "catalog"}
        agent.context["catalog"] = Context(**content, prefix=prefix, optional=True)
        if disabled:
            agent.context_manager.disable("catalog")

        snapshot = AgentSnapshot.from_agent(agent).model_dump_json()
        saved = AgentSnapshot.model_validate_json(snapshot)
        block = next(block for block in saved.context if block.key == "catalog")
        assert block.optional

        restored = _SimpleAgent()
        saved.restore(restored)
        assert restored.context_manager.is_optional("catalog")
        assert restored.context_manager.is_disabled("catalog") is disabled
        raw = dict(restored.context_manager._raw_items())["catalog"]
        if expression:
            assert isinstance(raw, DynamicContext)
            assert raw.expr == "self.value"
            assert restored.context_manager.is_static("catalog") is prefix
        else:
            assert raw == "catalog"
            # Literal placement remains unchanged by this snapshot-format addition.
            assert not restored.context_manager.is_static("catalog")

    @pytest.mark.parametrize("expression", [False, True])
    def test_legacy_block_is_required(self, expression):
        block = {"key": "catalog", "type": "dynamic", "expr": "self.value"}
        if not expression:
            block = {"key": "catalog", "type": "static", "value": "catalog"}
        restored = _SimpleAgent()
        restored.context["catalog"] = Context("previous", optional=True)
        AgentSnapshot.model_validate({"context": [block]}).restore(restored)

        assert not restored.context_manager.is_optional("catalog")

    @pytest.mark.parametrize("expression", [False, True])
    def test_optional_protected_block_stays_protected(self, expression):
        block = {"key": "state", "type": "static", "value": ["catalog"], "optional": True}
        if expression:
            block = {
                "key": "state",
                "type": "dynamic",
                "expr": "self.value",
                "optional": True,
                "prefix": True,
            }
        restored = _SimpleAgent()
        AgentSnapshot.model_validate({"context": [block], "disabled_context": ["state"]}).restore(
            restored
        )

        assert restored.context_manager.is_optional("state")
        assert restored.context_manager.is_disabled("state")
        with pytest.raises(ProtectedBlockError):
            restored.context_manager.set_dynamic("state", "None")
        if not expression:
            assert dict(restored.context_manager._raw_items())["state"] == ["catalog"]

    def test_optional_literal_keeps_arbitrary_value_type(self):
        agent = _SimpleAgent()
        agent.context_manager.set_dynamic("catalog", value=[1, {"x": 2}])
        agent.context_manager.set_optional("catalog", True)
        restored = _SimpleAgent()
        AgentSnapshot.model_validate_json(
            AgentSnapshot.from_agent(agent).model_dump_json()
        ).restore(restored)

        assert dict(restored.context_manager._raw_items())["catalog"] == [1, {"x": 2}]
        assert restored.context_manager.is_optional("catalog")

    def test_legacy_null_literal_still_suppresses_block(self):
        restored = _SimpleAgent()
        restored.context["catalog"] = "previous"
        AgentSnapshot.model_validate(
            {"context": [{"key": "catalog", "type": "static", "value": None}]}
        ).restore(restored)

        assert "catalog" not in restored.context_manager
        assert restored.context_manager.is_disabled("catalog")
