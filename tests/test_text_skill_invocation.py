# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-skill invocation policy, exact lookup and progressive disclosure."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from aster import Agent
from aster.agentdoc import doc
from aster.skill import Skill, TextSkill
from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient


def _write_skill(root, directory="review", *, metadata=None, body="FULL SKILL BODY", policy=None):
    path = root / directory
    path.mkdir(parents=True, exist_ok=True)
    fields = {"name": "Review changes", "description": "Inspect code changes."}
    fields.update(metadata or {})
    (path / "SKILL.md").write_text(f"---\n{yaml.safe_dump(fields)}---\n{body}\n", encoding="utf-8")
    if policy is not None:
        (path / "agents").mkdir(exist_ok=True)
        (path / "agents" / "openai.yaml").write_text(policy, encoding="utf-8")
    return path


@pytest.fixture
def registry():
    with patch("aster.skill_registry.entry_points", return_value=[]):
        registry = SkillRegistry(SimpleNamespace())
    yield registry
    registry.close()


def test_text_skill_metadata_preserves_directory_id_and_defaults(tmp_path):
    skill = TextSkill(path=_write_skill(tmp_path, "source_dir"))

    assert skill.id == "source-dir"
    assert skill.name == "Review changes"
    assert skill.argument_hint == "[task]"
    assert skill.user_invocable is True
    assert skill.allow_implicit_invocation is True


def test_text_skill_frontmatter_controls_invocation_metadata(tmp_path):
    skill = TextSkill(
        path=_write_skill(
            tmp_path,
            metadata={
                "argument-hint": "<issue> [notes]",
                "user-invocable": False,
                "disable-model-invocation": True,
            },
        ),
        id="custom-id",
    )

    assert skill.id == "custom-id"
    assert skill.name == "Review changes"
    assert skill.argument_hint == "<issue> [notes]"
    assert skill.user_invocable is False
    assert skill.allow_implicit_invocation is False


def test_multiline_description_keeps_skill_trigger_rules_in_catalog(registry, tmp_path):
    description = "Inspect code changes.\nUse when a patch conflicts or a review is requested."
    skill = TextSkill(path=_write_skill(tmp_path, metadata={"description": description}))
    registry.register("custom.review", skill)

    assert skill.description == description
    assert description in registry.status()
    assert "FULL SKILL BODY" not in registry.status()


@pytest.mark.parametrize("name", ["", "  ", None])
def test_empty_text_skill_name_falls_back_to_id(tmp_path, name):
    skill = TextSkill(path=_write_skill(tmp_path, metadata={"name": name}), id="custom-id")
    assert skill.name == "custom-id"


@pytest.mark.parametrize("disable_model", [False, True])
@pytest.mark.parametrize("allow_implicit", [False, True])
def test_openai_policy_and_frontmatter_must_both_allow_implicit_invocation(
    tmp_path, disable_model, allow_implicit
):
    skill = TextSkill(
        path=_write_skill(
            tmp_path,
            metadata={"disable-model-invocation": disable_model},
            policy=yaml.safe_dump({"policy": {"allow_implicit_invocation": allow_implicit}}),
        )
    )
    assert skill.allow_implicit_invocation is (allow_implicit and not disable_model)
    assert skill.user_invocable is True


@pytest.mark.parametrize("field", ["user-invocable", "disable-model-invocation"])
@pytest.mark.parametrize("value", ["false", 0, None, []])
def test_frontmatter_invocation_flags_reject_non_booleans(tmp_path, field, value):
    with pytest.raises(ValueError, match=f"{field!r} must be a boolean"):
        TextSkill(path=_write_skill(tmp_path, metadata={field: value}))


@pytest.mark.parametrize("value", ["false", 0, None, []])
def test_openai_invocation_policy_rejects_non_booleans_even_when_disabled(tmp_path, value):
    with pytest.raises(ValueError, match="allow_implicit_invocation.*boolean"):
        TextSkill(
            path=_write_skill(
                tmp_path,
                metadata={"disable-model-invocation": True},
                policy=yaml.safe_dump({"policy": {"allow_implicit_invocation": value}}),
            )
        )


@pytest.mark.parametrize("policy", ["[]", "policy: []", "policy: false"])
def test_openai_policy_rejects_invalid_structure(tmp_path, policy):
    with pytest.raises(ValueError, match="YAML mapping"):
        TextSkill(path=_write_skill(tmp_path, policy=policy))


def test_unrelated_openai_configuration_keeps_default_policy(tmp_path):
    skill = TextSkill(path=_write_skill(tmp_path, policy="interface:\n  display_name: Review\n"))
    assert skill.allow_implicit_invocation is True


def test_text_skills_returns_current_objects_without_exposing_mutable_cache(registry, tmp_path):
    old = TextSkill(path=_write_skill(tmp_path))
    registry.register("cmd.review", old)
    registry.register("test.tool", Skill(content="A Python tool."))
    assert registry.text_skills() == {"cmd.review": old}
    registry.text_skills().clear()
    assert registry.text_skills() == {"cmd.review": old}

    replacement = TextSkill(path=_write_skill(tmp_path, body="REPLACEMENT BODY"))
    registry.register("cmd.review", replacement)
    assert registry.text_skills() == {"cmd.review": replacement}


@pytest.mark.parametrize("selector", ["PLUGIN.REVIEW", "review CHANGES", "REVIEW"])
def test_text_skill_lookup_matches_full_name_name_and_id_case_insensitively(
    registry, tmp_path, selector
):
    skill = TextSkill(path=_write_skill(tmp_path))
    registry.register("plugin.review", skill)
    assert registry.resolve_text_skill(selector) == ("plugin.review", skill)


def test_fully_qualified_name_takes_priority_over_another_skills_alias(registry, tmp_path):
    exact = TextSkill(path=_write_skill(tmp_path, "first"))
    alias = TextSkill(path=_write_skill(tmp_path, "second", metadata={"name": "plugin.first"}))
    registry.register("plugin.first", exact)
    registry.register("plugin.second", alias)
    assert registry.resolve_text_skill("PLUGIN.FIRST") == ("plugin.first", exact)


@pytest.mark.parametrize("selector", ["Review changes", "review"])
def test_ambiguous_name_or_id_requires_full_registry_name(registry, tmp_path, selector):
    first = TextSkill(path=_write_skill(tmp_path / "a"))
    second = TextSkill(path=_write_skill(tmp_path / "b"))
    registry.register("one.first", first)
    registry.register("two.second", second)

    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve_text_skill(selector)
    assert registry.resolve_text_skill("one.first") == ("one.first", first)
    assert registry.resolve_text_skill("two.second") == ("two.second", second)


def test_shared_attribute_cannot_silently_resolve_to_another_text_skill(registry, tmp_path):
    registry.register("one.shared", TextSkill(path=_write_skill(tmp_path, "first")))
    registry.register("two.shared", TextSkill(path=_write_skill(tmp_path, "second")))

    for selector in ("one.shared", "two.shared"):
        with pytest.raises(ValueError, match="shared binding"):
            registry.resolve_text_skill(selector)
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve_text_skill("second")
    assert "Inspect code changes." not in registry.status()


def test_python_skill_overwriting_text_binding_cannot_be_read_as_text(registry, tmp_path):
    registry.register("one.shared", TextSkill(path=_write_skill(tmp_path)))
    registry.register("two.shared", Skill(content="Python tool"))
    assert registry.text_skills() == {}
    with pytest.raises(ValueError, match="not found"):
        registry.read_text_skill("one.shared")


def test_casefold_collision_with_unloaded_nontext_registration_is_rejected(registry, tmp_path):
    registry.register("cmd.review", TextSkill(path=_write_skill(tmp_path)))
    registry._discovered["CMD.REVIEW"] = SimpleNamespace(entry_point=None)
    with pytest.raises(ValueError, match="conflicts with other registered names"):
        registry.resolve_text_skill("cmd.review")


def test_casefold_collision_between_text_registrations_is_rejected(registry, tmp_path):
    registry.register("one.Review", TextSkill(path=_write_skill(tmp_path, "first")))
    registry.register("ONE.review", TextSkill(path=_write_skill(tmp_path, "second")))
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve_text_skill("one.Review")


def test_lookup_and_read_neither_expand_globs_nor_activate_skills(registry, tmp_path):
    skill = TextSkill(path=_write_skill(tmp_path))
    registry.register("cmd.[review]", skill)
    with patch.object(registry, "activate", side_effect=AssertionError("must not activate")):
        for selector in ("*", "cmd.*", "review?", "missing"):
            with pytest.raises(ValueError, match="not found"):
                registry.resolve_text_skill(selector)
        assert registry.resolve_text_skill("CMD.[REVIEW]") == ("cmd.[review]", skill)
        content = registry.read_text_skill("review")
    assert f"source_dir: {skill.source_dir}" in content
    assert content.endswith(type(skill).__doc__)
    assert registry.activated() == []


@pytest.mark.parametrize("user_invocable", [False, True])
@pytest.mark.parametrize("allow_implicit", [False, True])
def test_explicit_and_implicit_invocation_respect_independent_policies(
    registry, tmp_path, user_invocable, allow_implicit
):
    skill = TextSkill(
        path=_write_skill(
            tmp_path,
            metadata={
                "user-invocable": user_invocable,
                "disable-model-invocation": not allow_implicit,
            },
        )
    )
    registry.register("cmd.review", skill)
    for explicit, allowed in ((True, user_invocable), (False, allow_implicit)):
        if allowed:
            assert registry.resolve_text_skill("review", user_invoked=explicit) == (
                "cmd.review",
                skill,
            )
        else:
            with pytest.raises(ValueError, match="does not allow"):
                registry.resolve_text_skill("review", user_invoked=explicit)
    if allow_implicit:
        assert "FULL SKILL BODY" in registry.read_text_skill("review")
    else:
        with pytest.raises(ValueError, match="implicit invocation"):
            registry.read_text_skill("review")


def test_catalog_uses_text_skill_type_and_contains_only_eligible_summaries(registry, tmp_path):
    visible = TextSkill(path=_write_skill(tmp_path, "visible"))
    inactive = TextSkill(path=_write_skill(tmp_path, "inactive"))
    disabled = TextSkill(
        path=_write_skill(
            tmp_path,
            "disabled",
            metadata={"name": "No automatic use", "disable-model-invocation": True},
            body="DISABLED BODY",
        )
    )
    registry.register("custom.visible", visible)
    registry.register("custom.inactive", inactive)
    registry.register("custom.disabled", disabled)
    registry.activate(["custom.visible", "custom.disabled"])

    catalog = registry.status()
    assert "Text Skills" in catalog
    assert "custom.visible (Review changes): Inspect code changes." in catalog
    assert "custom.inactive (Review changes): Inspect code changes." in catalog
    assert str(visible.source_dir) in catalog
    assert "read_text_skill(" in catalog
    assert "source_dir" in catalog
    assert "current turn only" in catalog
    for absent in (
        "FULL SKILL BODY",
        "DISABLED BODY",
        "No automatic use",
        "custom.disabled",
        "Active Skills",
        "Available Skills",
    ):
        assert absent not in catalog


async def test_reload_refreshes_text_metadata_objects_and_full_instructions(registry, tmp_path):
    _write_skill(tmp_path, body="ORIGINAL BODY")
    registry.discover_skills_dirs([tmp_path])
    original = registry.text_skills()["cmd.review"]
    _write_skill(
        tmp_path,
        metadata={"name": "Changed name", "argument-hint": "<issue>", "user-invocable": False},
        body="UPDATED BODY",
        policy="policy:\n  allow_implicit_invocation: false\n",
    )

    assert await registry.reload("cmd.review") == "Reloaded cmd.review (self.review)"
    current = registry.text_skills()["cmd.review"]
    assert current is not original
    assert current is registry["cmd.review"]
    assert current.name == "Changed name"
    assert current.argument_hint == "<issue>"
    assert current.user_invocable is False
    assert current.allow_implicit_invocation is False
    assert "UPDATED BODY" in type(current).__doc__
    with pytest.raises(ValueError, match="implicit invocation"):
        registry.read_text_skill("Changed name")
    assert "Changed name" not in registry.status()

    _write_skill(tmp_path, metadata={"name": "Latest name"}, body="LATEST BODY", policy="{}")
    await registry.reload("cmd.review")
    latest = registry.text_skills()["cmd.review"]
    assert latest is not current
    assert latest.user_invocable is True
    assert latest.allow_implicit_invocation is True
    assert registry.resolve_text_skill("Latest name") == ("cmd.review", latest)
    content = registry.read_text_skill("review")
    assert "LATEST BODY" in content
    assert "UPDATED BODY" not in content
    with pytest.raises(ValueError, match="not found"):
        registry.resolve_text_skill("Changed name")


async def test_failed_reload_preserves_previous_invocation_policy_and_instructions(
    registry, tmp_path
):
    _write_skill(tmp_path, body="WORKING BODY")
    registry.discover_skills_dirs([tmp_path])
    original = registry.text_skills()["cmd.review"]
    _write_skill(tmp_path, metadata={"disable-model-invocation": "false"}, body="INVALID BODY")

    assert (await registry.reload("cmd.review")).startswith("Reload failed")
    assert registry.text_skills()["cmd.review"] is original
    assert "WORKING BODY" in registry.read_text_skill("review")


class _CatalogAgent(Agent, llm=FakeLLMClient()):
    async def handle(self) -> str:
        """Handle the current task."""
        ...


async def test_initial_model_context_has_catalog_but_no_unselected_skill_body(tmp_path):
    llm = FakeLLMClient.with_tool_call("return_result", {"result": "done"}, strict_exhaustion=True)
    agent = _CatalogAgent(llm=llm)
    with patch("aster.skill_registry.entry_points", return_value=[]):
        agent.skills = SkillRegistry(agent)
    body = f"UNSELECTED_SKILL_BODY_{tmp_path.name}"
    agent.skills.register("custom.review", TextSkill(path=_write_skill(tmp_path, body=body)))
    agent.skills.register(
        "custom.disabled",
        TextSkill(
            path=_write_skill(
                tmp_path,
                "disabled",
                metadata={"description": "PRIVATE SUMMARY", "disable-model-invocation": True},
                body="PRIVATE INSTRUCTIONS",
            )
        ),
    )
    try:
        assert await agent.handle() == "done"
        rendered = str(llm.calls[0].messages)
        assert "Inspect code changes." in rendered
        assert "read_text_skill(" in rendered
        assert body not in rendered
        assert "PRIVATE INSTRUCTIONS" not in rendered
        assert "PRIVATE SUMMARY" not in rendered
        assert body not in doc(agent)

        assert body in agent.skills.read_text_skill("review")
        assert body not in agent.skills.status()
        assert agent.skills.activated() == []
    finally:
        await agent.skills.aclose()
