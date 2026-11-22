# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared coding-agent skill command discovery and dispatch."""

import asyncio
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Literal

import pytest
from aster_cli.coding import CodingAgent, CodingSlashCommandRegistry

from aster.skill import Skill, TextSkill, slash_command
from aster.slash_dispatch import CoercionError
from aster.unifiedllm import FakeLLMClient


class _WorkflowSkill(Skill):
    @slash_command(
        "diagnose",
        argument_hint="<fast|deep>",
        completions=("fast", "deep"),
    )
    async def diagnose(self, mode: Literal["fast", "deep"]) -> str:
        """Diagnose the current failure."""
        return f"Diagnose in {mode} mode."


class _BackgroundWorkflowSkill(Skill):
    @slash_command("background")
    def background(self, args: str) -> str:
        channel = self._agent.queue_manager.queue("background-test", replace=True)

        async def produce() -> None:
            await asyncio.Event().wait()

        self._agent.queue_manager.spawn(produce(), channel=channel.name)
        return args


async def test_registry_discovers_metadata_and_dispatches_typed_arguments(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    agent.skills.register("test.workflow", _WorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        assert registry.commands()[0].name == "diagnose"
        assert registry.commands()[0].description == "Diagnose the current failure."
        assert registry.commands()[0].argument_hint == "<fast|deep>"
        assert registry.commands()[0].completions == ("fast", "deep")

        result = await registry.invoke("diagnose", "deep")

        assert result.command == "diagnose"
        assert result.text == "Diagnose in deep mode."
        assert result.output_to_agent is True
    finally:
        registry.close()
        await agent.close()


async def test_registry_reports_typed_argument_errors(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    agent.skills.register("test.workflow", _WorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        with pytest.raises(CoercionError, match="cannot convert"):
            await registry.invoke("diagnose", "invalid")
    finally:
        registry.close()
        await agent.close()


async def test_registry_refresh_callback_observes_new_skill_commands(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    registry = CodingSlashCommandRegistry(agent)
    updates: list[tuple[str, ...]] = []
    registry.set_on_change(
        lambda commands: updates.append(tuple(command.name for command in commands)),
        emit=True,
    )
    try:
        agent.skills.register("test.workflow", _WorkflowSkill())
        agent.skills.activate(["test.workflow"])

        assert updates == [(), ("diagnose",)]
    finally:
        registry.close()
        await agent.close()


async def test_sync_command_runs_on_agent_loop_and_can_spawn_background_job(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    agent.skills.register("test.background", _BackgroundWorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        result = await registry.invoke("background", "started")
        handle = agent.queue_manager.job("background-test")

        assert result.text == "started"
        assert handle is not None
        assert handle.state == "running"
    finally:
        registry.close()
        await agent.close()
    assert handle is not None
    assert handle.state == "cancelled"


async def test_async_command_is_cooperatively_cancellable_on_agent_loop(tmp_path):
    started = asyncio.Event()

    class AsyncWorkflowSkill(Skill):
        @slash_command("wait")
        async def wait(self) -> str:
            assert asyncio.get_running_loop() is host_loop
            started.set()
            await asyncio.Event().wait()
            return "never"

    host_loop = asyncio.get_running_loop()
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    agent.skills.register("test.async", AsyncWorkflowSkill())
    registry = CodingSlashCommandRegistry(agent)
    try:
        invocation = asyncio.create_task(registry.invoke("wait", ""))
        # Bounded: a regression that stops the coroutine reaching the host loop
        # would otherwise wedge here forever, so CI stalls instead of going red.
        await asyncio.wait_for(started.wait(), timeout=30)
        invocation.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, timeout=30)
    finally:
        registry.close()
        await agent.close()


async def test_commands_are_sorted_deduplicated_and_case_insensitive(tmp_path):
    """Three registry behaviours that no test exercised.

    Every existing case registers one command and reads it back in canonical
    case, so unsorted output, a dropped duplicate guard, and lost case
    normalisation were all invisible.
    """

    class _Beta(Skill):
        @slash_command("beta")
        def beta(self, args: str) -> str:
            """Beta."""
            return "beta"

    class _Alpha(Skill):
        @slash_command("alpha")
        def alpha(self, args: str) -> str:
            """Alpha."""
            return "alpha"

    class _AlphaAgain(Skill):
        @slash_command("alpha")
        def alpha(self, args: str) -> str:
            """Duplicate."""
            return "duplicate"

    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    # Registered out of order, and with a colliding name.
    # Registry names sort opposite to the command names they provide, so
    # _commands is built in [beta, alpha] order and sorting is observable.
    agent.skills.register("test.aaa", _Beta())
    agent.skills.register("test.zzz", _Alpha())
    agent.skills.register("test.zzzz", _AlphaAgain())
    registry = CodingSlashCommandRegistry(agent)
    try:
        names = [command.name for command in registry.commands()]
        assert names == sorted(names), names
        assert names.count("alpha") == 1, names
        assert registry.get("ALPHA") is not None, "lookup is not case-insensitive"
    finally:
        registry.close()
        await agent.close()


@pytest.mark.parametrize("raw_args", ["", 'add "two words"'])
@pytest.mark.parametrize(
    "postponed,annotation",
    [
        (False, "str"),
        (False, '"str"'),
        (True, "str"),
        (True, '"str"'),
        (True, "'str'"),
    ],
)
async def test_string_args_annotation_preserves_raw_input(
    tmp_path, raw_args, postponed, annotation
):
    """All direct and postponed spellings of str preserve the exact input."""
    source = "from __future__ import annotations\n" if postponed else ""
    source += (
        "class StringArgsSkill(Skill):\n"
        "    @slash_command('raw-input', output_to_agent=False)\n"
        f"    def raw_input(self, args: {annotation}):\n"
        '        """Return the raw command argument unchanged."""\n'
        "        return args\n"
    )
    namespace = {"__name__": __name__, "Skill": Skill, "slash_command": slash_command}
    exec(compile(source, "<raw-args-skill>", "exec", dont_inherit=True), namespace)
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    agent.skills.register("test.raw", namespace["StringArgsSkill"]())
    registry = CodingSlashCommandRegistry(agent)
    try:
        result = await registry.invoke("raw-input", raw_args)
        assert result.text == raw_args
    finally:
        registry.close()
        await agent.close()


async def test_no_text_skill_hub_is_advertised_without_skills(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, libs_dir=tmp_path / "libs")
    registry = CodingSlashCommandRegistry(agent)
    try:
        assert registry.commands() == ()
        assert registry.get("skill") is None
        with pytest.raises(CoercionError, match="(?i)skill"):
            await registry.invoke("cmd.missing", "write a report")
    finally:
        registry.close()
        await agent.close()


_TEXT_SKILL_BODY = "Inspect the reported failure.\nKeep {name} and $ARGUMENTS literal."


def _write_text_skill(
    skills_root: Path,
    name: str,
    *,
    body: str = _TEXT_SKILL_BODY,
    description: str = "Investigate a failure.",
    display_name: str = "Display name only",
    metadata: str = "",
) -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {display_name}\ndescription: {description}\n{metadata}---\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture
async def text_skill_runtime(tmp_path):
    skills_root = tmp_path / ".agents" / "skills"
    _write_text_skill(skills_root, "triage")
    agent = CodingAgent(
        llm=FakeLLMClient(),
        cwd=tmp_path,
        libs_dir=tmp_path / "libs",
        skills_dirs=[skills_root],
    )
    registry = CodingSlashCommandRegistry(agent)
    try:
        yield agent, registry, skills_root
    finally:
        registry.close()
        await agent.close()


async def test_text_skill_discovery_and_completion_do_not_activate_or_run_skill(
    text_skill_runtime, monkeypatch
):
    agent, registry, skills_root = text_skill_runtime
    active_before = agent.skills.activated()

    async def unexpected_script(*args, **kwargs):
        pytest.fail("Discovering or invoking a text skill must not execute its scripts")

    monkeypatch.setattr(TextSkill, "run_script", unexpected_script)
    agent.skills.discover_skills_dirs([skills_root])
    command = registry.get("triage")
    assert command is not None
    assert command.output_to_agent is True
    assert command.completions == ()
    assert command.argument_hint == "[task]"
    assert command.skill_name == "cmd.triage"
    assert command.aliases == ("cmd.triage",)
    assert registry.get("skill") is None
    assert agent.skills.activated() == active_before

    await registry.invoke("triage", "inspect this error")
    assert agent.skills.activated() == active_before


@pytest.mark.parametrize("selector", ["triage", "cmd.triage", "CMD.TRIAGE"])
async def test_text_skill_invocation_selects_directory_id_and_supplies_full_instructions(
    text_skill_runtime, selector
):
    agent, registry, skills_root = text_skill_runtime
    result = await registry.invoke(selector, "inspect this error")

    assert result.command == selector.casefold()
    assert result.args == "inspect this error"
    assert result.output_to_agent is True
    assert result.value["skill"] == "cmd.triage"
    assert result.value["source_dir"] == str((skills_root / "triage").resolve())
    assert result.value["instructions"] == type(agent.skills["cmd.triage"]).__doc__
    assert _TEXT_SKILL_BODY in result.value["instructions"]
    assert result.value["arguments"] == "inspect this error"
    assert _TEXT_SKILL_BODY in str(result)
    assert "inspect this error" in str(result)
    assert "cmd.triage" not in agent.skills.activated()


@pytest.mark.parametrize(
    "arguments",
    [
        "",
        'inspect "two words" in C:\\Work\\My Project\\file.py',
        "  keep the leading and trailing spaces  ",
        "\nfirst line\nsecond line\t  ",
        'an unmatched "quote and {value} $ARGUMENTS',
    ],
)
async def test_text_skill_task_arguments_are_not_shell_parsed_or_templated(
    text_skill_runtime, arguments
):
    _, registry, _ = text_skill_runtime
    result = await registry.invoke("triage", arguments)

    assert result.args == arguments
    assert result.value["arguments"] == arguments
    assert _TEXT_SKILL_BODY in result.value["instructions"]


async def test_text_skill_without_task_uses_an_empty_argument_string(text_skill_runtime):
    _, registry, _ = text_skill_runtime
    result = await registry.invoke("triage", "")
    assert result.value["arguments"] == ""


async def test_missing_qualified_text_skill_reports_error_without_activation(text_skill_runtime):
    agent, registry, _ = text_skill_runtime
    active_before = agent.skills.activated()

    with pytest.raises(CoercionError):
        await registry.invoke("cmd.missing", "explain the failure")

    assert agent.skills.activated() == active_before


async def test_text_skill_refreshes_completions_and_invokes_reloaded_body(text_skill_runtime):
    agent, registry, skills_root = text_skill_runtime
    updates = []
    registry.set_on_change(
        lambda commands: updates.append(tuple(c.name for c in commands)),
        emit=True,
    )
    _write_text_skill(skills_root, "summarize", body="Summarize the findings.")
    agent.skills.discover_skills_dirs([skills_root])
    assert updates == [("triage",), ("summarize", "triage")]

    _write_text_skill(skills_root, "triage", body="Use the revised diagnostic procedure.")
    await agent.skills.reload("cmd.triage")
    result = await registry.invoke("triage", "investigate")

    assert "Use the revised diagnostic procedure." in result.value["instructions"]
    assert _TEXT_SKILL_BODY not in result.value["instructions"]
    assert result.value["arguments"] == "investigate"
    assert "cmd.summarize" not in agent.skills.activated()


async def test_text_skill_short_name_conflicts_require_qualified_registry_name(
    text_skill_runtime, tmp_path
):
    agent, registry, _ = text_skill_runtime
    second_path = _write_text_skill(tmp_path / "extra", "another", body="Second skill body.")
    agent.skills.register("custom.another", TextSkill(path=second_path, id="triage"))

    with pytest.raises(CoercionError, match="(?i)triage"):
        await registry.invoke("triage", "inspect the failure")

    assert "cmd.triage" not in agent.skills.activated()
    assert "custom.another" not in agent.skills.activated()
    result = await registry.invoke("custom.another", "inspect the failure")
    assert result.value["skill"] == "custom.another"
    assert "Second skill body." in result.value["instructions"]
    assert "custom.another" not in agent.skills.activated()
    assert "cmd.triage" not in agent.skills.activated()


@pytest.mark.parametrize("selector", ["cmd.triage", "custom.triage", "triage"])
async def test_text_skill_rejects_shared_agent_binding_instead_of_invoking_wrong_skill(
    text_skill_runtime, tmp_path, selector
):
    agent, registry, _ = text_skill_runtime
    second_path = _write_text_skill(tmp_path / "extra", "replacement")
    agent.skills.register("custom.triage", TextSkill(path=second_path))
    active_before = agent.skills.activated()

    with pytest.raises((CoercionError, KeyError)):
        await registry.invoke(selector, "investigate")

    assert agent.skills.activated() == active_before


async def test_python_command_collision_keeps_qualified_text_skill_invocation(text_skill_runtime):
    agent, registry, _ = text_skill_runtime

    class InterceptSkill(Skill):
        @slash_command("triage", output_to_agent=False)
        def intercept(self, args: str) -> str:
            return "Python workflow"

    agent.skills.register("test.intercept", InterceptSkill())
    python_result = await registry.invoke("triage", "investigate")
    assert python_result.text == "Python workflow"
    result = await registry.invoke("cmd.triage", "investigate")

    assert result.value["skill"] == "cmd.triage"
    assert result.value["arguments"] == "investigate"
    assert [c.name for c in registry.commands()] == ["cmd.triage", "triage"]
    assert registry.skill_mentions("Use $triage")[0].value["skill"] == "cmd.triage"


async def test_text_skill_named_like_terminal_builtin_uses_qualified_name(
    text_skill_runtime,
):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(skills_root, "help", body="Explain the project terminology.")
    agent.skills.discover_skills_dirs([skills_root])

    result = await registry.invoke("cmd.help", "explain the vocabulary")

    assert result.value["skill"] == "cmd.help"
    assert "Explain the project terminology." in result.value["instructions"]
    assert registry.get("help") is None
    assert "cmd.help" not in agent.skills.activated()
    assert registry.skill_mentions("Use $help")[0].value["skill"] == "cmd.help"
    assert "cmd.triage" not in agent.skills.activated()


async def test_text_skill_duplicate_roots_keep_existing_discovery_precedence(
    text_skill_runtime, tmp_path
):
    agent, registry, skills_root = text_skill_runtime
    extra_root = tmp_path / "extra"
    _write_text_skill(extra_root, "triage", body="A lower-priority copy.")
    agent.skills.discover_skills_dirs([extra_root])

    result = await registry.invoke("triage", "inspect this failure")

    assert result.value["source_dir"] == str((skills_root / "triage").resolve())
    assert _TEXT_SKILL_BODY in result.value["instructions"]
    assert "A lower-priority copy." not in result.value["instructions"]


async def test_text_skill_selector_is_literal_when_name_contains_glob_characters(
    text_skill_runtime,
):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(skills_root, "triage[1]", body="Use the bracketed skill.")
    _write_text_skill(skills_root, "triage1", body="Use the unbracketed skill.")
    agent.skills.discover_skills_dirs([skills_root])

    result = await registry.invoke("cmd.triage[1]", "investigate")

    assert result.value["skill"] == "cmd.triage[1]"
    assert "Use the bracketed skill." in result.value["instructions"]
    assert "cmd.triage[1]" not in agent.skills.activated()
    assert "cmd.triage1" not in agent.skills.activated()
    assert "cmd.triage" not in agent.skills.activated()


async def test_text_skill_activation_does_not_activate_case_colliding_python_skill(
    text_skill_runtime,
):
    agent, registry, _ = text_skill_runtime

    class OtherSkill(Skill):
        """A separate Python skill with a case-colliding registry name."""

    agent.skills.register("cmd.TRIAGE", OtherSkill())
    active_before = agent.skills.activated()
    with pytest.raises(CoercionError, match="(?i)conflict"):
        await registry.invoke("cmd.triage", "investigate")

    assert agent.skills.activated() == active_before


async def test_text_skill_activation_does_not_load_case_colliding_entry_point(
    text_skill_runtime, monkeypatch
):
    agent, registry, _ = text_skill_runtime
    entry = EntryPoint(
        name="cmd.TRIAGE",
        value="unloaded_test_skill:OtherSkill",
        group="aster.skills",
    )

    def unexpected_load(self):
        pytest.fail("Selecting a text skill must not import a case-colliding plugin")

    monkeypatch.setattr("aster.skill_registry.entry_points", lambda **kwargs: [entry])
    monkeypatch.setattr(EntryPoint, "load", unexpected_load)
    agent.skills._discover()
    active_before = agent.skills.activated()

    with pytest.raises(CoercionError, match="(?i)conflict"):
        await registry.invoke("cmd.triage", "investigate")

    assert "cmd.TRIAGE" not in agent.skills.loaded()
    assert agent.skills.activated() == active_before


async def test_frontmatter_name_is_advertised_and_directory_alias_remains_usable(
    text_skill_runtime,
):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(
        skills_root,
        "triage",
        display_name="review-changes",
        metadata="argument-hint: '[target]'\n",
    )
    await agent.skills.reload("cmd.triage")

    assert [command.name for command in registry.commands()] == ["review-changes"]
    assert registry.get("review-changes").argument_hint == "[target]"
    assert registry.get("review-changes").aliases == ("cmd.triage", "triage")
    for selector in ("review-changes", "triage", "cmd.triage"):
        result = await registry.invoke(selector, "Review this diff.")
        assert result.value["skill"] == "cmd.triage"
    assert _TEXT_SKILL_BODY not in repr(registry.commands())


async def test_duplicate_frontmatter_names_require_qualified_selection(text_skill_runtime):
    agent, registry, skills_root = text_skill_runtime
    for name in ("triage", "another"):
        _write_text_skill(skills_root, name, display_name="review")
    await agent.skills.reload("cmd.triage")
    agent.skills.discover_skills_dirs([skills_root])

    assert [command.name for command in registry.commands()] == ["cmd.another", "cmd.triage"]
    with pytest.raises(CoercionError, match="(?i)ambiguous"):
        await registry.invoke("review", "Inspect this diff.")
    with pytest.raises(CoercionError, match="(?i)ambiguous"):
        registry.skill_mentions("Use $review for this diff.")
    assert registry.skill_mentions("Use $cmd.triage")[0].value["skill"] == "cmd.triage"


async def test_multiple_mentions_preserve_task_and_deduplicate_aliases(text_skill_runtime):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(
        skills_root, "summarize", display_name="summarize", body="Summarize findings."
    )
    agent.skills.discover_skills_dirs([skills_root])
    task = 'Use $summarize and $TRIAGE, then $cmd.triage.\nKeep "quotes" and C:\\Work\\file.py  '
    active_before = agent.skills.activated()

    results = registry.skill_mentions(task)

    assert [result.value["skill"] for result in results] == ["cmd.summarize", "cmd.triage"]
    assert all(result.args == task and result.value["arguments"] == task for result in results)
    assert agent.skills.activated() == active_before


@pytest.mark.parametrize(
    "task",
    [
        r"Read $PATH and \$triage literally.",
        "The variables $triage_more, $triage-more and $triage.extra are different.",
        "See https://example.test/$triage, https://example.test/?q=$triage and /tmp/$triage.",
        "Write `$triage` in the docs.",
        "Write ``code with ` and $triage`` in the docs.",
        "Example:\n```sh\n$triage\n```\nDescribe this example.",
        "Example:\n~~~~\n$triage\n~~~\n$triage\n~~~~\nDescribe it.",
        "Example:\n```sh\n$triage",
        "Example:\n\n    $triage\n",
        "Example:\n\n\t$triage\n",
        "Example:\n\n \t$triage\n",
        "> ~~~\n> $triage\n> ~~~",
        ">     $triage\n",
        "- ~~~\n  $triage\n  ~~~",
        "Use $$triage in a formula.",
    ],
)
async def test_mentions_inside_code_paths_escapes_or_longer_names_are_literal(
    text_skill_runtime, task
):
    _, registry, _ = text_skill_runtime
    assert registry.skill_mentions(task) == ()


@pytest.mark.parametrize(
    "opener,false_closer,code_prefix,closer",
    [
        ("~~~", "> ~~~", "", "~~~"),
        ("```", "> ```", "", "```"),
        ("~~~", "- ~~~", "", "~~~"),
        ("```", "- ```", "", "```"),
        ("- ~~~", "  > ~~~", "  ", "  ~~~"),
        ("> ~~~", "> > ~~~", "> ", "> ~~~"),
        ("> > ~~~", "> ~~~", "> > ", "> > ~~~"),
    ],
)
async def test_fence_content_cannot_change_its_container_and_expose_mentions(
    text_skill_runtime, opener, false_closer, code_prefix, closer
):
    _, registry, _ = text_skill_runtime
    task = f"Example:\n{opener}\n{false_closer}\n{code_prefix}$triage\n{closer}\n"

    assert registry.skill_mentions(task) == ()
    results = registry.skill_mentions(task + "Now use $triage.")
    assert len(results) == 1
    assert results[0].value["skill"] == "cmd.triage"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_prose_after_code_and_sentence_punctuation_can_mention_skill(
    text_skill_runtime, newline
):
    _, registry, _ = text_skill_runtime
    task = "Example: `$triage`\n```\n$triage\n```\nNow use ($triage)."
    task = task.replace("\n", newline)
    results = registry.skill_mentions(task)
    assert len(results) == 1
    assert results[0].value["arguments"] == task


async def test_user_disabled_skill_is_hidden_and_rejects_both_explicit_forms(text_skill_runtime):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(skills_root, "triage", metadata="user-invocable: false\n")
    await agent.skills.reload("cmd.triage")

    assert registry.commands() == ()
    with pytest.raises(CoercionError):
        await registry.invoke("triage", "Inspect this.")
    with pytest.raises(CoercionError):
        registry.skill_mentions("Use $triage to inspect this.")


async def test_implicit_disabled_skill_remains_explicitly_invocable(text_skill_runtime):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(skills_root, "triage", metadata="disable-model-invocation: true\n")
    await agent.skills.reload("cmd.triage")

    assert [command.name for command in registry.commands()] == ["triage"]
    assert (await registry.invoke("triage", "Inspect this.")).value["skill"] == "cmd.triage"
    assert registry.skill_mentions("Use $triage")[0].value["skill"] == "cmd.triage"


async def test_unicode_python_command_remains_case_insensitive(text_skill_runtime):
    agent, registry, _ = text_skill_runtime

    class UnicodeCommand(Skill):
        @slash_command("Straße", output_to_agent=False)
        def run(self, args: str) -> str:
            return args

    agent.skills.register("test.unicode", UnicodeCommand())
    assert registry.get("Straße") is not None
    assert (await registry.invoke("Straße", "data")).text == "data"


@pytest.mark.parametrize(
    "prefix", [r"A literal \` character, then ", "An unmatched ` character, then "]
)
async def test_literal_backticks_do_not_swallow_later_mentions(text_skill_runtime, prefix):
    _, registry, _ = text_skill_runtime
    assert registry.skill_mentions(prefix + "$triage")[0].value["skill"] == "cmd.triage"


async def test_unicode_skill_names_work_in_mentions_and_slash_commands(text_skill_runtime):
    agent, registry, skills_root = text_skill_runtime
    _write_text_skill(skills_root, "triage", display_name="Straße")
    await agent.skills.reload("cmd.triage")

    assert "straße" in registry.get("strasse").aliases
    for name in ("Straße", "strasse", "STRASSE"):
        assert (await registry.invoke(name, "Inspect")).value["skill"] == "cmd.triage"
        assert registry.skill_mentions(f"Use ${name}")[0].value["skill"] == "cmd.triage"
