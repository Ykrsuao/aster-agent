# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral discovery and dispatch for coding-agent skill commands."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aster.skill import Skill, TextSkill, get_slash_commands
from aster.slash_dispatch import CoercionError, SlashCommandResult, parse_typed_args

logger = logging.getLogger(__name__)

# These commands belong to the terminal. Text skills remain available by their
# qualified registry name and by $mention when a short name is already taken.
_HOST_COMMANDS = frozenset(
    "new sessions drafts resume rewind tree tools changes queue steer attach attachments "
    "reconnect retry undo model effort reasoning mode plan code permissions sandbox tasks "
    "skills mcp recovery stop search status hud context logs help cancel exit".split()
)
_COMMAND_NAME = re.compile(r"\w[\w.:\[\]-]*\Z")


@dataclass(frozen=True, slots=True)
class CodingSlashCommand:
    """One user-invocable command exposed by a loaded coding skill."""

    name: str
    description: str
    argument_hint: str | None
    completions: tuple[str, ...]
    output_to_agent: bool
    _method: Any = field(repr=False, compare=False)
    skill_name: str | None = None
    aliases: tuple[str, ...] = ()


class CodingSlashCommandRegistry:
    """Invoke Python skill commands and explicit text skills without UI coupling."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._commands: dict[str, CodingSlashCommand] = {}
        self._text_aliases: dict[str, list[str]] = {}
        self._on_change: Callable[[tuple[CodingSlashCommand, ...]], None] | None = None
        self._previous_registry = getattr(agent, "_command_registry", None)
        agent._command_registry = self
        self.refresh_skill_commands()

    def commands(self) -> tuple[CodingSlashCommand, ...]:
        return tuple(self._commands[name] for name in sorted(self._commands))

    def get(self, name: str) -> CodingSlashCommand | None:
        key = name.casefold()
        command = self._commands.get(key)
        if command is not None or key in _HOST_COMMANDS:
            return command
        matches = self._text_aliases.get(key)
        if matches:
            skill = self.agent.skills.text_skills()[matches[0]]
            return self._text_command(key, key, skill)
        # A qualified text-skill request is explicit even if discovery failed.
        if key.startswith("cmd.") and _COMMAND_NAME.fullmatch(key):
            return CodingSlashCommand(key, "Apply a text skill.", "[task]", (), True, None, key)
        return None

    def set_on_change(
        self,
        callback: Callable[[tuple[CodingSlashCommand, ...]], None] | None,
        *,
        emit: bool = False,
    ) -> None:
        self._on_change = callback
        if emit and callback is not None:
            callback(self.commands())

    def refresh_skill_commands(self) -> None:
        """Rebuild the command map after skill load, activation, or reload."""
        previous = self.commands()
        commands: dict[str, CodingSlashCommand] = {}
        loaded = self._loaded_skills()
        recovery = getattr(self.agent, "recovery", None)
        if getattr(self.agent, "_action_journal", None) is not None and recovery is not None:
            # Host reconciliation stays available even if a skill is unloaded,
            # and user skills cannot take its short command name.
            loaded = {
                "aster.recovery": recovery,
                **{name: skill for name, skill in loaded.items() if name != "aster.recovery"},
            }
        for skill_name, skill in loaded.items():
            for meta, method in get_slash_commands(skill):
                name = meta.name.casefold()
                if name in commands:
                    logger.warning(
                        "Ignoring duplicate slash command /%s from skill %s",
                        name,
                        skill_name,
                    )
                    continue
                doc = inspect.getdoc(method) or ""
                description = doc.splitlines()[0] if doc else f"Run /{name}"
                commands[name] = CodingSlashCommand(
                    name=name,
                    description=description,
                    argument_hint=meta.argument_hint,
                    completions=tuple(meta.completions),
                    output_to_agent=meta.output_to_agent,
                    _method=method,
                )
        text_skills = {
            name: skill for name, skill in loaded.items() if isinstance(skill, TextSkill)
        }
        aliases: dict[str, list[str]] = {}
        for name, skill in text_skills.items():
            for alias in {name.casefold(), skill.name.casefold(), skill.id.casefold()}:
                if _COMMAND_NAME.fullmatch(alias):
                    aliases.setdefault(alias, []).append(name)
        self._text_aliases = aliases
        python_commands = set(commands)
        for name, skill in text_skills.items():
            if not skill.user_invocable:
                continue
            preferred = skill.name if _COMMAND_NAME.fullmatch(skill.name) else skill.id
            key = preferred.casefold()
            if key in _HOST_COMMANDS or key in commands or len(aliases.get(key, ())) != 1:
                key = name.casefold()
            if not _COMMAND_NAME.fullmatch(key) or key in commands or key in _HOST_COMMANDS:
                logger.warning(
                    "No unambiguous slash command for text skill %s; use $%s", name, name
                )
                continue
            command_aliases = tuple(
                sorted(
                    alias
                    for alias in {
                        spelling
                        for original in (name, skill.name, skill.id)
                        for spelling in (original.lower(), original.casefold())
                    }
                    if alias != key
                    and _COMMAND_NAME.fullmatch(alias)
                    and alias.casefold() not in _HOST_COMMANDS
                    and alias.casefold() not in python_commands
                    and (
                        alias.casefold() == name.casefold()
                        or aliases.get(alias.casefold()) == [name]
                    )
                )
            )
            commands[key] = self._text_command(key, name, skill, aliases=command_aliases)
        self._commands = commands
        if self._on_change is not None and self.commands() != previous:
            self._on_change(self.commands())

    @staticmethod
    def _text_command(
        name: str, skill_name: str, skill: TextSkill, *, aliases: tuple[str, ...] = ()
    ) -> CodingSlashCommand:
        return CodingSlashCommand(
            name=name,
            description=skill.description,
            argument_hint=skill.argument_hint,
            completions=(),
            output_to_agent=True,
            _method=None,
            skill_name=skill_name,
            aliases=aliases,
        )

    def _loaded_skills(self) -> dict[str, Skill]:
        skills = getattr(self.agent, "skills", None)
        loaded = skills.loaded() if skills is not None and hasattr(skills, "loaded") else []
        result: dict[str, Skill] = {}
        for skill_name in loaded:
            try:
                skill = skills[skill_name]
            except (AttributeError, KeyError):
                continue
            if isinstance(skill, Skill):
                result[skill_name] = skill
        return result

    def _invoke_text_skill(self, selector: str, arguments: str) -> SlashCommandResult:
        try:
            name, skill = self.agent.skills.resolve_text_skill(selector, user_invoked=True)
        except ValueError as exc:
            raise CoercionError(str(exc), hint="[task]") from exc
        value = {
            "skill": name,
            "source_dir": str(skill.source_dir),
            "instructions": type(skill).__doc__ or "",
            "arguments": arguments,
        }
        return SlashCommandResult(
            command=selector.casefold(),
            args=arguments,
            value=value,
            text=(
                f"Apply text skill {name} to the current user task only. "
                "If a preview is truncated, read the full instructions and arguments from "
                "this result's value in notification['slash_commands'] before proceeding.\n\n"
                f"Resolve the skill's relative file references from: {value['source_dir']}\n\n"
                f"{value['instructions']}\n\nUser task (literal arguments):\n{arguments}"
            ),
            output_to_agent=True,
        )

    def skill_mentions(self, text: str) -> tuple[SlashCommandResult, ...]:
        """Resolve explicit $skills in prose without interpreting shell/code text.

        Every selected skill receives the same literal user task. Resolution has
        no activation or queue side effects, so a later invalid selection cannot
        leave a partially submitted turn.
        """
        if not self._text_aliases or "$" not in text:
            return ()
        spellings = set(self._text_aliases)
        for name, skill in self.agent.skills.text_skills().items():
            spellings.update(
                alias for alias in (name, skill.name, skill.id) if _COMMAND_NAME.fullmatch(alias)
            )
        alternatives = "|".join(
            re.escape(name) for name in sorted(spellings, key=lambda name: (-len(name), name))
        )
        pattern = re.compile(
            rf"(?<![\w\\/$=])\$({alternatives})(?![\w:\[\]-]|\.(?=\w))", re.IGNORECASE
        )
        results: list[SlashCommandResult] = []
        selected: set[str] = set()
        for match in pattern.finditer(_outside_code(text)):
            result = self._invoke_text_skill(match.group(1), text)
            name = result.value["skill"]
            if name not in selected:
                selected.add(name)
                results.append(result)
        return tuple(results)

    async def invoke(self, name: str, raw_args: str) -> SlashCommandResult:
        command = self.get(name)
        if command is None:
            raise KeyError(name)
        if command.skill_name is not None:
            result = self._invoke_text_skill(command.skill_name, raw_args)
            result.command = command.name
            return result
        kwargs = parse_typed_args(command._method, raw_args)
        # Commands execute on the agent's event loop, matching the native TUI.
        # Several existing synchronous skills call QueueManager.spawn(), which
        # requires that loop. Async commands can cancel cooperatively; a truly
        # blocking sync command needs the future per-agent process boundary to
        # provide a safe kill mechanism.
        value = command._method(**kwargs)
        if inspect.isawaitable(value):
            value = await value
        return SlashCommandResult(
            command=command.name,
            args=raw_args,
            value=value,
            text=str(value) if value is not None else None,
            output_to_agent=command.output_to_agent,
        )

    def close(self) -> None:
        self._on_change = None
        if getattr(self.agent, "_command_registry", None) is self:
            if self._previous_registry is None:
                delattr(self.agent, "_command_registry")
            else:
                self.agent._command_registry = self._previous_registry


def _outside_code(text: str) -> str:
    """Mask Markdown code while retaining the positions of ordinary prose."""
    text = re.sub(
        r"(?m)^(?: {0,3}>[ \t]?)*(?: {4}| {0,3}\t)[^\r\n]*",
        lambda match: " " * len(match[0]),
        text,
    )
    opener = re.compile(
        r"^[ \t]{0,3}(?P<quotes>(?:>[ \t]?)*)(?:(?:[-+*]|\d+[.)])[ \t]+)?"
        r"(?P<fence>`{3,}|~{3,})[^\n]*(?:\n|$)|(?<!\\)(?P<inline>`+)",
        re.MULTILINE,
    )
    parts: list[str] = []
    offset = 0
    while match := opener.search(text, offset):
        parts.append(text[offset : match.start()])
        fence = match.group("fence")
        if fence:
            # Quote markers inside the fence are literal unless its opener shares that depth.
            quote_depth = match.group("quotes").count(">")
            closing = re.compile(
                rf"^[ \t]{{0,3}}(?:>[ \t]?){{{quote_depth}}}"
                rf"{re.escape(fence[0])}{{{len(fence)},}}[ \t]*(?:\r?\n|$)",
                re.MULTILINE,
            )
        else:
            closing = re.compile(rf"(?<!`){re.escape(match.group('inline'))}(?!`)")
        end = closing.search(text, match.end())
        if end is None and fence is None:
            parts.append(match[0])
            offset = match.end()
            continue
        offset = end.end() if end else len(text)
        parts.append(" " * (offset - match.start()))
    parts.append(text[offset:])
    return "".join(parts)


__all__ = ["CodingSlashCommand", "CodingSlashCommandRegistry"]
