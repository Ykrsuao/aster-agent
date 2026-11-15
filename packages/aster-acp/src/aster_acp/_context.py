# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request summaries and bounded text for a read-only context inspector."""

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from aster.agents.summarization import summary_fork_active
from aster.context_blocks.models import BlockPart, ContextWindowStats, RenderedMessage
from aster.llm_types import CacheBoundary, LLMResponse, LLMUsage

_MAX_ITEMS = 64
_MAX_CONTENT_CHARS = 1_000_000
_CATEGORIES = (
    "system_prompt",
    "system_tools",
    "memory_files",
    "skills",
    "messages",
    "runtime_context",
    "other",
)
_BLOCK_CATEGORIES = {
    "system_prompt": "system_prompt",
    "strategy_prompt": "system_prompt",
    "self": "system_tools",
    "events_api": "system_tools",
    "python_tools": "system_tools",
    "repository_instructions": "memory_files",
    "skills": "skills",
    "skill_instructions": "skills",
    **dict.fromkeys(
        (
            "state",
            "execution_context",
            "execution_policy",
            "python_cell_context",
            "python_cell_state",
            "sandbox",
            "context_usage",
            "todo_status",
            "workspace_paths",
            "host_shell",
        ),
        "runtime_context",
    ),
}


def _json_chars(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError):
        return 0


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return ""


def _positive(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@dataclass(frozen=True, slots=True)
class ContextItem:
    category: str
    key: str
    chars: int
    content: str = field(default="", repr=False)
    content_chars: int = 0


@dataclass(frozen=True, slots=True)
class RequestComposition:
    generation_id: str
    window_tokens: int | None
    reserved_output_tokens: int | None
    output_reserve_is_fallback: bool
    items: tuple[ContextItem, ...]
    messages_count: int
    has_media: bool
    request_id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def capture(
        cls,
        messages: list[Any],
        *,
        generation_id: str,
        rendered_messages: tuple[RenderedMessage, ...] = (),
        context_stats: ContextWindowStats | None = None,
        tools: Any = None,
        output_model: Any = None,
    ) -> "RequestComposition":
        sizes: dict[tuple[str, str], int] = {}
        contents: dict[tuple[str, str], list[str]] = defaultdict(list)
        content_sizes: dict[tuple[str, str], int] = defaultdict(int)
        remaining_content = _MAX_CONTENT_CHARS
        candidates: dict[tuple[str, str], deque[RenderedMessage]] = defaultdict(deque)
        for message in rendered_messages:
            if message.content is not None and message.parts:
                candidates[(message.role.value, message.content)].append(message)

        def add(category: str, key: str, chars: int, text: str = "") -> None:
            nonlocal remaining_content
            if chars <= 0:
                return
            item = (category, key[:120])
            if item not in sizes and len(sizes) >= _MAX_ITEMS - 1:
                item = ("other", "Additional context")
            sizes[item] = sizes.get(item, 0) + chars
            if text:
                separator = "\n\n" if content_sizes[item] else ""
                content_sizes[item] += len(separator) + len(text)
                part = (
                    separator[:remaining_content]
                    + text[: max(remaining_content - len(separator), 0)]
                )
                if part:
                    contents[item].append(part)
                    remaining_content -= len(part)

        count = 0
        has_media = False
        for message in messages:
            if isinstance(message, CacheBoundary):
                continue
            count += 1
            wire = dict(message)
            role = wire.get("role", "")
            message_key = f"Message {count} ({role or 'unknown'})"
            content = wire.get("content")
            default_category = "system_prompt" if role == "system" else "messages"
            if isinstance(content, str):
                matches = candidates.get((role, content))
                source = matches.popleft() if matches else None
                if (
                    source
                    and source.parts
                    and "".join(
                        part.content if isinstance(part, BlockPart) else part.text
                        for part in source.parts
                    )
                    == content
                ):
                    framing = (
                        "runtime_context"
                        if any(
                            isinstance(part, BlockPart) and not part.key.startswith("event_")
                            for part in source.parts
                        )
                        else default_category
                    )
                    for part in source.parts:
                        if isinstance(part, BlockPart):
                            category = (
                                "messages"
                                if part.key.startswith("event_")
                                else _BLOCK_CATEGORIES.get(part.key, "other")
                            )
                            key = (
                                f"{message_key}: {part.key}" if category == "messages" else part.key
                            )
                            add(category, key, len(part.content), part.content)
                        else:
                            add(framing, "Context framing", len(part.text), part.text)
                else:
                    add(
                        default_category,
                        message_key,
                        len(content),
                        content,
                    )
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if isinstance(text, str):
                            add(default_category, message_key, len(text), text)
                    else:
                        # Image/audio/PDF token costs do not scale with encoded
                        # character length. Keep the provider total, omit the split.
                        has_media = True
            structure = {key: value for key, value in wire.items() if key != "content"}
            add(
                default_category,
                f"{message_key} structure / tool calls",
                _json_chars(structure),
                _json_text(structure),
            )
        add("system_tools", "API tool schemas", _json_chars(tools), _json_text(tools))
        if output_model is not None and callable(getattr(output_model, "model_json_schema", None)):
            schema = output_model.model_json_schema()
            add("system_tools", "Response schema", _json_chars(schema), _json_text(schema))

        return cls(
            generation_id=generation_id,
            window_tokens=_positive(context_stats.model_context_window) if context_stats else None,
            reserved_output_tokens=context_stats.reserved_output_tokens if context_stats else None,
            output_reserve_is_fallback=bool(
                context_stats and context_stats.output_reserve_is_fallback
            ),
            items=tuple(
                ContextItem(
                    category,
                    key,
                    chars,
                    "".join(contents[(category, key)]),
                    content_sizes[(category, key)],
                )
                for (category, key), chars in sizes.items()
            ),
            messages_count=count,
            has_media=has_media,
        )

    def metadata(self, usage: LLMUsage | None, model: str) -> dict[str, Any]:
        total_chars = sum(item.chars for item in self.items)
        available = usage is not None and usage.input_tokens_reported
        attributed = available and total_chars > 0 and not self.has_media
        details: list[dict[str, Any]] = []
        categories: list[dict[str, Any]] = []
        if attributed and usage is not None:
            # Integer largest remainders preserve the exact provider total,
            # including tiny categories and a legitimately reported zero.
            allocations = [
                divmod(usage.input_tokens * item.chars, total_chars) for item in self.items
            ]
            tokens = [allocation[0] for allocation in allocations]
            remaining = usage.input_tokens - sum(tokens)
            for index in sorted(
                range(len(tokens)), key=lambda index: (-allocations[index][1], index)
            )[:remaining]:
                tokens[index] += 1
            details = [
                {"category": item.category, "key": item.key, "chars": item.chars, "tokens": token}
                for item, token in zip(self.items, tokens, strict=True)
            ]
            categories = [
                {
                    "id": category,
                    "tokens": sum(
                        item["tokens"] for item in details if item["category"] == category
                    ),
                }
                for category in _CATEGORIES
                if any(item["category"] == category for item in details)
            ]
        return {
            "version": 1,
            "generationId": self.generation_id,
            "requestId": self.request_id,
            "model": model,
            "windowTokens": self.window_tokens,
            "reservedOutputTokens": self.reserved_output_tokens,
            "outputReserveIsFallback": self.output_reserve_is_fallback,
            "messages": self.messages_count,
            "attribution": "characters" if attributed else "unavailable",
            "unavailableReason": (
                None
                if attributed
                else "media"
                if self.has_media
                else "no-usage"
                if not available
                else "no-text"
            ),
            "categories": categories,
            "items": details,
            "contentItems": [
                {"category": item.category, "key": item.key, "chars": item.content_chars}
                for item in self.items
                if item.content_chars
            ],
        }

    def inspect(self, category: str, key: str | None = None) -> dict[str, Any]:
        if category not in _CATEGORIES:
            raise ValueError("Unknown context category.")
        items = [
            item
            for item in self.items
            if item.category == category and (key is None or item.key == key) and item.content_chars
        ]
        if not items:
            raise ValueError("This context item has no captured text.")
        return {
            "requestId": self.request_id,
            "category": category,
            "key": key,
            "source": "model_request",
            "sections": [
                {
                    "key": item.key,
                    "text": item.content,
                    "characters": item.content_chars,
                    "truncated": len(item.content) < item.content_chars,
                }
                for item in items
            ],
        }


def capture_output(response: LLMResponse) -> list[dict[str, Any]]:
    """Keep public assistant text and tool calls, excluding SDK/native state."""
    texts = [("Assistant text", response.content)] if response.content else []
    if response.tool_calls:
        calls = []
        for call in response.tool_calls:
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, ValueError):
                arguments = call.arguments
            calls.append({"id": call.id, "name": call.name, "arguments": arguments})
        texts.append(("Tool calls", _json_text(calls)))
    remaining = _MAX_CONTENT_CHARS
    sections = []
    for key, text in texts:
        captured = text[:remaining]
        remaining -= len(captured)
        sections.append(
            {
                "key": key,
                "text": captured,
                "characters": len(text),
                "truncated": len(captured) < len(text),
            }
        )
    return sections


def _noop(*args: Any, **kwargs: Any) -> None:
    pass


class ContextObserver:
    """Compose with tracing for one call; never install process-wide hooks."""

    before_agent_call = _noop
    after_agent_call = _noop
    before_generation = _noop
    after_generation = _noop
    before_code_execution = _noop
    after_code_execution = _noop
    before_method_invocation = _noop
    after_method_invocation = _noop
    before_tool_execution = _noop
    after_tool_execution = _noop

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.request: RequestComposition | None = None

    def on_messages_built(
        self,
        agent: Any,
        method_name: str,
        messages: list[Any],
        generation_id: str,
        **kwargs: Any,
    ) -> None:
        if agent is not self.agent or method_name != "handle" or summary_fork_active():
            return
        self.request = RequestComposition.capture(
            messages,
            generation_id=generation_id,
            rendered_messages=kwargs.get("rendered_messages", ()),
            context_stats=kwargs.get("context_stats"),
            tools=kwargs.get("tools"),
            output_model=kwargs.get("output_model"),
        )
