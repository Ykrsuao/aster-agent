# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context statistics describe the completed request without exporting its contents."""

import json
from pathlib import Path
from typing import Any

import pytest
from acp.schema import UsageUpdate
from aster_acp._context import ContextObserver, RequestComposition, capture_output
from aster_acp.event_bridge import ACPEventBridge
from aster_cli.coding import CodingAgent
from pydantic import BaseModel

from aster import Context
from aster.context_blocks.models import (
    BlockPart,
    ContextWindowStats,
    RenderedMessage,
    Role,
    TextPart,
)
from aster.llm_types import AssistantText, CacheBoundary, LLMResponse, LLMUsage, ToolCall
from aster.runtime.hooks import get_hooks, hooks_scope
from aster.unifiedllm import FakeLLMClient


class ReplySchema(BaseModel):
    message: str


def test_content_is_read_only_and_not_embedded_in_usage_metadata() -> None:
    text = "Actual system prompt 中文\nKeep the original newlines."
    source = rendered([BlockPart(key="system_prompt", content=text)])
    request = RequestComposition.capture(
        [{"role": "system", "content": source.content}],
        generation_id="request",
        rendered_messages=(source,),
    )
    metadata = request.metadata(LLMUsage(input_tokens=42), "model")
    assert text not in json.dumps(metadata, ensure_ascii=False)
    assert metadata["contentItems"][0]["key"] == "system_prompt"
    content = request.inspect("system_prompt", "system_prompt")
    assert content["requestId"] == metadata["requestId"]
    assert content["sections"] == [
        {
            "key": "system_prompt",
            "text": text,
            "characters": len(text),
            "truncated": False,
        }
    ]
    with pytest.raises(ValueError, match="no captured text"):
        request.inspect("system_prompt", "not-a-block")
    with pytest.raises(ValueError, match="Unknown context category"):
        request.inspect("invalid")


def test_content_and_output_capture_limits_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aster_acp._context._MAX_CONTENT_CHARS", 32)
    request = RequestComposition.capture(
        [{"role": "system", "content": "甲" * 100}],
        generation_id="request",
    )
    content = request.inspect("system_prompt")
    first = content["sections"][0]
    assert first["text"] == "甲" * 32
    assert first["characters"] == 100 and first["truncated"] is True
    assert sum(len(section["text"]) for section in content["sections"]) <= 32
    output = capture_output(
        LLMResponse(
            parts=(AssistantText(text="Visible output " * 20),),
            raw_response={"secret": "SDK_PRIVATE_STATE"},
        )
    )
    assert sum(len(section["text"]) for section in output) == 32
    assert output[0]["truncated"] is True
    assert "SDK_PRIVATE_STATE" not in json.dumps(output)


def test_plain_messages_keep_stable_individual_keys_and_their_own_tool_calls() -> None:
    messages = [
        {"role": "system", "content": "PRIVATE_SYSTEM"},
        {"role": "user", "content": "PRIVATE_REPEATED_QUESTION"},
        {
            "role": "assistant",
            "content": "PRIVATE_REPLY",
            "tool_calls": [{"id": "call-1", "function": {"name": "example", "arguments": "{}"}}],
        },
        {"role": "user", "content": "PRIVATE_REPEATED_QUESTION"},
        {"role": "tool", "content": "PRIVATE_TOOL_RESULT", "tool_call_id": "call-1"},
    ]
    requests = [
        RequestComposition.capture(
            [messages[0], CacheBoundary(), *messages[1:]], generation_id="request"
        )
        for _ in range(2)
    ]
    request = requests[0]
    metadata = request.metadata(LLMUsage(input_tokens=337), "model")
    assert metadata["messages"] == len(messages)
    assert metadata["contentItems"] == requests[1].metadata(None, "model")["contentItems"]
    assert "PRIVATE_" not in json.dumps(metadata)
    for index, message in enumerate(messages, 1):
        category = "system_prompt" if message["role"] == "system" else "messages"
        key = f"Message {index} ({message['role']})"
        sections = request.inspect(category, key)["sections"]
        assert len(sections) == 1
        assert sections[0]["text"] == message["content"]
        structure = request.inspect(category, f"{key} structure / tool calls")["sections"]
        assert len(structure) == 1
        assert json.loads(structure[0]["text"]) == {
            name: value for name, value in message.items() if name != "content"
        }
    assert sum(item["tokens"] for item in metadata["items"]) == 337
    assert sum(category["tokens"] for category in metadata["categories"]) == 337


def test_event_blocks_remain_individually_inspectable_across_messages() -> None:
    sources = (
        rendered(
            [
                BlockPart(key="event_one", content="PRIVATE_FIRST_EVENT"),
                BlockPart(key="event_two", content="PRIVATE_SECOND_EVENT"),
            ]
        ),
        rendered([BlockPart(key="event_one", content="PRIVATE_LATER_EVENT")]),
    )
    request = RequestComposition.capture(
        [{"role": "system", "content": source.content} for source in sources],
        generation_id="request",
        rendered_messages=sources,
    )
    metadata = request.metadata(LLMUsage(input_tokens=107), "model")
    expected = {
        "Message 1 (system): event_one": "PRIVATE_FIRST_EVENT",
        "Message 1 (system): event_two": "PRIVATE_SECOND_EVENT",
        "Message 2 (system): event_one": "PRIVATE_LATER_EVENT",
    }
    keys = {item["key"] for item in metadata["contentItems"] if item["category"] == "messages"}
    assert keys == expected.keys()
    for key, text in expected.items():
        sections = request.inspect("messages", key)["sections"]
        assert len(sections) == 1 and sections[0]["text"] == text
    assert sum(item.chars for item in request.items if item.category == "messages") == sum(
        map(len, expected.values())
    )
    assert sum(item["tokens"] for item in metadata["items"]) == 107
    assert sum(category["tokens"] for category in metadata["categories"]) == 107
    assert "PRIVATE_" not in json.dumps(metadata)


@pytest.mark.parametrize("capture_limit", [128, 1_000_000])
def test_individual_message_overflow_keeps_totals_and_capture_bounds(
    monkeypatch: pytest.MonkeyPatch, capture_limit: int
) -> None:
    monkeypatch.setattr("aster_acp._context._MAX_CONTENT_CHARS", capture_limit)
    messages = [
        {"role": "user", "content": f"PRIVATE_MESSAGE_{index:03d}_END"} for index in range(80)
    ]
    request = RequestComposition.capture(messages, generation_id="request")
    metadata = request.metadata(LLMUsage(input_tokens=9001), "model")
    assert metadata["messages"] == len(messages)
    assert len(metadata["items"]) == len(metadata["contentItems"]) == 64
    assert any(item["key"] == "Additional context" for item in metadata["items"])
    assert sum(item.chars for item in request.items) == sum(
        len(message["content"]) + len('{"role":"user"}') for message in messages
    )
    assert sum(item["tokens"] for item in metadata["items"]) == 9001
    assert sum(category["tokens"] for category in metadata["categories"]) == 9001
    assert "PRIVATE_" not in json.dumps(metadata)
    sections = [
        section
        for category in {item.category for item in request.items}
        for section in request.inspect(category)["sections"]
    ]
    assert sum(len(section["text"]) for section in sections) <= capture_limit
    if capture_limit == 128:
        assert any(section["truncated"] for section in sections)
    else:
        assert all(not section["truncated"] for section in sections)
        text = "\n".join(section["text"] for section in sections)
        assert all(text.count(message["content"]) == 1 for message in messages)


async def test_inspector_reads_the_exact_request_and_output_without_another_call(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        [response("execute_python", {"code": "print(42)"}, 3000, 0.1), reply()],
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    agent.context["system_prompt"] = Context("VIEW_SYSTEM 中文", prefix=True)
    agent.context["repository_instructions"] = Context("VIEW_RULES", prefix=True)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "session")  # type: ignore[arg-type]
    try:
        await agent.handle({"user_messages": ["VIEW_QUESTION"]})
        await bridge.flush()
        first, latest = map(context, client.updates)
        assert first["generationId"] == latest["generationId"]
        assert first["requestId"] != latest["requestId"]
        with pytest.raises(ValueError, match="no longer available"):
            bridge.inspect_context(first["requestId"], "system_prompt")
        system = bridge.inspect_context(latest["requestId"], "system_prompt", "system_prompt")
        assert system["sections"][0]["text"] == (
            "<system_prompt>\nVIEW_SYSTEM 中文\n</system_prompt>"
        )
        rules = bridge.inspect_context(latest["requestId"], "memory_files")
        assert "VIEW_RULES" in json.dumps(rules)
        messages = bridge.inspect_context(latest["requestId"], "messages")
        assert "VIEW_QUESTION" in json.dumps(messages)
        assert "42" in json.dumps(messages)
        output = bridge.inspect_context(latest["requestId"], "output")
        assert "visible reply" in json.dumps(output) and "return_result" in json.dumps(output)
        assert latest["outputAvailable"] is True
        assert "VIEW_SYSTEM" not in json.dumps(latest)
        assert llm.call_count == 2
        agent.event_manager.add(
            LLMResponse(
                generation_id="helper",
                parts=(AssistantText(text="UNRELATED_OUTPUT"),),
                usage=LLMUsage(input_tokens=90_000, cost_usd=0.1),
            )
        )
        await bridge.flush()
        assert bridge.inspect_context(latest["requestId"], "output") == output
        assert llm.call_count == 2
    finally:
        await bridge.close()
        await agent.close()


def rendered(parts: list[BlockPart | TextPart]) -> RenderedMessage:
    return RenderedMessage(
        role=Role.SYSTEM,
        content="".join(
            part.content if isinstance(part, BlockPart) else part.text for part in parts
        ),
        parts=parts,
    )


def test_block_boundaries_schemas_and_unicode_preserve_the_reported_total() -> None:
    source = rendered(
        [
            TextPart(text="<context>"),
            BlockPart(key="system_prompt", content="PRIVATE_SYSTEM"),
            BlockPart(key="repository_instructions", content="PRIVATE_RULES 中文"),
            BlockPart(key="skills", content="PRIVATE_SKILLS"),
            TextPart(text="</context>"),
        ]
    )
    request = RequestComposition.capture(
        [
            {"role": "system", "content": source.content},
            CacheBoundary(),
            {"role": "user", "content": "PRIVATE_QUESTION"},
        ],
        generation_id="request-1",
        rendered_messages=(source,),
        tools=[{"type": "function", "function": {"name": "example", "parameters": {}}}],
        output_model=ReplySchema,
        context_stats=ContextWindowStats(
            context_blocks_count=3,
            events_count=1,
            model_context_window=1000,
            reserved_output_tokens=200,
        ),
    )
    metadata = request.metadata(LLMUsage(input_tokens=997), "model")
    assert metadata["unavailableReason"] is None
    assert sum(item["tokens"] for item in metadata["items"]) == 997
    assert sum(category["tokens"] for category in metadata["categories"]) == 997
    categories = {category["id"] for category in metadata["categories"]}
    assert categories == {
        "system_prompt",
        "system_tools",
        "memory_files",
        "skills",
        "messages",
        "runtime_context",
    }
    items = {item["key"]: item for item in metadata["items"]}
    assert items["repository_instructions"]["chars"] == len("PRIVATE_RULES 中文")
    assert items["API tool schemas"]["tokens"] > 0
    assert items["Response schema"]["tokens"] > 0
    assert (metadata["windowTokens"], metadata["reservedOutputTokens"]) == (1000, 200)
    assert metadata["messages"] == 2
    assert "PRIVATE_" not in json.dumps(metadata)


def test_rewritten_messages_cannot_reuse_the_original_block_labels() -> None:
    source = rendered([BlockPart(key="repository_instructions", content="original")])
    request = RequestComposition.capture(
        [{"role": "system", "content": "replacement"}],
        generation_id="request",
        rendered_messages=(source,),
    )
    metadata = request.metadata(LLMUsage(input_tokens=20), "model")
    assert metadata["categories"] == [{"id": "system_prompt", "tokens": 20}]
    assert all(item["key"] != "repository_instructions" for item in metadata["items"])


def test_identical_messages_consume_their_own_block_boundaries() -> None:
    sources = tuple(
        rendered([BlockPart(key=key, content="same")])
        for key in ("repository_instructions", "skills")
    )
    request = RequestComposition.capture(
        [{"role": "system", "content": "same"}] * 2,
        generation_id="request",
        rendered_messages=sources,
    )
    metadata = request.metadata(LLMUsage(input_tokens=100), "model")
    assert {item["category"] for item in metadata["items"]} == {
        "memory_files",
        "skills",
        "system_prompt",
    }
    assert sum(item["tokens"] for item in metadata["items"]) == 100


@pytest.mark.parametrize("usage", [None, LLMUsage(input_tokens=40, input_tokens_reported=False)])
def test_unreported_input_usage_has_no_guessed_category_totals(usage: LLMUsage | None) -> None:
    request = RequestComposition.capture(
        [{"role": "user", "content": "some text"}], generation_id="request"
    )
    metadata = request.metadata(usage, "model")
    assert metadata["attribution"] == "unavailable"
    assert metadata["unavailableReason"] == "no-usage"
    assert metadata["categories"] == metadata["items"] == []


def test_reported_zero_is_available_and_detail_payload_is_bounded() -> None:
    source = rendered([BlockPart(key=f"block-{i}", content="private text") for i in range(200)])
    request = RequestComposition.capture(
        [{"role": "system", "content": source.content}],
        generation_id="request",
        rendered_messages=(source,),
    )
    for tokens in (0, 1, 9999):
        metadata = request.metadata(LLMUsage(input_tokens=tokens), "model")
        assert metadata["attribution"] == "characters"
        assert len(metadata["items"]) == 64
        assert any(item["key"] == "Additional context" for item in metadata["items"])
        assert sum(item["tokens"] for item in metadata["items"]) == tokens
        assert "private text" not in json.dumps(metadata)


@pytest.mark.parametrize("kind", ["image_url", "input_audio", "file"])
def test_media_keeps_provider_usage_without_attributing_encoded_bytes(kind: str) -> None:
    request = RequestComposition.capture(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": kind, kind: {"data": "PRIVATE_DATA" * 1000}},
                ],
            }
        ],
        generation_id="request",
    )
    metadata = request.metadata(LLMUsage(input_tokens=4000), "model")
    assert metadata["attribution"] == "unavailable"
    assert metadata["unavailableReason"] == "media"
    assert metadata["categories"] == metadata["items"] == []
    assert len(json.dumps(metadata)) < 1000
    content = request.inspect("messages", "Message 1 (user)")
    assert content["sections"][0]["text"] == "caption"
    assert "PRIVATE_DATA" not in json.dumps(content)


class RecordingClient:
    def __init__(self) -> None:
        self.updates: list[UsageUpdate] = []

    async def session_update(self, session_id: str, update: object, **kwargs: Any) -> None:
        if isinstance(update, UsageUpdate):
            self.updates.append(update)


def response(name: str, arguments: dict[str, Any], tokens: int, cost: float) -> LLMResponse:
    return LLMResponse(
        finish_reason="tool_calls",
        parts=(ToolCall(id=f"call-{tokens}", name=name, arguments=json.dumps(arguments)),),
        usage=LLMUsage(input_tokens=tokens, output_tokens=10, cost_usd=cost),
    )


def reply(tokens: int = 7000) -> LLMResponse:
    return response(
        "return_result",
        {"result": {"message": "visible reply", "explanation": "done"}},
        tokens,
        0.2,
    )


def context(update: UsageUpdate) -> dict[str, Any]:
    assert update.field_meta is not None
    return update.field_meta["aster"]["context"]


@pytest.mark.parametrize("rewrite", [False, True])
async def test_real_handle_observes_final_request_and_composes_with_existing_hooks(
    tmp_path: Path, rewrite: bool
) -> None:
    llm = FakeLLMClient([reply()], strict_exhaustion=True)
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    agent.context["repository_instructions"] = Context("PRIVATE_RULES", prefix=True)
    agent.context["skills"] = Context("PRIVATE_SKILLS", prefix=True)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "session")  # type: ignore[arg-type]
    observer = ContextObserver(agent)

    async def rewrite_request(ctx, nxt):
        ctx.messages = [
            {"role": "system", "content": "PRIVATE_REPLACEMENT"},
            {"role": "user", "content": "PRIVATE_QUESTION"},
        ]
        ctx.params["max_tokens"] = 8192
        return await nxt(ctx)

    if rewrite:
        agent.event_manager.intercept("llm_call", rewrite_request)
    try:
        with hooks_scope(observer):
            result = await agent.handle({"user_messages": ["PRIVATE_QUESTION"]})
            assert get_hooks() is observer
        await bridge.flush()
        assert result.message == "visible reply"
        assert llm.call_count == len(client.updates) == 1
        update = client.updates[0]
        metadata = context(update)
        assert update.used == sum(item["tokens"] for item in metadata["items"]) == 7000
        assert sum(category["tokens"] for category in metadata["categories"]) == 7000
        assert observer.request is not None
        assert metadata["messages"] == len(llm.last_messages)
        assert metadata["windowTokens"] == llm.context_window
        assert "PRIVATE_" not in json.dumps(metadata)
        categories = {category["id"] for category in metadata["categories"]}
        if rewrite:
            assert categories == {"system_prompt", "messages"}
            assert metadata["reservedOutputTokens"] == 8192
        else:
            assert {"memory_files", "skills", "system_tools"}.issubset(categories)
    finally:
        await bridge.close()
        await agent.close()


async def test_each_tool_turn_owns_its_composition_and_helpers_only_change_cost(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient(
        [response("execute_python", {"code": "print(42)"}, 3000, 0.1), reply()],
        strict_exhaustion=True,
    )
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "session")  # type: ignore[arg-type]
    try:
        await agent.handle({"user_messages": ["Run the calculation"]})
        await bridge.flush()
        assert [update.used for update in client.updates] == [3000, 7000]
        first, latest = map(context, client.updates)
        assert first["generationId"] == latest["generationId"]
        assert first["items"] != latest["items"]
        agent.event_manager.add(
            LLMResponse(generation_id="helper", usage=LLMUsage(input_tokens=90_000, cost_usd=0.4))
        )
        await bridge.flush()
        assert client.updates[-1].used == 7000
        assert context(client.updates[-1]) == latest
        assert client.updates[-1].cost is not None
        assert client.updates[-1].cost.amount == pytest.approx(0.7)
    finally:
        await bridge.close()
        await agent.close()


async def test_changed_model_does_not_republish_the_previous_requests_context(
    tmp_path: Path,
) -> None:
    llm = FakeLLMClient([reply()], strict_exhaustion=True)
    agent = CodingAgent(llm=llm, cwd=tmp_path)
    client = RecordingClient()
    bridge = ACPEventBridge(agent, client, "session")  # type: ignore[arg-type]
    try:
        await agent.handle({"user_messages": ["First question"]})
        await bridge.flush()
        replacement = FakeLLMClient([reply(4000)], strict_exhaustion=True)
        replacement._context_window = 256_000
        agent.set_llm(replacement)
        agent.event_manager.add(
            LLMResponse(generation_id="old-helper", usage=LLMUsage(input_tokens=100, cost_usd=0.1))
        )
        await bridge.flush()
        assert len(client.updates) == 2
        pending = client.updates[-1]
        assert pending.used == 0
        assert pending.field_meta is not None
        assert pending.field_meta["aster"]["usagePending"] is True
        assert "context" not in pending.field_meta["aster"]
        assert pending.field_meta["aster"]["hud"]["latest"] == {}
        assert pending.cost is not None
        assert pending.cost.amount == pytest.approx(0.3)
        await agent.handle({"user_messages": ["Second question"]})
        await bridge.flush()
        assert len(client.updates) == 3
        assert client.updates[-1].used == 4000
        assert context(client.updates[-1])["windowTokens"] == 256_000
    finally:
        await bridge.close()
        await agent.close()
        await llm.aclose()
