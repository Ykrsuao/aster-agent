# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandboxed cell may reach only what the agent publishes to generated code.

These drive the host-side broker dispatcher directly, so they need no microVM.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest

from aster import Agent, Context, hidden, spec
from aster.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from aster.context_blocks.models import DynamicContext
from aster.runtime.sandbox import broker_policy, wire
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.unifiedllm.fake import FakeLLMClient

_ran_on_host: list[str] = []


class _Notes:
    """A live helper object reached as ``self.notes``."""

    def __init__(self, owner: Any) -> None:
        self.agent = owner  # a public back-reference to the agent
        self.items: dict[str, int] = {"a": 1}
        self._secret = "nested-secret"

    def add(self, key: str, value: int) -> int:
        self.items[key] = value
        return len(self.items)

    def __getitem__(self, key: str) -> int:
        return self.items[key]

    def __len__(self) -> int:
        return len(self.items)


class _BrokeredAgent(Agent, llm=FakeLLMClient()):
    value: int = 41
    api_key: Annotated[str, hidden] = "sk-host-only"

    def __init__(self, *, show_context: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.notes = _Notes(self)
        self._token = "private-token"
        if show_context:
            spec(self, "context", hidden=False)

    def add_one(self, n: int) -> int:
        return n + 1

    async def async_double(self, n: int) -> int:
        return n * 2

    @hidden
    def wipe(self) -> str:
        _ran_on_host.append("wipe")
        return "wiped"

    def _private_helper(self) -> str:
        _ran_on_host.append("private")
        return "private"

    @spec(hidden=False)
    def _shown_helper(self) -> str:
        return "shown"


def _executor(agent: Any) -> SandboxedExecutor:
    executor = object.__new__(SandboxedExecutor)
    executor._agent = agent
    executor._max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error
    return executor


async def _call(agent: Any, path: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
    return await _executor(agent)._dispatch_tool_call(
        {"kind": "call", "path": path, "args": args, "kwargs": kwargs}
    )


async def _request(agent: Any, kind: str, path: list[str], **extra: Any) -> dict[str, Any]:
    return await _executor(agent)._dispatch_tool_call({"kind": kind, "path": path, **extra})


def _assert_refused(response: dict[str, Any]) -> None:
    assert response["ok"] is False, response
    assert response["error_type"] == "BrokerAccessError", response
    assert "not available to sandboxed code" in response["error"]


@pytest.fixture(autouse=True)
def _clear_host_log() -> None:
    _ran_on_host.clear()


# --- the escapes -------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "args", "kwargs"),
    [
        (["runtime", "evaluate_expression"], ("__import__('os').getcwd()",), {}),
        (["runtime", "execute_code"], ("import os",), {"validate": False}),
        (["event_manager", "run_middleware"], (), {}),
        (["context_manager", "set_dynamic"], ("k", "1 + 1"), {}),
        (["notes", "agent", "runtime", "evaluate_expression"], ("1 + 1",), {}),
        (["add_one", "__self__", "runtime", "evaluate_expression"], ("1 + 1",), {}),
        (["add_one", "__globals__", "__getitem__"], ("_ran_on_host",), {}),
        (["__class__", "__init__", "__globals__", "__getitem__"], ("os",), {}),
        (["__getattribute__"], ("runtime",), {}),
        (["__setattr__"], ("value", 0), {}),
        (["__dict__", "__getitem__"], ("_token",), {}),
    ],
)
async def test_guest_cannot_reach_host_code_execution(path, args, kwargs) -> None:
    _assert_refused(await _call(_BrokeredAgent(), path, *args, **kwargs))


async def test_hidden_and_private_methods_do_not_run() -> None:
    agent = _BrokeredAgent()
    _assert_refused(await _call(agent, ["wipe"]))
    _assert_refused(await _call(agent, ["_private_helper"]))
    assert _ran_on_host == []


@pytest.mark.parametrize(
    "path",
    [["api_key"], ["_token"], ["_llm"], ["llm"], ["runtime"], ["notes", "_secret"], ["__dict__"]],
)
async def test_guest_cannot_read_hidden_or_private_state(path) -> None:
    agent = _BrokeredAgent()
    for kind in ("attr", "bool", "iter"):
        response = await _request(agent, kind, path)
        _assert_refused(response)
        assert "sk-host-only" not in str(response)
        assert "private-token" not in str(response)


@pytest.mark.parametrize(
    "path", [["api_key"], ["_token"], ["runtime"], ["notes", "_secret"], ["__class__"]]
)
async def test_guest_cannot_overwrite_hidden_or_private_state(path) -> None:
    agent = _BrokeredAgent()
    runtime = agent.runtime
    _assert_refused(await _request(agent, "setattr", path, value="guest"))
    assert agent.api_key == "sk-host-only"
    assert agent._token == "private-token"
    assert agent.runtime is runtime
    assert agent.notes._secret == "nested-secret"


async def test_guest_cannot_introspect_hidden_objects() -> None:
    executor = _executor(_BrokeredAgent())
    executor._codec = wire.Codec()
    response = await executor._introspect(
        {"path": ["doc", "runtime"], "payload": executor._codec.dumps(((), {}))}
    )
    assert response["ok"] is False
    assert response["error_type"] == "BrokerAccessError"


async def test_variables_introspection_respects_instance_hidden_fields() -> None:
    agent = _BrokeredAgent()
    agent.notes.visible_note = "visible-marker"
    agent.notes.hidden_note = "hidden-marker"
    spec(agent.notes, "hidden_note", hidden=True)
    _assert_refused(await _request(agent, "attr", ["notes", "hidden_note"]))

    executor = _executor(agent)
    executor._codec = wire.Codec()
    response = await executor._introspect(
        {"path": ["variables", "notes"], "payload": executor._codec.dumps(((), {}))}
    )

    assert response["ok"] is True
    assert "hidden_note" not in response["result"]
    assert "hidden-marker" not in response["result"]
    assert "visible-marker" in response["result"]


# --- expression context blocks run on the host -------------------------------


@pytest.mark.parametrize(
    ("path", "args", "kwargs"),
    [
        (["context", "set"], ("k",), {"expr": "__import__('os').getcwd()"}),
        (["context", "set_static"], ("k",), {"expr": "1 + 1"}),
        (["context", "set_dynamic"], ("k", "1 + 1"), {}),
        (["context", "set_dynamic"], ("k",), {"expr": "1 + 1"}),
        (["context", "__setitem__"], ("k", Context(expr="1 + 1")), {}),
        (["context", "__setitem__"], ("k", DynamicContext("1 + 1")), {}),
    ],
)
async def test_guest_cannot_register_host_evaluated_expressions(path, args, kwargs) -> None:
    agent = _BrokeredAgent(show_context=True)
    _assert_refused(await _call(agent, path, *args, **kwargs))
    assert "k" not in agent.context


async def test_shown_context_still_accepts_literal_blocks() -> None:
    agent = _BrokeredAgent(show_context=True)
    assert (await _call(agent, ["context", "__setitem__"], "plan", "host plan"))["ok"]
    assert (await _call(agent, ["context", "set"], "detail", "host detail"))["ok"]
    assert (await _call(agent, ["context", "__setitem__"], "fixed", Context("v", prefix=True)))[
        "ok"
    ]
    assert agent.context["plan"] == "host plan"
    assert agent.context["detail"] == "host detail"
    assert (await _call(agent, ["context", "__contains__"], "plan"))["result"] is True


async def test_context_is_refused_until_the_agent_shows_it() -> None:
    _assert_refused(await _call(_BrokeredAgent(), ["context", "__setitem__"], "plan", "text"))


# --- the published surface keeps working -------------------------------------


async def test_public_methods_and_fields_are_brokered() -> None:
    agent = _BrokeredAgent()
    assert (await _call(agent, ["add_one"], 41))["result"] == 42
    doubled = await _call(agent, ["async_double"], 21)
    assert doubled["result"] == 42 and doubled["was_async"] is True
    assert (await _request(agent, "attr", ["value"]))["result"] == 41
    described = await _request(agent, "attr", ["add_one"])
    assert described["callable"] is True and described["is_async"] is False
    assert (await _request(agent, "setattr", ["value"], value=7))["ok"]
    assert agent.value == 7


async def test_private_method_opted_in_with_spec_is_brokered() -> None:
    assert (await _call(_BrokeredAgent(), ["_shown_helper"]))["result"] == "shown"


async def test_nested_objects_and_container_operators_are_brokered() -> None:
    agent = _BrokeredAgent()
    assert (await _call(agent, ["notes", "add"], "b", 2))["result"] == 2
    assert (await _call(agent, ["notes", "__getitem__"], "b"))["result"] == 2
    assert (await _call(agent, ["notes", "__len__"]))["result"] == 2
    assert (await _request(agent, "bool", ["notes"]))["result"] is True
    assert (await _request(agent, "iter", ["notes", "items"]))["result"] == ["a", "b"]


async def test_container_operator_is_refused_outside_a_call_or_mid_path() -> None:
    agent = _BrokeredAgent()
    _assert_refused(await _request(agent, "attr", ["notes", "__getitem__"]))
    _assert_refused(await _call(agent, ["notes", "__getitem__", "__self__"]))


async def test_deleting_an_attribute_checks_the_named_attribute() -> None:
    agent = _BrokeredAgent()
    agent.notes.scratch = 1
    assert (await _call(agent, ["notes", "__delattr__"], "scratch"))["ok"]
    assert not hasattr(agent.notes, "scratch")
    _assert_refused(await _call(agent, ["notes", "__delattr__"], "_secret"))
    assert agent.notes._secret == "nested-secret"


def test_resolve_reports_the_refused_segment() -> None:
    with pytest.raises(broker_policy.BrokerAccessError, match=r"self\.notes\.agent\.runtime "):
        broker_policy.resolve(_BrokeredAgent(), ["notes", "agent", "runtime", "execute_code"])
