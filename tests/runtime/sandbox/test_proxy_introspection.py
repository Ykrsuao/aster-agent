# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The guest's documented introspection helpers describe live host capabilities."""

import asyncio
import threading
from types import SimpleNamespace

import cloudpickle
import pytest

from aster.agentdoc import doc
from aster.agentdoc.introspect import methods, variables
from aster.runtime.context import ContextApi
from aster.runtime.context_manager import ContextManager
from aster.runtime.sandbox import wire
from aster.runtime.sandbox.bootstrap import dumps_namespace
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.executor import SandboxedExecutor
from aster.runtime.sandbox.worker import ChildBroker, ParentAgentProxy, build_namespace
from aster.strategies import CodeActStrategy
from aster.tools.todo import TodoManager

pytestmark = pytest.mark.sandbox


class _DocumentationHost:
    def __init__(self):
        self.todo = TodoManager()
        self.context_manager = ContextManager()
        self.context = ContextApi(self)
        self._lookups = 0

    @property
    def broken(self):
        raise ValueError("documentation target is unavailable")

    @property
    def observed(self):
        self._lookups += 1
        return self.todo


class _LoopbackConnection:
    """Replace only VM transport; use the real request codec and host dispatch."""

    def __init__(self, executor):
        self.executor = executor

    def send_bytes(self, data):
        self.message = wire.Codec().loads(data)

    def recv(self):
        response = asyncio.run(self.executor._service_tool_call(self.message))
        return cloudpickle.loads(dumps_namespace(response))


@pytest.fixture
def sandbox():
    agent = _DocumentationHost()
    call = SimpleNamespace(return_type=str, kwargs={}, method_name="reply")
    builtins = CodeActStrategy()._build_builtins(SimpleNamespace(agent=agent), call)
    executor = SandboxedExecutor(agent, SandboxConfig(), cell_timeout=None)
    broker = ChildBroker(_LoopbackConnection(executor), threading.RLock())
    proxy = ParentAgentProxy(broker, None)
    namespace = build_namespace(None, builtins, proxy, None, module_globals={})
    try:
        yield SimpleNamespace(agent=agent, executor=executor, proxy=proxy, namespace=namespace)
    finally:
        executor.close_sync()


@pytest.mark.parametrize(
    "path, expected",
    [
        (("todo",), "class TodoManager:"),
        (("context",), "class ContextApi:"),
        (("todo", "add"), "title: str"),
    ],
)
def test_doc_describes_host_skills_and_method_signatures(sandbox, path, expected):
    target = sandbox.proxy
    for name in path:
        target = getattr(target, name)
    text = sandbox.namespace["doc"](target, inline_depth=0)
    assert expected in text
    assert "_NestedProxy" not in text and "_BrokerCallable" not in text


@pytest.mark.parametrize(
    "target, kwargs, error_type",
    [
        (["missing"], {}, "AttributeError"),
        (["broken"], {}, "ValueError"),
        (["todo"], {"inline_depth": -1}, "ValueError"),
        (["todo", "add"], {"unknown_option": True}, "TypeError"),
    ],
)
def test_introspection_failures_are_recoverable_tool_errors(sandbox, target, kwargs, error_type):
    response = asyncio.run(
        sandbox.executor._service_tool_call(
            {
                "type": "tool_call",
                "tool_call_id": 17,
                "kind": "introspect",
                "path": ["doc", *target],
                "payload": wire.Codec().dumps(((), kwargs)),
            }
        )
    )
    assert response["type"] == "tool_result" and response["tool_call_id"] == 17
    assert response["ok"] is False and response["error_type"] == error_type
    assert "class ContextApi:" in sandbox.namespace["doc"](sandbox.proxy.context)


def test_introspection_helpers_and_import_aliases_describe_host_state(sandbox):
    sandbox.agent.todo.inspection_marker = 42
    namespace = build_namespace(
        None,
        {"api": doc, "help": doc, "method_api": methods, "field_api": variables},
        sandbox.proxy,
        None,
        module_globals={},
    )
    for name in ("doc", "api", "help"):
        assert "class TodoManager:" in namespace[name](sandbox.proxy.todo)
    for name in ("methods", "method_api"):
        assert "def add(" in namespace[name](sandbox.proxy.todo)
    for name in ("variables", "field_api"):
        assert "inspection_marker: int = 42" in namespace[name](sandbox.proxy.todo)
    assert "class _DocumentationHost:" in namespace["doc"](sandbox.proxy)


@pytest.mark.parametrize("value", [int, [int, str], {"answer": 42}])
def test_ordinary_documentation_keeps_native_behavior(sandbox, value):
    assert sandbox.namespace["doc"](value) == doc(value)


def test_user_defined_introspection_names_and_parameters_are_preserved(sandbox):
    def custom_doc(value):
        return f"custom:{value}"

    namespace = build_namespace(
        None,
        {"doc": custom_doc, "methods": 17, "variables": None, "help": custom_doc},
        sandbox.proxy,
        None,
        module_globals={},
    )
    assert namespace["doc"]("value") == "custom:value"
    assert namespace["help"]("help") == "custom:help"
    assert namespace["methods"] == 17 and namespace["variables"] is None


def test_stale_proxy_and_bad_options_raise_without_losing_the_session(sandbox):
    cached = sandbox.proxy.todo
    del sandbox.agent.todo
    with pytest.raises(AttributeError, match="todo"):
        sandbox.namespace["doc"](cached)
    with pytest.raises(TypeError, match="unexpected keyword"):
        sandbox.namespace["doc"](sandbox.proxy.context, unknown_option=True)
    assert "class ContextApi:" in sandbox.namespace["doc"](sandbox.proxy.context)


@pytest.mark.parametrize(
    "path",
    [None, [], False, "doc", ["unknown"], [[]], ["doc", 0], ["doc", None], ["doc", []]],
)
def test_introspection_rejects_invalid_function_and_target_paths(sandbox, path):
    response = asyncio.run(
        sandbox.executor._service_tool_call(
            {
                "type": "tool_call",
                "tool_call_id": 18,
                "kind": "introspect",
                "path": path,
                "payload": wire.Codec().dumps(((), {})),
            }
        )
    )
    assert response["ok"] is False and response["error_type"] == "CellSerializationError"


@pytest.mark.parametrize(
    "payload",
    [None, b"\xc1", wire.Codec().dumps([1]), wire.Codec().dumps(([], {}))],
)
def test_introspection_validates_arguments_before_resolving_the_target(sandbox, payload):
    response = asyncio.run(
        sandbox.executor._service_tool_call(
            {
                "type": "tool_call",
                "tool_call_id": 19,
                "kind": "introspect",
                "path": ["doc", "observed"],
                "payload": payload,
            }
        )
    )
    assert response["ok"] is False and response["error_type"] == "CellSerializationError"
    assert sandbox.agent._lookups == 0
