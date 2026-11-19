# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior shared by host cells and the guest's local execution entry point.

These tests use the real execution paths without starting a microVM. The host
adapter only carries returned REPL locals into the next call, as CodeAct does;
the guest persists those locals in its namespace directly.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from aster import Agent
from aster.errors.formatting import format_error_for_llm
from aster.events import _NO_RETURN, ExecutionResult, ExecutionSignal
from aster.runtime.actor import ActorRuntime
from aster.runtime.sandbox.cell_core import run_cell_source
from aster.unifiedllm import FakeLLMClient


class _CellAgent(Agent, llm=FakeLLMClient()):
    pass


@dataclass
class _CellSession:
    backend: str
    runtime: ActorRuntime | None = None
    namespace: dict[str, Any] = field(default_factory=dict)
    execution_namespace: dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0

    async def run(
        self, source: str, *, execution_count: int | None = None, validate: bool = True
    ) -> ExecutionResult:
        self.execution_count = (
            self.execution_count + 1 if execution_count is None else execution_count
        )
        if self.runtime is None:
            return await run_cell_source(
                source, self.namespace, execution_count=self.execution_count
            )
        result = await self.runtime.execute_code(
            source,
            builtins=self.namespace,
            namespace=self.execution_namespace,
            validate=validate,
            wrap_in_function=True,
            execution_count=self.execution_count,
        )
        self.namespace.update(result.captured_locals)
        self.namespace.update(result.defined_methods)
        return result


@pytest.fixture(params=["actor", "guest"])
async def cell_session(request: pytest.FixtureRequest) -> AsyncIterator[_CellSession]:
    agent = _CellAgent() if request.param == "actor" else None
    try:
        yield _CellSession(request.param, agent.runtime if agent is not None else None)
    finally:
        if agent is not None:
            await agent.aclose()


@pytest.mark.parametrize(
    ("source", "expected", "explicit"),
    [
        pytest.param("answer = 42", _NO_RETURN, False, id="assignment"),
        pytest.param("answer = 40\nanswer + 2", 42, False, id="last-expression"),
        pytest.param("len([1, 2, 3])", 3, False, id="last-call"),
        pytest.param("None", _NO_RETURN, False, id="implicit-none"),
        pytest.param("return None", None, True, id="explicit-none"),
        pytest.param("return", None, True, id="bare-return"),
        pytest.param("return 42", 42, True, id="explicit-value"),
        pytest.param("if True:\n    return 42\n99", 42, True, id="return-in-if"),
        pytest.param("for value in [42]:\n    return value", 42, True, id="return-in-loop"),
        pytest.param(
            "try:\n    return 41\nfinally:\n    return 42", 42, True, id="return-in-finally"
        ),
        pytest.param(
            "if False:\n    return 7\n42", None, True, id="unreached-return-disables-implicit"
        ),
    ],
)
async def test_return_values(cell_session: _CellSession, source, expected, explicit):
    result = await cell_session.run(source)

    assert result.error is None
    assert result.signal is None
    assert result.explicit_return is explicit
    assert result.has_return is (expected is not _NO_RETURN)
    if expected is _NO_RETURN:
        assert result.returned_value is _NO_RETURN
    else:
        assert result.returned_value == expected


async def test_explicit_return_skips_deep_unreachable_expression(cell_session: _CellSession):
    source = "return 1\n" + "+".join(["1"] * 600)

    result = await cell_session.run(source, validate=False)

    assert result.error is None
    assert result.returned_value == 1
    assert result.explicit_return


async def test_implicit_none_keeps_print_output(cell_session: _CellSession):
    result = await cell_session.run("print('visible')")

    assert result.error is None
    assert result.stdout == "visible\n"
    assert result.returned_value is _NO_RETURN
    assert not result.explicit_return


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("def helper():\n    return 42\nhelper()", id="function"),
        pytest.param("async def helper():\n    return 42\nawait helper()", id="async-function"),
        pytest.param(
            "class Helper:\n    def answer(self):\n        return 42\nHelper().answer()",
            id="class-method",
        ),
    ],
)
async def test_nested_returns_do_not_disable_last_expression(cell_session: _CellSession, source):
    result = await cell_session.run(source)

    assert result.error is None
    assert result.returned_value == 42
    assert not result.explicit_return


@pytest.mark.parametrize(
    ("source", "expected_events", "expected_value"),
    [
        pytest.param(
            "events.append('a'); events.append('b')", ["a", "b"], _NO_RETURN, id="two-calls"
        ),
        pytest.param(
            "events.append('a')\nevents.append('b'); events.append('c'); len(events)",
            ["a", "b", "c"],
            3,
            id="compound-final-line",
        ),
        pytest.param(
            "count = 3; events.append(str(count)); count + 1", ["3"], 4, id="assignment-and-call"
        ),
    ],
)
async def test_semicolons_keep_earlier_statements(
    cell_session: _CellSession, source, expected_events, expected_value
):
    cell_session.namespace["events"] = []

    result = await cell_session.run(source)

    assert result.error is None
    assert cell_session.namespace["events"] == expected_events
    assert result.returned_value == expected_value
    assert not result.explicit_return


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("label = 'é'; label", "é", id="two-byte-literal"),
        pytest.param("值 = 40; 值 + 2", 42, id="three-byte-identifier"),
        pytest.param("text = '🪐'; len(text)", 1, id="four-byte-literal"),
        pytest.param("label = 'é'; (\n    40\n    + 2\n)", 42, id="multiline-after-unicode"),
        pytest.param("sum([\n    1,\n    2,\n])", 3, id="multiline-call"),
    ],
)
async def test_last_expression_uses_python_source_columns(
    cell_session: _CellSession, source, expected
):
    result = await cell_session.run(source)

    assert result.error is None
    assert result.returned_value == expected
    assert not result.explicit_return


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param('"""first\n  second\n\nlast"""', "first\n  second\n\nlast", id="literal"),
        pytest.param(
            'message = """first\n  second\n\nlast"""\nmessage',
            "first\n  second\n\nlast",
            id="assigned-literal",
        ),
        pytest.param(
            'value = 2\nf"""first {value}\n  second {value + 1}\nlast"""',
            "first 2\n  second 3\nlast",
            id="f-string",
        ),
        pytest.param(
            'value = 2\nmessage = f"""first {value}\n  second {value + 1}\nlast"""\nmessage',
            "first 2\n  second 3\nlast",
            id="assigned-f-string",
        ),
        pytest.param(
            'f"""{\n1 = }"""',
            "\n1 = 1",
            id="f-string-debug-expression",
        ),
        pytest.param(
            'value = 2\nmessage = f"""{\n  value = \n}"""\nmessage',
            "\n  value = \n2",
            id="assigned-f-string-debug-expression",
        ),
        pytest.param(
            'f"""outer {f"""{\n1 = }"""}\ntail"""',
            "outer \n1 = 1\ntail",
            id="nested-f-string-debug-expression",
        ),
        pytest.param("'first\\\n  second'", "first  second", id="continued-string"),
    ],
)
async def test_wrapping_preserves_multiline_string_contents(
    cell_session: _CellSession, source, expected
):
    result = await cell_session.run(source)

    assert result.error is None
    assert result.returned_value == expected


async def test_carriage_return_inside_string_preserves_cell_state(cell_session: _CellSession):
    source = 's = """a\rb"""\ns'

    result = await cell_session.run(source)

    assert result.error is None
    # Python counts the bare carriage return as a source line. When that places
    # the final expression beyond the editable lines, the cell still executes.
    assert not result.has_return
    assert cell_session.namespace["s"] == "a\nb"

    resumed = await cell_session.run("s")

    assert resumed.error is None
    assert resumed.returned_value == "a\nb"


async def test_unclosed_string_is_reported_as_syntax_error(cell_session: _CellSession):
    result = await cell_session.run("s = '''\nunclosed triple", validate=False)

    assert isinstance(result.error, SyntaxError)
    assert not result.has_return
    assert "s" not in cell_session.namespace

    resumed = await cell_session.run("40 + 2")

    assert resumed.error is None
    assert resumed.returned_value == 42


@pytest.mark.parametrize("exit_kind", ["fallthrough", "return", "error", "signal", "signal-call"])
async def test_locals_persist_across_cell_exits(cell_session: _CellSession, exit_kind):
    signal = ExecutionSignal("return_result", {"result": 41})

    def finish():
        raise signal

    cell_session.namespace.update(finish=finish, completion=signal)
    exits = {
        "fallthrough": "pass",
        "return": "return answer\nunreached = 99",
        "error": "raise ValueError('boom')\nunreached = 99",
        "signal": "raise completion\nunreached = 99",
        "signal-call": "finish()\nunreached = 99",
    }
    source = (
        "answer = 41\nitems = ['before']\nitems.append('after')\n_private = 'hidden'\n"
        + exits[exit_kind]
    )

    result = await cell_session.run(source)

    if exit_kind == "error":
        assert isinstance(result.error, ValueError)
    else:
        assert result.error is None
    assert result.signal is (signal if exit_kind.startswith("signal") else None)
    assert cell_session.namespace["answer"] == 41
    assert cell_session.namespace["items"] == ["before", "after"]
    assert "unreached" not in cell_session.namespace
    assert "_private" not in cell_session.namespace

    resumed = await cell_session.run("answer = answer + 1\nitems.append(answer)\n(answer, items)")

    assert resumed.error is None
    assert resumed.returned_value == (42, ["before", "after", 42])


@pytest.mark.parametrize("asynchronous", [False, True], ids=["function", "async-function"])
async def test_persisted_methods_keep_original_source(cell_session: _CellSession, asynchronous):
    prefix = "async " if asynchronous else ""
    method_source = (
        f"{prefix}def helper(self, value):\n    # Preserve this comment.\n    return value * 2"
    )
    source = "# Cell preamble.\n\n" + method_source + "\n"

    result = await cell_session.run(source, execution_count=71)

    assert result.error is None
    assert set(result.defined_methods) == {"helper"}
    helper = result.defined_methods["helper"]
    assert helper is cell_session.namespace["helper"]
    assert helper.__code__.co_filename == "Cell In[71]"
    assert helper.__code__.co_firstlineno == 3
    assert inspect.getsource(helper) == method_source + "\n"
    assert helper._generated_source == method_source

    call = "helper(None, 21)"
    resumed = await cell_session.run("await " + call if asynchronous else call)

    assert resumed.error is None
    assert resumed.returned_value == 42


@pytest.mark.parametrize("asynchronous", [False, True], ids=["function", "async-function"])
@pytest.mark.parametrize("with_self", [False, True], ids=["plain", "self-parameter"])
async def test_private_helpers_persist_and_use_live_state(
    cell_session: _CellSession, asynchronous, with_self
):
    prefix = "async " if asynchronous else ""
    parameters = "self, value" if with_self else "value"
    arguments = "None, 2" if with_self else "2"
    call = ("await " if asynchronous else "") + f"_helper({arguments})"
    definition = f"{prefix}def _helper({parameters}):\n    return value + offset\n"

    defined = await cell_session.run("offset = 40\n_private = 'temporary'\n" + definition)

    assert defined.error is None
    assert "_helper" in cell_session.namespace
    assert "_private" not in cell_session.namespace
    assert "__repl_wrapper__" not in defined.captured_locals
    assert "__repl_captured_locals__" not in defined.captured_locals
    assert "_gvar" not in defined.captured_locals
    assert set(defined.defined_methods) == ({"_helper"} if with_self else set())

    resumed = await cell_session.run("offset = 41\n" + call)

    assert resumed.error is None
    assert resumed.returned_value == 43
    assert "_private" not in cell_session.namespace

    replaced = await cell_session.run(definition.replace(" + offset", " * offset"))
    assert replaced.error is None
    after_replacement = await cell_session.run("offset = 21\n" + call)
    assert after_replacement.error is None
    assert after_replacement.returned_value == 42


async def test_persisted_helpers_resolve_sibling_functions(cell_session: _CellSession):
    defined = await cell_session.run(
        "def increment(value):\n    return value + 1\n"
        "def answer(value):\n    return increment(value)"
    )

    assert defined.error is None
    assert not defined.defined_methods

    result = await cell_session.run("answer(41)")

    assert result.error is None
    assert result.returned_value == 42


async def test_persisted_helper_keeps_its_own_cell_location(cell_session: _CellSession):
    helper_source = 'def boom():\n    marker = "helper"\n    raise ValueError("x")\n'
    defined = await cell_session.run(helper_source, execution_count=71)
    assert defined.error is None

    # This changes the current cell's wrapper offset, not the old helper's lines.
    cell_session.namespace["persisted"] = 1
    failed = await cell_session.run("boom()", execution_count=73)

    assert isinstance(failed.error, ValueError)
    formatted = format_error_for_llm(failed.error, "boom()", line_offset=failed.wrapper_line_offset)
    assert "Cell In[73], line 1" in formatted
    assert "Cell In[71], line 3, in boom" in formatted
    assert 'raise ValueError("x")' in formatted
    assert "Cell In[71], line 1, in boom" not in formatted


@pytest.mark.parametrize("exit_kind", ["success", "error", "signal"])
async def test_helper_definitions_execute_once_in_source_order(
    cell_session: _CellSession, exit_kind
):
    events = []
    signal = ExecutionSignal()

    def default_value():
        events.append("default")
        return 42

    def decorate(function):
        events.append("decorate")
        return function

    def finish():
        raise signal

    cell_session.namespace.update(
        events=events, default_value=default_value, decorate=decorate, finish=finish
    )
    source = (
        "events.append('before')\n"
        "@decorate\ndef helper(self, value=default_value()):\n    return value\n"
        "events.append('after')\n"
    )
    source += {"success": "pass", "error": "raise ValueError('boom')", "signal": "finish()"}[
        exit_kind
    ]

    result = await cell_session.run(source)

    if exit_kind == "error":
        assert isinstance(result.error, ValueError)
    else:
        assert result.error is None
    assert result.signal is (signal if exit_kind == "signal" else None)
    assert events == ["before", "default", "decorate", "after"]


@pytest.mark.parametrize("annotation", ["int", "MissingCellType"])
async def test_annotation_evaluation_preserves_backend_policy(
    cell_session: _CellSession, annotation
):
    # compile() inherits future flags from its caller: the guest opts into
    # postponed annotations, while the host deliberately retains eager ones.
    source = f"def helper(self, value: {annotation}) -> {annotation}:\n    return value"

    result = await cell_session.run(source)

    if cell_session.backend == "actor" and annotation == "MissingCellType":
        assert isinstance(result.error, NameError)
        assert "MissingCellType" in str(result.error)
        assert not result.defined_methods
        return
    assert result.error is None
    expected = annotation if cell_session.backend == "guest" else int
    assert result.defined_methods["helper"].__annotations__ == {
        "value": expected,
        "return": expected,
    }
