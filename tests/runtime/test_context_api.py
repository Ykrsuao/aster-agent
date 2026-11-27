# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ContextApi — dict-like Skill wrapping agent.context_manager."""

import pytest

from aster import Agent
from aster.context_blocks import Context, DynamicContext
from aster.context_blocks.exceptions import DynamicNotResolvedError, ProtectedBlockError
from aster.runtime.context import ContextApi
from aster.unifiedllm import FakeLLMClient

_llm = FakeLLMClient()


class _TestAgent(Agent, llm=_llm):
    pass


def _ctx() -> ContextApi:
    return _TestAgent().context


# ── setitem ────────────────────────────────────────────────────────────────────


def test_setitem_and_getitem():
    ctx = _ctx()
    ctx["key"] = "value"
    assert ctx["key"] == "value"


def test_setitem_raises_on_protected_key():
    ctx = _ctx()
    with pytest.raises(ProtectedBlockError):
        ctx["system_prompt"] = "overwrite"


def test_set_plain_value_still_rejects_protected_key():
    with pytest.raises(ProtectedBlockError):
        _ctx().set("system_prompt", "overwrite")


@pytest.mark.parametrize("value", [{"key": "value"}, ["one", "two"]])
def test_set_plain_value_preserves_existing_structured_values(value):
    ctx = _ctx()
    ctx.set("data", value)
    assert ctx["data"] is value
    assert not ctx._context.is_optional("data")


def test_setitem_accepts_dynamic_context_value():
    ctx = _ctx()
    ctx["dynamic"] = DynamicContext("'expr'")
    assert "dynamic" in ctx._context


@pytest.mark.parametrize("expr", [None, "'catalog'"])
@pytest.mark.parametrize("prefix", [False, True])
def test_set_preserves_optional_independently_of_content_and_placement(expr, prefix):
    ctx = _ctx()
    ctx.set("catalog", "catalog" if expr is None else None, expr=expr, prefix=prefix, optional=True)
    assert ctx._context.is_optional("catalog")
    assert ctx._context.is_static("catalog") is prefix
    ctx.set("catalog", "required", prefix=prefix)
    assert not ctx._context.is_optional("catalog")


@pytest.mark.parametrize(
    "operation",
    ["assign", "context", "dynamic", "static", "override", "delete", "pop", "none"],
)
def test_replacing_or_removing_a_block_clears_optional(operation):
    cm = _ctx()._context
    cm["catalog"] = Context("catalog", prefix=True, optional=True)
    assert cm.is_optional("catalog")
    if operation == "assign":
        cm["catalog"] = "required"
    elif operation == "context":
        cm["catalog"] = Context("required")
    elif operation == "dynamic":
        cm.set_dynamic("catalog", "'required'")
    elif operation == "static":
        cm.set_static("catalog", "required")
    elif operation == "override":
        cm.apply_override("catalog", "required")
    elif operation == "delete":
        del cm["catalog"]
    elif operation == "pop":
        cm.pop("catalog")
    else:
        cm["catalog"] = None
    assert not cm.is_optional("catalog")


@pytest.mark.parametrize("operation", ["static", "dynamic", "remove", "none", "override"])
def test_protected_block_changes_clear_optional(operation):
    cm = _ctx()._context
    cm["system_prompt"] = Context("catalog", prefix=True, optional=True)
    assert cm.is_optional("system_prompt")
    if operation == "static":
        cm.set_static_protected("system_prompt", "required")
    elif operation == "dynamic":
        cm.set_dynamic_protected("system_prompt", value="required")
    elif operation == "remove":
        cm.remove_protected("system_prompt")
    elif operation == "none":
        cm["system_prompt"] = None
    else:
        cm.apply_override("system_prompt", None)
    assert not cm.is_optional("system_prompt")


def test_optional_survives_disable_enable_and_failed_replacement():
    cm = _ctx()._context
    cm["catalog"] = Context("catalog", optional=True)
    cm.disable("catalog")
    cm.enable("catalog")
    with pytest.raises(TypeError):
        cm.set_static("catalog", "value", expr="'invalid'")
    assert cm.is_optional("catalog")
    cm.set_optional("catalog", False)
    assert not cm.is_optional("catalog")
    with pytest.raises(KeyError):
        cm.set_optional("missing", True)


def test_class_and_instance_context_preserve_optional():
    class OptionalAgent(Agent, llm=_llm, context={"catalog": Context("class", optional=True)}):
        pass

    assert OptionalAgent().context_manager.is_optional("catalog")
    assert not OptionalAgent(context={"catalog": "required"}).context_manager.is_optional("catalog")
    assert OptionalAgent(
        context={"catalog": Context(expr="'instance'", prefix=True, optional=True)}
    ).context_manager.is_optional("catalog")


def test_set_static_expr_lands_in_static_partition():
    ctx = _ctx()
    ctx.set_static("live", expr="1 + 1")
    # Re-evaluated (DynamicContext marker) yet in the static/cacheable partition.
    assert ctx._context.is_static("live") is True
    assert isinstance(ctx._context._blocks["live"], DynamicContext)
    assert ctx._context._blocks["live"].expr == "1 + 1"


def test_set_static_rejects_both_value_and_expr():
    ctx = _ctx()
    with pytest.raises(TypeError, match="both"):
        ctx.set_static("k", "v", expr="1")


def test_set_static_rejects_neither():
    ctx = _ctx()
    with pytest.raises(TypeError, match="requires"):
        ctx.set_static("k")


# ── getitem ────────────────────────────────────────────────────────────────────


def test_getitem_raises_for_missing_key():
    ctx = _ctx()
    with pytest.raises(KeyError):
        _ = ctx["missing"]


def test_getitem_dynamic_raises_before_resolve():
    ctx = _ctx()
    ctx.set_dynamic("dyn", "'hello'")
    with pytest.raises(DynamicNotResolvedError):
        _ = ctx["dyn"]


def test_getitem_dynamic_returns_after_resolve():
    ctx = _ctx()
    ctx.set_dynamic("dyn", "'hello'")
    ctx._context._update_resolved({"dyn": "hello"})
    assert ctx["dyn"] == "hello"


# ── set_dynamic ────────────────────────────────────────────────────────────────


def test_set_dynamic_raises_on_protected_key():
    ctx = _ctx()
    with pytest.raises(ProtectedBlockError):
        ctx.set_dynamic("system_prompt", "'expr'")


# ── delitem ────────────────────────────────────────────────────────────────────


def test_delitem():
    ctx = _ctx()
    ctx["key"] = "value"
    del ctx["key"]
    assert "key" not in ctx


def test_delitem_raises_for_missing_key():
    ctx = _ctx()
    with pytest.raises(KeyError):
        del ctx["missing"]


def test_delitem_raises_for_protected_key():
    ctx = _ctx()
    ctx._context._blocks["guarded"] = "value"
    ctx._context.protected_keys.add("guarded")
    with pytest.raises(ProtectedBlockError):
        del ctx["guarded"]


# ── contains / len / iter / repr ───────────────────────────────────────────────


def test_contains():
    ctx = _ctx()
    ctx["a"] = 1
    assert "a" in ctx
    assert "b" not in ctx


def test_len_and_iter():
    ctx = _ctx()
    ctx["a"] = 1
    ctx["b"] = 2
    assert len(ctx) == 2
    assert set(ctx) == {"a", "b"}


def test_repr():
    ctx = _ctx()
    ctx["x"] = 1
    assert "1 block" in repr(ctx)


# ── keys / _raw_items ──────────────────────────────────────────────────────────


def test_keys():
    ctx = _ctx()
    ctx["a"] = 1
    ctx["b"] = 2
    assert set(ctx.keys()) == {"a", "b"}


# ── get ────────────────────────────────────────────────────────────────────────


def test_get_returns_value():
    ctx = _ctx()
    ctx["key"] = "value"
    assert ctx.get("key") == "value"


def test_get_returns_default_for_missing():
    ctx = _ctx()
    assert ctx.get("missing", "default") == "default"
    assert ctx.get("missing") is None


# ── pop ────────────────────────────────────────────────────────────────────────


def test_pop_static_block():
    ctx = _ctx()
    ctx["key"] = "value"
    assert ctx.pop("key") == "value"
    assert "key" not in ctx


def test_pop_dynamic_block():
    ctx = _ctx()
    ctx._context._blocks["dyn"] = DynamicContext("'hello'")
    ctx._context._dynamic_cache["dyn"] = "hello"
    assert ctx.pop("dyn") == "hello"
    assert "dyn" not in ctx


def test_pop_returns_default_for_missing():
    ctx = _ctx()
    assert ctx.pop("missing", "default") == "default"


def test_pop_raises_for_missing_without_default():
    ctx = _ctx()
    with pytest.raises(KeyError):
        ctx.pop("missing")


def test_pop_raises_for_protected_key():
    ctx = _ctx()
    ctx._context._blocks["guarded"] = "value"
    ctx._context.protected_keys.add("guarded")
    with pytest.raises(ProtectedBlockError):
        ctx.pop("guarded")
