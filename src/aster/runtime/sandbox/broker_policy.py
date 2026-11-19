# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a sandboxed cell may reach on the host through the ``self.*`` broker.

The guest names a dotted attribute path and the host resolves it against the
live agent. The guest is untrusted, so the host accepts only the surface the
agent author published to generated code: the names ``doc(self)`` shows.

- A name starting with ``_`` is refused, unless it is a method opted back in
  with ``@spec(hidden=False)``.
- A member hidden with ``@hidden``, ``Annotated[T, hidden]`` or
  ``spec(..., hidden=True)`` is refused. That covers every framework attribute
  on ``Agent`` (``runtime``, ``event_manager``, ``context_manager``, ...), which
  would otherwise let a cell run code in the host process.
- The container operators the guest proxies forward (``self.vars["k"]``,
  ``len(self.bag)``, ``del self.v.name``) are accepted only as the final
  segment of a call.
- Registering an expression context block is refused: the host evaluates the
  expression on later turns, outside the sandbox.

Each rule is applied at every segment, so ``self.tool.agent.runtime`` is
refused at ``runtime`` exactly as ``self.runtime`` is.
"""

from __future__ import annotations

import inspect
from typing import Any

# Operators forwarded by the guest's nested proxies (see worker._NestedProxy).
_CONTAINER_OPERATORS = frozenset(
    {"__getitem__", "__setitem__", "__delitem__", "__contains__", "__len__", "__delattr__"}
)


class BrokerAccessError(PermissionError):
    """A sandboxed cell addressed a host member that generated code may not use."""


def _refuse(path: list[str], reason: str) -> BrokerAccessError:
    return BrokerAccessError(
        f"self.{'.'.join(path)} is not available to sandboxed code: {reason}. "
        "Only the members shown by doc(self) can be used from a sandboxed cell."
    )


def _static_member(obj: Any, name: str) -> Any:
    """Look *name* up without running descriptors or ``__getattr__``."""
    try:
        return inspect.getattr_static(obj, name)
    except AttributeError:
        return None


def _is_hidden(obj: Any, name: str) -> bool:
    from aster.agentdoc._visibility import is_hidden_field, is_hidden_method

    if is_hidden_field(obj, name):
        return True
    member = _static_member(obj, name)
    return is_hidden_method(member) or is_hidden_method(getattr(member, "__func__", None))


def _is_shown_private_method(obj: Any, name: str) -> bool:
    """True for a ``_private`` method opted in with ``@spec(hidden=False)``."""
    member = _static_member(obj, name)
    function = getattr(member, "__func__", member)
    if not callable(function):
        return False
    return (
        getattr(function, "_agentdoc_hidden", None) is False
        or getattr(member, "_agentdoc_hidden", None) is False
    )


def check_member(obj: Any, name: str, path: list[str]) -> None:
    """Refuse *name* on *obj* unless generated code is allowed to use it."""
    if not isinstance(name, str) or not name:
        raise _refuse(path, "the attribute name is not valid")
    if name.startswith("__"):
        raise _refuse(path, f"{name!r} is an internal attribute")
    if name.startswith("_") and not _is_shown_private_method(obj, name):
        raise _refuse(path, f"{name!r} is private")
    if _is_hidden(obj, name):
        raise _refuse(path, f"{name!r} is hidden")


def resolve(agent: Any, path: list[str], *, operator: bool = False) -> Any:
    """Resolve a guest-supplied dotted *path* on *agent*, checking each segment.

    With ``operator=True`` the final segment may be one of the container
    operators the guest proxies forward; it is still refused anywhere else.
    """
    obj = agent
    last = len(path) - 1
    for index, name in enumerate(path):
        if not (operator and index == last and name in _CONTAINER_OPERATORS):
            check_member(obj, name, path[: index + 1])
        obj = getattr(obj, name)
    return obj


def _registers_expression(value: Any) -> bool:
    from aster.context_blocks.models import Context, DynamicContext

    return isinstance(value, DynamicContext) or (
        isinstance(value, Context) and value.expr is not None
    )


def check_call(target: Any, path: list[str], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Refuse calls whose arguments the host would later execute as code."""
    from aster.runtime.context import ContextApi
    from aster.runtime.context_manager import ContextManager

    owner = getattr(target, "__self__", None)
    if path and path[-1] == "__delattr__" and args:
        # ``del self.v.name`` deletes an attribute the guest names in an argument.
        check_member(owner, args[0], [*path[:-1], str(args[0])])
    if not isinstance(owner, ContextApi | ContextManager):
        return
    reason = "expression context blocks are evaluated on the host, outside the sandbox"
    if any(_registers_expression(value) for value in (*args, *kwargs.values())):
        raise _refuse(path, reason)
    try:
        bound = inspect.signature(target).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return  # The call itself reports the bad arguments.
    if bound.arguments.get("expr") is not None:
        raise _refuse(path, reason)
