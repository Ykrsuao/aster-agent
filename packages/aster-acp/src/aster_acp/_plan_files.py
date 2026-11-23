# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch literal Plan file requests without evaluating their Python syntax."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aster_cli.tools.workspace_files import WorkspaceFiles

from aster.events import ExecutionResult

_METHODS = {
    "read": (WorkspaceFiles.read, ("path", "start_line", "end_line")),
    "list": (WorkspaceFiles.list, ("path", "limit")),
    "search": (WorkspaceFiles.search, ("query", "path", "glob", "limit")),
}


@dataclass(frozen=True)
class _Request:
    method: str
    arguments: tuple[tuple[str, Any], ...]


def _parse(code: str) -> _Request | None:
    if not isinstance(code, str) or len(code) > 8192:
        return None
    try:
        tree = ast.parse(code)
        if any(index >= 128 for index, _ in enumerate(ast.walk(tree))):
            return None
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    expression = tree.body[0].value
    if not isinstance(expression, ast.Await) or not isinstance(expression.value, ast.Call):
        return None
    call = expression.value
    target = call.func
    if not (
        isinstance(target, ast.Attribute)
        and target.attr in _METHODS
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "files"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "self"
    ):
        return None
    names = _METHODS[target.attr][1]
    if len(call.args) > len(names):
        return None
    values = dict(zip(names, call.args, strict=False))
    for keyword in call.keywords:
        if keyword.arg not in names or keyword.arg in values:
            return None
        values[keyword.arg] = keyword.value
    if any(not isinstance(value, ast.Constant) for value in values.values()):
        return None
    return _Request(target.attr, tuple(sorted((name, node.value) for name, node in values.items())))


@dataclass(frozen=True)
class PlanFileExecutor:
    _request: _Request
    _files: WorkspaceFiles
    _arguments: dict[str, Any]
    _allowed: Callable[[], bool]
    _timeout: float | None

    async def run_cell(self, code: str, *, execution_count: int = 1) -> ExecutionResult:
        return await self.run_cell_with_timeout(
            code, timeout=self._timeout, execution_count=execution_count
        )

    async def run_cell_with_timeout(
        self, code: str, *, timeout: float | None, execution_count: int = 1
    ) -> ExecutionResult:
        if _parse(code) != self._request or not self._allowed():
            return ExecutionResult(
                error=PermissionError(
                    "The Plan request or permission changed. The code was not run."
                )
            )
        try:
            method = _METHODS[self._request.method][0]
            value = await asyncio.wait_for(method(self._files, **self._arguments), timeout=timeout)
            return ExecutionResult(returned_value=value)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return ExecutionResult(error=error)


def plan_file_executor(
    code: str,
    *,
    files: WorkspaceFiles,
    allowed: Callable[[], bool],
    timeout: float | None,
    host_path: Callable[[str], str] | None = None,
) -> PlanFileExecutor | None:
    request = _parse(code)
    if request is None:
        return None
    arguments = dict(request.arguments)
    if host_path is not None:
        path = arguments.get("path", ".")
        if not isinstance(path, str):
            return None
        try:
            arguments["path"] = host_path(path)
        except (OSError, ValueError):
            return None
    return PlanFileExecutor(request, files, arguments, allowed, timeout)
