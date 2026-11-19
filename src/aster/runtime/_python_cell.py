# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Python execution-cell source semantics for the host and sandbox.

Compilation and execution stay with each caller: ``compile`` inherits the
caller's future flags, and each backend owns its namespace and error handling.
"""

import ast
import linecache
import symtable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aster.events import _NO_RETURN


@dataclass(frozen=True)
class CellSource:
    """Prepared source and metadata, without compiled or executed user code."""

    source: str
    filename: str
    tree: ast.Module
    helper_ast: ast.Module | None
    explicit_return: bool
    implicit_return_added: bool
    wrapper_line_offset: int
    _original_source: str
    _method_sources: dict[str, str]
    execution_tree: ast.Module

    def execution_source(self) -> ast.Module:
        """Register the source for execution tracebacks and return it to compile."""
        self._register_source(self._original_source)
        return self.execution_tree

    def finish_helpers(self, namespace: dict[str, Any]) -> dict[str, Any]:
        """Finish source registration after the caller has bound the helpers.

        Definitions execute once, in source order, with their original locations.
        """
        for name, source in self._method_sources.items():
            fn = namespace.get(name)
            if callable(fn):
                try:
                    fn._generated_source = source  # pyright: ignore[reportFunctionMemberAccess]
                except (AttributeError, TypeError):
                    # Built-in or decorated callables may reject attributes.
                    pass
        return {
            name: namespace[name] for name in self._method_sources if callable(namespace.get(name))
        }

    def returned_value(self, value: Any) -> Any:
        """Explicit returns include None; implicit returns suppress None."""
        if self.explicit_return or (self.implicit_return_added and value is not None):
            return value
        return _NO_RETURN

    def _register_source(self, source: str) -> None:
        linecache.cache[self.filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            self.filename,
        )


def prepare_cell(
    source: str,
    filename: str,
    *,
    variables: Mapping[str, Any],
    wrap_in_function: bool,
) -> CellSource:
    """Prepare a cell while preserving its source lines and helper definitions.

    ``variables`` is the caller's persisted state: host strategy builtins or the
    sandbox namespace. Public bindings and top-level helpers become wrapper globals
    so helpers resolve current session state rather than closed-over cell snapshots.
    """
    tree = ast.parse(source, filename=filename)
    method_sources: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.args.args and node.args.args[0].arg == "self":
                method_source = ast.get_source_segment(source, node)
                if method_source:
                    method_sources[node.name] = method_source

    finder = _TopLevelReturnFinder()
    for node in tree.body:
        finder.visit(node)
        if finder.found:
            break
    explicit_return = finder.found
    original_source = source
    implicit_return_added = False
    if wrap_in_function and tree.body and not explicit_return:
        last_stmt = tree.body[-1]
        if isinstance(last_stmt, ast.Expr):
            # Insert into the original text: unparsing loses comments and line
            # numbers. AST columns are UTF-8 byte offsets, including after ';'.
            lines = source.split("\n")
            # Python also counts bare carriage returns as newlines. Preserve the
            # existing fallback when its line number is outside this source view.
            if 1 <= last_stmt.lineno <= len(lines):
                line = lines[last_stmt.lineno - 1].encode("utf-8")
                col = last_stmt.col_offset
                lines[last_stmt.lineno - 1] = (line[:col] + b"return " + line[col:]).decode("utf-8")
                source = "\n".join(lines)
                implicit_return_added = True

    helper_defs: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    helper_names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and not node.name.startswith("__")
    }
    helper_ast = ast.Module(body=helper_defs, type_ignores=[]) if helper_defs else None
    execution_tree = ast.parse(original_source, filename=filename)
    if implicit_return_added:
        last_expression = execution_tree.body[-1]
        assert isinstance(last_expression, ast.Expr)
        execution_tree.body[-1] = ast.copy_location(
            ast.Return(value=last_expression.value), last_expression
        )
    wrapper_line_offset = 0
    if wrap_in_function:
        symbols = symtable.symtable(source, filename, "exec").get_symbols()
        global_vars = sorted(
            name
            for name in set(variables)
            | {
                symbol.get_name()
                for symbol in symbols
                if symbol.is_assigned() or symbol.is_imported() or symbol.is_declared_global()
            }
            if (not name.startswith("_") or name in helper_names)
            and name not in ("self", "asyncio")
        )
        # An AST wrapper preserves source positions and string contents without
        # evaluating definitions again to repair their line numbers.
        wrapper = ast.parse(f"""async def __repl_wrapper__():
    try:
        pass
    finally:
        for _gvar in {tuple(global_vars)!r}:
            if _gvar in globals():
                __repl_captured_locals__[_gvar] = globals()[_gvar]
""")
        function = wrapper.body[0]
        assert isinstance(function, ast.AsyncFunctionDef)
        guarded = function.body[0]
        assert isinstance(guarded, ast.Try)
        guarded.body = _CellAnnotations().visit(execution_tree).body or [ast.Pass()]
        if global_vars:
            function.body.insert(0, ast.Global(names=global_vars))
        execution_tree = ast.fix_missing_locations(wrapper)

    return CellSource(
        source=source,
        filename=filename,
        tree=tree,
        helper_ast=helper_ast,
        explicit_return=explicit_return,
        implicit_return_added=implicit_return_added,
        wrapper_line_offset=wrapper_line_offset,
        _original_source=original_source,
        _method_sources=method_sources,
        execution_tree=execution_tree,
    )


class _TopLevelReturnFinder(ast.NodeVisitor):
    """Find returns in cell control flow, excluding nested functions/classes."""

    def __init__(self) -> None:
        self.found = False

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass


class _CellAnnotations(ast.NodeTransformer):
    """Global cell assignments cannot carry local-variable annotations."""

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
            if node.value is None:
                return ast.copy_location(ast.Pass(), node)
            return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        return node

    def visit_Return(self, node: ast.Return) -> ast.Return:
        return node
