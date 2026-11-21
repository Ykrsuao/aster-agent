# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute literal file operations without evaluating their Python spelling.

This adapter owns its file tools. Neither a REPL namespace nor the live agent's
mutable shell supplies the implementation of an automatically accepted edit.
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import os
import stat
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from aster_cli.coding.activity import (
    _MAX_DIFF_INPUT_CHARS,
    _MAX_EVENT_TEXT_CHARS,
    FileEdit,
    _edit_diff,
    _line_count,
    _omitted_diff,
)
from aster_cli.coding.instructions import _opened_file_path

from aster.agentdoc import pformat
from aster.events import ExecutionResult
from aster.runtime.event_manager import EventManager
from aster.tools._bounded_io import FileTooLargeError, read_line_range, read_whole_file_checked
from aster.tools.shell_tools import FileWrite, Match

_MAX_SOURCE_BYTES = 128 * 1024
_MAX_AST_NODES = 4096
_MAX_PROTECTED_ENTRIES = 16_384
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Read:
    path: str
    lines: tuple[int, int] | None


@dataclass(frozen=True)
class _Write:
    path: str
    content: str


@dataclass(frozen=True)
class _Replace:
    path: str
    old: str
    new: str


_Operation = _Read | _Write | _Replace

# Directories whose contents are executed later (hooks, skills, libraries) or
# hold credentials and settings. Touching them always needs the user's approval.
_PROTECTED_DIRECTORIES = frozenset({".git", ".aster", ".agents", ".cursor", ".claude"})


def _needs_approval(target: Path, roots: Iterable[Path]) -> bool:
    """True when *target* is outside every root or enters a protected directory.

    Check both the requested spelling and the opened path so resolving a link
    cannot erase a protected component from the requested spelling.
    """
    inside = False
    for root in roots:
        try:
            relative = target.relative_to(root)
        except ValueError:
            continue
        inside = True
        for part in relative.parts:
            # Windows ignores trailing dots and spaces and reads "name:stream" as
            # "name", so those spellings could address a protected directory.
            if (part != ".." and part != part.rstrip(". ")) or ":" in part:
                return True
            if part.lower() in _PROTECTED_DIRECTORIES:
                return True
    return not inside


def _protected_target(target: Path, roots: Iterable[Path]) -> bool:
    """Protect host configuration trees, including links inside linked libraries.

    A bounded scan costs no more as repositories grow: uncertain paths fall back
    to ordinary approval. Recheck on use because links can change after preparation.
    """
    pending = [(root / name, True) for root in roots for name in _PROTECTED_DIRECTORIES]
    seen: set[Path] = set()
    remaining = _MAX_PROTECTED_ENTRIES
    try:
        while pending:
            path, linked = pending.pop()
            resolved = path.resolve() if linked else path
            if linked and target.is_relative_to(resolved):
                return True
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                with os.scandir(resolved) as entries:
                    for entry in entries:
                        remaining -= 1
                        if remaining < 0:
                            return True
                        if entry.is_symlink() or entry.is_junction():
                            pending.append((Path(entry.path), True))
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append((Path(entry.path), False))
            except (FileNotFoundError, NotADirectoryError):
                pass
    except (OSError, ValueError, RuntimeError):
        return True
    return False


def _arguments(call: ast.Call, names: tuple[str, ...]) -> dict[str, ast.expr] | None:
    if len(call.args) > len(names):
        return None
    values = dict(zip(names, call.args, strict=False))
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in names or keyword.arg in values:
            return None
        values[keyword.arg] = keyword.value
    return values


def _string(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and type(node.value) is str else None


def _integer(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def _parse_operation(code: str) -> _Operation | None:
    if not isinstance(code, str) or len(code) > _MAX_SOURCE_BYTES:
        return None
    try:
        if len(code.encode("utf-8")) > _MAX_SOURCE_BYTES:
            return None
        tree = ast.parse(code)
        if any(index >= _MAX_AST_NODES for index, _ in enumerate(ast.walk(tree))):
            return None
    except (SyntaxError, ValueError, UnicodeError, RecursionError, MemoryError):
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
        and isinstance(target.value, ast.Attribute)
        and target.value.attr == "shell"
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "self"
    ):
        return None
    names = {
        "read": ("path", "lines"),
        "write_file": ("path", "content"),
        "replace": ("target", "old_or_new", "new"),
    }.get(target.attr)
    if names is None or (values := _arguments(call, names)) is None:
        return None
    path = _string(values.get(names[0]))
    if path is None:
        return None
    if target.attr == "read":
        lines = values.get("lines")
        if lines is None or isinstance(lines, ast.Constant) and lines.value is None:
            return _Read(path, None)
        if isinstance(lines, ast.Tuple) and len(lines.elts) == 2:
            start, end = (_integer(value) for value in lines.elts)
            if start is not None and end is not None:
                return _Read(path, (start, end))
        return None
    if target.attr == "write_file":
        content = _string(values.get("content"))
        return _Write(path, content) if content is not None else None
    old, new = _string(values.get("old_or_new")), _string(values.get("new"))
    return _Replace(path, old, new) if old is not None and new is not None else None


@dataclass(frozen=True)
class FileEditExecutor:
    """A single approved operation implementing the runtime's executor interface."""

    _operation: _Operation
    _cwd: str
    _event_manager: EventManager
    _max_file_bytes: int
    _timeout: float | None
    _result_path: str
    _host_path: Path
    _roots: tuple[Path, ...]
    _identity: tuple[int, int] | None
    _parent_identity: tuple[int, int]

    async def run_cell(self, code: str, *, execution_count: int = 1) -> ExecutionResult:
        """Run a standalone request with the timeout selected during preparation."""
        return await self.run_cell_with_timeout(
            code, execution_count=execution_count, timeout=self._timeout
        )

    async def run_cell_with_timeout(
        self, code: str, *, timeout: float | None, execution_count: int = 1
    ) -> ExecutionResult:
        """Recheck the final source after middleware, then run only the stored operation."""
        if _parse_operation(code) != self._operation:
            return ExecutionResult(
                error=PermissionError(
                    "The file operation changed after approval. The modified code was not run."
                )
            )
        try:
            value = await asyncio.wait_for(self._run_operation(), timeout=timeout)
            return ExecutionResult(returned_value=value)
        except asyncio.CancelledError:
            raise
        except FileTooLargeError as error:
            # The bounded reader owns a duplicate fd; display the user's path.
            description = str(error)[len(error.path) :]
            error.path = self._result_path
            error.args = (self._result_path + description,)
            return ExecutionResult(error=error)
        except Exception as error:
            return ExecutionResult(error=error)

    async def _run_operation(self) -> Match | FileWrite:
        operation = self._operation
        path = self._host_path
        creating = isinstance(operation, _Write) and self._identity is None
        flags = os.O_RDONLY if isinstance(operation, _Read) else os.O_RDWR
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
        if creating:
            flags |= os.O_CREAT | os.O_EXCL
        # Never truncate while opening: the handle must pass the boundary check first.
        try:
            fd = os.open(path, flags, 0o666)
        except FileExistsError:
            raise PermissionError(
                "The file appeared after preparation; request approval again."
            ) from None
        validated = False
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            try:
                opened_path = _opened_file_path(fd)
                named = path.stat()
                parent = path.parent.stat()
                allowed = (
                    stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and (self._identity is None or identity == self._identity)
                    and identity == (named.st_dev, named.st_ino)
                    and self._parent_identity == (parent.st_dev, parent.st_ino)
                    and opened_path == path
                    and not _needs_approval(opened_path, self._roots)
                    and not _protected_target(opened_path, self._roots)
                )
            except (OSError, ValueError):
                allowed = False
            if not allowed:
                reason = "The file changed or is outside the approved boundary."
                if creating:
                    reason += " No content was written; an empty file may remain."
                raise PermissionError(reason)
            validated = True
            # The bounded readers own their handle. A duplicate keeps every read on
            # this verified file even if its pathname is replaced afterwards.
            if isinstance(operation, _Read):
                if operation.lines is None:
                    content = read_whole_file_checked(
                        os.dup(fd),
                        self._max_file_bytes,
                        hint=(
                            "Read a region instead: read(path, lines=(start, end)). To allow "
                            "whole-file reads this large, construct "
                            "ShellTools(max_file_bytes=...)."
                        ),
                    )
                    start, end = 1, len(content.splitlines(keepends=True))
                else:
                    start, end = operation.lines
                    start = max(1, start)
                    content, end = read_line_range(
                        os.dup(fd), start, end, max_line_chars=self._max_file_bytes
                    )
                return Match(self._result_path, start, end, content, resolved_path=path)
            old_text = None
            start = end = None
            if isinstance(operation, _Replace):
                content = read_whole_file_checked(
                    os.dup(fd),
                    self._max_file_bytes,
                    hint=(
                        "Replacing text requires rebuilding the file. To edit a file this "
                        "large, construct ShellTools(max_file_bytes=...)."
                    ),
                )
                count = content.count(operation.old)
                if count == 0:
                    raise ValueError(
                        f"old text not found in {self._result_path}. "
                        "It must match exactly once — check whitespace and indentation."
                    )
                if count > 1:
                    raise ValueError(
                        f"old text matched {count} times in {self._result_path}. "
                        "It must match exactly once — add surrounding context to make it unique."
                    )
                old_text, new_text = operation.old, operation.new
                if len(content) <= _MAX_EVENT_TEXT_CHARS:
                    start = content.count("\n", 0, content.index(old_text)) + 1
                    end = start + max(1, _line_count(old_text)) - 1
                written = content.replace(old_text, new_text, 1)
            else:
                new_text = written = operation.content
                if not creating:
                    try:
                        with open(os.dup(fd)) as preview:
                            old_text = preview.read(_MAX_DIFF_INPUT_CHARS + 1)
                    except (OSError, UnicodeError):
                        pass
                    if old_text and len(old_text) <= _MAX_EVENT_TEXT_CHARS:
                        start, end = 1, _line_count(old_text)
            # Writing through the checked descriptor avoids reopening a replaced path.
            with open(os.dup(fd), "r+b") as raw, io.TextIOWrapper(raw) as writer:
                writer.seek(0)
                writer.write(written)
                writer.truncate()
            diff_path = (
                path.relative_to(self._cwd).as_posix()
                if path.is_relative_to(self._cwd)
                else path.as_posix().lstrip("/")
            )
            diff, diff_complete = (
                _omitted_diff(diff_path, "previous file content could not be read")
                if not creating and old_text is None
                else _edit_diff(
                    diff_path,
                    old_text or "",
                    new_text,
                    start,
                    whole_file=isinstance(operation, _Write),
                )
            )
            event = FileEdit(
                path=str(path),
                operation="create" if creating else "update",
                old_text=pformat(old_text, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True)
                if old_text is not None
                else None,
                new_text=pformat(new_text, max_string=_MAX_EVENT_TEXT_CHARS, unquote_strings=True),
                start_line=start,
                end_line=end,
                diff=diff,
                content_complete=(
                    creating or old_text is not None and len(old_text) <= _MAX_EVENT_TEXT_CHARS
                )
                and len(new_text) <= _MAX_EVENT_TEXT_CHARS,
                diff_complete=diff_complete,
            )
            try:
                self._event_manager.add(event)
            except Exception:
                logger.debug("Failed to emit shell activity", exc_info=True)
            if isinstance(operation, _Write):
                line_count = written.count("\n") + (
                    1 if written and not written.endswith("\n") else 0
                )
                return FileWrite(
                    path=self._result_path,
                    message=f"Created {self._result_path} ({line_count} lines)",
                    new_text=new_text,
                )
            return FileWrite(
                path=self._result_path,
                message=f"Edited {self._result_path}",
                diff=f"--- a/{self._result_path}\n+++ b/{self._result_path}",
                new_text=new_text,
            )
        finally:
            os.close(fd)
            if creating and not validated:
                # shortcut: retain a refused empty file; pathname cleanup can delete
                # a replacement. Upgrade with handle-relative creation/cleanup.
                logger.warning("File creation was refused before writing content: %s", path)


def file_edit_executor(
    code: str,
    *,
    cwd: str | Path,
    event_manager: EventManager,
    max_file_bytes: int,
    timeout: float | None,
    host_path: Callable[[str], str] | None = None,
    workspace: str | Path | None = None,
    extra_roots: Iterable[tuple[str | Path, bool]] = (),
) -> FileEditExecutor | None:
    """Prepare one literal read/write/replace; all other Python needs normal approval.

    No source is evaluated, and no file is opened until execution. The
    working directory is captured now so later shell navigation cannot retarget
    the approved operation. A sandbox maps only the target path, leaving file
    content and the source used for the final approval check untouched. The
    runtime may override the cooperative timeout after execution middleware.

    Only targets inside *workspace* (the working directory when omitted) and
    outside its protected directories are prepared. *extra_roots* lists further
    directories the operator granted as ``(path, writable)``; a read-only one
    admits reads only. Anything else returns ``None``, so the request goes
    through normal approval. Existing files must have one hard link; creating
    a missing parent directory also requires normal approval.
    """
    operation = _parse_operation(code)
    if operation is None or sys.platform not in {"win32", "linux"}:
        return None
    directory = Path(cwd).resolve()
    target = host_path(operation.path) if host_path is not None else operation.path
    roots = [Path(workspace or directory).resolve()]
    roots += [
        Path(root).resolve()
        for root, writable in extra_roots
        if writable or isinstance(operation, _Read)
    ]
    requested = directory / target
    if _needs_approval(requested, roots):
        return None
    try:
        resolved = requested.resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if _needs_approval(resolved, roots) or _protected_target(resolved, roots):
        return None
    try:
        parent = resolved.parent.stat()
        try:
            info = resolved.stat()
        except FileNotFoundError:
            info = None
        if info is not None and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            return None
        if info is not None and isinstance(operation, _Write) and not os.access(resolved, os.R_OK):
            return None
    except OSError:
        # Creating directories safely needs platform-specific traversal; ask instead.
        return None
    return FileEditExecutor(
        operation,
        str(directory),
        event_manager,
        max_file_bytes,
        timeout,
        target,
        resolved,
        tuple(roots),
        (info.st_dev, info.st_ino) if info is not None else None,
        (parent.st_dev, parent.st_ino),
    )
