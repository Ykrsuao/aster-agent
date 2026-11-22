# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, read-only workspace inspection without a shell or generated code."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

_FILE_BYTES = 2 * 1024 * 1024
_SEARCH_BYTES = 16 * 1024 * 1024
_SEARCH_ENTRIES = 2000
_OUTPUT_CHARS = 64_000
_SKIP_DIRECTORIES = frozenset(
    {".git", ".venv", "venv", "node_modules", "target", "dist", "build", "__pycache__"}
)


def _integer(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return value


def _windows_open(path: Path) -> int:
    """Pin the named object against replacement, including directory junctions."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    # READ, share READ/WRITE (not DELETE), OPEN_EXISTING, open directories and
    # reparse points themselves. Every component is checked before proceeding.
    handle = create(str(path), 0x80000000, 3, None, 3, 0x02200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise


class WorkspaceFiles:
    """Read, list and search UTF-8 project files within a fixed workspace.

    Links, reparse points, multiply linked files and special files are refused.
    Results and work are bounded; a truncated search is not proof of absence.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        try:
            info = self._root.stat()
            self._identity = (info.st_dev, info.st_ino)
        except OSError:
            self._identity = None

    def _relative(self, path: str) -> Path:
        if not isinstance(path, str) or not path or len(path) > 4096 or "\0" in path:
            raise ValueError("Use a nonempty workspace path of at most 4096 characters.")
        requested = Path(path)
        try:
            relative = requested.relative_to(self._root) if requested.is_absolute() else requested
        except ValueError:
            raise PermissionError("The path is outside this workspace.") from None
        if requested.drive and not requested.is_absolute():
            raise PermissionError("Drive-relative paths are not supported.")
        if len(relative.parts) > 64 or any(
            part == ".."
            or ":" in part
            or "\\" in part
            or part != part.rstrip(". ")
            or part.split(".", 1)[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }
            for part in relative.parts
        ):
            raise PermissionError("Use an ordinary path inside this workspace, without '..'.")
        return relative

    @contextmanager
    def _open(self, relative: Path, *, directory: bool = False) -> Iterator[int]:
        """Open each component without following links; retain all parent handles."""
        from aster_cli.coding.instructions import _opened_file_path

        relative = self._relative(relative.as_posix())
        if sys.platform not in {"linux", "win32"}:
            raise OSError("Workspace inspection is available on Linux and Windows.")
        with ExitStack() as stack:
            parent_fd = None
            path = self._root
            parts = (None, *relative.parts)
            for index, part in enumerate(parts):
                if part is not None:
                    path = path / part
                is_directory = index < len(parts) - 1 or directory or part is None
                if sys.platform == "win32":
                    fd = _windows_open(path)
                else:
                    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
                    if is_directory:
                        flags |= os.O_DIRECTORY
                    fd = os.open(path if part is None else part, flags, dir_fd=parent_fd)
                stack.callback(os.close, fd)
                info = os.fstat(fd)
                valid_type = (
                    stat.S_ISDIR(info.st_mode) if is_directory else stat.S_ISREG(info.st_mode)
                )
                if (
                    not valid_type
                    or getattr(info, "st_file_attributes", 0) & 0x400
                    or (not is_directory and info.st_nlink != 1)
                    or _opened_file_path(fd) != path
                    or (part is None and (info.st_dev, info.st_ino) != self._identity)
                ):
                    raise PermissionError(
                        "The workspace path changed or refers to a link or special file."
                    )
                parent_fd = fd
            yield fd
            if _opened_file_path(fd) != path:
                raise PermissionError("The workspace path changed while it was being inspected.")

    def _bytes(self, relative: Path, budget: int) -> tuple[bytes, bool]:
        with self._open(relative) as fd:
            if os.fstat(fd).st_size > budget:
                return b"", False
            with os.fdopen(os.dup(fd), "rb") as stream:
                value = stream.read(budget + 1)
            return value[:budget], len(value) <= budget

    @contextmanager
    def _entries(self, relative: Path) -> Iterator[Iterator[os.DirEntry[str]]]:
        with self._open(relative, directory=True) as fd:
            target = self._root / relative if sys.platform == "win32" else fd
            with os.scandir(target) as entries:
                yield entries

    @staticmethod
    def _kind(entry: os.DirEntry) -> str:
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return "link"
        if stat.S_ISDIR(info.st_mode):
            return "directory"
        return "file" if stat.S_ISREG(info.st_mode) else "special"

    async def read(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> dict[str, Any]:
        """Read numbered UTF-8 text: default 200 lines, at most 500 lines/64K chars.

        Files larger than 2 MiB are refused. Paths are relative to the workspace;
        absolute paths inside it are also accepted. No links are followed.
        """
        start = _integer(start_line, "start_line", 10_000_000)
        end = start + 199 if end_line is None else _integer(end_line, "end_line", 10_000_000)
        if end < start or end - start >= 500:
            raise ValueError("Choose a range of 1 to 500 lines.")
        relative = self._relative(path)
        raw, complete = self._bytes(relative, _FILE_BYTES)
        if not complete:
            raise ValueError("Workspace reads are limited to files of at most 2 MiB.")
        if b"\0" in raw:
            raise ValueError("Workspace reads require UTF-8 text, not binary data.")
        lines = raw.decode("utf-8").splitlines(keepends=True)
        selected = "".join(lines[start - 1 : end])
        text = selected[:_OUTPUT_CHARS]
        return {
            "path": relative.as_posix(),
            "start_line": start,
            "end_line": start + len(text.splitlines()) - 1,
            "text": text,
            "truncated": end < len(lines) or len(selected) > len(text),
        }

    async def list(self, path: str = ".", limit: int = 200) -> dict[str, Any]:
        """List up to 500 directory entries; links are named but never traversed."""
        limit = _integer(limit, "limit", 500)
        relative = self._relative(path)
        found = []
        with self._entries(relative) as entries:
            for entry in entries:
                found.append({"name": entry.name, "kind": self._kind(entry)})
                if len(found) > limit:
                    break
        return {
            "path": relative.as_posix(),
            "entries": sorted(found[:limit], key=lambda entry: entry["name"]),
            "truncated": len(found) > limit,
        }

    async def search(
        self, query: str, path: str = ".", glob: str = "*", limit: int = 50
    ) -> dict[str, Any]:
        """Find literal, case-sensitive text in UTF-8 files below a directory.

        glob filters relative filenames. Skips links, binary/large files and
        common dependency/build directories. Limits: 200 matches, 2000 entries,
        16 MiB read, 64K output characters and five seconds per request.
        """
        limit = _integer(limit, "limit", 200)
        if (
            not isinstance(query, str)
            or not 1 <= len(query) <= 256
            or "\n" in query
            or "\r" in query
        ):
            raise ValueError("query must be 1 to 256 characters on one line (literal text).")
        if not isinstance(glob, str) or not 1 <= len(glob) <= 256:
            raise ValueError("glob must be a nonempty filename pattern of at most 256 characters.")
        root = self._relative(path)
        pending = [root]
        matches = []
        scanned = used = output_chars = skipped = 0
        truncated = False
        deadline = time.monotonic() + 5
        while pending and not truncated:
            directory = pending.pop()
            with ExitStack() as inspection:
                try:
                    entries = inspection.enter_context(self._entries(directory))
                except (OSError, ValueError):
                    if directory == root:
                        raise
                    skipped += 1
                    continue
                for entry in entries:
                    await asyncio.sleep(0)
                    if (
                        scanned >= _SEARCH_ENTRIES
                        or used >= _SEARCH_BYTES
                        or time.monotonic() >= deadline
                    ):
                        truncated = True
                        break
                    scanned += 1
                    try:
                        kind = self._kind(entry)
                        relative = directory / entry.name
                        if kind == "directory":
                            if entry.name not in _SKIP_DIRECTORIES:
                                pending.append(relative)
                            continue
                        if kind != "file":
                            skipped += 1
                            continue
                        if not fnmatch.fnmatchcase(relative.relative_to(root).as_posix(), glob):
                            continue
                        raw, complete = self._bytes(
                            relative, min(_FILE_BYTES, _SEARCH_BYTES - used)
                        )
                        used += len(raw)
                        if not complete or b"\0" in raw:
                            skipped += 1
                            continue
                        content = raw.decode("utf-8")
                    except (OSError, UnicodeError, ValueError):
                        skipped += 1
                        continue
                    for line_number, line in enumerate(content.splitlines(), 1):
                        position = line.find(query)
                        if position < 0:
                            continue
                        snippet = line[max(0, position - 100) : position + 400]
                        size = len(relative.as_posix()) + len(snippet) + 64
                        if len(matches) >= limit or output_chars + size > _OUTPUT_CHARS:
                            truncated = True
                            break
                        matches.append(
                            {"path": relative.as_posix(), "line": line_number, "text": snippet}
                        )
                        output_chars += size
                    if truncated:
                        break
        return {
            "matches": matches,
            "truncated": truncated,
            "skipped": skipped,
            "scanned_entries": scanned,
        }
