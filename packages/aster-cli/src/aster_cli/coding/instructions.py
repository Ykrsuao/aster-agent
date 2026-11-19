# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository instruction discovery for interactive coding agents."""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_agent_instruction_files(working_directory: str | Path) -> tuple[Path, ...]:
    """Return applicable ``AGENTS.md`` files from repository root to cwd.

    A file in a deeper directory is appended after its parent instruction file,
    so the resulting context naturally gives the most local instructions the
    final word. Discovery stops at the nearest Git worktree root. Outside a Git
    worktree, only the working directory itself is considered.
    """
    cwd = _without_extended_prefix(str(Path(working_directory).resolve()))
    root = _git_root(cwd)
    directories = [cwd]
    if root is not None:
        distance = len(cwd.relative_to(root).parts)
        directories = list(reversed((cwd, *cwd.parents[:distance])))

    files = []
    for directory in directories:
        path = directory / "AGENTS.md"
        try:
            if path.is_file() and _without_extended_prefix(
                str(path.resolve(strict=True))
            ).is_relative_to(root or cwd):
                files.append(path)
        except (OSError, RuntimeError):
            continue
    return tuple(files)


#: Per-file and combined caps on repository instructions. The content is
#: workspace-controlled and lands in a prefix context block on every turn, so an
#: unbounded read is a memory and context-window risk at session setup.
_MAX_INSTRUCTION_FILE_CHARS = 100_000
_MAX_INSTRUCTION_TOTAL_CHARS = 200_000
_SECTION_SEPARATOR = "\n\n---\n\n"


def render_agent_instructions(working_directory: str | Path) -> str:
    """Render applicable repository instructions as one bounded context block.

    Reads are bounded rather than truncated after the fact: the content is
    workspace-controlled and lands in a prefix context block on every turn, so
    reading a huge file in full before discarding most of it would still cost
    the memory. The budget covers the rendered text — headers, separators and
    truncation markers included — not just the retained file content.

    Only ordinary, singly linked files within the repository (or cwd outside
    Git) are loaded. Internal symbolic links are allowed; external symbolic
    links and files with multiple hardlinks are skipped. The opened handle is
    checked before reading so a path replacement cannot redirect the read.
    """
    cwd = _without_extended_prefix(str(Path(working_directory).resolve()))
    root = _git_root(cwd) or cwd
    sections: list[str] = []
    remaining = _MAX_INSTRUCTION_TOTAL_CHARS
    for path in discover_agent_instruction_files(working_directory):
        if remaining <= 0:
            logger.warning("Skipping repository instructions from %s: total limit reached", path)
            continue
        budget = min(_MAX_INSTRUCTION_FILE_CHARS, remaining)
        try:
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(path, flags), "r", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    continue
                if not _opened_file_path(stream.fileno()).is_relative_to(root):
                    continue
                # One char past the budget is enough to know it was cut.
                content = stream.read(budget + 1)
        except (OSError, UnicodeError):
            continue
        truncated = len(content) > budget
        if truncated:
            content = content[:budget]
            logger.warning("Truncating repository instructions from %s at %d chars", path, budget)
        content = content.strip()
        if not content:
            continue
        if truncated:
            content += "\n\n[... truncated ...]"
        section = f"Instructions from {path}:\n\n{content}"
        remaining -= len(section) + len(_SECTION_SEPARATOR)
        sections.append(section)
    return _SECTION_SEPARATOR.join(sections)


def _opened_file_path(fd: int) -> Path:
    """Resolve the already-open file, never the potentially replaced pathname."""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        get_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
        get_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
        get_path.restype = wintypes.DWORD
        handle = msvcrt.get_osfhandle(fd)
        size = get_path(handle, None, 0, 0)
        if not size:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_unicode_buffer(size + 1)
        written = get_path(handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise OSError("Cannot resolve opened repository instruction file")
        return _without_extended_prefix(buffer.value)
    if sys.platform == "linux":
        path = os.readlink(f"/proc/self/fd/{fd}")
        if path.endswith(" (deleted)"):
            raise OSError("Repository instruction file was removed while opening")
        return Path(path)
    raise OSError("Safe repository instruction reads are unavailable on this platform")


def _without_extended_prefix(path: str) -> Path:
    if path.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[8:]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return Path(path)


def _git_root(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return None
