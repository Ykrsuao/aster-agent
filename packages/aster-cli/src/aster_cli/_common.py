# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for aster CLI commands.

Import from here instead of duplicating helpers across command modules:

    from aster_cli._common import find_project_root, format_size
"""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the user's project root by walking up from *start* or the current directory.

    Walk upward, returning the first ancestor (including the starting directory)
    that contains a ``pyproject.toml``. Fall back to the starting directory if
    none is found. With no argument, start at :func:`Path.cwd`.

    Walking up from the cwd (rather than from this module's ``__file__``) ensures
    the *user's* project is resolved. Resolving from ``__file__`` would instead
    find the installed CLI package's own ``pyproject.toml`` in an editable or
    monorepo install, which is never what a user of ``aster`` wants.

    Uses a local implementation to keep CLI startup fast (avoids importing
    the heavy aster core package at module level).
    """
    cwd = (start if start is not None else Path.cwd()).resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cwd


def format_size(size_bytes: int) -> str:
    """Format a byte count as human-readable (e.g. '4.2 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def load_workspace_trust():
    """Return ``aster.workspace_trust`` without importing the framework.

    The terminal launcher draws before the framework is imported. The module
    is self-contained, so it is loaded from its file when the framework is not
    already present. The launcher and the agent are separate processes, so the
    copy's settings never need to match another one.
    """
    import importlib.util
    import sys

    loaded = sys.modules.get("aster.workspace_trust") or sys.modules.get("_aster_workspace_trust")
    if loaded is not None:
        return loaded
    package = importlib.util.find_spec("aster")
    if package is None or not package.submodule_search_locations:
        raise ImportError("aster is not installed")
    path = Path(next(iter(package.submodule_search_locations))) / "workspace_trust.py"
    spec = importlib.util.spec_from_file_location("_aster_workspace_trust", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module
