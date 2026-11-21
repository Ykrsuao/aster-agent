# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether a workspace's own files may configure and extend Aster.

A repository can carry files that Aster would otherwise act on as soon as it is
opened: Python skills and libraries that are imported, ``llm_config.yaml`` that
chooses where API keys are sent, ``secrets.yaml`` that sets environment
variables, and saved sessions whose context expressions are evaluated. Opening
a repository must not be enough to run its code, so the interactive hosts use
those files only for workspaces the user has trusted.

The check is off until a host calls :func:`enforce`. A script that imports
Aster as a library runs in the developer's own project and is unaffected.

Trusted workspaces are recorded in ``trusted_workspaces.json`` in the user
directory. Trusting a directory also trusts the directories inside it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep imports lightweight for the terminal launcher; the Windows write lock
# loads its dependency only when a decision is saved. See aster_cli._common.
DIR_NAME = ".aster"
_STORE_FILENAME = "trusted_workspaces.json"
_TRUST_ALL_ENV_VAR = "ASTER_TRUST_ALL_WORKSPACES"
_PROJECT_DIR_ENV_VAR = "ASTER_PROJECT_DIR"

# Paths inside a workspace that Aster imports, evaluates or takes settings from.
_WORKSPACE_CONTENT = (
    Path(DIR_NAME) / "settings.yaml",
    Path(DIR_NAME) / "config.toml",
    Path(DIR_NAME) / "llm_config.yaml",
    Path(DIR_NAME) / "secrets.yaml",
    Path(DIR_NAME) / "libs",
    Path(DIR_NAME) / "sessions",
    Path(".agents") / "skills",
    Path(".cursor") / "skills",
    Path(".claude") / "skills",
    Path(".claude") / "commands",
)

_enforced = False
_reported: set[str] = set()


def enforce() -> None:
    """Turn the check on for this process. Interactive hosts call this at startup."""
    global _enforced
    _enforced = True


def is_enforced() -> bool:
    return _enforced


@contextmanager
def enforced(value: bool = True) -> Iterator[None]:
    """Set the check for a block, restoring the previous setting afterwards."""
    global _enforced
    previous = _enforced
    _enforced = value
    try:
        yield
    finally:
        _enforced = previous


def _key(path: str | Path) -> str:
    """Identify a directory the way the file system does."""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _store_path() -> Path:
    # Same location as aster.paths.get_user_dir.
    override = os.environ.get("ASTER_USER_DIR")
    if override:
        return Path(override) / _STORE_FILENAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "aster" / _STORE_FILENAME


def trusted_workspaces() -> list[str]:
    """Return the recorded directories. An unreadable record trusts nothing."""
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return []
    entries = data.get("workspaces") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, str) and entry]


@contextmanager
def _write_lock() -> Iterator[None]:
    """Serialize the complete read-modify-write across Aster processes."""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    if os.name == "nt":
        from filelock import FileLock

        with FileLock(lock_path):
            yield
        return
    import fcntl

    # Keep the sidecar so waiters never lock different inodes after replacement.
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _write(entries: list[str]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps({"version": 1, "workspaces": entries}, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def trust(workspace: str | Path) -> Path:
    """Record *workspace* as trusted and return its resolved path.

    Raises OSError when the record cannot be saved.
    """
    resolved = Path(workspace).expanduser().resolve()
    with _write_lock():
        entries = trusted_workspaces()
        if _key(resolved) not in {_key(entry) for entry in entries}:
            _write([*entries, str(resolved)])
    return resolved


def untrust(workspace: str | Path) -> bool:
    """Remove *workspace* from the record. Return whether it was listed."""
    key = _key(workspace)
    with _write_lock():
        entries = trusted_workspaces()
        kept = [entry for entry in entries if _key(entry) != key]
        if len(kept) == len(entries):
            return False
        _write(kept)
    return True


def is_trusted(workspace: str | Path) -> bool:
    """True when *workspace* may configure and extend Aster.

    Always true while the check is off, and when ``ASTER_TRUST_ALL_WORKSPACES``
    is set to a true value for unattended runs.
    """
    if not _enforced:
        return True
    if os.environ.get(_TRUST_ALL_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        candidate = Path(_key(workspace))
    except (OSError, RuntimeError, ValueError):
        return False
    for entry in trusted_workspaces():
        try:
            trusted = Path(_key(entry))
        except (OSError, RuntimeError, ValueError):
            continue
        if candidate == trusted or trusted in candidate.parents:
            return True
    return False


def allows_project_dir(project_dir: str | Path) -> bool:
    """True when files in a project-local directory (``<workspace>/.aster``) may be used.

    A directory named by ``ASTER_PROJECT_DIR`` was chosen by whoever started the
    process, not by the repository, and is always allowed.
    """
    if not _enforced:
        return True
    directory = Path(project_dir)
    override = os.environ.get(_PROJECT_DIR_ENV_VAR)
    try:
        if override and _key(Path(override).resolve()) == _key(directory.resolve()):
            return True
    except (OSError, RuntimeError, ValueError):
        return False
    workspace = directory.parent if directory.name == DIR_NAME else directory
    allowed = is_trusted(workspace)
    if not allowed:
        _report(workspace)
    return allowed


def _report(workspace: Path) -> None:
    key = str(workspace)
    if key in _reported:
        return
    _reported.add(key)
    logger.warning(
        "Ignoring project files in %s because this folder is not trusted. "
        "Run `aster trust` in it to use them.",
        workspace,
    )


def ignored_content(workspace: str | Path) -> list[Path]:
    """Return what an untrusted *workspace* contains that Aster will not use.

    Empty for a trusted workspace, and for one with nothing to ignore, so hosts
    can stay silent when the question would change nothing.
    """
    root = Path(workspace).expanduser()
    if is_trusted(root):
        return []
    found = []
    for relative in _WORKSPACE_CONTENT:
        path = root / relative
        try:
            present = any(path.iterdir()) if path.is_dir() else path.is_file()
        except OSError:
            present = False
        if present:
            found.append(relative)
    return found


def restricted_notice(workspace: str | Path) -> str | None:
    """Describe what is being ignored in an untrusted *workspace*, if anything."""
    ignored = ignored_content(workspace)
    if not ignored:
        return None
    names = ", ".join(path.as_posix() for path in ignored)
    return (
        f"This folder is not trusted, so its own Aster files are not used: {names}. "
        "Skills, libraries, model settings, secrets and saved sessions from the folder "
        "are ignored. Run `aster trust` in the folder to use them."
    )


__all__ = [
    "allows_project_dir",
    "enforce",
    "enforced",
    "ignored_content",
    "is_enforced",
    "is_trusted",
    "restricted_notice",
    "trust",
    "trusted_workspaces",
    "untrust",
]
