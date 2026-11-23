# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-aware settings shared by interactive coding-agent hosts."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aster import workspace_trust
from aster.layered_config import load_layered_yaml

if TYPE_CHECKING:
    from aster.task_budget import TaskBudgetLimits

_SETTINGS_FILENAME = "settings.yaml"
_SETTINGS_ENV_VAR = "ASTER_SETTINGS"
_LEGACY_CONFIG_FILENAME = "config.toml"
_WORKSPACE_SKILL_DIRS = (
    Path(".agents/skills"),
    Path(".cursor/skills"),
    Path(".claude/skills"),
    Path(".claude/commands"),
)
_USER_SKILL_DIRS = (
    Path(".agents/skills"),
    Path(".claude/skills"),
    Path(".claude/commands"),
)

logger = logging.getLogger(__name__)


def load_coding_skills_dirs(
    workspace: str | Path,
    *,
    explicit: Iterable[str | Path] = (),
) -> list[Path]:
    """Return existing skill roots for one coding-agent workspace.

    The shared ``coding.additional_skills_dirs`` setting is preferred for new
    configuration. ``tui.additional_skills_dirs`` remains supported while the
    TUI migrates to the shared section. Relative configured paths are resolved
    against the active workspace, not the ACP server process's checkout.
    """
    root = Path(workspace).expanduser().resolve()
    settings = load_layered_yaml(
        _SETTINGS_FILENAME,
        _SETTINGS_ENV_VAR,
        project_dir=root / ".aster",
    )

    configured: list[str | Path] = []
    configured.extend(_setting_paths(settings, "coding"))
    configured.extend(_setting_paths(settings, "tui"))
    # A workspace's old config.toml is still a workspace layer. Do not let an
    # unrelated user-level settings.yaml silently suppress it. A modern
    # workspace settings file supersedes the legacy file, and an explicit
    # ASTER_SETTINGS file remains a full override.
    # An untrusted workspace contributes no settings and no skill directories:
    # Python skills found in a directory are imported.
    trusted = workspace_trust.is_trusted(root)
    project_settings = (
        _read_project_settings(root / ".aster" / _SETTINGS_FILENAME) if trusted else {}
    )
    # Presence of the key, not truthiness of its value: `additional_skills_dirs: []`
    # is an explicit "none", and treating it as absent silently resurrected the
    # legacy paths the user had just emptied.
    modern_key_set = any(
        isinstance(project_settings.get(section), Mapping)
        and "additional_skills_dirs" in project_settings[section]
        for section in ("coding", "tui")
    )
    if trusted and not os.environ.get(_SETTINGS_ENV_VAR) and not modern_key_set:
        configured.extend(_legacy_project_paths(root / ".aster" / _LEGACY_CONFIG_FILENAME))

    candidates: list[Path] = []
    candidates.extend(_resolve_configured(path, root) for path in explicit)
    candidates.extend(_resolve_configured(path, root) for path in configured)
    candidates.extend(root / path for path in _WORKSPACE_SKILL_DIRS)
    candidates.extend(Path.home() / path for path in _USER_SKILL_DIRS)

    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        # Check the source before links can resolve outside an untrusted folder.
        if not trusted and candidate.is_relative_to(root):
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        if not trusted and (resolved == root or root in resolved.parents):
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def load_required_checks(workspace: str | Path) -> dict[str, Any]:
    """Load coding checks through existing settings layers; reject invalid declarations."""
    from aster_cli.coding.verification import RequiredCheck

    root = Path(workspace).expanduser().resolve()
    settings = load_layered_yaml(_SETTINGS_FILENAME, _SETTINGS_ENV_VAR, project_dir=root / ".aster")
    coding = settings.get("coding")
    if coding is None:
        return {}
    if not isinstance(coding, Mapping):
        raise ValueError("coding settings must be a mapping")
    checks = coding.get("required_checks", {})
    if not isinstance(checks, Mapping):
        raise ValueError("coding.required_checks must be a mapping")
    result = {}
    for name, value in checks.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("required check names must be nonblank strings")
        result[name] = RequiredCheck.model_validate(value)
    return result


def load_task_budget(workspace: str | Path) -> TaskBudgetLimits | None:
    """Read limits for a new task; existing tasks retain their saved declaration."""
    from aster.task_budget import TaskBudgetLimits

    root = Path(workspace).expanduser().resolve()
    settings = load_layered_yaml(_SETTINGS_FILENAME, _SETTINGS_ENV_VAR, project_dir=root / ".aster")
    coding = settings.get("coding")
    if coding is None:
        return None
    if not isinstance(coding, Mapping):
        raise ValueError("coding settings must be a mapping")
    value = coding.get("task_budget")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("coding.task_budget must be a mapping")
    return TaskBudgetLimits.model_validate(dict(value))


def _project_memory_document(workspace: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read only the explicit settings belonging to this trusted workspace."""
    import yaml

    root = Path(workspace).expanduser().resolve()
    if not workspace_trust.is_trusted(root):
        raise ValueError("Project memory requires a trusted workspace.")
    path = root / ".aster" / _SETTINGS_FILENAME
    if not path.resolve().is_relative_to(root):
        raise ValueError("Project memory settings must stay inside the workspace.")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        value = {}
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError("Project memory settings could not be read.") from None
    if not isinstance(value, dict):
        raise ValueError("Project settings must be a mapping.")
    coding = value.get("coding", {})
    if not isinstance(coding, dict):
        raise ValueError("coding settings must be a mapping.")
    memory = coding.get("project_memory", {})
    if not isinstance(memory, dict):
        raise ValueError("coding.project_memory must be a mapping.")
    return path, value


def load_project_memory_settings(workspace: str | Path) -> dict[str, Any]:
    """Memory is opt-in per trusted project; user/global settings cannot enable it."""
    if not workspace_trust.is_trusted(Path(workspace).expanduser().resolve()):
        return {}
    _, document = _project_memory_document(workspace)
    result = document.get("coding", {}).get("project_memory", {})
    if type(result.get("enabled", False)) is not bool:
        raise ValueError("coding.project_memory.enabled must be true or false.")
    return dict(result)


def disable_project_memory(workspace: str | Path) -> None:
    """Persist the explicit host /memory off command without changing other settings."""
    import tempfile

    import yaml

    path, document = _project_memory_document(workspace)
    document.setdefault("coding", {}).setdefault("project_memory", {})["enabled"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".settings-", delete=False
        ) as stream:
            staged = Path(stream.name)
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def _setting_paths(settings: Mapping[str, Any], section: str) -> list[str | Path]:
    value = settings.get(section)
    if not isinstance(value, Mapping):
        return []
    paths = value.get("additional_skills_dirs")
    if isinstance(paths, (str, Path)):
        return [paths]
    if not isinstance(paths, list):
        return []
    return [path for path in paths if isinstance(path, (str, Path))]


def _resolve_configured(path: str | Path, workspace: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else workspace / candidate


def _read_project_settings(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.warning("Failed to read coding settings %s: %s", path, exc)
        return {}
    return value if isinstance(value, Mapping) else {}


def _legacy_project_paths(path: Path) -> list[str | Path]:
    """Read the pre-settings-YAML ``[tui].libs_dirs`` compatibility key."""
    if not path.is_file():
        return []
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Failed to read legacy coding settings %s: %s", path, exc)
        return []
    tui = data.get("tui")
    if not isinstance(tui, Mapping):
        return []
    value = tui.get("libs_dirs")
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


__all__ = ["load_coding_skills_dirs", "load_required_checks", "load_task_budget"]
