# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose and remember Aster's default without loading the framework or credentials."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aster_cli._common import find_project_root, load_workspace_trust


def _user_dir() -> Path:
    user_override = os.environ.get("ASTER_USER_DIR")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(xdg) if xdg else Path.home() / ".config"
    return Path(user_override) if user_override else config_home / "aster"


def _config_paths(workspace: Path, filename: str, env_var: str) -> list[Path]:
    """Mirror user/project/override layers as seen from the backend's workspace cwd."""
    user_dir = _user_dir()
    project_override = os.environ.get("ASTER_PROJECT_DIR")
    project_dir = (
        Path(project_override) if project_override else find_project_root(workspace) / ".aster"
    )
    candidates = [user_dir / filename]
    if _project_file_allowed(workspace, project_dir, filename):
        candidates.append(project_dir / filename)
    candidates.extend(
        Path(raw.strip()).expanduser()
        for raw in os.environ.get(env_var, "").split(",")
        if raw.strip()
    )

    paths: list[Path] = []
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = workspace / candidate
        try:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        # The shared config loader keeps the last occurrence of a resolved path.
        if resolved in paths:
            paths.remove(resolved)
        paths.append(resolved)
    return paths


def _project_file_allowed(workspace: Path, project_dir: Path, filename: str) -> bool:
    """A repository's settings apply only once the user trusts the folder."""
    if os.environ.get("ASTER_PROJECT_DIR"):
        return True
    directory = project_dir if project_dir.is_absolute() else workspace / project_dir
    if not (directory / filename).is_file():
        return True
    return load_workspace_trust().allows_project_dir(directory)


def _read_config(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        # YAML exceptions can contain source lines, so keep config values out of
        # the user-facing diagnostic.
        raise ValueError(
            f"Cannot read model configuration in {path}. Fix it or pass --model."
        ) from exc
    return data if isinstance(data, dict) else {}


def save_tui_model(workspace: Path, model: str) -> Path:
    """Remember a confirmed model in the highest existing settings layer.

    With no settings file, create the selected workspace's project settings.
    The caller's working directory and model environment variables are untouched.
    Raises ValueError for invalid settings and OSError for a failed save.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A nonblank model name is required to save the default.")
    workspace = workspace.resolve()
    paths = _config_paths(workspace, "settings.yaml", "ASTER_SETTINGS")
    if paths:
        path = paths[-1]
    else:
        project_override = os.environ.get("ASTER_PROJECT_DIR")
        directory = (
            Path(project_override) if project_override else find_project_root(workspace) / ".aster"
        )
        if not project_override and not load_workspace_trust().allows_project_dir(directory):
            # A file written into an untrusted folder would not be read back.
            directory = _user_dir()
        if not directory.is_absolute():
            directory = workspace / directory
        path = (directory / "settings.yaml").resolve()
    try:
        with _settings_write_lock(path):
            _write_tui_model(path, model)
    except OSError:
        raise OSError(
            f"Cannot save default model in {path}. Check file access and retry."
        ) from None
    return path


@contextmanager
def _settings_write_lock(path: Path) -> Iterator[None]:
    """Use a stable sidecar because an atomic replacement changes the data file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    if os.name == "nt":
        from filelock import FileLock

        with FileLock(lock_path, timeout=5):
            yield
        return
    import fcntl
    import time

    with lock_path.open("a") as stream:
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Settings file is locked by another writer.") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    # Retain the POSIX sidecar so waiters never lock different inodes.


def _write_tui_model(path: Path, model: str) -> None:
    import stat
    import tempfile

    import yaml

    try:
        original = path.read_bytes()
    except FileNotFoundError:
        original = None
    try:
        source = (original or b"").decode("utf-8")
        data = yaml.safe_load(source)
    except (UnicodeError, yaml.YAMLError):
        raise ValueError(
            f"Cannot save default model in {path}: invalid YAML or encoding."
        ) from None
    if data is None:
        data = {}
    if not isinstance(data, dict) or not isinstance(data.get("tui", {}), (dict, type(None))):
        raise ValueError(f"Cannot save default model in {path}: settings and tui must be mappings.")
    if (data.get("tui") or {}).get("default_model") == model:
        return
    data = {**data, "tui": {**(data.get("tui") or {}), "default_model": model}}
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = _model_settings_text(source, data, model)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if original is not None and os.name != "nt":
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        if (path.read_bytes() if path.exists() else None) != original:
            raise ValueError(f"Cannot save default model in {path}: settings changed; retry.")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _model_settings_text(source: str, settings: dict[str, Any], model: str) -> str:
    """Keep ordinary scalar edits local; serialize complex YAML as parsed data."""
    import json

    import yaml

    if not any(
        isinstance(token, (yaml.AnchorToken, yaml.AliasToken, yaml.TagToken))
        for token in yaml.scan(source)
    ):
        node = yaml.compose(source)
        for name in ("tui", "default_model"):
            if not isinstance(node, yaml.MappingNode):
                break
            node = next((value for key, value in node.value if key.value == name), None)
        if (
            isinstance(node, yaml.ScalarNode)
            and node.start_mark.line == node.end_mark.line
            and node.start_mark.index < node.end_mark.index
        ):
            candidate = (
                source[: node.start_mark.index] + json.dumps(model) + source[node.end_mark.index :]
            )
            if yaml.safe_load(candidate) == settings:
                return candidate
    newline = "\r\n" if "\r\n" in source else "\n"
    return yaml.safe_dump(settings, sort_keys=False, allow_unicode=True).replace("\n", newline)


def resolve_tui_model(workspace: Path, *, allow_unselected: bool = False) -> str:
    """Return a settings default or the sole explicitly configured chat alias.

    Called only when the CLI and model environment variables supply no model.
    Bundled providers are not user preferences and are never loaded here.
    """
    workspace = workspace.resolve()
    default: object = None
    for path in _config_paths(workspace, "settings.yaml", "ASTER_SETTINGS"):
        settings = _read_config(path)
        if "tui" not in settings:
            continue
        tui = settings["tui"]
        if not isinstance(tui, dict):
            default = None
        elif "default_model" in tui:
            default = tui["default_model"]
    if default is not None:
        if not isinstance(default, str) or not default.strip():
            raise ValueError("tui.default_model must be a nonblank model name, or pass --model.")
        return default.strip()

    aliases: dict[str, bool] = {}
    for path in _config_paths(workspace, "llm_config.yaml", "ASTER_LLM_CONFIG"):
        models = _read_config(path).get("models")
        if not isinstance(models, dict):
            continue
        for name, config in models.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(config, dict):
                aliases[name] = config.get("client_type", "completion") in (
                    "completion",
                    "responses",
                )
            else:
                aliases.pop(name, None)
    candidates = sorted(name for name, eligible in aliases.items() if eligible)
    if len(candidates) == 1:
        return candidates[0]

    if allow_unselected:
        return ""

    instruction = "Pass --model, set ASTER_MODEL, or set tui.default_model in .aster/settings.yaml."
    if not candidates:
        raise ValueError(f"No default or configured chat model was found. {instruction}")
    names = ", ".join(candidates[:5]) + (", ..." if len(candidates) > 5 else "")
    raise ValueError(f"Multiple configured chat models ({names}). {instruction}")
