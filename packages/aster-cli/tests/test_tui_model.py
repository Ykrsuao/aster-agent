# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aster defaults use the backend's workspace and explicit configuration layers."""

import os
from pathlib import Path

import pytest
from aster_cli._common import find_project_root
from aster_cli._tui_model import _config_paths, resolve_tui_model


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    selected = tmp_path / "selected workspace"
    selected.mkdir()
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    for name in (
        "ASTER_PROJECT_DIR",
        "ASTER_SETTINGS",
        "ASTER_LLM_CONFIG",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    return selected


def write_config(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_first_run_can_open_the_in_app_model_picker(workspace):
    assert resolve_tui_model(workspace, allow_unselected=True) == ""


def test_multiple_aliases_can_be_chosen_inside_aster(workspace):
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  one: {}\n  two: {}\n")
    assert resolve_tui_model(workspace, allow_unselected=True) == ""


def test_project_default_overrides_user_and_does_not_need_model_registry(workspace, tmp_path):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: project-model\n")
    write_config(workspace / ".aster/llm_config.yaml", "models: [invalid YAML")

    assert resolve_tui_model(workspace) == "project-model"


def test_user_default_is_used_without_project_default(workspace, tmp_path):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  theme: dark\n")

    assert resolve_tui_model(workspace) == "user-model"


def test_settings_overrides_are_workspace_relative_and_last_wins(workspace, tmp_path, monkeypatch):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: project-model\n")
    write_config(workspace / "first.yaml", "tui:\n  default_model: first-model\n")
    write_config(workspace / "second.yaml", "tui:\n  default_model: last-model\n")
    write_config(Path.cwd() / "second.yaml", "tui:\n  default_model: wrong-cwd-model\n")
    monkeypatch.setenv("ASTER_SETTINGS", " first.yaml, , missing.yaml, second.yaml ")

    assert resolve_tui_model(workspace) == "last-model"


def test_repeated_config_paths_keep_the_last_layer_position(workspace, tmp_path, monkeypatch):
    user = write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: user-model\n")
    project = write_config(
        workspace / ".aster/settings.yaml", "tui:\n  default_model: project-model\n"
    )
    monkeypatch.setenv("ASTER_SETTINGS", f"{user},{project},{user}")

    assert _config_paths(workspace, "settings.yaml", "ASTER_SETTINGS") == [project, user]
    assert resolve_tui_model(workspace) == "user-model"


@pytest.mark.parametrize(
    "replacement", ["tui: null\n", "tui: disabled\n", "tui:\n  default_model: null\n"]
)
def test_settings_deletion_clears_earlier_default(workspace, tmp_path, replacement):
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: old-default\n")
    write_config(workspace / ".aster/settings.yaml", replacement)
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  only-chat: {}\n")

    assert resolve_tui_model(workspace) == "only-chat"


@pytest.mark.parametrize("value", ["''", "'   '", "42", "[]"])
def test_invalid_explicit_default_is_not_replaced_by_an_inferred_alias(workspace, value):
    write_config(workspace / ".aster/settings.yaml", f"tui:\n  default_model: {value}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  only-chat: {}\n")

    with pytest.raises(ValueError, match="tui.default_model.*nonblank"):
        resolve_tui_model(workspace)


def test_default_accepts_a_model_name_without_a_registered_alias(workspace):
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: openai/my-model\n")

    assert resolve_tui_model(workspace) == "openai/my-model"


def test_selected_workspace_uses_nearest_project_root(workspace):
    write_config(workspace / "pyproject.toml", "[project]\n")
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: selected-project\n")
    write_config(Path.cwd() / "pyproject.toml", "[project]\n")
    write_config(Path.cwd() / ".aster/settings.yaml", "tui:\n  default_model: caller-project\n")
    nested = workspace / "src/nested"
    nested.mkdir(parents=True)

    assert resolve_tui_model(nested) == "selected-project"
    assert find_project_root(nested) == workspace
    assert find_project_root() == Path.cwd()


def test_project_directory_override_is_relative_to_selected_workspace(workspace, monkeypatch):
    write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: ordinary-project\n")
    write_config(workspace / "custom/settings.yaml", "tui:\n  default_model: override-project\n")
    write_config(Path.cwd() / "custom/settings.yaml", "tui:\n  default_model: wrong-cwd-model\n")
    monkeypatch.setenv("ASTER_PROJECT_DIR", "custom")

    assert resolve_tui_model(workspace) == "override-project"


@pytest.mark.parametrize(
    ("variable", "directory"),
    [("ASTER_USER_DIR", "custom-user"), ("XDG_CONFIG_HOME", "custom-user/aster")],
)
def test_relative_user_directories_follow_backend_cwd(workspace, monkeypatch, variable, directory):
    monkeypatch.delenv("ASTER_USER_DIR")
    monkeypatch.setenv(variable, "custom-user")
    write_config(workspace / directory / "settings.yaml", "tui:\n  default_model: selected-user\n")
    write_config(Path.cwd() / directory / "settings.yaml", "tui:\n  default_model: caller-user\n")

    assert resolve_tui_model(workspace) == "selected-user"


def test_user_directory_override_wins_over_xdg(workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    write_config(tmp_path / "xdg/aster/settings.yaml", "tui:\n  default_model: xdg-model\n")
    write_config(tmp_path / "user/settings.yaml", "tui:\n  default_model: explicit-user\n")

    assert resolve_tui_model(workspace) == "explicit-user"


def test_user_directory_falls_back_to_home_config(workspace, tmp_path, monkeypatch):
    monkeypatch.delenv("ASTER_USER_DIR")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    write_config(
        tmp_path / "home/.config/aster/settings.yaml", "tui:\n  default_model: home-model\n"
    )

    assert resolve_tui_model(workspace) == "home-model"


def test_environment_file_paths_expand_tilde(workspace, tmp_path, monkeypatch):
    original_expanduser = os.path.expanduser

    def expanduser(value):
        if value == "~" or value.startswith(("~/", "~\\")):
            return str(tmp_path / "home") + value[1:]
        return original_expanduser(value)

    monkeypatch.setattr(os.path, "expanduser", expanduser)
    write_config(tmp_path / "home/model-settings.yaml", "tui:\n  default_model: home-override\n")
    monkeypatch.setenv("ASTER_SETTINGS", "~/model-settings.yaml")

    assert resolve_tui_model(workspace) == "home-override"


@pytest.mark.parametrize("directory_var", ["ASTER_USER_DIR", "ASTER_PROJECT_DIR"])
def test_directory_overrides_preserve_literal_tilde(workspace, monkeypatch, directory_var):
    monkeypatch.setenv(directory_var, "~/custom")
    write_config(workspace / "~/custom/settings.yaml", "tui:\n  default_model: literal-directory\n")

    assert resolve_tui_model(workspace) == "literal-directory"


def test_user_singleton_alias_is_inferred(workspace, tmp_path):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")

    assert resolve_tui_model(workspace) == "user-chat"


def test_separate_user_and_project_aliases_are_ambiguous(workspace, tmp_path):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  project-chat: {}\n")

    with pytest.raises(ValueError, match="Multiple configured chat models") as result:
        resolve_tui_model(workspace)
    assert "user-chat" in str(result.value)
    assert "project-chat" in str(result.value)
    assert "tui.default_model" in str(result.value)
    assert "--model" in str(result.value)


def test_later_alias_replaces_the_whole_entry(workspace, tmp_path):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  model: {client_type: decision}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  model: {}\n")

    assert resolve_tui_model(workspace) == "model"


@pytest.mark.parametrize("replacement", ["null", "invalid", "{client_type: decision}"])
def test_removed_or_ineligible_alias_does_not_remain_a_candidate(workspace, tmp_path, replacement):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  removed: {}\n  remaining: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", f"models:\n  removed: {replacement}\n")

    assert resolve_tui_model(workspace) == "remaining"


@pytest.mark.parametrize("models", ["null", "[]", "invalid"])
def test_nonmapping_models_section_does_not_delete_lower_aliases(workspace, tmp_path, models):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", f"models: {models}\n")

    assert resolve_tui_model(workspace) == "user-chat"


def test_only_chat_compatible_aliases_are_candidates(workspace):
    write_config(
        workspace / ".aster/llm_config.yaml",
        "models:\n"
        "  decisions: {client_type: decision}\n"
        "  unknown: {client_type: unsupported}\n"
        "  null-client: {client_type: null}\n"
        "  malformed: {client_type: []}\n"
        "  chat: {client_type: responses}\n",
    )

    assert resolve_tui_model(workspace) == "chat"


def test_model_file_overrides_are_workspace_relative_and_last_wins(
    workspace, tmp_path, monkeypatch
):
    write_config(tmp_path / "user/llm_config.yaml", "models:\n  user-chat: {}\n")
    write_config(workspace / ".aster/llm_config.yaml", "models:\n  project-chat: {}\n")
    write_config(workspace / "first.yaml", "models:\n  user-chat: null\n")
    write_config(workspace / "last.yaml", "models:\n  project-chat: null\n  selected-chat: {}\n")
    write_config(Path.cwd() / "last.yaml", "models:\n  wrong-chat: {}\n")
    monkeypatch.setenv("ASTER_LLM_CONFIG", "first.yaml, last.yaml")

    assert resolve_tui_model(workspace) == "selected-chat"


def test_no_models_reports_how_to_set_a_default(workspace):
    with pytest.raises(ValueError, match="No default or configured chat model") as result:
        resolve_tui_model(workspace)

    assert "ASTER_MODEL" in str(result.value)
    assert "tui.default_model" in str(result.value)


@pytest.mark.parametrize("filename", ["settings.yaml", "llm_config.yaml"])
def test_malformed_yaml_reports_path_without_file_contents(workspace, filename):
    path = write_config(workspace / ".aster" / filename, "sensitive_test_value: [unfinished\n")

    with pytest.raises(ValueError, match="Cannot read model configuration") as result:
        resolve_tui_model(workspace)

    assert str(path) in str(result.value)
    assert "sensitive_test_value" not in str(result.value)
    assert "--model" in str(result.value)


def test_saved_model_creates_workspace_default_for_next_launch(workspace):
    from aster_cli._tui_model import save_tui_model

    path = save_tui_model(workspace, "chosen-model")

    assert path == workspace / ".aster/settings.yaml"
    assert resolve_tui_model(workspace) == "chosen-model"
    assert not (Path.cwd() / ".aster/settings.yaml").exists()


@pytest.mark.parametrize("layer", ["user", "project", "override"])
def test_save_changes_only_highest_existing_settings_layer(workspace, tmp_path, monkeypatch, layer):
    import yaml
    from aster_cli._tui_model import save_tui_model

    original = "tui:\n  default_model: previous\n  theme: dark\nother:\n  enabled: true\n"
    user = write_config(tmp_path / "user/settings.yaml", original)
    paths = [user]
    if layer in {"project", "override"}:
        paths.append(write_config(workspace / ".aster/settings.yaml", original))
    if layer == "override":
        paths.append(write_config(workspace / "first.yaml", original))
        paths.append(write_config(workspace / "last.yaml", original))
        monkeypatch.setenv("ASTER_SETTINGS", "first.yaml, last.yaml, missing.yaml")
        write_config(Path.cwd() / "last.yaml", original)

    saved = save_tui_model(workspace, "new-model")

    assert saved == paths[-1]
    assert resolve_tui_model(workspace) == "new-model"
    assert yaml.safe_load(saved.read_text(encoding="utf-8")) == {
        "tui": {"default_model": "new-model", "theme": "dark"},
        "other": {"enabled": True},
    }
    for lower in paths[:-1]:
        assert lower.read_text(encoding="utf-8") == original
    if layer == "override":
        assert (Path.cwd() / "last.yaml").read_text(encoding="utf-8") == original


@pytest.mark.parametrize("override", [None, "custom-settings", "~/literal-settings", "absolute"])
def test_save_creates_project_settings_in_the_selected_workspace(
    workspace, tmp_path, monkeypatch, override
):
    from aster_cli._tui_model import save_tui_model

    write_config(workspace / "pyproject.toml", "[project]\n")
    nested = workspace / "src/nested"
    nested.mkdir(parents=True)
    if override == "absolute":
        override = str(tmp_path / "external-settings")
    if override:
        monkeypatch.setenv("ASTER_PROJECT_DIR", override)
        directory = Path(override)
        if not directory.is_absolute():
            directory = nested / directory
    else:
        directory = workspace / ".aster"
    monkeypatch.setenv("ASTER_SETTINGS", "nonexistent.yaml")

    saved = save_tui_model(nested, "nested-model")

    assert saved == directory / "settings.yaml"
    assert resolve_tui_model(nested) == "nested-model"
    assert not (Path.cwd() / ".aster/settings.yaml").exists()
    assert not (nested / "nonexistent.yaml").exists()


@pytest.mark.parametrize(
    ("variable", "directory"),
    [("ASTER_USER_DIR", "custom-user"), ("XDG_CONFIG_HOME", "custom-user/aster")],
)
def test_save_in_untrusted_workspace_uses_relative_user_directory(
    workspace, monkeypatch, variable, directory
):
    from aster_cli._tui_model import save_tui_model

    from aster import workspace_trust

    monkeypatch.delenv("ASTER_USER_DIR")
    monkeypatch.delenv("ASTER_TRUST_ALL_WORKSPACES", raising=False)
    monkeypatch.setenv(variable, "custom-user")
    caller = Path.cwd()
    with workspace_trust.enforced():
        assert not workspace_trust.is_trusted(workspace)
        saved = save_tui_model(workspace, "selected-model")
        assert saved == workspace / directory / "settings.yaml"
        assert resolve_tui_model(workspace) == "selected-model"

    assert Path.cwd() == caller
    assert not (caller / directory / "settings.yaml").exists()
    assert not (workspace / ".aster").exists()


@pytest.mark.parametrize(
    "content",
    [
        "sensitive_test_value: [unfinished\n",
        "[sensitive_test_value]\n",
        "sensitive_test_value\n",
        "false\n",
        "tui: sensitive_test_value\n",
        "tui: [sensitive_test_value]\n",
    ],
)
def test_save_rejects_invalid_settings_without_replacing_or_exposing_contents(workspace, content):
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", content)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="Cannot save default model") as result:
        save_tui_model(workspace, "new-model")

    assert path.read_bytes() == original
    assert str(path) in str(result.value)
    assert "sensitive_test_value" not in str(result.value)
    assert result.value.__cause__ is None


@pytest.mark.parametrize("content", ["", "null\n", "tui: null\n"])
def test_save_accepts_empty_settings_and_null_sections(workspace, content):
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", content)

    assert save_tui_model(workspace, "new-model") == path
    assert resolve_tui_model(workspace) == "new-model"


@pytest.mark.parametrize("boundary", ["replace", "fsync"])
def test_save_failure_keeps_original_and_removes_temporary_files(workspace, monkeypatch, boundary):
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: previous\n")
    original = path.read_bytes()

    def fail(*args):
        assert path.read_bytes() == original
        if boundary == "replace":
            assert Path(args[0]).parent == path.parent
            assert Path(args[1]) == path
        raise PermissionError("sensitive_test_value")

    monkeypatch.setattr(os, boundary, fail)
    with pytest.raises(OSError, match="Cannot save default model") as result:
        save_tui_model(workspace, "new-model")

    assert path.read_bytes() == original
    assert "sensitive_test_value" not in str(result.value)
    assert result.value.__cause__ is None
    assert [p for p in path.parent.iterdir() if not p.name.endswith(".lock")] == [path]


def test_save_waits_for_other_settings_writer_and_preserves_its_update(workspace):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from contextlib import ExitStack

    import yaml
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: previous\n")
    lock_path = path.with_name(".settings.yaml.lock")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with ExitStack() as stack:
            if os.name == "nt":
                from filelock import FileLock

                stack.enter_context(FileLock(lock_path, timeout=1))
            else:
                import fcntl

                lock = stack.enter_context(lock_path.open("a"))
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            pending = pool.submit(save_tui_model, workspace, "new-model")
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.2)
            path.write_text("tui:\n  default_model: previous\n  theme: changed\n", encoding="utf-8")
        assert pending.result(timeout=5) == path
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "tui": {"default_model": "new-model", "theme": "changed"}
    }


@pytest.mark.parametrize("model", ["route/#one: 'quoted' \"name\"", "line\nbreak", "模型/新档位"])
def test_save_preserves_comments_and_line_endings_while_quoting_model_safely(workspace, model):
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", "")
    prefix = b"# user preferences\r\ntui:\r\n  theme: 'dark'\r\n  default_model: "
    suffix = b"  # selected model\r\nother: {enabled: true}\r\n"
    path.write_bytes(prefix + b"old" + suffix)

    assert save_tui_model(workspace, model) == path
    assert resolve_tui_model(workspace) == model
    assert path.read_bytes().startswith(prefix)
    assert path.read_bytes().endswith(suffix)


@pytest.mark.parametrize(
    "content",
    [
        "other: &model original\ntui:\n  default_model: *model\n",
        "tui:\n  default_model: &model original\nother: *model\n",
        "base: &settings {default_model: original, theme: dark}\n"
        "tui: *settings\nother: *settings\n",
    ],
)
def test_save_does_not_change_values_shared_by_yaml_anchors(workspace, content):
    import yaml
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", content)
    original_other = yaml.safe_load(content)["other"]

    save_tui_model(workspace, "new-model")

    assert resolve_tui_model(workspace) == "new-model"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["other"] == original_other


def test_saving_current_model_leaves_file_bytes_unchanged(workspace, monkeypatch):
    from aster_cli._tui_model import save_tui_model

    path = write_config(
        workspace / ".aster/settings.yaml", "# keep this comment\ntui: {default_model: current}\n"
    )
    original = path.read_bytes()

    def unexpected_replace(*args):
        pytest.fail("Saving an unchanged default should not replace the file")

    monkeypatch.setattr(os, "replace", unexpected_replace)
    assert save_tui_model(workspace, "current") == path
    assert path.read_bytes() == original


@pytest.mark.parametrize("model", ["", "   ", None, 42])
def test_save_rejects_nonblank_model_contract_violations(workspace, model):
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: previous\n")
    original = path.read_bytes()

    with pytest.raises(ValueError, match="nonblank model name"):
        save_tui_model(workspace, model)

    assert path.read_bytes() == original


def test_save_times_out_without_replacing_settings_locked_by_another_writer(workspace):
    from contextlib import ExitStack

    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: previous\n")
    original = path.read_bytes()
    lock_path = path.with_name(".settings.yaml.lock")
    with ExitStack() as stack:
        if os.name == "nt":
            from filelock import FileLock

            stack.enter_context(FileLock(lock_path, timeout=1))
        else:
            import fcntl

            lock = stack.enter_context(lock_path.open("a"))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with pytest.raises(OSError, match="Cannot save default model"):
            save_tui_model(workspace, "new-model")
    assert path.read_bytes() == original


def test_save_does_not_overwrite_a_concurrent_edit_outside_the_lock(workspace, monkeypatch):
    from aster_cli._tui_model import save_tui_model

    path = write_config(workspace / ".aster/settings.yaml", "tui:\n  default_model: previous\n")
    other_editor = b"tui:\n  default_model: edited-elsewhere\n  theme: changed\n"
    original_fsync = os.fsync

    def edit_during_save(fd):
        path.write_bytes(other_editor)
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", edit_during_save)
    with pytest.raises(ValueError, match="settings changed"):
        save_tui_model(workspace, "new-model")
    assert path.read_bytes() == other_editor
    assert [p for p in path.parent.iterdir() if not p.name.endswith(".lock")] == [path]
