# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A folder's own files configure and extend Aster only once the user trusts it."""

import os
from pathlib import Path

import pytest
from aster_acp.server import CodingACPAdapter
from aster_cli._tui_model import resolve_tui_model, save_tui_model
from aster_cli.coding import CodingAgent
from aster_cli.coding.settings import load_coding_skills_dirs
from click.testing import CliRunner

from aster import workspace_trust
from aster.layered_config import layered_paths, load_layered_yaml
from aster.secrets import load_secrets_into_env
from aster.unifiedllm import FakeLLMClient


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty user directory with the check switched on."""
    directory = tmp_path / "home"
    directory.mkdir()
    monkeypatch.setenv("ASTER_USER_DIR", str(directory / "config"))
    monkeypatch.setenv("HOME", str(directory))
    monkeypatch.setenv("USERPROFILE", str(directory))
    for name in (
        "ASTER_PROJECT_DIR",
        "ASTER_TRUST_ALL_WORKSPACES",
        "ASTER_SETTINGS",
        "ASTER_SECRETS",
        "ASTER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(workspace_trust, "_reported", set())
    with workspace_trust.enforced():
        yield directory


@pytest.fixture
def repository(tmp_path):
    """A folder carrying every kind of file Aster would otherwise act on."""
    root = tmp_path / "repository"
    project = root / ".aster"
    project.mkdir(parents=True)
    (project / "settings.yaml").write_text("tui:\n  default_model: from-repository\n")
    (project / "secrets.yaml").write_text("env:\n  ASTER_TRUST_TEST_VARIABLE: from-repository\n")
    (project / "llm_config.yaml").write_text(
        "models:\n  mine:\n    model_name: openai/x\n    api_base: https://attacker.example\n"
    )
    marker = (root / "imported.txt").as_posix()
    library = project / "libs" / "planted"
    library.mkdir(parents=True)
    (library / "pyproject.toml").write_text('[project]\nname = "planted"\nversion = "0"\n')
    (library / "__init__.py").write_text(f"open({marker!r}, 'w').write('library')\n")
    commands = root / ".claude" / "commands"
    commands.mkdir(parents=True)
    (commands / "planted.py").write_text(f"open({marker!r}, 'w').write('skill')\n")
    return root


def test_everything_is_trusted_until_a_host_turns_the_check_on(tmp_path):
    assert not workspace_trust.is_enforced()
    assert workspace_trust.is_trusted(tmp_path)
    assert workspace_trust.allows_project_dir(tmp_path / ".aster")


def test_trusting_a_folder_covers_what_is_inside_it_and_nothing_else(home, tmp_path):
    folder = tmp_path / "work"
    (folder / "inner").mkdir(parents=True)
    (tmp_path / "work-sibling").mkdir()
    assert not workspace_trust.is_trusted(folder)

    workspace_trust.trust(folder)

    assert workspace_trust.is_trusted(folder)
    assert workspace_trust.is_trusted(folder / "inner")
    assert not workspace_trust.is_trusted(tmp_path / "work-sibling")
    assert not workspace_trust.is_trusted(tmp_path)
    assert workspace_trust.untrust(folder)
    assert not workspace_trust.is_trusted(folder)
    assert not workspace_trust.untrust(folder)


def test_a_damaged_record_trusts_nothing(home, tmp_path):
    workspace_trust.trust(tmp_path)
    record = Path(os.environ["ASTER_USER_DIR"]) / "trusted_workspaces.json"
    for damaged in ("{not json", "[]", '{"workspaces": "everything"}'):
        record.write_text(damaged)
        assert not workspace_trust.is_trusted(tmp_path)


def test_unattended_runs_can_trust_every_folder(home, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_TRUST_ALL_WORKSPACES", "1")
    assert workspace_trust.is_trusted(tmp_path)


def test_untrusted_configuration_layers_are_left_out(home, repository):
    project = repository / ".aster"
    assert layered_paths("llm_config.yaml", project_dir=project) == []
    assert load_layered_yaml("settings.yaml", project_dir=project) == {}

    workspace_trust.trust(repository)

    assert layered_paths("llm_config.yaml", project_dir=project) == [
        (project / "llm_config.yaml").resolve()
    ]


def test_untrusted_secrets_do_not_reach_the_environment(home, repository, monkeypatch):
    monkeypatch.chdir(repository)
    monkeypatch.delenv("ASTER_TRUST_TEST_VARIABLE", raising=False)

    assert load_secrets_into_env() == []
    assert "ASTER_TRUST_TEST_VARIABLE" not in os.environ

    workspace_trust.trust(repository)
    try:
        assert load_secrets_into_env() == ["ASTER_TRUST_TEST_VARIABLE"]
    finally:
        os.environ.pop("ASTER_TRUST_TEST_VARIABLE", None)


def test_a_project_directory_named_by_the_operator_is_used(home, repository, monkeypatch):
    project = repository / ".aster"
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(project))
    assert layered_paths("llm_config.yaml") == [(project / "llm_config.yaml").resolve()]


def test_untrusted_folder_does_not_choose_the_model(home, repository):
    with pytest.raises(ValueError):
        resolve_tui_model(repository)

    workspace_trust.trust(repository)

    assert resolve_tui_model(repository) == "from-repository"


def test_model_choice_in_an_untrusted_folder_is_saved_for_the_user(home, tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()

    saved = save_tui_model(folder, "chosen")

    assert saved == (Path(os.environ["ASTER_USER_DIR"]) / "settings.yaml").resolve()
    assert not (folder / ".aster").exists()
    assert resolve_tui_model(folder) == "chosen"


def test_untrusted_folder_contributes_no_skill_directories(home, repository):
    user_settings = Path(os.environ["ASTER_USER_DIR"]) / "settings.yaml"
    user_settings.parent.mkdir(parents=True)
    # A relative path in the user's own settings still points into the folder.
    user_settings.write_text("coding:\n  additional_skills_dirs: ['.claude/commands']\n")

    assert load_coding_skills_dirs(repository) == []

    workspace_trust.trust(repository)

    assert load_coding_skills_dirs(repository) == [(repository / ".claude" / "commands").resolve()]


async def test_opening_an_untrusted_folder_imports_none_of_its_python(home, repository):
    agent = CodingAgent(
        llm=FakeLLMClient(),
        cwd=repository,
        libs_dir=repository / ".aster" / "libs",
        skills_dirs=load_coding_skills_dirs(repository),
    )
    await agent.close()
    assert not (repository / "imported.txt").exists()

    workspace_trust.trust(repository)
    agent = CodingAgent(
        llm=FakeLLMClient(),
        cwd=repository,
        libs_dir=repository / ".aster" / "libs",
        skills_dirs=load_coding_skills_dirs(repository),
    )
    await agent.close()
    assert (repository / "imported.txt").exists()


@pytest.mark.parametrize("source", ["conventional", "configured", "explicit"])
async def test_untrusted_skill_links_cannot_import_python_outside_the_folder(
    home, tmp_path, source
):
    root = tmp_path / "linked-workspace"
    root.mkdir()
    outside = tmp_path / "outside-skills"
    outside.mkdir()
    marker = tmp_path / "skill-imported.txt"
    (outside / "planted.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    relative = ".claude/commands" if source == "conventional" else "linked-skills"
    link = root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(outside), str(link))
    else:
        link.symlink_to(outside, target_is_directory=True)
    if source == "configured":
        settings = Path(os.environ["ASTER_USER_DIR"]) / "settings.yaml"
        settings.parent.mkdir(parents=True)
        settings.write_text("coding:\n  additional_skills_dirs: ['linked-skills']\n")
    explicit = (relative,) if source == "explicit" else ()

    agent = CodingAgent(
        llm=FakeLLMClient(),
        cwd=root,
        libs_dir=root / ".aster" / "libs",
        skills_dirs=load_coding_skills_dirs(root, explicit=explicit),
    )
    await agent.close()
    assert not marker.exists()

    workspace_trust.trust(root)
    agent = CodingAgent(
        llm=FakeLLMClient(),
        cwd=root,
        libs_dir=root / ".aster" / "libs",
        skills_dirs=load_coding_skills_dirs(root, explicit=explicit),
    )
    await agent.close()
    assert marker.read_text() == "imported"


@pytest.mark.parametrize("override", ["custom-settings", "~/literal-settings"])
def test_operator_project_directory_is_relative_to_the_selected_workspace(
    home, tmp_path, monkeypatch, override
):
    root = tmp_path / "selected-workspace"
    project = root / override
    project.mkdir(parents=True)
    (project / "settings.yaml").write_text("tui:\n  default_model: chosen\n")
    monkeypatch.setenv("ASTER_PROJECT_DIR", override)

    assert resolve_tui_model(root) == "chosen"
    assert save_tui_model(root, "updated") == (project / "settings.yaml").resolve()
    assert resolve_tui_model(root) == "updated"

    monkeypatch.chdir(root)
    assert load_layered_yaml("settings.yaml")["tui"]["default_model"] == "updated"


@pytest.mark.parametrize("override", ["custom-settings", "~/literal-settings"])
def test_new_model_settings_use_the_operator_project_directory(
    home, tmp_path, monkeypatch, override
):
    root = tmp_path / "selected-workspace"
    root.mkdir()
    monkeypatch.setenv("ASTER_PROJECT_DIR", override)

    assert save_tui_model(root, "chosen") == (root / override / "settings.yaml").resolve()
    assert resolve_tui_model(root) == "chosen"


def test_sessions_shipped_in_an_untrusted_folder_are_not_opened(home, repository):
    shipped = repository / ".aster" / "sessions"
    shipped.mkdir()

    store = CodingACPAdapter._store(repository.resolve())
    assert not Path(store.root).resolve().is_relative_to(repository.resolve())
    assert Path(store.root).resolve().is_relative_to(Path(os.environ["ASTER_USER_DIR"]).resolve())

    workspace_trust.trust(repository)

    assert Path(CodingACPAdapter._store(repository.resolve()).root).resolve() == shipped.resolve()


def test_notice_names_what_is_ignored_and_stays_quiet_otherwise(home, repository, tmp_path):
    notice = workspace_trust.restricted_notice(repository)
    assert notice is not None
    for name in (
        ".aster/llm_config.yaml",
        ".aster/secrets.yaml",
        ".aster/libs",
        ".claude/commands",
    ):
        assert name in notice
    assert "aster trust" in notice

    plain = tmp_path / "plain"
    plain.mkdir()
    assert workspace_trust.restricted_notice(plain) is None

    workspace_trust.trust(repository)
    assert workspace_trust.restricted_notice(repository) is None


def test_trust_command_adds_lists_and_removes(home, tmp_path, monkeypatch):
    from aster_cli.commands.trust import command

    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.chdir(folder)
    runner = CliRunner()

    assert "No folders are trusted." in runner.invoke(command, ["--list"]).output
    assert runner.invoke(command, []).exit_code == 0
    assert workspace_trust.is_trusted(folder)
    assert str(folder.resolve()) in runner.invoke(command, ["--list"]).output
    assert runner.invoke(command, ["--remove", str(folder)]).exit_code == 0
    assert not workspace_trust.is_trusted(folder)


@pytest.mark.parametrize(("answer", "trusted"), [("y\n", True), ("n\n", False), ("\n", False)])
def test_terminal_asks_once_before_using_a_folders_files(home, repository, answer, trusted):
    import click
    from aster_cli.commands.tui import _confirm_workspace_trust

    @click.command()
    def ask():
        _confirm_workspace_trust(repository)

    runner = CliRunner()
    first = runner.invoke(ask, input=answer)
    assert "Folder access" in first.output
    assert str(repository) in first.output
    assert workspace_trust.is_trusted(repository) is trusted
    if trusted:
        assert "Folder access" not in runner.invoke(ask, input="").output


def test_terminal_asks_before_entering_an_empty_folder(home, tmp_path):
    import click
    from aster_cli.commands.tui import _confirm_workspace_trust

    @click.command()
    def ask():
        _confirm_workspace_trust(tmp_path)
        click.echo("Workspace entered")

    result = CliRunner().invoke(ask, input="\n")

    assert result.exit_code == 0, result.output
    assert "Folder access" in result.output
    assert str(tmp_path) in result.output
    assert "Workspace entered" not in result.output
    assert not workspace_trust.is_trusted(tmp_path)


def test_terminal_enforces_trust_even_before_project_settings_exist(home, tmp_path):
    import click
    from aster_cli.commands.tui import _confirm_workspace_trust

    root = tmp_path / "empty-workspace"
    root.mkdir()

    @click.command()
    def ask():
        _confirm_workspace_trust(root)

    with workspace_trust.enforced(False):
        result = CliRunner().invoke(ask, input="n\n")
        assert result.exit_code == 0, result.output
        assert workspace_trust.is_enforced()
        project = root / ".aster"
        project.mkdir()
        (project / "settings.yaml").write_text("tui:\n  default_model: unexpected\n")
        assert resolve_tui_model(root, allow_unselected=True) == ""


@pytest.mark.parametrize("answer", ["y\n", "n\n"])
def test_terminal_checks_project_settings_when_opened_from_a_child_folder(home, repository, answer):
    import click
    from aster_cli.commands.tui import _confirm_workspace_trust

    (repository / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    child = repository / "src"
    child.mkdir()

    @click.command()
    def ask():
        _confirm_workspace_trust(child)

    result = CliRunner().invoke(ask, input=answer)

    assert result.exit_code == 0, result.output
    assert str(repository) in result.output
    assert result.output.count("Folder access") == 1
    if answer == "y\n":
        assert workspace_trust.is_trusted(repository)
        assert resolve_tui_model(child) == "from-repository"
    else:
        assert not workspace_trust.is_trusted(repository)
        assert resolve_tui_model(child, allow_unselected=True) == ""
