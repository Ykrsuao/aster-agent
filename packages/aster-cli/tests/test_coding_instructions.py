# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository instruction reads stay within the selected workspace."""

import os
from pathlib import Path

import pytest
from aster_cli.coding import instructions


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")


def test_instruction_discovery_and_read_skip_external_symlinks(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-marker", encoding="utf-8")
    _symlink(repository / "AGENTS.md", outside)

    assert instructions.discover_agent_instruction_files(repository) == ()
    assert instructions.render_agent_instructions(repository) == ""


def test_instruction_read_preserves_internal_symlinks_and_hierarchy(tmp_path):
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    target = tmp_path / "rules.md"
    target.write_text("shared-rule", encoding="utf-8")
    _symlink(tmp_path / "AGENTS.md", target)
    (nested / "AGENTS.md").write_text("nested-rule", encoding="utf-8")

    rendered = instructions.render_agent_instructions(nested)

    assert rendered.index("shared-rule") < rendered.index("nested-rule")


def test_instruction_read_uses_cwd_boundary_without_git(tmp_path):
    directory = tmp_path / "project"
    directory.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-marker", encoding="utf-8")
    _symlink(directory / "AGENTS.md", outside)

    assert instructions.render_agent_instructions(directory) == ""


def test_instruction_read_checks_boundary_again_after_discovery(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-marker", encoding="utf-8")
    monkeypatch.setattr(instructions, "discover_agent_instruction_files", lambda _: (outside,))

    assert instructions.render_agent_instructions(repository) == ""


def test_instruction_read_skips_hardlinked_files(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-marker", encoding="utf-8")
    try:
        (repository / "AGENTS.md").hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Hardlinks are unavailable: {exc}")

    assert instructions.render_agent_instructions(repository) == ""


def test_instruction_read_validates_the_open_file_not_its_current_name(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    instruction_file = repository / "AGENTS.md"
    instruction_file.write_text("inside-rule", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-marker", encoding="utf-8")
    real_open = os.open

    def replaced_open(path, flags, *args, **kwargs):
        # Model replacement during open; the visible pathname remains harmless.
        target = outside if Path(path) == instruction_file else path
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replaced_open)

    assert instructions.render_agent_instructions(repository) == ""


def test_instruction_read_fails_closed_if_the_opened_path_is_unavailable(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("inside-rule", encoding="utf-8")

    def unavailable(_):
        raise OSError("Opened path unavailable")

    monkeypatch.setattr(instructions, "_opened_file_path", unavailable)

    assert instructions.render_agent_instructions(tmp_path) == ""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named pipes only")
def test_instruction_read_rejects_pipes_without_blocking(tmp_path, monkeypatch):
    fifo = tmp_path / "AGENTS.md"
    os.mkfifo(fifo)
    monkeypatch.setattr(instructions, "discover_agent_instruction_files", lambda _: (fifo,))

    assert instructions.render_agent_instructions(tmp_path) == ""


def test_instruction_read_outside_git_loads_only_cwd(tmp_path):
    (tmp_path / "AGENTS.md").write_text("outside-rule", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "AGENTS.md").write_text("inside-rule", encoding="utf-8")

    rendered = instructions.render_agent_instructions(project)

    assert "inside-rule" in rendered
    assert "outside-rule" not in rendered


@pytest.mark.skipif(os.name != "nt", reason="Windows extended paths only")
def test_instruction_read_accepts_an_extended_workspace_path(tmp_path):
    (tmp_path / "AGENTS.md").write_text("inside-rule", encoding="utf-8")
    extended_path = "\\\\?\\" + str(tmp_path.resolve())

    assert "inside-rule" in instructions.render_agent_instructions(extended_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Linux open-file path semantics")
def test_instruction_read_rejects_deleted_path_with_a_surviving_hardlink(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "AGENTS.md"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-marker", encoding="utf-8")
    source.hardlink_to(outside)
    real_open = os.open

    def unlinked_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == source:
            source.unlink()
        return fd

    monkeypatch.setattr(os, "open", unlinked_open)

    assert instructions.render_agent_instructions(repository) == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows handle paths only")
@pytest.mark.parametrize(
    "handle_path, expected",
    [
        (r"\\?\C:\workspace\AGENTS.md", r"C:\workspace\AGENTS.md"),
        (r"\\?\UNC\server\share\AGENTS.md", r"\\server\share\AGENTS.md"),
    ],
)
def test_windows_handle_paths_normalize_dos_and_unc_prefixes(
    tmp_path, monkeypatch, handle_path, expected
):
    import ctypes
    from types import SimpleNamespace

    def get_path(handle, buffer, size, flags):
        if buffer is None:
            return len(handle_path) + 1
        buffer.value = handle_path
        return len(handle_path)

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: SimpleNamespace(GetFinalPathNameByHandleW=get_path),
    )
    path = tmp_path / "AGENTS.md"
    path.write_text("inside-rule", encoding="utf-8")

    with path.open() as stream:
        assert instructions._opened_file_path(stream.fileno()) == Path(expected)
