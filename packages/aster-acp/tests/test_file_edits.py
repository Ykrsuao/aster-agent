# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Automatically accepted file operations never execute the proposed Python."""

import asyncio
import os
import sys
from pathlib import Path

import aster_acp._file_edits as file_edits
import pytest
from aster_acp._file_edits import FileEditExecutor, file_edit_executor
from aster_cli.coding.activity import ActivityShellTools, FileEdit

from aster.runtime.event_manager import EventManager
from aster.tools._bounded_io import FileTooLargeError, read_line_range, read_whole_file_checked
from aster.tools.shell_tools import FileWrite, Match, ShellTools


@pytest.fixture
def observed_events():
    manager = EventManager()
    events = []
    manager.on("*", events.append)
    return manager, events


def _executor(code, cwd, event_manager, *, max_file_bytes=4096, timeout=1, host_path=None):
    executor = file_edit_executor(
        code,
        cwd=cwd,
        event_manager=event_manager,
        max_file_bytes=max_file_bytes,
        timeout=timeout,
        host_path=host_path,
    )
    assert executor is not None
    return executor


async def test_real_read_write_replace_return_values_and_file_activity(tmp_path, observed_events):
    manager, events = observed_events
    write = "await self.shell.write_file('note.txt', 'one\\ntwo\\n')"
    result = await _executor(write, tmp_path, manager).run_cell(write, execution_count=3)
    assert result.success and result.has_return and not result.explicit_return
    assert isinstance(result.returned_value, FileWrite)
    assert result.returned_value.new_text == "one\ntwo\n"
    assert result.stdout == ""
    assert not result.captured_locals

    read = "await self.shell.read(lines=(2, 2), path='note.txt')"
    result = await _executor(read, tmp_path, manager).run_cell(read)
    assert result.success and isinstance(result.returned_value, Match)
    assert result.returned_value.text == "two\n"
    assert Path(result.returned_value.resolved_path) == tmp_path / "note.txt"

    replace = "await self.shell.replace('note.txt', 'two', new='three')"
    result = await _executor(replace, tmp_path, manager).run_cell(replace)
    assert result.success and isinstance(result.returned_value, FileWrite)
    assert result.returned_value.new_text == "three"
    assert (tmp_path / "note.txt").read_text() == "one\nthree\n"

    edits = [event for event in events if isinstance(event, FileEdit)]
    assert [(event.operation, event.old_text, event.new_text) for event in edits] == [
        ("create", None, "one\ntwo\n"),
        ("update", "two", "three"),
    ]
    assert all(event.path == str(tmp_path / "note.txt") for event in edits)
    assert "-two" in edits[1].diff and "+three" in edits[1].diff


@pytest.mark.parametrize("operation", ["read", "write_file", "replace"])
async def test_maps_only_the_target_and_preserves_path_looking_content(
    tmp_path, observed_events, operation
):
    manager, _ = observed_events
    unmapped = tmp_path / "note.txt"
    unmapped.write_text("unmapped file", encoding="utf-8")
    mapped = tmp_path / "mounted" / "note.txt"
    mapped.parent.mkdir()
    original = "/workspace/content-only.txt"
    replacement = "/allowed/0/replacement-only.txt"
    if operation != "write_file":
        mapped.write_text(original, encoding="utf-8")
    mapped_paths = []

    def host_path(path):
        assert path == "note.txt", "Only the file target may be translated"
        mapped_paths.append(path)
        return str(mapped)

    code = {
        "read": "await self.shell.read('note.txt')",
        "write_file": f"await self.shell.write_file('note.txt', {original!r})",
        "replace": f"await self.shell.replace('note.txt', {original!r}, {replacement!r})",
    }[operation]
    executor = _executor(code, tmp_path, manager, host_path=host_path)
    result = await executor.run_cell(code)

    assert result.success, result.error
    assert mapped_paths == ["note.txt"]
    assert unmapped.read_text(encoding="utf-8") == "unmapped file"
    expected = replacement if operation == "replace" else original
    assert mapped.read_text(encoding="utf-8") == expected
    if operation == "read":
        assert isinstance(result.returned_value, Match)
        assert result.returned_value.text == original
        assert Path(result.returned_value.resolved_path) == mapped


async def test_mapped_path_does_not_replace_the_original_source_approval(tmp_path, observed_events):
    manager, events = observed_events
    mapped = tmp_path / "mapped.txt"
    approved = "await self.shell.write_file('note.txt', 'safe')"
    executor = _executor(approved, tmp_path, manager, host_path=lambda path: str(mapped))
    changed = f"await self.shell.write_file({str(mapped)!r}, 'safe')"

    result = await executor.run_cell(changed)

    assert isinstance(result.error, PermissionError)
    assert not result.has_return
    assert list(tmp_path.iterdir()) == []
    assert events == []


@pytest.mark.parametrize(
    "arguments", ["'note.txt'", "'note.txt', None", "path='note.txt', lines=None"]
)
async def test_read_accepts_omitted_or_explicit_none_range(tmp_path, observed_events, arguments):
    manager, _ = observed_events
    (tmp_path / "note.txt").write_text("whole file")
    code = f"await self.shell.read({arguments})"
    result = await _executor(code, tmp_path, manager).run_cell(code)
    assert result.success and result.returned_value.text == "whole file"


async def test_empty_text_and_signed_integer_range_are_literal_values(tmp_path, observed_events):
    manager, _ = observed_events
    code = "await self.shell.write_file(content='', path='empty.txt')"
    result = await _executor(code, tmp_path, manager).run_cell(code)
    assert result.success and (tmp_path / "empty.txt").read_bytes() == b""
    (tmp_path / "note.txt").write_text("one\ntwo\n")
    read = "await self.shell.read('note.txt', (-1, +1))"
    result = await _executor(read, tmp_path, manager).run_cell(read)
    assert result.success and result.returned_value.text == "one\n"


@pytest.mark.parametrize(
    "code",
    [
        "",
        "pass",
        "self.shell.write_file('note.txt', 'bad')",
        "result = await self.shell.write_file('note.txt', 'bad')",
        "return await self.shell.read('note.txt')",
        "await self.shell.write_file('note.txt', 'bad'); print('after')",
        "print('before')\nawait self.shell.write_file('note.txt', 'bad')",
        "await self.shell.write_file('note.txt', 'bad')\nawait self.shell.read('note.txt')",
        "'extra statement'\nawait self.shell.write_file('note.txt', 'bad')",
        "await self.shell.run('echo bad')",
        "await self.repo.write_file('note.txt', 'bad')",
        "await other.shell.write_file('note.txt', 'bad')",
        "await self.shell.write_file(path, 'bad')",
        "await self.shell.write_file('note.txt', f'bad')",
        "await self.shell.write_file('note.txt', 'a' + 'b')",
        "await self.shell.write_file('note.txt', content())",
        "await self.shell.write_file(*('note.txt', 'bad'))",
        "await self.shell.write_file(**{'path': 'note.txt', 'content': 'bad'})",
        "await self.shell.write_file('note.txt', 'bad', unknown=True)",
        "await self.shell.write_file('note.txt', 'bad', path='other.txt')",
        "await self.shell.write_file(path='note.txt', path='other.txt', content='bad')",
        "await self.shell.write_file('note.txt')",
        "await self.shell.write_file('note.txt', b'bad')",
        "await self.shell.write_file('note.txt', None)",
        "await self.shell.read('note.txt', (True, 2))",
        "await self.shell.read('note.txt', (1, 2.0))",
        "await self.shell.read('note.txt', [1, 2])",
        "await self.shell.read('note.txt', (1, 2, 3))",
        "await self.shell.read('note.txt', (1, end))",
        "await self.shell.replace(match, 'new')",
        "await self.shell.replace('note.txt', 'old')",
        "await self.shell.replace('note.txt', 'old', None)",
        "await self.shell.replace('note.txt', 'old', 'new', 'extra')",
        "await self.shell.read(",
    ],
)
def test_other_python_does_not_prepare_or_perform_operations(tmp_path, observed_events, code):
    manager, events = observed_events
    assert (
        file_edit_executor(
            code, cwd=tmp_path, event_manager=manager, max_file_bytes=4096, timeout=None
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []
    assert events == []


@pytest.mark.parametrize(
    "changed",
    [
        "await self.shell.write_file('other.txt', 'safe')",
        "await self.shell.write_file('note.txt', 'different')",
        "await self.shell.read('note.txt')",
        "await self.shell.write_file('note.txt', 'safe'); print('extra')",
    ],
)
async def test_rechecks_code_changed_by_middleware_before_any_effect(
    tmp_path, observed_events, changed
):
    manager, events = observed_events
    approved = "await self.shell.write_file('note.txt', 'safe')"
    executor = _executor(approved, tmp_path, manager)
    result = await executor.run_cell(changed)
    assert isinstance(result.error, PermissionError)
    assert "not run" in str(result.error)
    assert not result.has_return
    assert list(tmp_path.iterdir()) == []
    assert events == []


async def test_equivalent_spelling_keeps_the_same_approved_operation(tmp_path, observed_events):
    manager, _ = observed_events
    code = "await self.shell.write_file('note.txt', 'safe')"
    executor = _executor(code, tmp_path, manager)
    result = await executor.run_cell(
        '# A middleware comment\nawait self.shell.write_file(content="safe", path="note.txt")'
    )
    assert result.success and (tmp_path / "note.txt").read_text() == "safe"


@pytest.mark.parametrize(
    "code,error_type",
    [
        ("await self.shell.read('missing.txt')", FileNotFoundError),
        ("await self.shell.replace('note.txt', 'absent', 'new')", ValueError),
        ("await self.shell.replace('note.txt', 'same', 'new')", ValueError),
    ],
)
async def test_file_errors_return_execution_errors_without_success_activity(
    tmp_path, observed_events, code, error_type
):
    manager, events = observed_events
    (tmp_path / "note.txt").write_text("same same")
    result = await _executor(code, tmp_path, manager).run_cell(code)
    assert isinstance(result.error, error_type)
    assert not result.has_return
    assert (tmp_path / "note.txt").read_text() == "same same"
    assert events == []


@pytest.mark.parametrize(
    "code",
    [
        "await self.shell.read('large.txt')",
        "await self.shell.replace('large.txt', 'large', 'small')",
    ],
)
async def test_respects_configured_file_size_limit(tmp_path, observed_events, code):
    manager, events = observed_events
    (tmp_path / "large.txt").write_text("large contents")
    result = await _executor(code, tmp_path, manager, max_file_bytes=4).run_cell(code)
    assert isinstance(result.error, FileTooLargeError)
    assert result.error.path == "large.txt"
    assert (tmp_path / "large.txt").read_text() == "large contents"
    assert events == []


@pytest.mark.parametrize("read", ["whole", "range", "empty", "oversize"])
def test_bounded_readers_close_their_owned_descriptor(tmp_path, read):
    path = tmp_path / "note.txt"
    path.write_text("one\ntwo\n")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    if read == "whole":
        assert read_whole_file_checked(fd, 4096, hint="") == "one\ntwo\n"
    elif read == "oversize":
        with pytest.raises(FileTooLargeError):
            read_whole_file_checked(fd, 2, hint="")
    else:
        end = 1 if read == "range" else 0
        assert read_line_range(fd, 1, end, max_line_chars=4096) == (
            "one\n" if end else "",
            end,
        )
    with pytest.raises(OSError):
        os.fstat(fd)


async def test_owns_its_shell_and_captures_the_supplied_current_directory(
    tmp_path, observed_events, monkeypatch
):
    manager, _ = observed_events
    active = tmp_path / "active"
    active.mkdir()
    caller = ShellTools(cwd=str(active))
    caller_session = caller.session

    async def changed_method(*args, **kwargs):
        raise AssertionError("The caller's rebound shell must not be invoked")

    monkeypatch.setattr(caller, "write_file", changed_method)
    content = "__import__('pathlib').Path('unexpected.txt').write_text('not executed')"
    code = f"await self.shell.write_file('note.txt', {content!r})"
    executor = _executor(code, caller.cwd, manager)
    caller.cwd = tmp_path
    try:
        result = await executor.run_cell(code)
        assert result.success and (active / "note.txt").read_text() == content
        assert not (tmp_path / "note.txt").exists()
        assert not (active / "unexpected.txt").exists()
        assert caller.cwd == tmp_path
        assert caller.session is caller_session and not caller_session._started
        assert caller.write_file is changed_method
    finally:
        await ShellTools.close(caller)


@pytest.mark.parametrize("outcome", ["success", "error"])
async def test_opened_file_is_closed_on_every_execution_outcome(
    tmp_path, observed_events, monkeypatch, outcome
):
    manager, _ = observed_events
    (tmp_path / "note.txt").write_text("original")
    closed = []
    original_close = os.close

    def close(fd):
        original_close(fd)
        closed.append(fd)

    monkeypatch.setattr(os, "close", close)
    code = (
        "await self.shell.replace('note.txt', 'absent', 'safe')"
        if outcome == "error"
        else "await self.shell.write_file('note.txt', 'safe')"
    )
    result = await _executor(code, tmp_path, manager).run_cell(code)
    if outcome == "error":
        assert isinstance(result.error, ValueError)
    else:
        assert result.success
    assert len(closed) == 1
    with pytest.raises(OSError):
        os.fstat(closed[0])
    assert (tmp_path / "note.txt").read_text() == ("safe" if outcome == "success" else "original")


async def test_expired_timeout_does_not_begin_the_write(tmp_path, observed_events):
    manager, events = observed_events
    code = "await self.shell.write_file('note.txt', 'safe')"
    result = await _executor(code, tmp_path, manager, timeout=0).run_cell(code)
    assert isinstance(result.error, TimeoutError)
    assert not (tmp_path / "note.txt").exists()
    assert events == []


async def test_cancellation_before_file_execution_has_no_side_effects(
    tmp_path, observed_events, monkeypatch
):
    manager, events = observed_events
    started = asyncio.Event()

    async def paused_write(executor):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(FileEditExecutor, "_run_operation", paused_write)
    code = "await self.shell.write_file('note.txt', 'safe')"
    task = asyncio.create_task(_executor(code, tmp_path, manager, timeout=None).run_cell(code))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not (tmp_path / "note.txt").exists()
    assert events == []


@pytest.mark.parametrize(
    "code",
    [
        "await self.shell.write_file('note.txt', '" + "x" * (128 * 1024) + "')",
        "await self.shell.write_file('note.txt', '" + "汉" * (64 * 1024) + "')",
        "await self.shell.read(" + ",".join("1" for _ in range(5000)) + ")",
        "await self.shell.read(" + "(" * 5000 + "'note.txt'" + ")" * 5000 + ")",
        "await self.shell.write_file('note.txt', '\ud800')",
    ],
    ids=["source-bytes", "unicode-bytes", "many-nodes", "deep-nesting", "invalid-unicode"],
)
def test_unreviewable_source_is_rejected_without_side_effects(tmp_path, observed_events, code):
    manager, events = observed_events
    assert (
        file_edit_executor(
            code, cwd=tmp_path, event_manager=manager, max_file_bytes=4096, timeout=None
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []
    assert events == []


def _prepared(code, cwd, manager, *, workspace=None, host_path=None):
    return file_edit_executor(
        code,
        cwd=cwd,
        event_manager=manager,
        max_file_bytes=4096,
        timeout=1,
        host_path=host_path,
        workspace=workspace,
    )


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "sub/../../outside.txt",
        "{outside}",
        "//attacker/share/loot.txt",
        ".git/hooks/pre-commit",
        ".GIT/hooks/pre-commit",
        "sub/.git/config",
        ".aster/secrets.yaml",
        ".aster/libs/pkg/__init__.py",
        ".claude/commands/run.py",
        ".agents/skills/x.py",
        ".cursor/skills/x.py",
        ".git./hooks/pre-commit",
        ".git /hooks/pre-commit",
        ".git::$INDEX_ALLOCATION/hooks/pre-commit",
        "note.txt:stream",
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "await self.shell.read({path!r})",
        "await self.shell.write_file({path!r}, 'x')",
        "await self.shell.replace({path!r}, 'a', 'b')",
    ],
)
def test_targets_outside_the_workspace_or_in_protected_directories_need_approval(
    tmp_path, observed_events, path, template
):
    manager, events = observed_events
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    code = template.format(path=path.format(outside=outside.as_posix()))

    assert _prepared(code, workspace, manager) is None
    assert outside.read_text() == "secret"
    assert events == []


def test_workspace_is_the_boundary_after_the_shell_changes_directory(tmp_path, observed_events):
    manager, _ = observed_events
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    inside = "await self.shell.write_file('note.txt', 'x')"
    assert _prepared(inside, workspace / "sub", manager, workspace=workspace) is not None
    assert _prepared(inside, elsewhere, manager, workspace=workspace) is None
    up = "await self.shell.write_file('../note.txt', 'x')"
    assert _prepared(up, workspace / "sub", manager, workspace=workspace) is not None


def test_mapped_sandbox_path_is_judged_by_its_host_target(tmp_path, observed_events):
    manager, _ = observed_events
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = "await self.shell.read('/workspace/note.txt')"

    inside = _prepared(code, workspace, manager, host_path=lambda _: str(workspace / "note.txt"))
    assert inside is not None
    escaped = _prepared(code, workspace, manager, host_path=lambda _: str(tmp_path / "note.txt"))
    assert escaped is None


def test_link_leading_out_of_the_workspace_needs_approval(tmp_path, observed_events):
    manager, _ = observed_events
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "link"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(outside), str(link))  # needs no extra privilege
    else:
        link.symlink_to(outside, target_is_directory=True)

    assert _prepared("await self.shell.write_file('link/x.txt', 'x')", workspace, manager) is None
    assert list(outside.iterdir()) == []


def _link_directory(link, target):
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("protected", [".git", ".aster", ".agents", ".cursor", ".claude"])
@pytest.mark.parametrize("direct", [False, True])
def test_protected_directory_links_and_their_targets_need_approval(
    tmp_path, observed_events, protected, direct
):
    manager, events = observed_events
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text("original")
    _link_directory(tmp_path / protected, config)
    path = "config/settings.yaml" if direct else f"{protected}/settings.yaml"
    for code in (
        f"await self.shell.read({path!r})",
        f"await self.shell.write_file({path!r}, 'changed')",
        f"await self.shell.replace({path!r}, 'original', 'changed')",
    ):
        assert _prepared(code, tmp_path, manager) is None
    assert (config / "settings.yaml").read_text() == "original"
    assert events == []


async def test_protected_directory_link_created_after_preparation_blocks_edit(
    tmp_path, observed_events
):
    manager, events = observed_events
    config = tmp_path / "config"
    config.mkdir()
    target = config / "settings.yaml"
    target.write_text("original")
    code = "await self.shell.write_file('config/settings.yaml', 'changed')"
    executor = _executor(code, tmp_path, manager)
    _link_directory(tmp_path / ".aster", config)

    result = await executor.run_cell(code)

    assert isinstance(result.error, PermissionError)
    assert target.read_text() == "original"
    assert events == []


async def test_ordinary_internal_directory_link_still_allows_edit(tmp_path, observed_events):
    manager, _ = observed_events
    sources = tmp_path / "sources"
    sources.mkdir()
    _link_directory(tmp_path / "shortcut", sources)
    code = "await self.shell.write_file('shortcut/note.txt', 'safe')"

    result = await _executor(code, tmp_path, manager).run_cell(code)

    assert result.success, result.error
    assert (sources / "note.txt").read_text() == "safe"


def test_links_within_protected_libraries_protect_transitive_targets(tmp_path, observed_events):
    manager, _ = observed_events
    workspace = tmp_path / "workspace"
    (workspace / ".aster").mkdir(parents=True)
    libraries = workspace / "libraries"
    libraries.mkdir()
    sources = workspace / "sources"
    sources.mkdir()
    _link_directory(workspace / ".aster" / "libs", libraries)
    _link_directory(libraries / "pkg", sources)
    code = "await self.shell.write_file('__init__.py', 'changed')"

    assert _prepared(code, sources, manager, workspace=workspace) is None


def test_overlapping_grant_cannot_allow_another_roots_protected_target(tmp_path, observed_events):
    manager, _ = observed_events
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    _link_directory(external / ".aster", workspace)
    code = "await self.shell.write_file('note.txt', 'changed')"

    assert (
        file_edit_executor(
            code,
            cwd=workspace,
            event_manager=manager,
            max_file_bytes=4096,
            timeout=1,
            extra_roots=[(external, True)],
        )
        is None
    )


def test_protected_scan_falls_back_to_approval_when_bounded(tmp_path, observed_events, monkeypatch):
    manager, _ = observed_events
    (tmp_path / ".aster").mkdir()
    (tmp_path / ".aster" / "settings.yaml").write_text("original")
    monkeypatch.setattr(file_edits, "_MAX_PROTECTED_ENTRIES", 0)

    assert _prepared("await self.shell.write_file('note.txt', 'safe')", tmp_path, manager) is None


def test_unreadable_protected_tree_requires_approval(tmp_path, observed_events, monkeypatch):
    manager, _ = observed_events
    protected = tmp_path / ".aster"
    protected.mkdir()
    original = os.scandir

    def unreadable(path):
        if Path(path) == protected:
            raise PermissionError("Cannot inspect protected content")
        return original(path)

    monkeypatch.setattr(os, "scandir", unreadable)

    assert _prepared("await self.shell.write_file('note.txt', 'safe')", tmp_path, manager) is None


def test_operator_granted_directories_extend_the_boundary(tmp_path, observed_events):
    manager, _ = observed_events
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    writable = tmp_path / "writable"
    readonly = tmp_path / "readonly"
    writable.mkdir()
    readonly.mkdir()

    def prepared(code):
        return file_edit_executor(
            code,
            cwd=workspace,
            event_manager=manager,
            max_file_bytes=4096,
            timeout=1,
            extra_roots=[(writable, True), (readonly, False)],
        )

    def path(root, name="note.txt"):
        return (root / name).as_posix()

    assert prepared(f"await self.shell.write_file({path(writable)!r}, 'x')") is not None
    assert prepared(f"await self.shell.read({path(writable)!r})") is not None
    assert prepared(f"await self.shell.read({path(readonly)!r})") is not None
    assert prepared(f"await self.shell.write_file({path(readonly)!r}, 'x')") is None
    assert prepared(f"await self.shell.replace({path(readonly)!r}, 'a', 'b')") is None
    assert prepared(f"await self.shell.write_file({path(writable, '.git/hooks/x')!r}, 'x')") is None
    assert prepared(f"await self.shell.read({path(tmp_path)!r})") is None


@pytest.mark.parametrize("operation", ["read", "write_file", "replace"])
@pytest.mark.parametrize("protected", [False, True])
def test_hardlinked_targets_require_approval(tmp_path, observed_events, operation, protected):
    manager, events = observed_events
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = workspace / ".aster" / "secrets.yaml" if protected else tmp_path / "outside.txt"
    original.parent.mkdir(exist_ok=True)
    original.write_text("secret")
    (workspace / "note.txt").hardlink_to(original)
    code = {
        "read": "await self.shell.read('note.txt')",
        "write_file": "await self.shell.write_file('note.txt', 'changed')",
        "replace": "await self.shell.replace('note.txt', 'secret', 'changed')",
    }[operation]

    assert _prepared(code, workspace, manager) is None
    assert original.read_text() == "secret"
    assert events == []


@pytest.mark.parametrize("operation", ["read", "write_file", "replace"])
@pytest.mark.parametrize("when", ["before_open", "after_open"])
async def test_replaced_target_never_reads_or_changes_external_file(
    tmp_path, observed_events, monkeypatch, operation, when
):
    manager, events = observed_events
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "note.txt"
    target.write_text("original")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    code = {
        "read": "await self.shell.read('note.txt')",
        "write_file": "await self.shell.write_file('note.txt', 'changed')",
        "replace": "await self.shell.replace('note.txt', 'secret', 'changed')",
    }[operation]
    executor = _executor(code, workspace, manager)

    def replace_target():
        target.unlink()
        target.hardlink_to(outside)

    if when == "before_open":
        replace_target()
    else:
        real_open = os.open

        def open_then_replace(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if Path(path) == target:
                replace_target()
            return fd

        monkeypatch.setattr(os, "open", open_then_replace)

    result = await executor.run_cell(code)

    assert isinstance(result.error, PermissionError)
    assert not result.has_return
    assert outside.read_text() == "secret"
    assert events == []


async def test_new_file_created_after_preparation_requires_approval(tmp_path, observed_events):
    manager, events = observed_events
    code = "await self.shell.write_file('note.txt', 'changed')"
    executor = _executor(code, tmp_path, manager)
    (tmp_path / "note.txt").write_text("new writer's content")

    result = await executor.run_cell(code)

    assert isinstance(result.error, PermissionError)
    assert (tmp_path / "note.txt").read_text() == "new writer's content"
    assert events == []


def test_new_file_with_missing_parent_requires_approval(tmp_path, observed_events):
    manager, events = observed_events
    code = "await self.shell.write_file('missing/note.txt', 'changed')"
    assert _prepared(code, tmp_path, manager) is None
    assert not (tmp_path / "missing").exists()
    assert events == []


def test_write_without_read_permission_uses_normal_approval(tmp_path, observed_events, monkeypatch):
    manager, events = observed_events
    target = tmp_path / "note.txt"
    target.write_text("original")
    real_access = os.access

    def access(path, mode):
        return False if Path(path) == target and mode == os.R_OK else real_access(path, mode)

    monkeypatch.setattr(os, "access", access)

    assert (
        _prepared("await self.shell.write_file('note.txt', 'changed')", tmp_path, manager) is None
    )
    assert target.read_text() == "original"
    assert events == []


async def test_parent_replaced_while_creating_file_does_not_write_content(
    tmp_path, observed_events, monkeypatch
):
    manager, events = observed_events
    workspace = tmp_path / "workspace"
    parent = workspace / "files"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    code = "await self.shell.write_file('files/note.txt', 'private content')"
    executor = _executor(code, workspace, manager)
    real_open = os.open

    def redirect_then_open(path, flags, *args, **kwargs):
        if Path(path) == parent / "note.txt":
            parent.rmdir()
            if sys.platform == "win32":
                import _winapi

                _winapi.CreateJunction(str(outside), str(parent))
            else:
                parent.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", redirect_then_open)
    result = await executor.run_cell(code)

    assert isinstance(result.error, PermissionError)
    assert not result.has_return
    assert (outside / "note.txt").read_bytes() == b""
    assert events == []


async def test_unavailable_opened_path_does_not_read_or_write(
    tmp_path, observed_events, monkeypatch
):
    manager, events = observed_events
    (tmp_path / "note.txt").write_text("original")
    code = "await self.shell.write_file('note.txt', 'changed')"

    def unavailable(fd):
        raise OSError("handle path unavailable")

    monkeypatch.setattr(file_edits, "_opened_file_path", unavailable)
    result = await _executor(code, tmp_path, manager).run_cell(code)

    assert isinstance(result.error, PermissionError)
    assert (tmp_path / "note.txt").read_text() == "original"
    assert events == []


async def test_refused_creation_never_unlinks_another_writers_replacement(
    tmp_path, observed_events, monkeypatch
):
    manager, events = observed_events
    target = tmp_path / "note.txt"
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("other writer's content")
    code = "await self.shell.write_file('note.txt', 'private content')"
    real_close = os.close

    def unavailable(fd):
        raise OSError("handle path unavailable")

    def close_then_replace(fd):
        real_close(fd)
        replacement.replace(target)

    monkeypatch.setattr(file_edits, "_opened_file_path", unavailable)
    monkeypatch.setattr(os, "close", close_then_replace)
    result = await _executor(code, tmp_path, manager).run_cell(code)

    assert isinstance(result.error, PermissionError)
    assert "No content was written" in str(result.error)
    assert target.read_text() == "other writer's content"
    assert events == []


@pytest.mark.parametrize(
    ("original", "method", "arguments"),
    [
        (None, "write_file", ("new\ncontent\n",)),
        ("one\ntwo\n", "write_file", ("short\n",)),
        ("", "write_file", ("new\vline",)),
        ("one\ntwo\n", "replace", ("two", "three")),
        pytest.param(
            "line\n" * 14000 + "last\n", "replace", ("last", "changed"), id="large-replace"
        ),
        ("", "replace", ("", "new")),
        ("one\ntwo\n", "read", ()),
        ("one\ntwo\n", "read", ((-1, 1),)),
        ("one\ntwo\n", "read", ((1, 0),)),
        ("one\ntwo\n", "read", ((5, 9),)),
        ("one\vtwo\nthree", "read", ((2, 3),)),
    ],
)
async def test_file_results_and_activity_match_normal_tools(
    tmp_path, observed_events, original, method, arguments
):
    manager, events = observed_events
    target = tmp_path / "note.txt"
    if original is not None:
        target.write_text(original)
    shell = ShellTools(cwd=str(tmp_path))
    activity = ActivityShellTools(shell, manager)
    try:
        expected = await getattr(activity, method)("note.txt", *arguments)
    finally:
        await shell.close()
    expected_events = [event for event in events if isinstance(event, FileEdit)]
    expected_content = target.read_bytes()
    if original is None:
        target.unlink()
    else:
        target.write_text(original)
    events.clear()
    params = ", ".join(repr(value) for value in ("note.txt", *arguments))
    code = f"await self.shell.{method}({params})"

    result = await _executor(code, tmp_path, manager, max_file_bytes=100_000).run_cell(code)

    assert result.success, result.error
    assert vars(result.returned_value) == vars(expected)
    assert target.read_bytes() == expected_content
    fields = (
        "path",
        "operation",
        "old_text",
        "new_text",
        "start_line",
        "end_line",
        "diff",
        "content_complete",
        "diff_complete",
    )
    assert [tuple(getattr(event, field) for field in fields) for event in events] == [
        tuple(getattr(event, field) for field in fields) for event in expected_events
    ]
