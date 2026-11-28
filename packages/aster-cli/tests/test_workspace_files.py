# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only inspection covers real files and OS path boundaries."""

import asyncio
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from aster_cli.tools import workspace_files
from aster_cli.tools.workspace_files import WorkspaceFiles


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as error:
        pytest.skip(f"Symbolic links unavailable: {error}")


async def test_utf8_read_listing_and_literal_search_preserve_files(tmp_path):
    source = tmp_path / "src" / "例子.py"
    source.parent.mkdir()
    original = "# 中文 🙂\r\ndef needle():\r\n    return 'a.*b'\r\n".encode()
    source.write_bytes(original)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("a.*b", encoding="utf-8")
    reader = WorkspaceFiles(tmp_path)

    listing = await reader.list("src")
    assert listing["entries"] == [{"name": "例子.py", "kind": "file"}]
    assert not listing["truncated"]
    result = await reader.read("src/例子.py", start_line=2, end_line=3)
    assert result["text"] == "def needle():\r\n    return 'a.*b'\r\n"
    assert (result["start_line"], result["end_line"], result["truncated"]) == (2, 3, False)
    assert (await reader.read(str(source), 1, 1))["text"] == "# 中文 🙂\r\n"
    found = await reader.search("a.*b", glob="*.py")
    assert found["matches"] == [{"path": "src/例子.py", "line": 3, "text": "    return 'a.*b'"}]
    assert not found["truncated"]
    assert not (await reader.search("NEEDLE"))["matches"]
    assert source.read_bytes() == original


@pytest.mark.parametrize("method", ["read", "list", "search"])
@pytest.mark.parametrize(
    "spelling", ["../outside", "absolute", "child/../../outside", "name:stream", "CON"]
)
async def test_rejects_paths_outside_the_workspace_and_special_spellings(
    tmp_path, method, spelling
):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside canary", encoding="utf-8")
    path = str(outside) if spelling == "absolute" else spelling
    reader = WorkspaceFiles(root)
    with pytest.raises(PermissionError):
        if method == "search":
            await reader.search("canary", path=path)
        else:
            await getattr(reader, method)(path)


async def test_links_and_hardlinks_cannot_reveal_an_outside_canary(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("outside canary", encoding="utf-8")
    _symlink(root / "file-link", secret)
    _symlink(root / "directory-link", outside)
    os.link(secret, root / "hard-link")
    reader = WorkspaceFiles(root)

    for path in ("file-link", "hard-link", "directory-link/secret.txt"):
        with pytest.raises(OSError):
            await reader.read(path)
    with pytest.raises(OSError):
        await reader.list("directory-link")
    result = await reader.search("canary")
    assert result["matches"] == []
    assert result["skipped"] == 3
    assert "outside canary" not in repr(await reader.list())


async def test_hardlinked_files_are_refused_without_symlink_privileges(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("hardlink canary", encoding="utf-8")
    os.link(outside, root / "linked.txt")
    reader = WorkspaceFiles(root)
    with pytest.raises(PermissionError):
        await reader.read("linked.txt")
    result = await reader.search("canary")
    assert result["matches"] == [] and result["skipped"] == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows junction boundary")
async def test_windows_junctions_are_not_traversed(tmp_path):
    import _winapi

    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("junction canary", encoding="utf-8")
    _winapi.CreateJunction(str(outside), str(root / "junction"))
    reader = WorkspaceFiles(root)
    assert (await reader.list())["entries"] == [{"name": "junction", "kind": "link"}]
    with pytest.raises(OSError):
        await reader.read("junction/secret.txt")
    with pytest.raises(OSError):
        await reader.list("junction")
    assert (await reader.search("canary"))["matches"] == []


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="POSIX special files")
async def test_fifo_is_refused_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    reader = WorkspaceFiles(tmp_path)
    with pytest.raises(PermissionError):
        await reader.read("pipe")
    assert (await reader.search("anything"))["skipped"] == 1


async def test_replaced_workspace_root_is_not_silently_reauthorized(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    reader = WorkspaceFiles(root)
    root.rename(tmp_path / "original")
    root.mkdir()
    (root / "secret.txt").write_text("replacement canary", encoding="utf-8")
    with pytest.raises(PermissionError):
        await reader.read("secret.txt")
    with pytest.raises(PermissionError):
        await reader.list()


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="POSIX directory rename")
@pytest.mark.parametrize("operation", ["read", "list"])
async def test_directory_replaced_after_open_cannot_redirect_inspection(
    tmp_path, monkeypatch, operation
):
    root = tmp_path / "workspace"
    directory = root / "files"
    directory.mkdir(parents=True)
    (directory / "note.txt").write_text("original", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.txt").write_text("outside canary", encoding="utf-8")
    moved = tmp_path / "moved"
    reader = WorkspaceFiles(root)
    real_open = os.open
    opened = []

    def open_then_replace(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened.append(fd)
        if os.fspath(path) == "files" and kwargs.get("dir_fd") is not None:
            directory.rename(moved)
            directory.symlink_to(outside, target_is_directory=True)
        return fd

    monkeypatch.setattr(os, "open", open_then_replace)
    with pytest.raises(PermissionError):
        if operation == "read":
            await reader.read("files/note.txt")
        else:
            await reader.list("files")

    assert moved.is_dir()
    assert (outside / "note.txt").read_text(encoding="utf-8") == "outside canary"
    assert (moved / "note.txt").read_text(encoding="utf-8") == "original"
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_opened_handle_is_checked_before_reading(tmp_path, monkeypatch):
    from aster_cli.coding import instructions

    (tmp_path / "data.txt").write_text("must not be read", encoding="utf-8")
    reader = WorkspaceFiles(tmp_path)
    opened_path = instructions._opened_file_path

    def escaped(fd):
        value = opened_path(fd)
        return tmp_path.parent / "outside.txt" if value.name == "data.txt" else value

    monkeypatch.setattr(instructions, "_opened_file_path", escaped)
    with pytest.raises(PermissionError):
        await reader.read("data.txt")


async def test_results_and_search_work_are_bounded_and_report_incompleteness(tmp_path, monkeypatch):
    (tmp_path / "many.txt").write_text("needle\n" * 20, encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"needle\0binary")
    (tmp_path / "invalid.txt").write_bytes(b"\xff")
    reader = WorkspaceFiles(tmp_path)
    assert (await reader.list(limit=1))["truncated"]
    limited = await reader.search("needle", limit=1)
    assert len(limited["matches"]) == 1 and limited["truncated"]
    monkeypatch.setattr(workspace_files, "_OUTPUT_CHARS", 10)
    read = await reader.read("many.txt")
    assert len(read["text"]) == 10 and read["truncated"]
    monkeypatch.setattr(workspace_files, "_OUTPUT_CHARS", 64_000)
    monkeypatch.setattr(workspace_files, "_FILE_BYTES", 64)
    with pytest.raises(ValueError, match="limited"):
        await reader.read("many.txt")
    incomplete = await reader.search("needle")
    assert incomplete["matches"] == [] and incomplete["skipped"] == 3
    monkeypatch.setattr(workspace_files, "_SEARCH_ENTRIES", 1)
    assert (await reader.search("needle"))["truncated"]


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("read", {"path": "file.txt", "start_line": True}),
        ("read", {"path": "file.txt", "start_line": 0}),
        ("read", {"path": "file.txt", "end_line": 501}),
        ("read", {"path": "file.txt", "start_line": 3, "end_line": 2}),
        ("list", {"limit": 501}),
        ("search", {"query": ""}),
        ("search", {"query": "a\nb"}),
        ("search", {"query": "a", "limit": True}),
        ("search", {"query": "a", "glob": "*" * 257}),
    ],
)
async def test_invalid_limits_and_queries_are_rejected(tmp_path, method, arguments):
    with pytest.raises(ValueError):
        await getattr(WorkspaceFiles(tmp_path), method)(**arguments)


async def test_search_skips_an_inaccessible_child_but_refuses_an_inaccessible_root(
    tmp_path, monkeypatch
):
    (tmp_path / "denied").mkdir()
    (tmp_path / "visible.txt").write_text("needle", encoding="utf-8")
    reader = WorkspaceFiles(tmp_path)
    entries = reader._entries

    @contextmanager
    def deny_child(relative):
        if relative.name == "denied":
            raise PermissionError("synthetic unreadable directory")
        with entries(relative) as value:
            yield value

    monkeypatch.setattr(reader, "_entries", deny_child)
    result = await reader.search("needle")
    assert [item["path"] for item in result["matches"]] == ["visible.txt"]
    assert result["skipped"] == 1
    with pytest.raises(PermissionError):
        await reader.search("needle", path="denied")


async def test_cancelled_search_releases_directory_handles(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    for index in range(20):
        (root / f"{index}.txt").write_text("needle", encoding="utf-8")
    reader = WorkspaceFiles(root)
    started = asyncio.Event()
    read_bytes = reader._bytes

    def observe(*args):
        started.set()
        return read_bytes(*args)

    monkeypatch.setattr(reader, "_bytes", observe)
    task = asyncio.create_task(reader.search("needle"))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # On Windows an unreleased directory handle would prevent this rename.
    root.rename(tmp_path / "closed")
