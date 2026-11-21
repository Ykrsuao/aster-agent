# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve host file targets through the sandbox's public path interface."""

from __future__ import annotations

import builtins
from pathlib import PureWindowsPath
from types import SimpleNamespace

import pytest

from aster.runtime.sandbox import microsandbox_executor as backend
from aster.runtime.sandbox.config import FileRule, SandboxConfig
from aster.runtime.sandbox.executor import SandboxedExecutor


def test_guest_workspace_file_resolves_before_vm_start(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "sys", SimpleNamespace(platform="win32"))
    executor = SandboxedExecutor(object(), SandboxConfig(workspace=str(tmp_path)), cell_timeout=5)

    assert PureWindowsPath(executor.host_path("/workspace/src/main.py")) == (
        PureWindowsPath(str(tmp_path)) / "src" / "main.py"
    )


@pytest.fixture
def windows_executor(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "sys", SimpleNamespace(platform="win32"))
    return SandboxedExecutor(
        object(),
        SandboxConfig(
            workspace=str(tmp_path / "work"),
            allow=(
                FileRule(path=str(tmp_path / "reference")),
                FileRule(path=str(tmp_path / "scratch"), access="read_write"),
            ),
        ),
        cell_timeout=5,
    )


@pytest.mark.parametrize(
    "guest, host_directory",
    [("/allowed/0/docs/readme.md", "reference"), ("/allowed/1/docs/readme.md", "scratch")],
)
def test_explicit_guest_mounts_resolve_to_their_host_directories(
    windows_executor, tmp_path, guest, host_directory
):
    assert PureWindowsPath(windows_executor.host_path(guest)) == (
        PureWindowsPath(str(tmp_path)) / host_directory / "docs" / "readme.md"
    )


@pytest.mark.parametrize(
    "path",
    [
        "docs/readme.md",
        "./docs/readme.md",
        "../readme.md",
        r"C:\project\readme.md",
        "D:/project/readme.md",
        r"\\server\share\readme.md",
    ],
)
def test_relative_and_host_absolute_paths_remain_unchanged(windows_executor, path):
    assert windows_executor.host_path(path) == path


@pytest.mark.parametrize(
    "path",
    [
        "/workspace-other/readme.md",
        "/workspaces/readme.md",
        "/allowed/0-other/readme.md",
        "/allowed/01/readme.md",
        "/allowed/10/readme.md",
        "/unmapped/readme.md",
    ],
)
def test_similar_guest_prefixes_do_not_match_a_mount(windows_executor, path):
    assert windows_executor.host_path(path) == path


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_non_windows_hosts_keep_existing_path_spellings(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(backend, "sys", SimpleNamespace(platform=platform))
    executor = SandboxedExecutor(
        object(),
        SandboxConfig(
            workspace=str(tmp_path),
            allow=(FileRule(path=str(tmp_path / "reference")),),
        ),
        cell_timeout=5,
    )

    for path in (
        str(tmp_path / "src" / "main.py"),
        str(tmp_path / "reference" / "readme.md"),
        "src/main.py",
        "/workspace/readme.md",
        "/allowed/0/readme.md",
    ):
        assert executor.host_path(path) == path


def test_path_resolution_does_not_import_sdk_or_require_existing_mounts(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        if name == "microsandbox" or name.startswith("microsandbox."):
            pytest.fail("Resolving a host path must not import the VM SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    monkeypatch.setattr(backend, "sys", SimpleNamespace(platform="win32"))
    missing = tmp_path / "not-created"
    executor = SandboxedExecutor(
        object(),
        SandboxConfig(allow=(FileRule(path=str(missing)),)),
        cell_timeout=5,
    )

    assert PureWindowsPath(executor.host_path("/allowed/0/readme.md")) == (
        PureWindowsPath(str(missing)) / "readme.md"
    )
    assert not missing.exists()
