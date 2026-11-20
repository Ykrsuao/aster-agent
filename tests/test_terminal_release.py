# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distribution invariants and corruption checks for the Windows terminal bundle."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tomllib
import types
import zipfile
from pathlib import Path

import pytest

from scripts import build_terminal_release as release


def _wheel(
    directory,
    name,
    *,
    version="1.2.3",
    tag=None,
    duplicate_metadata=False,
    purelib="false",
    wheel_tags=None,
    terminal=b"MZfake-test-terminal",
):
    normalized = name.replace("-", "_")
    tag = tag or ("py3-none-win_amd64" if name == "aster-cli" else "py3-none-any")
    path = directory / f"{normalized}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        prefix = f"{normalized}-{version}.dist-info"
        archive.writestr(
            f"{prefix}/METADATA", f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
        )
        if duplicate_metadata:
            archive.writestr(
                f"{normalized}_extra-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n",
            )
        archive.writestr(
            f"{prefix}/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: {purelib}\n"
            + "".join(
                f"Tag: {value}\n" for value in (wheel_tags if wheel_tags is not None else [tag])
            ),
        )
        if name in ("aster-cli", "aster-acp"):
            command = "aster" if name == "aster-cli" else name
            archive.writestr(f"{normalized}/__init__.py", "def main():\n    print('fixture')\n")
            archive.writestr(
                f"{prefix}/entry_points.txt", f"[console_scripts]\n{command} = {normalized}:main\n"
            )
        if name == "aster-cli":
            archive.writestr("aster_cli/bin/aster.exe", terminal)
        archive.writestr(
            f"{prefix}/RECORD",
            "".join(f"{item},,\n" for item in [*archive.namelist(), f"{prefix}/RECORD"]),
        )
    return path


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "wheels").mkdir()
    for package in release.PACKAGES:
        _wheel(tmp_path / "wheels", package)
    (tmp_path / "constraints.txt").write_text("click==8.2.1\n", encoding="utf-8")
    return tmp_path


def test_platform_hook_rejects_wrong_architecture_and_never_uses_universal_tag(
    tmp_path, monkeypatch
):
    interface = types.ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = object
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    spec = importlib.util.spec_from_file_location(
        "aster_cli_build_hook", release.ROOT / "packages/aster-cli/hatch_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hook = module.CustomBuildHook()
    monkeypatch.delenv("ASTER_BUNDLE_TUI", raising=False)
    build_data = {"force_include": {}}
    hook.initialize("standard", build_data)
    assert build_data == {"force_include": {}}

    binary = tmp_path / "aster.exe"
    header = bytearray(64)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 60, 64)
    binary.write_bytes(header + b"PE\0\0\x64\x86")
    monkeypatch.setenv("ASTER_BUNDLE_TUI", str(binary))
    hook.initialize("standard", build_data)
    assert build_data["tag"] == "py3-none-win_amd64"
    assert build_data["pure_python"] is False
    assert build_data["force_include"] == {str(binary.resolve()): "aster_cli/bin/aster.exe"}
    with pytest.raises(ValueError, match="editable"):
        hook.initialize("editable", {})
    binary.write_bytes(header + b"PE\0\0\x4c\x01")
    with pytest.raises(ValueError, match="Windows x64"):
        hook.initialize("standard", {"force_include": {}})

    config = tomllib.loads(
        (release.ROOT / "packages/aster-cli/pyproject.toml").read_text(encoding="utf-8")
    )
    assert "src/aster_cli/bin/**" in config["tool"]["hatch"]["build"]["targets"]["wheel"]["exclude"]


def test_bundle_records_all_hashes_version_and_dirty_provenance(bundle):
    archive = release.assemble_bundle(bundle, commit="a" * 40, dirty=True)
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.2.3"
    assert manifest["commit"] == "a" * 40
    assert manifest["dirty"] is True
    assert set(manifest["packages"]) == set(release.PACKAGES)
    for name, digest in manifest["files"].items():
        assert release.sha256(bundle / name) == digest
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == set(manifest["files"]) | {"MANIFEST.json", "SHA256SUMS"}
    assert not any(
        Path(name).is_absolute() or ".." in Path(name).parts for name in manifest["files"]
    )


@pytest.mark.parametrize("failure", ["mismatched_version", "universal_binary", "missing_package"])
def test_bundle_refuses_incompatible_or_incomplete_packages(bundle, failure):
    wheels = bundle / "wheels"
    existing = next(wheels.glob("aster_cli-*.whl"))
    existing.unlink()
    if failure == "mismatched_version":
        _wheel(wheels, "aster-cli", version="2.0.0")
    elif failure == "universal_binary":
        _wheel(wheels, "aster-cli", tag="py3-none-any")
    with pytest.raises(ValueError):
        release.inspect_wheels(wheels)


def test_inspect_wheels_accepts_valid_bundle(bundle):
    version, packages = release.inspect_wheels(bundle / "wheels")
    assert version == "1.2.3"
    assert packages == {
        name: f"{name.replace('-', '_')}-1.2.3-"
        f"{'py3-none-win_amd64' if name == 'aster-cli' else 'py3-none-any'}.whl"
        for name in release.PACKAGES
    }


@pytest.mark.parametrize("package", release.PACKAGES)
def test_inspect_wheels_rejects_multiple_metadata_files(bundle, package):
    wheels = bundle / "wheels"
    _wheel(wheels, package, duplicate_metadata=True)
    with pytest.raises(ValueError, match="expected one distribution metadata file"):
        release.inspect_wheels(wheels)


@pytest.mark.parametrize(
    ("purelib", "wheel_tags"),
    [
        pytest.param("true", ["py3-none-win_amd64"], id="pure-python"),
        pytest.param("false", ["py3-none-any"], id="universal-tag"),
        pytest.param("false", ["py3-none-win32"], id="win32-tag"),
        pytest.param("false", ["py3-none-win_arm64"], id="arm64-tag"),
        pytest.param("false", ["py3-none-win_amd64", "py3-none-any"], id="extra-tag"),
        pytest.param("false", ["py3-none-win_amd64"] * 2, id="duplicate-tag"),
    ],
)
def test_inspect_wheels_rejects_invalid_internal_cli_declarations(bundle, purelib, wheel_tags):
    wheels = bundle / "wheels"
    path = _wheel(wheels, "aster-cli", purelib=purelib, wheel_tags=wheel_tags)
    assert path.name == "aster_cli-1.2.3-py3-none-win_amd64.whl"
    with pytest.raises(ValueError, match="The bundled CLI must be a Windows platform wheel"):
        release.inspect_wheels(wheels)


@pytest.mark.parametrize(
    "terminal",
    [
        pytest.param(b"not-a-native-terminal", id="non-native"),
        pytest.param(b"", id="empty"),
        pytest.param(b"prefix-MZfake-test-terminal", id="non-leading-mz"),
    ],
)
def test_inspect_wheels_rejects_cli_terminal_without_mz_signature(bundle, terminal):
    wheels = bundle / "wheels"
    _wheel(wheels, "aster-cli", terminal=terminal)
    with pytest.raises(ValueError, match="The bundled CLI is missing its native terminal"):
        release.inspect_wheels(wheels)


@pytest.mark.skipif(sys.platform != "win32", reason="Native PowerShell installer")
@pytest.mark.parametrize("failure", ["corrupt_wheel", "missing_hash", "path_traversal"])
def test_installer_rejects_corruption_before_calling_uv(bundle, failure):
    release.assemble_bundle(bundle, commit="a" * 40, dirty=False)
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if failure == "corrupt_wheel":
        next((bundle / "wheels").glob("aster_cli-*.whl")).write_bytes(b"corrupt")
    elif failure == "missing_hash":
        del manifest["files"]["constraints.txt"]
    else:
        manifest["packages"]["aster-cli"] = "../../unexpected.whl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    environment = os.environ.copy()
    system = Path(os.environ["SystemRoot"]) / "System32"
    environment["PATH"] = os.pathsep.join((str(system), str(system / "WindowsPowerShell/v1.0")))
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(bundle / "install.ps1"),
            "-Offline",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert (
        "checksum mismatch" in result.stderr
        or "bundle file list" in result.stderr
        or "package version or platform" in result.stderr
    )


@pytest.mark.skipif(
    sys.platform != "win32" or not shutil.which("uv"), reason="Windows uv installer"
)
def test_installer_preserves_foreign_launcher_and_reinstalls_its_own_tool(bundle, tmp_path):
    release.assemble_bundle(bundle, commit="a" * 40, dirty=False)
    foreign_bin = tmp_path / "foreign-bin"
    foreign_bin.mkdir()
    foreign = foreign_bin / "aster.exe"
    foreign.write_bytes(b"user-managed-launcher")

    def install(tools, binaries):
        return subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(bundle / "install.ps1"),
                "-Python",
                sys.executable,
                "-ToolsDirectory",
                str(tools),
                "-BinDirectory",
                str(binaries),
                "-Offline",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    blocked = install(tmp_path / "foreign-tools", foreign_bin)
    assert blocked.returncode != 0
    assert "Executable already exists" in blocked.stderr
    assert foreign.read_bytes() == b"user-managed-launcher"

    for _ in range(2):
        result = install(tmp_path / "owned-tools", tmp_path / "owned-bin")
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "owned-bin/aster.exe").is_file()
        assert (tmp_path / "owned-bin/aster-acp.exe").is_file()
