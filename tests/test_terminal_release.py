# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distribution invariants and corruption checks for the Windows terminal bundle."""

from __future__ import annotations

import hashlib
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
from scripts import terminal_release_smoke as smoke


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
    requirements=(),
    extras=(),
):
    normalized = name.replace("-", "_")
    tag = tag or ("py3-none-win_amd64" if name == "aster-cli" else "py3-none-any")
    path = directory / f"{normalized}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        prefix = f"{normalized}-{version}.dist-info"
        archive.writestr(
            f"{prefix}/METADATA",
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
            + "".join(f"Provides-Extra: {extra}\n" for extra in extras)
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements),
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


def _installed_package(version="1.2.3"):
    return {
        "versions": dict.fromkeys(release.PACKAGES, version),
        "native_sha256": hashlib.sha256(b"MZfake-test-terminal").hexdigest(),
        "bundled_terminal": True,
    }


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


@pytest.mark.skipif(
    sys.platform != "win32" or not shutil.which("uv"), reason="Windows uv installer"
)
@pytest.mark.parametrize("explicit_python", [False, True], ids=["reuse-python", "explicit-python"])
def test_installer_preserves_sandbox_with_new_dependency_bounds(bundle, tmp_path, explicit_python):
    _wheel(
        bundle / "wheels",
        "aster",
        extras=("sandbox", "arc"),
        requirements=(
            "cloudpickle>=2,<3; extra == 'sandbox'",
            "microsandbox>=1,<3; extra == 'sandbox'",
            "cloudpickle>=5; python_version < '3.12' and extra == 'sandbox'",
            "microsandbox>=5; sys_platform == 'not-windows' and extra == 'sandbox'",
            "rich>=1; extra == 'arc'",
            "arc-agi>=1; extra == 'arc'",
        ),
    )
    release.assemble_bundle(bundle, commit="a" * 40, dirty=False)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    old_dependencies = [
        _wheel(dependencies, name)
        for name in ("cloudpickle", "microsandbox", "unrelated-probe", "rich")
    ]
    _wheel(dependencies, "cloudpickle", version="2.0.0")
    tools, binaries = tmp_path / "tools", tmp_path / "bin"
    environment = {
        **os.environ,
        "UV_TOOL_DIR": str(tools),
        "UV_TOOL_BIN_DIR": str(binaries),
        "UV_CACHE_DIR": str(tmp_path / "cache"),
        "UV_FIND_LINKS": str(dependencies),
        "UV_NO_INDEX": "1",
        "UV_PYTHON_DOWNLOADS": "never",
    }
    uv = shutil.which("uv")
    seed = [
        uv,
        "tool",
        "install",
        "--no-config",
        "--offline",
        "--python",
        sys.executable,
        str(next((bundle / "wheels").glob("aster_cli-*.whl"))),
    ]
    for dependency in old_dependencies:
        seed += ["--with", str(dependency)]
    result = subprocess.run(seed, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    python = tools / "aster-cli/Scripts/python.exe"
    inspect = (
        "import importlib.metadata as m,json,sys; "
        "print(json.dumps({'python':sys._base_executable, "
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    )
    before = json.loads(subprocess.check_output([str(python), "-I", "-c", inspect], text=True))
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(bundle / "install.ps1"),
        "-ToolsDirectory",
        str(tools),
        "-BinDirectory",
        str(binaries),
        "-Offline",
    ]
    if explicit_python:
        command += ["-Python", sys.executable]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    after = json.loads(subprocess.check_output([str(python), "-I", "-c", inspect], text=True))
    assert Path(after["python"]) == Path(before["python"])
    assert after["packages"]["cloudpickle"] == "2.0.0"
    assert after["packages"]["microsandbox"] == "1.2.3"
    assert "unrelated-probe" not in after["packages"]
    assert "arc-agi" not in after["packages"]
    assert set(release.PACKAGES) <= after["packages"].keys()


def test_installed_smoke_binds_versions_and_native_bytes_to_clean_bundle(bundle):
    release.assemble_bundle(bundle, commit="a" * 40, dirty=False)
    manifest_path = bundle / "MANIFEST.json"
    package = _installed_package()

    bound = smoke.bind_bundle(package, manifest_path)

    assert bound == {
        **package,
        "bundle_commit": "a" * 40,
        "bundle_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    assert "bundle_commit" not in package


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("dirty", "clean terminal bundle"),
        ("installed-version", "clean terminal bundle"),
        ("missing-installed-package", "clean terminal bundle"),
        ("missing-bundle-package", "clean terminal bundle"),
        ("platform", "clean terminal bundle"),
        ("replaced-cli-wheel", "CLI wheel hash does not match"),
        ("native", "does not match the bundled executable"),
    ],
)
def test_installed_smoke_rejects_bundle_mismatch(bundle, failure, message):
    release.assemble_bundle(bundle, commit="a" * 40, dirty=False)
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = _installed_package()
    if failure == "dirty":
        manifest["dirty"] = True
    elif failure == "installed-version":
        package["versions"]["aster-memory"] = "1.2.2"
    elif failure == "missing-installed-package":
        del package["versions"]["aster-bench"]
    elif failure == "missing-bundle-package":
        del manifest["packages"]["aster-acp"]
    elif failure == "platform":
        manifest["platform"] = "win_arm64"
    elif failure == "replaced-cli-wheel":
        _wheel(bundle / "wheels", "aster-cli", terminal=b"MZreplacement-terminal")
    elif failure == "native":
        package["native_sha256"] = hashlib.sha256(b"MZdifferent-installed-terminal").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        smoke.bind_bundle(package, manifest_path)


@pytest.mark.parametrize(
    ("before", "after"),
    [("1.2.3", "1.2.3"), ("1.2.3", "1.2.2"), ("1.10.0", "1.9.0")],
    ids=["reinstall", "downgrade", "numeric-downgrade"],
)
def test_upgrade_evidence_rejects_reinstall_and_downgrade(before, after):
    with pytest.raises(ValueError, match="older version; reinstall is insufficient"):
        smoke.require_upgrade(_installed_package(before), _installed_package(after))


@pytest.mark.parametrize("phase", ["seed", "installed"])
@pytest.mark.parametrize("failure", ["missing-package", "mixed-versions"])
def test_upgrade_evidence_requires_all_five_packages_at_one_version(phase, failure):
    seed, package = _installed_package("1.2.2"), _installed_package()
    versions = (seed if phase == "seed" else package)["versions"]
    if failure == "missing-package":
        del versions["aster-acp"]
        message = "all five package versions"
    else:
        versions["aster-acp"] = "1.1.0"
        message = "older version; reinstall is insufficient"

    with pytest.raises(ValueError, match=message):
        smoke.require_upgrade(seed, package)


def test_upgrade_evidence_compares_versions_numerically():
    smoke.require_upgrade(_installed_package("1.9.0"), _installed_package("1.10.0"))


async def test_upgrade_smoke_preserves_saved_input_and_continues_without_replaying(
    tmp_path, monkeypatch
):
    root = tmp_path / "smoke"
    (root / "user").mkdir(parents=True)
    for name in tuple(os.environ):
        if name.startswith("ASTER_"):
            monkeypatch.delenv(name)
    for name, value in {
        "ASTER_USER_DIR": str(root / "user"),
        "ASTER_PROJECT_DIR": str(root / "workspace" / ".aster"),
        "ASTER_TRUST_ALL_WORKSPACES": "1",
        "ASTER_MODEL_METADATA_REFRESH": "False",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    seed_package, package = _installed_package("1.2.2"), _installed_package()

    seed = await smoke.session_check(root, "seed", seed_package)
    assert seed["session_seeded"] is True
    state = json.loads((root / "smoke-state.json").read_text(encoding="utf-8"))
    preserved = {relative: (root / relative).read_bytes() for relative in state["preserved_files"]}
    continued = root / "workspace" / "continued.txt"

    with pytest.raises(ValueError, match="older version; reinstall is insufficient"):
        await smoke.session_check(root, "verify", seed_package, verify_upgrade=True)
    assert not continued.exists()

    report = await smoke.session_check(root, "verify", package, verify_upgrade=True)

    assert report["versions"] == package["versions"]
    assert report["seed_package"] == seed_package
    assert report["seed_versions"] == seed_package["versions"]
    for flag in (
        "upgrade_verified",
        "cross_version",
        "session_reopened_without_execution",
        "continued_after_install",
        "saved_input_bytes_preserved",
    ):
        assert report[flag] is True, flag
    assert continued.read_text(encoding="utf-8") == "continued\n"
    assert (root / "workspace" / "result.txt").read_text(encoding="utf-8") == "user changes\n"
    for relative, before in preserved.items():
        assert (root / relative).read_bytes() == before, relative
