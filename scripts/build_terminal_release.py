# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a Windows terminal bundle. Run with uv run python; nothing is published."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("aster", "aster-cli", "aster-acp", "aster-memory", "aster-bench")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inspect_wheels(directory: Path) -> tuple[str, dict[str, str]]:
    packages = {}
    versions = set()
    for path in sorted(directory.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            metadata_paths = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"{path.name}: expected one distribution metadata file")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            name, version = metadata["Name"], metadata["Version"]
            if name not in PACKAGES or name in packages:
                raise ValueError(f"Unexpected or duplicate distribution: {name}")
            tag = "py3-none-win_amd64" if name == "aster-cli" else "py3-none-any"
            if path.name != f"{name.replace('-', '_')}-{version}-{tag}.whl":
                raise ValueError(f"{path.name}: unexpected version or platform tag")
            if name == "aster-cli":
                wheel = BytesParser().parsebytes(
                    archive.read(metadata_paths[0].replace("METADATA", "WHEEL"))
                )
                if wheel["Root-Is-Purelib"] != "false" or wheel.get_all("Tag") != [tag]:
                    raise ValueError("The bundled CLI must be a Windows platform wheel")
                if not archive.read("aster_cli/bin/aster.exe").startswith(b"MZ"):
                    raise ValueError("The bundled CLI is missing its native terminal")
            packages[name] = path.name
            versions.add(version)
    if set(packages) != set(PACKAGES) or len(versions) != 1:
        raise ValueError("Expected all five Aster wheels at the same version")
    return versions.pop(), packages


def assemble_bundle(directory: Path, *, commit: str, dirty: bool) -> Path:
    version, packages = inspect_wheels(directory / "wheels")
    shutil.copyfile(ROOT / "scripts/install_terminal.ps1", directory / "install.ps1")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(ROOT / name, directory / name)
    (directory / "README.txt").write_text(
        "Aster terminal for native Windows x64\n\n"
        "Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n"
        "Extract this entire archive, then run in PowerShell:\n"
        "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\install.ps1\n"
        "Run only a bundle from a trusted source. This execution-policy setting\n"
        "applies to that PowerShell process; it does not change the machine policy.\n"
        "Run the installer from a newer bundle to upgrade. Close Aster first.\n"
        "The installer replaces the aster-cli uv tool; it does not modify your\n"
        "model configuration, workspaces, saved conversations or drafts.\n"
        "It refuses to overwrite command files owned by another installation.\n"
        "Use -ToolsDirectory and -BinDirectory to install separately if names conflict.\n"
        "Run 'uv tool update-shell' once if the printed command directory is not on PATH.\n"
        "Then run 'aster connect' to configure a model, and 'aster -w PATH' to open a workspace.\n\n"
        "Python 3.12 and third-party dependencies may be downloaded by uv.\n"
        "All five Aster packages and the Rust terminal are included; Rust is not required.\n"
        "-Offline uses only an already available Python and uv dependency cache.\n"
        "-ToolsDirectory and -BinDirectory select isolated installation locations.\n"
        "SHA256SUMS detects corruption; it is not a publisher signature.\n",
        encoding="utf-8",
    )
    files = {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != ".gitignore"
    }
    manifest = {
        "schema_version": 1,
        "platform": "win_amd64",
        "version": version,
        "commit": commit,
        "dirty": dirty,
        "created_at": datetime.now(UTC).isoformat(),
        "packages": packages,
        "files": files,
    }
    manifest_path = directory / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files["MANIFEST.json"] = sha256(manifest_path)
    (directory / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())), encoding="utf-8"
    )
    archive_path = directory / f"aster-terminal-{version}-windows-x64.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted((*files, "SHA256SUMS")):
            archive.write(directory / name, name)
    print(f"Bundle: {archive_path}\nSHA256: {sha256(archive_path)}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist/terminal-windows")
    parser.add_argument(
        "--skip-cargo-build", action="store_true", help="Use the existing release binary (CI only)"
    )
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Build this bundle on native Windows x64")
    directory = args.out_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error(f"Output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("ASTER_BUNDLE_TUI", None)
    if not args.skip_cargo_build:
        subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                "packages/aster-tui/Cargo.toml",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    binary = ROOT / "packages/aster-tui/target/release/aster.exe"
    if not binary.is_file():
        parser.error(f"Native release binary not found: {binary}")
    for package in PACKAGES:
        build_env = environment | (
            {"ASTER_BUNDLE_TUI": str(binary)} if package == "aster-cli" else {}
        )
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--no-sources",
                "--package",
                package,
                "--out-dir",
                str(directory / "wheels"),
            ],
            cwd=ROOT,
            env=build_env,
            check=True,
        )
    subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--no-default-groups",
            "--no-dev",
            "--extra",
            "tui",
            "--extra",
            "memory",
            "--extra",
            "bench",
            "--no-emit-workspace",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--output-file",
            str(directory / "constraints.txt"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    )
    assemble_bundle(directory, commit=commit, dirty=dirty)


if __name__ == "__main__":
    main()
