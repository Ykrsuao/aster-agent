# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trust changes from separate hosts preserve each other's decisions."""

import subprocess
import sys
from pathlib import Path

import pytest

from aster import workspace_trust

_WRITER = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("_trust_store", sys.argv[1])
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)
assert "aster" not in sys.modules
assert "filelock" not in sys.modules

if sys.argv[4] == "pause":
    write = trust._write
    def paused_write(entries):
        print("read", flush=True)
        assert sys.stdin.readline()
        write(entries)
    trust._write = paused_write
else:
    print("ready", flush=True)
getattr(trust, sys.argv[2])(sys.argv[3])
"""


def _command(operation: str, folder: Path, *, pause: bool = False) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-c",
        _WRITER,
        workspace_trust.__file__,
        operation,
        str(folder),
        "pause" if pause else "run",
    ]


@pytest.fixture
def record(tmp_path, monkeypatch):
    directory = tmp_path / "user"
    monkeypatch.setenv("ASTER_USER_DIR", str(directory))
    monkeypatch.delenv("ASTER_TRUST_ALL_WORKSPACES", raising=False)
    return directory / "trusted_workspaces.json"


@pytest.mark.parametrize("operation", ["trust", "untrust"])
def test_concurrent_changes_preserve_other_hosts_decisions(record, tmp_path, operation):
    existing = tmp_path / "existing"
    added = tmp_path / "added"
    existing.mkdir()
    added.mkdir()
    if operation == "untrust":
        workspace_trust.trust(existing)

    processes = []
    try:
        first = subprocess.Popen(
            _command("trust", added, pause=True),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(first)
        assert first.stdout.readline().strip() == "read"
        second = subprocess.Popen(
            _command(operation, existing),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(second)
        assert second.stdout.readline().strip() == "ready"
        # Let a writer without the lock finish against the stale snapshot.
        try:
            second.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
        _, errors = first.communicate("\n", timeout=10)
        assert first.returncode == 0, errors
        _, errors = second.communicate(timeout=10)
        assert second.returncode == 0, errors
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

    expected = {str(added.resolve())}
    if operation == "trust":
        expected.add(str(existing.resolve()))
    assert set(workspace_trust.trusted_workspaces()) == expected


@pytest.mark.parametrize("operation", ["trust", "untrust"])
def test_failed_write_preserves_record_and_releases_lock(record, tmp_path, monkeypatch, operation):
    original = tmp_path / "original"
    added = tmp_path / "added"
    workspace_trust.trust(original)
    contents = record.read_bytes()

    def fail_replace(*_args):
        raise OSError("replace failed")

    with monkeypatch.context() as patch:
        patch.setattr(workspace_trust.os, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            getattr(workspace_trust, operation)(added if operation == "trust" else original)
    assert record.read_bytes() == contents
    assert not list(record.parent.glob("*.tmp"))

    result = subprocess.run(
        _command("trust", added), capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr
    assert set(workspace_trust.trusted_workspaces()) == {
        str(original.resolve()),
        str(added.resolve()),
    }
