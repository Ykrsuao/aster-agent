# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OAuth cache updates must preserve credentials across processes and failed writes."""

import json
import subprocess
import sys

from aster.mcp import oauth

_CACHE_WRITER = """
import sys
from pathlib import Path
from aster.mcp import oauth

cache = Path(sys.argv[1])
role = sys.argv[2]
oauth._token_cache_path = lambda: cache
original_read = Path.read_text

def read_snapshot(path, *args, **kwargs):
    content = original_read(path, *args, **kwargs)
    if path == cache and role == "first":
        print("snapshot-read", flush=True)
        assert sys.stdin.readline() == "resume\\n"
    return content

Path.read_text = read_snapshot
print("ready", flush=True)
oauth._save_cached_token(role, oauth.OAuthToken(access_token="test-token"))
"""


def test_concurrent_processes_preserve_both_oauth_cache_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path))
    oauth._save_cached_token("existing", oauth.OAuthToken(access_token="original-token"))
    cache = oauth._token_cache_path()
    processes = []
    try:
        first = subprocess.Popen(
            [sys.executable, "-c", _CACHE_WRITER, str(cache), "first"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        processes.append(first)
        assert first.stdout.readline().strip() == "ready"
        assert first.stdout.readline().strip() == "snapshot-read"
        second = subprocess.Popen(
            [sys.executable, "-c", _CACHE_WRITER, str(cache), "second"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        processes.append(second)
        assert second.stdout.readline().strip() == "ready"
        # A competing writer either finishes (the old race) or waits for the lock.
        try:
            second.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        _, first_error = first.communicate("resume\n", timeout=15)
        _, second_error = second.communicate(timeout=15)
        assert first.returncode == 0, first_error
        assert second.returncode == 0, second_error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()

    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert set(saved) == {"existing", "first", "second"}
    assert saved["existing"]["access_token"] == "original-token"


def test_failed_cache_replacement_preserves_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path))
    oauth._save_cached_token("existing", oauth.OAuthToken(access_token="original-token"))
    cache = oauth._token_cache_path()
    original = cache.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated cache replacement failure")

    monkeypatch.setattr(oauth.os, "replace", fail_replace)
    oauth._save_cached_token("new", oauth.OAuthToken(access_token="new-token"))

    assert cache.read_bytes() == original
    assert not list(cache.parent.glob(f".{cache.name}.*.tmp"))


_CACHE_READER = """
import sys
from pathlib import Path
from aster.mcp import oauth

cache = Path(sys.argv[1])
oauth._token_cache_path = lambda: cache
original_read = Path.read_text

def read_while_paused(path, *args, **kwargs):
    if path != cache:
        return original_read(path, *args, **kwargs)
    with path.open("r", **kwargs) as stream:
        print("reader-open", flush=True)
        assert sys.stdin.readline() == "resume\\n"
        return stream.read()

Path.read_text = read_while_paused
token = oauth._load_cached_token("existing")
assert token is not None and token.access_token == "original-token"
"""


def test_concurrent_process_reader_preserves_oauth_cache_update(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path))
    oauth._save_cached_token("existing", oauth.OAuthToken(access_token="original-token"))
    cache = oauth._token_cache_path()
    processes = []
    try:
        reader = subprocess.Popen(
            [sys.executable, "-c", _CACHE_READER, str(cache)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        processes.append(reader)
        assert reader.stdout.readline().strip() == "reader-open"
        writer = subprocess.Popen(
            [sys.executable, "-c", _CACHE_WRITER, str(cache), "new"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        processes.append(writer)
        assert writer.stdout.readline().strip() == "ready"
        # Windows read handles deny replacement, so writers must await the reader.
        try:
            writer.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        _, reader_error = reader.communicate("resume\n", timeout=15)
        _, writer_error = writer.communicate(timeout=15)
        assert reader.returncode == 0, reader_error
        assert writer.returncode == 0, writer_error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()

    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert set(saved) == {"existing", "new"}
    assert saved["existing"]["access_token"] == "original-token"
    assert saved["new"]["access_token"] == "test-token"
