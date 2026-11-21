# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the _memory_monitor module.

Tests resource limits (RLIMIT_AS) and hard-kill behavior in isolated
subprocesses to avoid poisoning the test runner's own process limits.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
import tracemalloc
from pathlib import Path

import pytest

from eval_pipeline._memory_monitor import (
    MemoryMonitor,
    enable_tracking,
    get_rss_mb,
)


class TestEnableTracking:
    def test_enables_tracemalloc(self):
        """tracemalloc.start() is called and tracing is active."""
        was_tracing = tracemalloc.is_tracing()
        try:
            if was_tracing:
                tracemalloc.stop()
            enable_tracking()
            assert tracemalloc.is_tracing()
        finally:
            if not was_tracing:
                tracemalloc.stop()

    def test_idempotent(self):
        """Calling enable_tracking twice doesn't raise."""
        was_tracing = tracemalloc.is_tracing()
        try:
            enable_tracking()
            enable_tracking()  # should not raise
            assert tracemalloc.is_tracing()
        finally:
            if not was_tracing:
                tracemalloc.stop()


class TestGetRssMb:
    def test_returns_positive_float(self):
        rss = get_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0

    def test_reasonable_range(self):
        """Current process RSS should be between 1 MB and 100 GB."""
        rss = get_rss_mb()
        assert 1.0 < rss < 100_000.0

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows working-set API")
    def test_windows_rss_tracks_resident_allocation(self):
        script = textwrap.dedent("""\
            import mmap
            from eval_pipeline._memory_monitor import get_rss_mb

            before = get_rss_mb()
            allocation = mmap.mmap(-1, 32 * 1024 * 1024)
            for offset in range(0, len(allocation), 4096):
                allocation[offset] = 1
            after = get_rss_mb()
            assert after > before + 24, (before, after)
            allocation.close()
            released = get_rss_mb()
            assert released < after - 24, (after, released)
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, result.stderr


class TestHardLimit:
    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has no RLIMIT_AS")
    def test_set_and_clear(self):
        """set_hard_limit / clear_hard_limit round-trips without error.

        Run in a subprocess to avoid permanently lowering the test process's
        hard limit (which can't be raised back without privileges).
        """
        script = textwrap.dedent("""\
            import resource
            from eval_pipeline._memory_monitor import set_hard_limit, clear_hard_limit, _get_vas_mb

            _, orig_hard = resource.getrlimit(resource.RLIMIT_AS)
            vas_before = _get_vas_mb()

            applied = set_hard_limit(512)
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            assert hard == orig_hard, f"hard limit should not change"

            if applied:
                expected = int((vas_before + 512) * 1024 * 1024)
                # Clamp to hard ceiling
                if orig_hard != resource.RLIM_INFINITY and expected > orig_hard:
                    expected = orig_hard
                # Allow small tolerance for allocations between _get_vas_mb and setrlimit
                assert abs(soft - expected) < 10 * 1024 * 1024, f"soft={soft}, expected≈{expected}"

                clear_hard_limit()
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                assert soft == orig_hard, f"soft should be restored to hard ceiling"

            print("OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Subprocess failed: stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
        assert "OK" in result.stdout

    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS enforcement requires Linux")
    def test_memory_error_on_exceed(self):
        """Allocating beyond the hard limit raises MemoryError.

        We run this in a subprocess to avoid poisoning the test process's
        RLIMIT_AS (which would cause pytest's own reporting to fail).
        """
        script = textwrap.dedent("""\
            from eval_pipeline._memory_monitor import set_hard_limit
            # Allow 50MB growth — then try to allocate 500MB
            set_hard_limit(50)
            try:
                _big = bytearray(500 * 1024 * 1024)  # 500 MB — should exceed
                print("NO_ERROR")
            except MemoryError:
                print("MEMORY_ERROR")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "MEMORY_ERROR" in result.stdout, (
            f"Expected MemoryError but got: stdout={result.stdout!r}, stderr={result.stderr!r}"
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows RLIMIT_AS fallback")
    def test_windows_reports_secondary_limit_unavailable(self):
        from eval_pipeline._memory_monitor import clear_hard_limit, set_hard_limit

        assert set_hard_limit(512) is False
        clear_hard_limit()


class TestMemoryMonitor:
    def test_tracks_peak_rss(self, tmp_path: Path):
        """Monitor tracks peak RSS over its lifetime."""
        monitor = MemoryMonitor(
            limit_mb=99999,  # high limit so soft limit won't fire
            trace_dir=str(tmp_path),
            sample_id="test_peak",
            poll_interval=0.1,
        )
        monitor.start()
        time.sleep(0.3)  # Let a few polls happen
        monitor.stop()

        assert monitor.peak_rss_mb > 0
        assert not monitor.soft_limit_hit
        assert monitor.diag_file is None

    def test_soft_limit_fires_and_writes_diagnostics(self, tmp_path: Path, monkeypatch):
        """When RSS exceeds the soft limit, diagnostics file is written."""
        current_rss = get_rss_mb()
        # Cross only the soft threshold; the real hard-kill check runs in a child.
        limit_mb = max(1024, int(current_rss * 4))

        monitor = MemoryMonitor(
            limit_mb=limit_mb,
            trace_dir=str(tmp_path),
            sample_id="test_diag_内存_🧪",
            soft_pct=current_rss * 0.5 / limit_mb,
            poll_interval=0.1,
        )
        unexpected_kills = []
        monkeypatch.setattr(monitor, "_hard_kill", unexpected_kills.append)

        # Enable tracemalloc so the diagnostics capture works
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start(5)
        try:
            monitor.start()
            deadline = time.monotonic() + 10
            while monitor.diag_file is None and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            monitor.stop()
            if not was_tracing:
                tracemalloc.stop()

        assert not unexpected_kills
        assert monitor.soft_limit_hit
        assert monitor.diag_file is not None

        diag_path = Path(monitor.diag_file)
        assert diag_path.exists()

        content = diag_path.read_text(encoding="utf-8")
        assert "MEMORY DIAGNOSTICS" in content
        assert "test_diag_内存_🧪" in content
        assert "TOP 25 MEMORY ALLOCATORS" in content
        assert "TOP 25 OBJECT TYPES BY COUNT" in content
        assert "THREAD STACK TRACES" in content
        assert "RSS HISTORY" in content

    def test_rss_history_recorded(self, tmp_path: Path):
        """RSS history is accumulated over the monitor lifetime."""
        monitor = MemoryMonitor(
            limit_mb=99999,
            trace_dir=str(tmp_path),
            sample_id="test_history",
            poll_interval=0.05,
        )
        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        assert len(monitor._rss_history) >= 2
        # Timestamps should be monotonically increasing
        times = [t for t, _ in monitor._rss_history]
        assert times == sorted(times)

    def test_hard_limit_exits_only_its_subprocess(self, tmp_path: Path):
        script = textwrap.dedent("""\
            import sys
            import time
            from eval_pipeline._memory_monitor import MemoryMonitor

            monitor = MemoryMonitor(
                limit_mb=1,
                trace_dir=sys.argv[1],
                sample_id="hard-limit",
                proto_out=sys.stdout.buffer,
                poll_interval=0.02,
            )
            monitor.start()
            time.sleep(10)
            raise AssertionError("The memory limit did not stop the child")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        assert result.returncode == 137, (result.stdout, result.stderr)
        error = json.loads(result.stdout)
        assert error["error_type"] == "MemoryError"
        assert error["passed"] is False
        assert error["peak_rss_mb"] > 1
        diagnostic = Path(error["memory_diag_file"]).read_text(encoding="utf-8")
        assert "HARD KILL" in diagnostic


class TestErrorClassification:
    """Verify that MemoryError patterns are classified correctly."""

    def test_memory_error(self):
        from eval_pipeline._utils import classify_error_type

        assert classify_error_type("MemoryError: process exceeded 4096 MB") == "MemoryError"

    def test_memory_soft_limit(self):
        from eval_pipeline._utils import classify_error_type

        assert (
            classify_error_type("Memory soft limit hit: 3500 MB (limit: 4096 MB)")
            == "MemoryWarning"
        )

    def test_existing_timeout_still_works(self):
        from eval_pipeline._utils import classify_error_type

        assert classify_error_type("TimeoutError: exceeded 30s") == "TimeoutError"
