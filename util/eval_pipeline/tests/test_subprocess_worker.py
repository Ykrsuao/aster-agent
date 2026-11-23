# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the subprocess worker (JSON stdin → JSON stdout).

Verifies that:
- The subprocess worker can reconstruct agents and scorers from config JSON
- Results are valid EvalTestResult JSON
- Tracing is set up fresh (no inherited state)
- The SubprocessEngine properly orchestrates Popen workers
"""

import json
import subprocess
import sys

import pytest


def _build_task_input(**overrides) -> dict:
    """Build a minimal SubprocessTaskInput dict."""
    base = {
        "agent_spec": {
            "agent_module": "eval_pipeline._test_fixtures",
            "agent_class": "DummyAgent",
            "method": "classify",
            "client_config": {},
        },
        "task": {
            "id": "test_001",
            "input": [[], {"text": "great product"}],
            "expected": "positive",
            "run_id": 1,
        },
        "scorers": [
            {
                "name": "exact_match",
                "weight": 1.0,
                "scorer_class": "eval_pipeline.scoring.ExactMatchScorer",
                "scorer_kwargs": {},
            }
        ],
        "sample_id": "test_sample_001",
        "model": "dummy",
        "test_name": "sentiment",
        "display_name": "sentiment/test_001/dummy/run1",
        "tier": "stable",
        "trace_dir": "/tmp/test_traces",
        "use_otlp": False,
        "pass_threshold": 0.5,
    }
    base.update(overrides)
    return base


DummyAgent = None  # imported from _test_fixtures by subprocess workers, not needed here


class TestSubprocessWorker:
    """Test the subprocess_worker.py script via Popen."""

    @pytest.mark.parametrize(
        "usages, known, total, complete",
        [
            ([{"prompt": 0, "completion": 0, "total": 0}], 0, 0, True),
            ([{"prompt": 8, "completion": 2, "total": 10}, None], 8, None, False),
            ([{"prompt": 0}], 0, 0, True),
        ],
    )
    def test_usage_survives_raw_spans_worker_json_and_experiment_file(
        self, tmp_path, usages, known, total, complete
    ):
        from eval_pipeline.eval_parser import EvalFileParser
        from eval_pipeline.eval_types import EvalTestResult
        from eval_pipeline.experiment_writer import ExperimentWriter

        task_input = _build_task_input(
            trace_dir=str(tmp_path / "traces"),
            agent_spec={
                "agent_module": "eval_pipeline._test_fixtures",
                "agent_class": "UsageTraceAgent",
                "method": "classify",
                "client_config": {},
            },
            task={
                "id": "usage",
                "input": [[], {"text": json.dumps(usages)}],
                "expected": "positive",
            },
        )
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        result = EvalTestResult.model_validate_json(proc.stdout)
        assert result.passed, result.error
        assert result.token_usage.records == len(usages)
        assert result.token_usage.known_input_tokens == known
        assert result.token_usage.input_tokens_complete is complete
        assert result.input_tokens == total
        if any(usage is None or "completion" not in usage for usage in usages):
            assert result.output_tokens is None
            assert result.token_usage.output_tokens_complete is False
        writer = ExperimentWriter(tmp_path / "results", "usage")
        writer.start()
        writer.append_result(result)
        writer.finalize()
        _, results, completion, _ = EvalFileParser().parse_file(writer.file_path)
        assert results[0].token_usage == result.token_usage == completion.token_usage
        assert completion.total_input_tokens == total

    @pytest.mark.parametrize("memory_limit_mb", [None, 4096])
    def test_basic_execution(self, tmp_path, memory_limit_mb):
        """Worker executes a task and returns valid EvalTestResult JSON."""
        task_input = _build_task_input(trace_dir=str(tmp_path), memory_limit_mb=memory_limit_mb)
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"Worker failed: {proc.stderr.decode()}"
        result = json.loads(proc.stdout)
        assert result["test_id"] == "test_001_dummy_run1"
        assert result["passed"] is True
        assert result["output"] == "positive"
        assert "exact_match" in result["scores"]
        assert result["token_usage"]["records"] == 0
        assert result["token_usage"]["input_tokens_complete"] is False
        assert result["input_tokens"] is None

    def test_disabled_trace_files_do_not_write_usage_or_eval_spans(self, tmp_path):
        task_input = _build_task_input(
            trace_dir=str(tmp_path),
            write_trace_file=False,
            agent_spec={
                "agent_module": "eval_pipeline._test_fixtures",
                "agent_class": "UsageTraceAgent",
                "method": "classify",
                "client_config": {},
            },
            task={
                "id": "usage",
                "input": [[], {"text": '[{"prompt": 8}]'}],
                "expected": "positive",
            },
        )
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr.decode()
        result = json.loads(proc.stdout)
        assert result["passed"] is True and result["input_tokens"] is None
        assert result["token_usage"]["input_tokens_complete"] is False
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_wrong_answer(self, tmp_path):
        """Worker correctly reports failure when output doesn't match expected."""
        task_input = _build_task_input(
            trace_dir=str(tmp_path),
            task={
                "id": "test_002",
                "input": [[], {"text": "great product"}],
                "expected": "negative",  # Wrong — agent will say "positive"
                "run_id": 1,
            },
        )
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0
        result = json.loads(proc.stdout)
        assert result["passed"] is False

    def test_invalid_json_input(self):
        """Worker returns error result (not crash) on invalid JSON."""
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=b"not json\n",
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0  # Worker doesn't crash
        result = json.loads(proc.stdout)
        assert result["passed"] is False
        assert "error" in result
        assert result["error_type"] == "WorkerError"

    def test_missing_agent_module(self, tmp_path):
        """Worker returns error result when agent module doesn't exist."""
        task_input = _build_task_input(
            trace_dir=str(tmp_path),
            agent_spec={
                "agent_module": "nonexistent.module",
                "agent_class": "FakeAgent",
                "method": "run",
                "client_config": {},
            },
        )
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0  # Should still return result, not crash
        result = json.loads(proc.stdout)
        assert result["passed"] is False
        assert result["error"] is not None

    def test_result_is_valid_pydantic(self, tmp_path):
        """Worker output can be parsed as EvalTestResult."""
        from eval_pipeline.eval_types import EvalTestResult

        task_input = _build_task_input(trace_dir=str(tmp_path))
        proc = subprocess.run(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            input=json.dumps(task_input).encode(),
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0
        result = EvalTestResult.model_validate_json(proc.stdout)
        assert result.test_id == "test_001_dummy_run1"
        assert result.passed is True


class TestPersistentWorker:
    """Test the persistent worker protocol (multiple tasks over one stdin/stdout)."""

    def test_multiple_tasks_on_one_worker(self, tmp_path):
        """A single persistent worker handles multiple tasks sequentially."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            results = []
            for i in range(3):
                task = _build_task_input(
                    trace_dir=str(tmp_path),
                    task={
                        "id": f"multi_{i}",
                        "input": [[], {"text": "great product"}],
                        "expected": "positive",
                        "run_id": 1,
                    },
                    sample_id=f"multi_{i}",
                )
                proc.stdin.write(json.dumps(task).encode() + b"\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                assert line, f"Worker closed stdout after task {i}"
                results.append(json.loads(line))

            assert len(results) == 3
            for r in results:
                assert r["passed"] is True
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)

    def test_stdout_redirect_protects_protocol(self, tmp_path):
        """Agent print() goes to stderr, not stdout, so JSON protocol is safe."""
        # Use a fixture agent that prints to stdout
        agent_code = tmp_path / "noisy_agent.py"
        agent_code.write_text(
            "class NoisyAgent:\n"
            "    def __init__(self, llm=None): pass\n"
            "    async def classify(self, text):\n"
            "        print('NOISE FROM AGENT')\n"
            "        return 'positive'\n"
        )
        task = _build_task_input(
            trace_dir=str(tmp_path),
            agent_spec={
                "agent_module": "",
                "agent_class": "NoisyAgent",
                "method": "classify",
                "client_config": {},
                "agent_file": str(agent_code),
            },
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            proc.stdin.write(json.dumps(task).encode() + b"\n")
            proc.stdin.flush()
            # Read result line, then close stdin to let worker exit
            result_line = proc.stdout.readline()
            proc.stdin.close()
            proc.wait(timeout=10)
            stdout = result_line
            stderr = proc.stderr.read()
            # The result JSON should be valid (not corrupted by print)
            result = json.loads(stdout)
            assert result["passed"] is True
            assert result["output"] == "positive"
            # The print output should be on stderr and identify the owning task.
            assert b"[test_001] NOISE FROM AGENT" in stderr
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_output_prefixer_supports_writelines(self):
        """Stream wrapper preserves writelines() and prefixes each line."""
        import io

        from eval_pipeline.subprocess_worker import _TaskOutputPrefixer

        stream = io.StringIO()
        prefixer = _TaskOutputPrefixer(stream, "task_001")

        prefixer.writelines(["first\n", "second\n"])

        assert stream.getvalue() == "[task_001] first\n[task_001] second\n"

    def test_output_prefixer_treats_carriage_return_as_line_separator(self):
        """Carriage-return-only progress output should not suppress the next prefix."""
        import io

        from eval_pipeline.subprocess_worker import _TaskOutputPrefixer

        stream = io.StringIO()
        prefixer = _TaskOutputPrefixer(stream, "task_001")

        prefixer.write("progress\r")
        prefixer.write("done\n")

        assert stream.getvalue() == "[task_001] progress\r[task_001] done\n"

    def test_error_on_bad_json_then_good_task(self, tmp_path):
        """Bad JSON produces error result, but next task still works."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "eval_pipeline.subprocess_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Send bad JSON
            proc.stdin.write(b"not json\n")
            proc.stdin.flush()
            line1 = proc.stdout.readline()
            r1 = json.loads(line1)
            assert r1["passed"] is False
            assert r1["error_type"] == "WorkerError"

            # Send good task — worker should still work
            task = _build_task_input(
                trace_dir=str(tmp_path),
                task={
                    "id": "after_error",
                    "input": [[], {"text": "great product"}],
                    "expected": "positive",
                    "run_id": 1,
                },
            )
            proc.stdin.write(json.dumps(task).encode() + b"\n")
            proc.stdin.flush()
            line2 = proc.stdout.readline()
            r2 = json.loads(line2)
            assert r2["passed"] is True
        finally:
            proc.stdin.close()
            proc.wait(timeout=5)


class TestSubprocessEngine:
    """Test the SubprocessEngine orchestration."""

    @pytest.mark.asyncio
    async def test_run_tasks(self, tmp_path):
        """SubprocessEngine runs tasks via JSON subprocess workers."""
        from eval_pipeline.concurrency import (
            ConcurrencyConfig,
            SubprocessEngine,
        )

        engine = SubprocessEngine()
        config = ConcurrencyConfig(max_concurrent=2)

        task_ids = ["s1", "s2"]
        task_data = {}
        for i, tid in enumerate(task_ids):
            task_input = _build_task_input(
                trace_dir=str(tmp_path),
                sample_id=tid,
                task={
                    "id": f"task_{i + 1:03d}",
                    "input": [[], {"text": "great product"}],
                    "expected": "positive",
                    "run_id": 1,
                },
            )
            task_data[tid] = json.dumps(task_input)

        results = await engine.run_tasks(
            task_ids=task_ids,
            task_data=task_data,
            config=config,
        )

        assert len(results) == 2
        for r in results:
            assert not isinstance(r, Exception), f"Task failed: {r}"
            assert r.passed is True

    @pytest.mark.asyncio
    async def test_run_evaluation_subprocess(self, tmp_path):
        """Integration: run_evaluation() with engine_type='subprocess' end-to-end."""
        from unittest.mock import MagicMock

        from eval_pipeline.eval_types import Tier
        from eval_pipeline.models import Task
        from eval_pipeline.pipeline import PipelineConfig, Sample, run_evaluation
        from eval_pipeline.scoring import ExactMatchScorer, ScorerConfig

        scorer = ExactMatchScorer()
        scorer_config = ScorerConfig(
            name="exact_match",
            weight=1.0,
            scorer=scorer,
            scorer_class="eval_pipeline.scoring.ExactMatchScorer",
            scorer_kwargs={},
        )

        samples = [
            Sample(
                task=Task(
                    id="int_001",
                    input=((), {"text": "great product"}),
                    expected="positive",
                    run_id=1,
                ),
                method="classify",
                agent_class="DummyAgent",
                scorers=[scorer_config],
                agent_factory=lambda: None,  # unused in subprocess path
                model="dummy",
                test_name="sentiment",
                sample_id="int_001",
                tier=Tier.STABLE,
                agent_module="eval_pipeline._test_fixtures",
                client_config={},
                scorer_specs=[
                    {
                        "name": "exact_match",
                        "weight": 1.0,
                        "scorer_class": "eval_pipeline.scoring.ExactMatchScorer",
                        "scorer_kwargs": {},
                    }
                ],
            ),
        ]

        writer = MagicMock()
        config = PipelineConfig(
            trace_dir=tmp_path,
            engine_type="subprocess",
            pass_threshold=0.5,
        )

        results = await run_evaluation(samples, config, writer, max_concurrent=1)

        assert len(results) == 1
        result = results[0]
        assert not isinstance(result, Exception), f"Task failed: {result}"
        assert result.passed is True
        assert writer.append_result.called
