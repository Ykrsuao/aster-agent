# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The acceptance harness must reject broken work and bound uncooperative work."""

import json
import sys
import time

import pytest

from scripts import terminal_acceptance as acceptance


def test_all_frozen_verifiers_reject_seeds_and_accept_reference_solutions():
    tasks, digest = acceptance.load_tasks()
    assert len(tasks) == 10 and len(digest) == 64
    acceptance.self_check(tasks)


def test_usage_unknown_is_not_reported_as_zero():
    known = {
        "input_tokens": 8,
        "input_tokens_reported": True,
        "output_tokens": 2,
        "output_tokens_reported": True,
        "cost_usd": 0.01,
        "cost_known": True,
    }
    mixed = acceptance.usage_totals([known, None])
    assert mixed["input_tokens"] is None and mixed["cost_usd"] is None
    assert mixed["known_cost_subtotal_usd"] == 0.01
    zero = acceptance.usage_totals([{**known, "cost_usd": 0}])
    assert zero["cost_usd"] == 0
    assert acceptance.usage_totals([])["cost_usd"] is None
    assert (
        acceptance.usage_totals([{**known, "output_tokens_reported": False}])["output_tokens"]
        is None
    )


def test_process_timeout_kills_blocking_code_and_bounds_output(tmp_path):
    started = time.monotonic()
    code, output, timed_out = acceptance.run_process(
        [
            sys.executable,
            "-c",
            "import sys,time; input(); print('x'*200000, flush=True); time.sleep(30)",
        ],
        cwd=tmp_path,
        text="go\n",
        timeout=0.5,
    )
    assert timed_out and code != 0 and time.monotonic() - started < 10
    assert 0 < len(output) <= 65536


def test_public_test_cannot_be_replaced_to_pass(tmp_path):
    task = acceptance.load_tasks()[0][6]
    acceptance.write_files(tmp_path, task["files"])
    acceptance.write_files(tmp_path, task["solution"])
    (tmp_path / "test_public.py").write_text("pass", encoding="utf-8")
    assert acceptance.verify(task, tmp_path) == (False, "protected_file_changed")


def test_invalid_utf8_public_test_is_a_failed_task_not_a_runner_error(tmp_path):
    task = acceptance.load_tasks()[0][6]
    acceptance.write_files(tmp_path, task["files"])
    (tmp_path / "test_public.py").write_bytes(b"\xff\xfe")
    assert acceptance.verify(task, tmp_path) == (False, "protected_file_changed")


def test_unreadable_public_test_is_a_failed_task_not_a_runner_error(tmp_path, monkeypatch):
    from pathlib import Path

    task = acceptance.load_tasks()[0][6]
    acceptance.write_files(tmp_path, task["files"])
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.name == "test_public.py":
            raise PermissionError("test file is locked")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert acceptance.verify(task, tmp_path) == (False, "protected_file_unreadable")


def test_cross_file_verifier_rejects_copied_converter(tmp_path):
    task = acceptance.load_tasks()[0][4]
    acceptance.write_files(tmp_path, task["solution"])
    (tmp_path / "report.py").write_text(
        "def describe(value, unit):\n"
        "    factors = {'m': 1, 'cm': .01, 'mm': .001}\n"
        "    if unit not in factors: raise ValueError(unit)\n"
        "    return f'{value * factors[unit]:g} m'\n",
        encoding="utf-8",
    )
    assert not acceptance.verify(task, tmp_path)[0]


@pytest.mark.parametrize("max_calls, expected", [(2, "completed"), (1, "call_limit")])
async def test_worker_uses_real_coding_agent_with_scripted_model(
    tmp_path, monkeypatch, max_calls, expected
):
    from aster.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage, ToolCall

    task = acceptance.load_tasks()[0][0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    acceptance.write_files(workspace, task["files"])
    code = f"await self.shell.write_file('pagination.py', {task['solution']['pagination.py']!r})"
    llm = FakeLLMClient(
        [
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(id="edit", name="execute_python", arguments=json.dumps({"code": code}))
                ],
                finish_reason="tool_calls",
                usage=LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0),
            ),
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="done",
                        name="return_result",
                        arguments=json.dumps(
                            {"result": {"message": "done", "explanation": "fixed"}}
                        ),
                    )
                ],
                finish_reason="tool_calls",
                usage=LLMUsage(input_tokens=10, output_tokens=5, cost_usd=0),
            ),
        ],
        strict_exhaustion=True,
    )
    monkeypatch.chdir(acceptance.ROOT)
    for name in ("ASTER_PROJECT_DIR", "ASTER_USER_DIR", "ASTER_SETTINGS"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setattr("aster.unifiedllm.get_llm_client", lambda *a, **kw: llm)
    monkeypatch.setattr("aster.unifiedllm.registry.load_model_entries", lambda *a: {"scripted": {}})
    monkeypatch.setattr("aster.llm_config.llm_config_chain", lambda: [])
    monkeypatch.setattr("aster.secrets.load_secrets_into_env", lambda: [])
    monkeypatch.setattr("aster.tracing.enable_tracing", lambda **kw: None)
    meter = tmp_path / "usage.json"
    await acceptance.worker(
        {
            "workspace": str(workspace),
            "meter": str(meter),
            "model": "scripted",
            "prompt": task["prompt"],
            "max_calls": max_calls,
            "max_output_tokens": 4096,
        }
    )
    result = json.loads(meter.read_text(encoding="utf-8"))
    assert result["status"] == expected
    assert result["attempted_calls"] == max_calls
    assert result["cost_usd"] == 0
    assert acceptance.verify(task, workspace)[0]


def test_live_options_are_required_and_output_cannot_overwrite(tmp_path):
    with pytest.raises(SystemExit):
        acceptance.main([])
    if sys.platform == "win32":
        with pytest.raises(FileExistsError):
            acceptance.main(["--model", "test", "--output", str(tmp_path)])
    else:
        with pytest.raises(SystemExit):
            acceptance.main(["--model", "test", "--output", str(tmp_path)])


def test_error_reason_does_not_export_exception_text(tmp_path):
    task = {"files": {}, "verify": "raise RuntimeError('api-key-do-not-export')"}
    passed, reason = acceptance.verify(task, tmp_path)
    assert not passed and reason == "verification_failed:RuntimeError"
    assert "api-key" not in reason


def test_report_preserves_verification_when_worker_hits_call_limit(tmp_path, monkeypatch):
    from argparse import Namespace
    from pathlib import Path

    task = acceptance.load_tasks()[0][0]
    original = acceptance.run_process

    def run_process(args, **kwargs):
        if "--worker" not in args:
            return original(args, **kwargs)
        payload = json.loads(kwargs["text"])
        acceptance.write_files(Path(payload["workspace"]), task["solution"])
        acceptance.write_json(
            Path(payload["meter"]),
            {"status": "call_limit", "attempted_calls": 8, "completed_calls": 8},
        )
        return 0, "", False

    monkeypatch.setattr(acceptance, "run_process", run_process)
    monkeypatch.setattr(acceptance, "candidate", lambda: {"commit": "test", "dirty": True})
    output = tmp_path / "report"
    args = Namespace(output=output, model="test", timeout=10, max_calls=8, max_output_tokens=4096)
    assert acceptance.run_suite(args, [task], "fixture-hash") == 1
    result = json.loads((output / "report.json").read_text(encoding="utf-8"))["tasks"][0]
    assert result["passed"] is False and result["reason"] == "call_limit"
    assert result["verification_passed"] is True and result["verification_reason"] is None
