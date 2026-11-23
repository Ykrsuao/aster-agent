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


@pytest.mark.parametrize(
    "max_calls, reply, expected",
    [
        pytest.param(2, {"message": "done", "explanation": "fixed"}, "completed", id="2-completed"),
        pytest.param(
            1, {"message": "done", "explanation": "fixed"}, "call_limit", id="1-call_limit"
        ),
        pytest.param(
            2,
            {"message": "reply only", "explanation": "task pending", "complete_task": False},
            "needs_input",
            id="reply-only",
        ),
        pytest.param(
            2, {"question": "Which task should I finish?"}, "needs_input", id="needs-input"
        ),
    ],
)
async def test_worker_uses_real_coding_agent_with_scripted_model(
    tmp_path, monkeypatch, max_calls, reply, expected
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
                        arguments=json.dumps({"result": reply}),
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
    assert result["attempted_calls"] == max_calls
    assert result["error"] is None
    assert result["cost_usd"] == 0
    assert acceptance.verify(task, workspace)[0]
    assert result["status"] == expected
    assert result["measurement_kind"] == "offline-scripted"
    assert result["model_identity"]["model"] == "fake-model"
    assert result["model_fingerprint"] == acceptance.fingerprint_model(result["model_identity"])


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


@pytest.mark.parametrize(
    "worker_status, correct_patch, expected_reason",
    [
        ("call_limit", True, "call_limit"),
        ("needs_input", True, "needs_input"),
        ("completed", False, "verification_failed:AssertionError"),
        ("completed", True, None),
    ],
)
def test_report_requires_completion_and_independent_verification(
    tmp_path, monkeypatch, worker_status, correct_patch, expected_reason
):
    from argparse import Namespace
    from pathlib import Path

    task = acceptance.load_tasks()[0][0]
    original = acceptance.run_process

    def run_process(args, **kwargs):
        if "--worker" not in args:
            return original(args, **kwargs)
        payload = json.loads(kwargs["text"])
        if correct_patch:
            acceptance.write_files(Path(payload["workspace"]), task["solution"])
        acceptance.write_json(
            Path(payload["meter"]),
            {"status": worker_status, "attempted_calls": 8, "completed_calls": 8},
        )
        return 0, "", False

    monkeypatch.setattr(acceptance, "run_process", run_process)
    monkeypatch.setattr(acceptance, "candidate", lambda **kwargs: {"commit": "test", "dirty": True})
    output = tmp_path / "report"
    args = Namespace(output=output, model="test", timeout=10, max_calls=8, max_output_tokens=4096)
    assert acceptance.run_suite(args, [task], "fixture-hash") == (
        0 if expected_reason is None else 1
    )
    result = json.loads((output / "report.json").read_text(encoding="utf-8"))["tasks"][0]
    assert result["passed"] is (expected_reason is None)
    assert result["reason"] == expected_reason
    assert result["verification_passed"] is correct_patch
    assert result["verification_reason"] == (None if correct_patch else expected_reason)
    assert result["agent_completed"] is (worker_status == "completed")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["repeat_id"] == 1
    assert report["report_complete"] is False  # A selected subset is not full-suite evidence.


def _known_usage():
    return {
        "input_tokens": 8,
        "input_tokens_reported": True,
        "output_tokens": 2,
        "output_tokens_reported": True,
        "cached_input_tokens": 6,
        "reasoning_tokens": 1,
        "cost_usd": 0,
        "cost_known": True,
    }


def _evidence_report(repeat):
    from aster.unifiedllm import FakeLLMClient

    identity = acceptance.model_identity(FakeLLMClient())
    fingerprint = acceptance.fingerprint_model(identity)
    metadata = {
        "model_identity": identity,
        "model_fingerprint": fingerprint,
        "measurement_kind": "offline-scripted",
    }
    return {
        "schema_version": 1,
        "repeat_id": repeat,
        "report_complete": True,
        "candidate": {"commit": "a" * 40, "dirty": False, "diff_sha256": "b" * 64},
        "fixture_sha256": acceptance.load_tasks()[1],
        "runner_sha256": "c" * 64,
        "model_alias": f"alias-{repeat}",
        **metadata,
        "limits": {
            "timeout_seconds": 180,
            "max_calls": 8,
            "max_output_tokens": 4096,
            "endpoint_retries": 0,
        },
        "tasks": [
            {
                "id": task["id"],
                "status": "completed",
                "agent_completed": True,
                "verification_passed": True,
                "passed": True,
                "elapsed_seconds": 1.0,
                "manual_interventions": 0,
                "recovery_scope": "not_exercised",
                **metadata,
                **acceptance.usage_totals([_known_usage()]),
            }
            for task in acceptance.load_tasks()[0]
        ],
    }


def _aggregate(reports, expected_repeats=2):
    return acceptance.aggregate_reports(
        reports,
        expected_task_ids=[task["id"] for task in acceptance.load_tasks()[0]],
        expected_repeats=expected_repeats,
    )


def test_serial_call_rejection_remains_an_aggregatable_failed_observation():
    report = _evidence_report(1)
    report["tasks"][0].update(
        status="concurrent_call_rejected", agent_completed=False, passed=False
    )
    result = _aggregate([report], expected_repeats=1)
    assert result["observations"] == 10
    assert result["agent_completed"] == result["passed"] == 9
    assert result["verification_passed"] == 10


def test_usage_preserves_partial_subtotals_and_dispatched_requests_without_responses():
    usage = _known_usage()
    complete = acceptance.usage_totals([usage])
    assert complete["total_tokens"] == 10  # Cached/reasoning tokens are subsets.
    assert complete["cost_usd"] == 0 and complete["cost_complete"] is True
    partial = acceptance.usage_totals([usage], attempted_calls=2)
    assert partial["input_tokens"] is None and partial["input_tokens_complete"] is False
    assert partial["known_input_tokens"] == 8 and partial["known_total_tokens"] == 10
    assert partial["cost_usd"] is None and partial["known_cost_subtotal_usd"] == 0
    assert partial["cost_complete"] is False
    assert acceptance.usage_totals([None])["known_input_tokens"] == 0
    with pytest.raises(ValueError):
        acceptance.usage_totals([usage], attempted_calls=0)


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_raw_usage_is_not_a_known_token_count(value):
    result = acceptance.usage_totals([{**_known_usage(), "input_tokens": value}])
    assert result["input_tokens"] is None and result["known_input_tokens"] == 0
    assert result["total_tokens"] is None


def test_model_fingerprint_uses_resolved_settings_without_credentials():
    import hashlib

    from aster.unifiedllm import FakeLLMClient

    llm = FakeLLMClient()
    llm.config.update(
        {
            "api_base": "https://alice:password@gateway.example:8443/v1/secret?key=private#credential",
            "api_key": "do-not-export",
            "headers": {"Authorization": "do-not-export"},
            "extra_body": {"private": "do-not-export"},
            "temperature": 0.25,
        }
    )
    identity = acceptance.model_identity(llm)
    assert identity["endpoint_origin"] == "https://gateway.example:8443"
    assert identity["sampling"] == {"temperature": 0.25}
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    assert all(
        secret not in encoded
        for secret in ("alice", "password", "private", "do-not-export", "Authorization")
    )
    digest = acceptance.fingerprint_model(identity)
    assert digest == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    llm.config["temperature"] = 0.5
    assert acceptance.fingerprint_model(acceptance.model_identity(llm)) != digest


def test_fingerprint_includes_effective_safe_sampling_and_request_overrides():
    from aster.unifiedllm import FakeLLMClient

    llm = FakeLLMClient()
    llm.config["extra_body"] = {"reasoning_effort": "low", "private": "do-not-export"}
    identity = acceptance.model_identity(llm, overrides={"temperature": 0.25})
    assert identity["sampling"] == {"reasoning_effort": "low", "temperature": 0.25}
    llm.config["extra_body"]["reasoning_effort"] = "high"
    assert acceptance.fingerprint_model(
        acceptance.model_identity(llm)
    ) != acceptance.fingerprint_model(identity)


@pytest.mark.parametrize(
    "patch",
    [
        {"api_key": "secret"},
        {"endpoint_origin": "https://user:secret@example.org/v1?token=x"},
        {"sampling": {"temperature": float("nan")}},
        {"sampling": {"seed": True}},
        {"sampling": {"max_tokens": 1.5}},
        {"sampling": {"headers": {"Authorization": "secret"}}},
    ],
)
def test_fingerprint_rejects_unknown_or_credential_bearing_identity(patch):
    identity = _evidence_report(1)["model_identity"]
    with pytest.raises(ValueError):
        acceptance.fingerprint_model({**identity, **patch})


def test_aggregate_separates_completion_verification_and_usage_knownness():
    reports = [_evidence_report(1), _evidence_report(2)]
    row = reports[0]["tasks"][0]
    row.update(status="needs_input", agent_completed=False, passed=False, manual_interventions=1)
    row.update(acceptance.usage_totals([_known_usage(), None]))
    summary = _aggregate(reports)
    assert summary["observations"] == 20
    assert summary["agent_completed"] == summary["passed"] == 19
    assert summary["verification_passed"] == 20
    assert summary["agent_completed_rate"] == 0.95 and summary["verification_passed_rate"] == 1
    assert summary["manual_interventions"] == 1 and summary["recovery_scopes"] == ["not_exercised"]
    assert summary["elapsed_seconds"] == 20
    assert summary["attempted_calls"] == summary["completed_calls"] == 21
    assert summary["known_input_tokens"] == 160 and summary["input_tokens"] is None
    assert summary["known_total_tokens"] == 200 and summary["total_tokens_complete"] is False
    assert summary["known_cost_subtotal_usd"] == 0 and summary["cost_usd"] is None


@pytest.mark.parametrize(
    "tamper",
    [
        lambda rs: rs.pop(),
        lambda rs: rs[1].update(repeat_id=1),
        lambda rs: rs[1].update(repeat_id=3),
        lambda rs: rs[1].update(report_complete=False),
        lambda rs: rs[1].update(candidate={"commit": "d" * 40, "dirty": False}),
        lambda rs: rs[1].update(fixture_sha256="d" * 64),
        lambda rs: rs[1].update(runner_sha256="d" * 64),
        lambda rs: rs[1].update(measurement_kind="live-model"),
        lambda rs: rs[1]["limits"].update(max_calls=9),
        lambda rs: rs[1]["tasks"].pop(),
        lambda rs: rs[1]["tasks"][1].update(id=rs[1]["tasks"][0]["id"]),
        lambda rs: rs[1]["tasks"].append(rs[1]["tasks"][0]),
        lambda rs: rs[1]["tasks"][0].update(model_fingerprint="d" * 64),
        lambda rs: rs[1]["tasks"][0].update(agent_completed=False),
        lambda rs: rs[1]["tasks"][0].update(attempted_calls=2),
        lambda rs: rs[1]["tasks"][0].update(known_input_tokens=-1),
    ],
)
def test_aggregation_rejects_mixed_incomplete_or_contradictory_evidence(tamper):
    reports = [_evidence_report(1), _evidence_report(2)]
    tamper(reports)
    with pytest.raises(ValueError):
        _aggregate(reports)


def test_repeats_write_separate_reports_and_can_be_summarized_offline(tmp_path, monkeypatch):
    import subprocess
    from argparse import Namespace
    from pathlib import Path

    sample = _evidence_report(1)
    # Reports inside the checkout must not change the source identity between repeats.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "source.py").write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Offline Test",
            "-c",
            "user.email=offline@example.invalid",
            "commit",
            "-qm",
            "test fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    monkeypatch.setattr(acceptance, "verify", lambda *args: (True, None))
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")

    def scripted_worker(args, **kwargs):
        payload = json.loads(kwargs["text"])
        acceptance.write_json(Path(payload["meter"]), sample["tasks"][0])
        return 0, "", False

    monkeypatch.setattr(acceptance, "run_process", scripted_worker)
    tasks, fixture_hash = acceptance.load_tasks()
    args = Namespace(
        output=tmp_path / "runs",
        model="offline",
        timeout=180,
        max_calls=8,
        max_output_tokens=4096,
        repeats=2,
        repeat_id=1,
    )
    assert acceptance.run_repeats(args, tasks, fixture_hash) == 0
    paths = [args.output / f"repeat-{repeat}" / "report.json" for repeat in (1, 2)]
    for repeat, path in enumerate(paths, 1):
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["report_complete"] is True and report["repeat_id"] == repeat
        assert len(report["tasks"]) == 10
        assert report["source_run_id"] == 123 and report["source_run_attempt"] == 2
        assert report["candidate"]["dirty"] is False
    assert (
        acceptance.main(
            [
                "--aggregate",
                *map(str, paths),
                "--repeats",
                "2",
                "--output",
                str(tmp_path / "aggregate"),
            ]
        )
        == 0
    )
    combined = json.loads((tmp_path / "aggregate" / "aggregate.json").read_text(encoding="utf-8"))
    assert combined["passed"] == 20 and combined["cost_usd"] == 0
    original = acceptance.candidate(artifact_dir=args.output)
    (tmp_path / "source.py").write_text("changed = True\n", encoding="utf-8")
    changed = acceptance.candidate(artifact_dir=args.output)
    assert changed["dirty"] is True and changed["diff_sha256"] != original["diff_sha256"]


def test_timeout_overrides_a_stale_completed_meter(tmp_path, monkeypatch):
    from argparse import Namespace
    from pathlib import Path

    sample = _evidence_report(1)
    monkeypatch.setattr(acceptance, "candidate", lambda **kwargs: sample["candidate"])
    monkeypatch.setattr(acceptance, "verify", lambda *args: (True, None))

    def timeout(args, **kwargs):
        payload = json.loads(kwargs["text"])
        acceptance.write_json(Path(payload["meter"]), sample["tasks"][0])
        return 1, "", True

    monkeypatch.setattr(acceptance, "run_process", timeout)
    task = acceptance.load_tasks()[0][0]
    args = Namespace(
        output=tmp_path / "timeout", model="offline", timeout=1, max_calls=8, max_output_tokens=4096
    )
    assert acceptance.run_suite(args, [task], "f" * 64) == 1
    row = json.loads((args.output / "report.json").read_text(encoding="utf-8"))["tasks"][0]
    assert row["status"] == "timeout" and row["agent_completed"] is False
    assert row["verification_passed"] is True and row["passed"] is False
    assert row["input_tokens"] is None and row["input_tokens_complete"] is False
    assert row["known_input_tokens"] == 8 and row["cost_complete"] is False


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_nonfinite_timeout_is_rejected_before_worker_dispatch(
    tmp_path, monkeypatch, capsys, timeout
):
    monkeypatch.setattr(
        acceptance, "run_repeats", lambda *args: pytest.fail("must reject before model work")
    )
    with pytest.raises(SystemExit) as error:
        acceptance.main(["--model", "unused", "--output", str(tmp_path), f"--timeout={timeout}"])
    assert error.value.code == 2
    assert "all limits must be finite and positive" in capsys.readouterr().err
