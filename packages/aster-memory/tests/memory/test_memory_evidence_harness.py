# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The offline evidence harness must measure tool results, not reward its own claims."""

import copy
import hashlib
import json

import pytest

from aster.unifiedllm import FakeLLMClient
from experiments.memory_evidence.run import (
    ARMS,
    FIXTURES,
    ROOT,
    load_fixture,
    run_case,
    run_experiment,
    seed_store,
)
from experiments.memory_evidence.scoring import score_case, summarize_arm


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("USERPROFILE", str(user))
    monkeypatch.setenv("ASTER_USER_DIR", str(user / "aster"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(user / "project"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.delenv("ASTER_LLM_CONFIG", raising=False)


async def test_real_memory_controls_survive_reopen_and_are_scored_independently():
    cases, fixture_hash = load_fixture()
    report = await run_experiment(repeats=1)
    assert report["report_complete"]
    assert report["measurement_kind"] == "offline-scripted"
    assert report["real_model_benefit_verified"] is False
    assert report["fixture_sha256"] == fixture_hash
    assert len(fixture_hash) == len(report["runner_sha256"]) == 64
    for relative, expected_hash in report["source_files_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash
    assert set(report["arms"]) == set(ARMS)
    for arm, data in report["arms"].items():
        assert data["scores"]["cases"] == len(cases)
        expected_successes = len(cases) if arm in {"current", "corrected"} else 0
        assert data["scores"]["answer_correct"]["true_count"] == expected_successes
        assert data["scores"]["task_completed"]["true_count"] == expected_successes
        assert data["scores"]["stale_adoption"]["true_count"] == (
            len(cases) if arm == "stale" else 0
        )
        assert data["scores"]["retrieval_covered_facts"]["retrieved_facts"] == (
            0 if arm == "off" else 2 * len(cases)
        )
        assert data["scores"]["retrieval_current_facts"]["retrieved_facts"] == (
            2 * len(cases) if arm in {"current", "corrected"} else 0
        )
        assert data["model_calls"] == len(cases)
        assert data["external_provider_calls"] == 0
        assert data["usage"]["input_tokens"] is None
        assert data["usage"]["output_tokens"] is None
        assert data["usage"]["cost_usd"] is None
        assert not data["usage"]["input_tokens_complete"]
        assert not data["usage"]["cost_complete"]
        assert data["stores"][0]["explicit_corrections"] == (
            2 * len(cases) if arm == "corrected" else 0
        )
        failures = [
            (row["case_id"], row["error_type"]) for row in data["rows"] if row["error_type"]
        ]
        assert not failures, failures
        assert all(row["repeat_id"] == 1 for row in data["rows"])
        assert all(
            row["recovery_scope"] == "new-agent-existing-memory-store" for row in data["rows"]
        )
    serialized = json.dumps(report)
    for case in cases:
        assert case["question"] not in serialized
        for value in [*case["current"].values(), *case["obsolete"].values()]:
            assert value not in serialized


async def test_answer_changes_with_actual_store_contents_without_changing_the_gold(tmp_path):
    cases, _ = load_fixture()
    gold = cases[0]
    seeded = copy.deepcopy(gold)
    seeded["current"][gold["primary_key"]] = "deliberately-replaced-store-value"
    path = tmp_path / "modified.sqlite"
    await seed_store(path, [seeded], "current")
    row, _ = await run_case(path, gold, "current", 1, top_k=3, timeout=30)
    assert row["error_type"] is None
    assert (
        row["answer_evidence"]["answer_sha256"]
        == hashlib.sha256(b"deliberately-replaced-store-value").hexdigest()
    )
    assert row["agent_completed"] is True
    assert row["scores"]["answer_correct"] is False
    assert row["verification_passed"] is False


async def test_failed_model_call_does_not_become_zero_usage_or_a_scored_answer(tmp_path):
    case = load_fixture()[0][0]
    row, _ = await run_case(
        tmp_path / "unused.sqlite",
        case,
        "off",
        1,
        top_k=3,
        timeout=30,
        client_factory=lambda: FakeLLMClient(strict_exhaustion=True),
    )
    assert row["error_type"]
    assert row["response_returned"] is False
    assert row["model_calls"] > 0
    assert row["usage"]["attempted_calls"] == row["model_calls"]
    assert row["usage"]["completed_calls"] == 0
    assert row["scores"]["answer_correct"] is None
    assert row["scores"]["retrieval_covered_facts"] is None
    assert row["scores"]["stale_adoption"] is None
    assert row["scores"]["task_completed"] is False
    assert row["usage"]["cost_usd"] is None
    summary = summarize_arm([row])
    assert summary["answer_correct"]["unknown_cases"] == 1
    assert summary["answer_correct"]["rate"] is None


def test_correct_primary_answer_is_separate_from_stale_task_fields_and_completion_claim():
    case = load_fixture()[0][0]
    primary, secondary = case["current"]
    notes = [
        {"memory_id": key, "subject": case["subject"], "key": key, "value": value}
        for key, value in case["current"].items()
    ]
    answer = {
        "answer": case["current"][primary],
        "task_fields": {primary: case["current"][primary], secondary: case["obsolete"][secondary]},
        "citations": {key: key for key in case["current"]},
        "claimed_complete": True,
    }
    scores = score_case(case, answer, [{"complete": True, "notes": notes}])
    assert scores["answer_correct"] is True
    assert scores["stale_adoption"] is True
    assert scores["task_completed"] is False
    answer["task_fields"] = dict(case["current"])
    assert score_case(case, answer, [{"complete": True, "notes": notes}])["task_completed"]
    assert not score_case(case, answer, [])["task_completed"]


@pytest.mark.parametrize("setting", [{"repeats": 0}, {"top_k": 0}, {"timeout": float("nan")}])
async def test_invalid_limits_fail_before_running(setting):
    with pytest.raises(ValueError):
        await run_experiment(**setting)


def test_fixture_rejects_ambiguous_gold_and_reports_actual_file_hash(tmp_path):
    document = json.loads(FIXTURES.read_text(encoding="utf-8"))
    document["cases"].append(document["cases"][0])
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_fixture(path)
