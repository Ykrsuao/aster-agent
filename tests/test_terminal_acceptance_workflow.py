# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep live acceptance isolated, serial, and bound to release evidence provenance."""

from pathlib import Path

import pytest
import yaml

from scripts import check_release_evidence as evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workflow():
    return yaml.load(
        (ROOT / evidence.TASK_WORKFLOW).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )


def test_live_acceptance_requires_explicit_manual_authorization(workflow):
    assert set(workflow["on"]) == {"workflow_dispatch"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"confirm_live", "candidate_sha", "model_fingerprint"}
    assert inputs["confirm_live"]["type"] == "boolean"
    assert inputs["confirm_live"]["default"] == "false"
    assert all(item["required"] == "true" for item in inputs.values())
    job = next(iter(workflow["jobs"].values()))
    assert job["if"] == "github.event_name == 'workflow_dispatch' && inputs.confirm_live"
    assert job["environment"] == "terminal-acceptance"
    assert not job.get("continue-on-error")


def test_provider_requests_and_workflow_runs_remain_serial(workflow):
    assert len(workflow["jobs"]) == 1
    assert workflow["concurrency"] == {
        "group": evidence.TASK_ARTIFACT,
        "cancel-in-progress": "false",
    }
    job = next(iter(workflow["jobs"].values()))
    assert "strategy" not in job
    assert job["name"] == evidence.TASK_JOB
    assert job["runs-on"].startswith("windows-")
    assert job["timeout-minutes"] == "45"


def test_candidate_and_secrets_are_passed_only_to_the_validating_driver(workflow):
    assert workflow["permissions"] == {}
    assert "env" not in workflow
    job = next(iter(workflow["jobs"].values()))
    assert job["permissions"] == {"contents": "read"}
    assert "env" not in job
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": "0",
        "persist-credentials": "false",
    }
    driver = next(step for step in steps if "terminal_acceptance_ci.py" in step.get("run", ""))
    assert driver["env"] == {
        "CANDIDATE_SHA": "${{ inputs.candidate_sha }}",
        "EXPECTED_MODEL_FINGERPRINT": "${{ inputs.model_fingerprint }}",
        "DEFAULT_BRANCH": "${{ github.event.repository.default_branch }}",
        "TERMINAL_ACCEPTANCE_MODEL_CONFIG": "${{ vars.TERMINAL_ACCEPTANCE_MODEL_CONFIG }}",
        "TERMINAL_ACCEPTANCE_API_KEY": "${{ secrets.TERMINAL_ACCEPTANCE_API_KEY }}",
    }
    assert '--candidate-sha "$env:CANDIDATE_SHA"' in driver["run"]
    assert '--model-fingerprint "$env:EXPECTED_MODEL_FINGERPRINT"' in driver["run"]
    assert "--output (Join-Path $env:RUNNER_TEMP 'terminal-task-acceptance')" in driver["run"]
    assert "$LASTEXITCODE" in driver["run"]
    assert "continue-on-error" not in driver
    assert "${{" not in driver["run"]
    for step in steps:
        if step is not driver:
            assert "secrets." not in str(step)
            assert "TERMINAL_ACCEPTANCE_API_KEY" not in str(step)
    self_check = next(step for step in steps if "--self-check" in step.get("run", ""))
    assert steps.index(self_check) < steps.index(driver)
    assert not self_check.get("env")


def test_artifact_has_the_consumers_name_and_only_the_sanitized_report(workflow):
    steps = next(iter(workflow["jobs"].values()))["steps"]
    uploads = [
        step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    upload = uploads[0]
    assert upload["with"]["name"] == evidence.TASK_ARTIFACT
    assert upload["with"]["path"] == "${{ runner.temp }}/terminal-task-acceptance/report.json"
    assert upload["with"]["if-no-files-found"] == "error"
    assert "include-hidden-files" not in upload["with"]
    assert upload["if"] == "always() && steps.report.outputs.present == 'true'"
    locator = next(step for step in steps if step.get("id") == "report")
    assert locator["if"] == "always()"
    assert "Test-Path -LiteralPath $report -PathType Leaf" in locator["run"]
    assert "terminal-task-acceptance/report.json" in locator["run"]
    assert "present=true" in locator["run"]
    assert steps.index(locator) < steps.index(upload)
