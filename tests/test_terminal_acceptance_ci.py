# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise CI admission and report contracts without any provider requests."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import terminal_acceptance as acceptance
from scripts import terminal_acceptance_ci as ci

CONFIG = {
    "model_name": "openai/acceptance-test",
    "client_type": "completion",
    "api_base": "https://gateway.example/v1",
    "temperature": 0,
}
IDENTITY = {
    "client_type": "aster.unifiedllm.unifiedllm.CompletionClient",
    "model": CONFIG["model_name"],
    "endpoint_origin": "https://gateway.example",
    "sampling": {"temperature": 0, "max_tokens": 4096},
    "reasoning_level": None,
}
FINGERPRINT = acceptance.fingerprint_model(IDENTITY)


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "source.txt").write_text("candidate", encoding="utf-8")
    commands = [
        ["init", "-q", "-b", "main"],
        ["add", "source.txt"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "candidate",
        ],
    ]
    for command in commands:
        subprocess.run(["git", *command], cwd=checkout, capture_output=True, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", sha], cwd=checkout, check=True)
    temp = tmp_path / "runner-temp"
    temp.mkdir()
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "inputs": {
                    "confirm_live": "true",
                    "candidate_sha": sha,
                    "model_fingerprint": FINGERPRINT,
                },
                "repository": {"full_name": "owner/repo", "default_branch": "main"},
            }
        ),
        encoding="utf-8",
    )
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_SHA": sha,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_WORKFLOW_REF": "owner/repo/.github/workflows/terminal-acceptance.yml@refs/heads/main",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "DEFAULT_BRANCH": "main",
        "RUNNER_TEMP": str(temp),
        ci.KEY_ENV: "test-credential-never-log",
        ci.CONFIG_ENV: json.dumps(CONFIG),
    }.items():
        monkeypatch.setenv(name, value)
    # Simulate only this module's platform gate, preserving pathlib's host semantics.
    monkeypatch.setattr(
        ci, "os", SimpleNamespace(name="nt", environ=os.environ, devnull=os.devnull)
    )
    monkeypatch.setattr(acceptance, "ROOT", checkout)

    async def identity(_config):
        return IDENTITY.copy()

    monkeypatch.setattr(ci, "resolved_identity", identity)
    return SimpleNamespace(sha=sha, root=checkout, output=temp / "report", event=event)


def options(dispatch):
    return [
        "--candidate-sha",
        dispatch.sha,
        "--model-fingerprint",
        FINGERPRINT,
        "--output",
        str(dispatch.output),
    ]


def report_for(args, tasks, fixture_hash):
    args.output.mkdir()
    report = {
        "schema_version": 1,
        "report_complete": True,
        "measurement_kind": "live-model",
        "repeat_id": 1,
        "candidate": acceptance.candidate(),
        "fixture_sha256": fixture_hash,
        "runner_sha256": ci.evidence.sha256(Path(acceptance.__file__).read_bytes()),
        "model_identity": IDENTITY.copy(),
        "model_fingerprint": FINGERPRINT,
        "source_run_id": 123,
        "source_run_attempt": 2,
        "limits": ci.evidence.TASK_LIMITS.copy(),
        "total": len(tasks),
        "passed": len(tasks),
        "tasks": [
            {
                "id": task["id"],
                "passed": True,
                "agent_completed": True,
                "verification_passed": True,
                "status": "completed",
                "model_identity": IDENTITY.copy(),
                "model_fingerprint": FINGERPRINT,
            }
            for task in tasks
        ],
    }
    return report


def test_success_uses_fixed_serial_budget_and_restores_private_environment(
    dispatch, monkeypatch, capsys
):
    monkeypatch.setenv("ASTER_LLM_CONFIG", "unrelated-registry")
    monkeypatch.setenv("ASTER_SECRETS", "unrelated-secrets")
    paths = []

    def runner(args, tasks, fixture_hash):
        assert (args.timeout, args.max_calls, args.max_output_tokens) == (180, 8, 4096)
        assert args.serial_calls and args.require_live_model
        assert args.expected_model_fingerprint == FINGERPRINT
        registry = Path(os.environ["ASTER_LLM_CONFIG"])
        paths.append(registry)
        saved = json.loads(registry.read_text(encoding="utf-8"))
        assert saved["models"][ci.MODEL_ALIAS]["api_key_env"] == ci.KEY_ENV
        assert os.environ[ci.KEY_ENV] not in registry.read_text(encoding="utf-8")
        assert not Path(os.environ["ASTER_SECRETS"]).exists()
        assert os.environ["ASTER_MODEL_METADATA_REFRESH"] == "False"
        print("test-credential-never-log raw transcript")
        acceptance.write_json(args.output / "report.json", report_for(args, tasks, fixture_hash))
        return 0

    monkeypatch.setattr(acceptance, "run_suite", runner)
    assert ci.main(options(dispatch)) == 0
    assert os.environ["ASTER_LLM_CONFIG"] == "unrelated-registry"
    assert os.environ["ASTER_SECRETS"] == "unrelated-secrets"
    assert all(not path.exists() for path in paths)
    captured = capsys.readouterr()
    assert "all frozen tasks verified" in captured.out
    assert "test-credential" not in captured.out + captured.err
    assert "raw transcript" not in captured.out + captured.err


@pytest.mark.parametrize(
    "change",
    [
        "no_confirm",
        "wrong_event",
        "wrong_sha",
        "wrong_workflow",
        "unreachable",
        "dirty",
        "output_in_source",
        "existing_output",
        "missing_key",
        "wrong_fingerprint",
        "stale_attempt",
    ],
)
def test_preflight_refuses_without_running_tasks(dispatch, monkeypatch, change):
    def unexpected(*_args):
        pytest.fail("must not start tasks")

    monkeypatch.setattr(acceptance, "run_suite", unexpected)
    event = json.loads(dispatch.event.read_text(encoding="utf-8"))
    if change == "no_confirm":
        event["inputs"]["confirm_live"] = "false"
        dispatch.event.write_text(json.dumps(event), encoding="utf-8")
    elif change == "wrong_event":
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    elif change == "wrong_sha":
        monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    elif change == "wrong_workflow":
        monkeypatch.setenv(
            "GITHUB_WORKFLOW_REF", "owner/repo/.github/workflows/ci.yml@refs/heads/main"
        )
    elif change == "unreachable":
        subprocess.run(
            ["git", "update-ref", "-d", "refs/remotes/origin/main"], cwd=dispatch.root, check=True
        )
    elif change == "dirty":
        (dispatch.root / "source.txt").write_text("changed", encoding="utf-8")
    elif change == "output_in_source":
        dispatch.output = dispatch.root / "reports"
    elif change == "existing_output":
        dispatch.output.mkdir()
    elif change == "missing_key":
        monkeypatch.delenv(ci.KEY_ENV)
    elif change == "wrong_fingerprint":

        async def different(_config):
            return {**IDENTITY, "model": "openai/different"}

        monkeypatch.setattr(ci, "resolved_identity", different)
    elif change == "stale_attempt":
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "0")
    assert ci.main(options(dispatch)) == 1


@pytest.mark.parametrize(
    "change",
    [
        "offline",
        "identity",
        "missing_task",
        "duplicate_task",
        "failed_task",
        "wrong_attempt",
        "source_changed",
        "wrong_budget",
    ],
)
def test_final_report_must_match_the_consumer_contract(dispatch, monkeypatch, change):
    def runner(args, tasks, fixture_hash):
        report = report_for(args, tasks, fixture_hash)
        if change == "offline":
            report["measurement_kind"] = "offline-scripted"
        elif change == "identity":
            report["model_identity"]["model"] = "openai/other"
        elif change == "missing_task":
            report["tasks"].pop()
        elif change == "duplicate_task":
            report["tasks"][-1] = report["tasks"][0]
        elif change == "failed_task":
            report["tasks"][0]["verification_passed"] = False
        elif change == "wrong_attempt":
            report["source_run_attempt"] = 1
        elif change == "source_changed":
            (dispatch.root / "source.txt").write_text("changed", encoding="utf-8")
        elif change == "wrong_budget":
            report["limits"]["max_calls"] = 9
        acceptance.write_json(args.output / "report.json", report)
        return 0

    monkeypatch.setattr(acceptance, "run_suite", runner)
    assert ci.main(options(dispatch)) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"api_key": "secret"},
        {"api_key_env": "OTHER_KEY"},
        {"base_url": "https://other.example"},
        {"provider_credentials": []},
        {"extra_body": {}},
        {"retry_config": {}},
        {"temperature": float("nan")},
        {"top_p": float("inf")},
        {"context_window": True},
        {"api_base": "https://user:secret@gateway.example"},
        {"api_base": "https://gateway.example?key=secret"},
        {"api_base": "http://gateway.example"},
        {"api_base": "https://gateway.example\n"},
        {"client_type": "fake"},
        {"model_name": "otherprovider/model"},
    ],
)
def test_model_config_rejects_credential_and_budget_bypasses(extra):
    with pytest.raises((ci.evidence.EvidenceError, ValueError)):
        ci.model_config(json.dumps({**CONFIG, **extra}))


def test_duplicate_json_configuration_is_rejected():
    with pytest.raises(ci.evidence.EvidenceError):
        ci.model_config('{"model_name":"openai/a","model_name":"openai/b"}')


def test_describe_model_is_offline_and_never_uses_a_real_key(monkeypatch, capsys):
    monkeypatch.setenv(ci.CONFIG_ENV, json.dumps(CONFIG))
    monkeypatch.delenv(ci.KEY_ENV, raising=False)
    observed = []

    class Client:
        async def aclose(self):
            observed.append("closed")

    def construct(alias, config, **kwargs):
        assert alias == ci.MODEL_ALIAS and kwargs["api_key"] == "fingerprint-only"
        assert os.environ["ASTER_MODEL_METADATA_REFRESH"] == "False"
        return Client()

    monkeypatch.setattr("aster.unifiedllm.registry.client_from_config", construct)
    monkeypatch.setattr(acceptance, "model_identity", lambda _client: IDENTITY.copy())
    assert ci.main(["--describe-model"]) == 0
    assert observed == ["closed"]
    assert json.loads(capsys.readouterr().out)["model_fingerprint"] == FINGERPRINT


def test_constructor_errors_do_not_disclose_configuration(monkeypatch, capsys):
    monkeypatch.setenv(ci.CONFIG_ENV, json.dumps(CONFIG))

    def fail(*_args, **_kwargs):
        raise RuntimeError("credential-private-value")

    monkeypatch.setattr("aster.unifiedllm.registry.client_from_config", fail)
    assert ci.main(["--describe-model"]) == 1
    captured = capsys.readouterr()
    assert "credential-private-value" not in captured.out + captured.err


@pytest.mark.parametrize(
    "client_type, cap_key", [("completion", "max_tokens"), ("responses", "max_output_tokens")]
)
def test_actual_client_identity_is_resolved_offline_and_keeps_reply_cap(
    monkeypatch, tmp_path, client_type, cap_key
):
    import httpx

    def no_network(*_args, **_kwargs):
        pytest.fail("offline identity resolution attempted network access")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    config = ci.model_config(json.dumps({**CONFIG, "client_type": client_type}))
    with ci.isolated_environment(tmp_path, config):
        identity = ci.asyncio.run(ci.resolved_identity(config))
    assert identity["sampling"][cap_key] == 4096
    changed = {**identity, "sampling": {**identity["sampling"], cap_key: 8192}}
    assert acceptance.fingerprint_model(changed) != acceptance.fingerprint_model(identity)


@pytest.mark.parametrize("confirmation", [1, 0, "false", False, None])
def test_dispatch_confirmation_requires_true_boolean_or_string(dispatch, confirmation):
    event = json.loads(dispatch.event.read_text(encoding="utf-8"))
    event["inputs"]["confirm_live"] = confirmation
    dispatch.event.write_text(json.dumps(event), encoding="utf-8")
    with pytest.raises(ci.evidence.EvidenceError):
        ci.dispatch_context(dispatch.sha, FINGERPRINT, dispatch.output)


def test_final_tag_dispatch_on_reachable_candidate_is_allowed(dispatch, monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v1.2.3")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repo/.github/workflows/terminal-acceptance.yml@refs/tags/v1.2.3",
    )
    run = ci.dispatch_context(dispatch.sha, FINGERPRINT, dispatch.output)
    assert run == {"workflow_run": {"id": 123}, "run_attempt": 2}
