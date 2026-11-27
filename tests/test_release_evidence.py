# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline release provenance and fail-closed publication checks."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
import tarfile
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OLD_SHA = "b" * 40
MODEL_IDENTITY = {
    "client_type": "test.provider.Client",
    "model": "resolved-model",
    "endpoint_origin": "https://models.example.com",
    "sampling": {"temperature": 0, "max_tokens": 4096},
    "reasoning_level": None,
}
FINGERPRINT = hashlib.sha256(
    json.dumps(
        MODEL_IDENTITY, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
).hexdigest()
TAG = "v1.2.3"
OLD_TAG = "v1.2.2"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "_release_evidence", ROOT / "scripts/check_release_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def wheel(name: str, version: str, *, native: bool = False) -> bytes:
    metadata = f"{name.replace('-', '_')}-{version}.dist-info"
    files = {
        f"{metadata}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
        f"{metadata}/WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: "
            + ("false\nTag: py3-none-win_amd64\n" if native else "true\nTag: py3-none-any\n")
        ).encode(),
    }
    if native:
        files["aster_cli/bin/aster.exe"] = b"MZ-test-terminal-" + version.encode()
    return zip_bytes(files)


def sdist(name: str, version: str) -> bytes:
    content = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{name}-{version}/PKG-INFO")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def terminal_bundle(gate, version: str, sha: str) -> bytes:
    files = {
        "install.ps1": b"# fixture only; never executed",
        "constraints.txt": b"",
        "LICENSE": b"fixture",
    }
    packages = {}
    for name in sorted(gate.PACKAGES):
        platform = "py3-none-win_amd64" if name == "aster-cli" else "py3-none-any"
        filename = f"{name.replace('-', '_')}-{version}-{platform}.whl"
        packages[name] = filename
        files[f"wheels/{filename}"] = wheel(name, version, native=name == "aster-cli")
    hashes = {name: gate.sha256(data) for name, data in files.items()}
    manifest = {
        "schema_version": 1,
        "platform": "win_amd64",
        "version": version,
        "commit": sha,
        "dirty": False,
        "packages": packages,
        "files": hashes,
    }
    files["MANIFEST.json"] = encoded(manifest)
    checksums = {**hashes, "MANIFEST.json": gate.sha256(files["MANIFEST.json"])}
    files["SHA256SUMS"] = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
    ).encode()
    return zip_bytes(files)


@pytest.fixture
def evidence(gate):
    class OfflineGitHub(gate.GitHub):
        def __init__(self):
            super().__init__("owner/aster", "never-used")
            self.responses = {}
            self.requests = []

        def _request(self, path, *, download=False):
            self.requests.append((path, download))
            require_path = path.split("?", 1)[0]
            if require_path not in self.responses:
                raise gate.EvidenceError(f"No trusted evidence: {path}")
            value = self.responses[require_path]
            return value if isinstance(value, bytes) else encoded(value)

    api = OfflineGitHub()
    api.responses[""] = {"id": 11, "full_name": "owner/aster", "default_branch": "main"}
    api.responses[f"/compare/{SHA}...main"] = {"status": "ahead", "base_commit": {"sha": SHA}}
    for tag, sha in ((TAG, SHA), (OLD_TAG, OLD_SHA)):
        api.responses[f"/git/ref/tags/{tag}"] = {"object": {"type": "commit", "sha": sha}}
    for path, workflow_id in ((gate.CI_WORKFLOW, 100), (gate.TASK_WORKFLOW, 101)):
        api.responses[f"/actions/workflows/{Path(path).name}"] = {"id": workflow_id, "path": path}
    for run_id, workflow_id, workflow, jobs in (
        (42, 101, gate.TASK_WORKFLOW, {gate.TASK_JOB}),
        (43, 100, gate.CI_WORKFLOW, gate.REQUIRED_JOBS),
    ):
        api.responses[f"/actions/runs/{run_id}"] = {
            "id": run_id,
            "workflow_id": workflow_id,
            "path": workflow,
            "event": "workflow_dispatch",
            "head_sha": SHA,
            "repository": {"id": 11},
            "head_repository": {"id": 11},
            "status": "completed",
            "conclusion": "success",
            "run_attempt": 1,
        }
        rows = [
            {
                "id": index,
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "head_sha": SHA,
            }
            for index, name in enumerate(sorted(jobs), 1)
        ]
        api.responses[f"/actions/runs/{run_id}/attempts/1/jobs"] = {
            "total_count": len(rows),
            "jobs": rows,
        }
    fixture = (ROOT / "tests/fixtures/terminal_acceptance/tasks.json").read_bytes()
    runner = (ROOT / "scripts/terminal_acceptance.py").read_bytes()
    for name, data in (
        ("tests/fixtures/terminal_acceptance/tasks.json", fixture),
        ("scripts/terminal_acceptance.py", runner),
    ):
        api.responses[f"/contents/{name}"] = {
            "type": "file",
            "encoding": "base64",
            "size": len(data),
            "content": base64.b64encode(data).decode(),
        }
    report = {
        "schema_version": 1,
        "candidate": {"commit": SHA, "dirty": False},
        "fixture_sha256": gate.sha256(fixture),
        "runner_sha256": gate.sha256(runner),
        "model_fingerprint": FINGERPRINT,
        "model_identity": MODEL_IDENTITY.copy(),
        "measurement_kind": "live-model",
        "repeat_id": 1,
        "report_complete": True,
        "source_run_id": 42,
        "source_run_attempt": 1,
        "limits": gate.TASK_LIMITS.copy(),
        "total": 10,
        "passed": 10,
        "tasks": [
            {
                "id": task["id"],
                "category": task["category"],
                "passed": True,
                "agent_completed": True,
                "model_identity": MODEL_IDENTITY.copy(),
                "model_fingerprint": FINGERPRINT,
                "verification_passed": True,
                "status": "completed",
            }
            for task in json.loads(fixture)["tasks"]
        ],
    }
    candidate_zip = terminal_bundle(gate, "1.2.3", SHA)
    baseline_zip = terminal_bundle(gate, "1.2.2", OLD_SHA)
    api.responses[f"/releases/tags/{OLD_TAG}"] = {
        "tag_name": OLD_TAG,
        "draft": False,
        "prerelease": False,
        "published_at": "2026-11-01T00:00:00Z",
        "assets": [
            {
                "id": 500,
                "name": "aster-terminal-1.2.2-windows-x64.zip",
                "digest": f"sha256:{gate.sha256(baseline_zip)}",
            }
        ],
    }
    api.responses["/releases/assets/500"] = baseline_zip

    def publish_artifact(artifact_id, name, run_id, data):
        api.responses[f"/actions/artifacts/{artifact_id}"] = {
            "id": artifact_id,
            "name": name,
            "expired": False,
            "digest": f"sha256:{gate.sha256(data)}",
            "workflow_run": {
                "id": run_id,
                "repository_id": 11,
                "head_repository_id": 11,
                "head_sha": SHA,
            },
        }
        api.responses[f"/actions/artifacts/{artifact_id}/zip"] = data

    def refresh_task():
        publish_artifact(700, gate.TASK_ARTIFACT, 42, zip_bytes({"report.json": encoded(report)}))

    refresh_task()
    _, baseline = gate.baseline_evidence(api, OLD_TAG, "1.2.3")
    report_bytes, task_source = gate.task_evidence(api, 700, SHA, FINGERPRINT)
    package = gate.bundle_evidence(candidate_zip, SHA, "1.2.3")
    smoke = {
        **package,
        "session_reopened_without_execution": True,
        "continued_after_install": True,
        "saved_input_bytes_preserved": True,
    }
    sources = {"baseline": baseline, "task": task_source}
    bundle_files = gate.archive_files(candidate_zip)
    files = {
        "terminal/aster-terminal-1.2.3-windows-x64.zip": candidate_zip,
        "terminal/MANIFEST.json": bundle_files["MANIFEST.json"],
        "terminal/SHA256SUMS": bundle_files["SHA256SUMS"],
        "evidence/task-report.json": report_bytes,
        "evidence/sources.json": encoded(sources),
        "evidence/install.json": encoded(smoke),
        "evidence/upgrade.json": encoded(
            {
                **smoke,
                "cross_version": True,
                "upgrade_verified": True,
                "seed_package": baseline["package"],
            }
        ),
    }
    for name in gate.PACKAGES:
        normalized = name.replace("-", "_")
        files[f"dist/{normalized}-1.2.3-py3-none-any.whl"] = wheel(name, "1.2.3")
        files[f"dist/{normalized}-1.2.3.tar.gz"] = sdist(name, "1.2.3")
    manifest = {
        "schema_version": 1,
        "repository": "owner/aster",
        "candidate_sha": SHA,
        "tag": TAG,
        "version": "1.2.3",
        "run_id": 43,
        "run_attempt": 1,
        "sources": sources,
        "files": {name: gate.sha256(data) for name, data in files.items()},
    }

    def refresh_candidate(*, update_hashes=True):
        if update_hashes:
            manifest["files"] = {name: gate.sha256(data) for name, data in files.items()}
        publish_artifact(
            800,
            gate.candidate_name(TAG, SHA, 1),
            43,
            zip_bytes({**files, "release-evidence.json": encoded(manifest)}),
        )

    refresh_candidate()
    api.requests.clear()
    return SimpleNamespace(
        api=api,
        report=report,
        manifest=manifest,
        files=files,
        refresh_task=refresh_task,
        refresh_candidate=refresh_candidate,
        candidate_zip=candidate_zip,
        package=package,
    )


def test_fetch_materializes_only_verified_original_bytes(gate, evidence, tmp_path, monkeypatch):
    def unexpected_command(*args, **kwargs):
        pytest.fail("Fetching accepted evidence must not build or run a model")

    monkeypatch.setattr(gate.subprocess, "check_output", unexpected_command)
    destination = tmp_path / "accepted"
    gate.fetch(evidence.api, TAG, 800, destination)
    for name, original in evidence.files.items():
        assert (destination / name).read_bytes() == original
    assert (destination / "release-evidence.json").read_bytes() == encoded(evidence.manifest)
    assert ("/actions/runs/43/attempts/1/jobs?per_page=100&page=1", False) in evidence.api.requests
    assert (f"/contents/scripts/terminal_acceptance.py?ref={SHA}", False) in evidence.api.requests


def test_prepare_seal_fetch_round_trip_is_offline(gate, evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "clean_checkout", lambda tag: SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", "43")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    source = tmp_path / "built"
    source.mkdir()
    (source / "candidate.zip").write_bytes(evidence.candidate_zip)
    inputs = tmp_path / "inputs"
    args = SimpleNamespace(
        tag=TAG,
        upgrade_from_tag=OLD_TAG,
        task_artifact_id=700,
        model_fingerprint=FINGERPRINT,
        bundle=source,
        destination=inputs,
    )
    gate.prepare(evidence.api, args)
    assert (inputs / "baseline/install.ps1").is_file()
    assert (inputs / "candidate/MANIFEST.json").is_file()
    staging = tmp_path / "staging"
    files = dict(evidence.files)
    for name in ("sources.json", "task-report.json"):
        files[f"evidence/{name}"] = (inputs / "evidence" / name).read_bytes()
    gate.write_files(staging, files)
    args.directory = staging
    gate.seal(evidence.api, args)
    sealed = (staging / "release-evidence.json").read_bytes()
    data = zip_bytes({**files, "release-evidence.json": sealed})
    evidence.api.responses["/actions/artifacts/800/zip"] = data
    evidence.api.responses["/actions/artifacts/800"]["digest"] = f"sha256:{gate.sha256(data)}"
    gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")
    with pytest.raises(gate.EvidenceError, match="already sealed"):
        gate.seal(evidence.api, args)


@pytest.mark.parametrize(
    "field,value",
    [("expired", True), ("name", "dist"), ("digest", None), ("digest", "sha256:" + "0" * 64)],
)
def test_rejects_unproven_or_corrupt_artifact(gate, evidence, tmp_path, field, value):
    evidence.api.responses["/actions/artifacts/800"][field] = value
    with pytest.raises(gate.EvidenceError):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")
    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize(
    "field,value", [("head_sha", OLD_SHA), ("repository_id", 12), ("head_repository_id", 12)]
)
def test_rejects_artifact_from_other_commit_or_repository(gate, evidence, tmp_path, field, value):
    evidence.api.responses["/actions/artifacts/800"]["workflow_run"][field] = value
    with pytest.raises(gate.EvidenceError):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", OLD_SHA),
        ("workflow_id", 101),
        ("path", "other.yml"),
        ("event", "pull_request"),
        ("event", "push"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("head_repository", {"id": 12}),
        ("run_attempt", 2),
    ],
)
def test_rejects_untrusted_run_or_stale_attempt(gate, evidence, tmp_path, field, value):
    evidence.api.responses["/actions/runs/43"][field] = value
    with pytest.raises(gate.EvidenceError):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


@pytest.mark.parametrize("failure", ["missing", "skipped", "failed", "duplicate", "wrong-sha"])
def test_requires_every_current_candidate_job(gate, evidence, tmp_path, failure):
    data = evidence.api.responses["/actions/runs/43/attempts/1/jobs"]
    job = next(row for row in data["jobs"] if row["name"] == "release-upgrade")
    if failure == "missing":
        data["jobs"].remove(job)
    elif failure == "duplicate":
        data["jobs"].append(copy.deepcopy(job))
    elif failure == "wrong-sha":
        job["head_sha"] = OLD_SHA
    else:
        job["conclusion"] = "skipped" if failure == "skipped" else "failure"
    data["total_count"] = len(data["jobs"])
    with pytest.raises(gate.EvidenceError, match="Required candidate job"):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


def test_candidate_must_be_in_default_branch(gate, evidence, tmp_path):
    evidence.api.responses[f"/compare/{SHA}...main"]["status"] = "diverged"
    with pytest.raises(gate.EvidenceError, match="default branch"):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


def test_absent_trusted_task_workflow_cannot_be_replaced_by_manual_json(gate, evidence, tmp_path):
    del evidence.api.responses[f"/actions/workflows/{Path(gate.TASK_WORKFLOW).name}"]
    with pytest.raises(gate.EvidenceError, match="No trusted evidence"):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")
    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", OLD_SHA),
        ("workflow_id", 100),
        ("path", ".github/workflows/ci.yml"),
        ("event", "push"),
        ("conclusion", "failure"),
    ],
)
def test_task_report_requires_its_own_successful_trusted_execution(gate, evidence, field, value):
    evidence.api.responses["/actions/runs/42"][field] = value
    with pytest.raises(gate.EvidenceError):
        gate.task_evidence(evidence.api, 700, SHA, FINGERPRINT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_fingerprint", "f" * 64),
        ("schema_version", True),
        ("report_complete", False),
        ("measurement_kind", "offline-scripted"),
        ("repeat_id", 0),
        ("repeat_id", True),
        ("source_run_id", 43),
        ("source_run_id", True),
        ("source_run_attempt", 2),
        ("source_run_attempt", True),
        ("fixture_sha256", "0" * 64),
        ("runner_sha256", "0" * 64),
        ("candidate", {"commit": SHA, "dirty": True}),
        ("candidate", {"commit": OLD_SHA, "dirty": False}),
        ("passed", True),
        ("total", 9),
    ],
)
def test_report_must_match_candidate_and_execution_contract(gate, evidence, field, value):
    evidence.report[field] = value
    evidence.refresh_task()
    with pytest.raises(gate.EvidenceError):
        gate.task_evidence(evidence.api, 700, SHA, FINGERPRINT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_completed", False),
        ("model_fingerprint", "f" * 64),
        ("model_identity", {**MODEL_IDENTITY, "client_type": "different.provider.Client"}),
    ],
)
def test_every_task_must_finish_with_the_same_resolved_model(gate, evidence, field, value):
    evidence.report["tasks"][0][field] = value
    evidence.refresh_task()
    with pytest.raises(gate.EvidenceError):
        gate.task_evidence(evidence.api, 700, SHA, FINGERPRINT)


def test_model_identity_must_hash_to_the_requested_fingerprint(gate, evidence):
    evidence.report["model_identity"]["model"] = "another-model"
    evidence.refresh_task()
    with pytest.raises(gate.EvidenceError, match="identity fingerprint"):
        gate.task_evidence(evidence.api, 700, SHA, FINGERPRINT)


@pytest.mark.parametrize(
    "identity",
    [
        {key: value for key, value in MODEL_IDENTITY.items() if key != "sampling"},
        {**MODEL_IDENTITY, "version": 1},
        {**MODEL_IDENTITY, "client_type": ""},
        {**MODEL_IDENTITY, "model": ""},
        {**MODEL_IDENTITY, "endpoint_origin": False},
        {**MODEL_IDENTITY, "sampling": []},
        {**MODEL_IDENTITY, "reasoning_level": False},
    ],
)
def test_model_identity_requires_the_shared_five_field_schema(gate, evidence, identity):
    fingerprint = gate.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    )
    evidence.report["model_identity"] = identity
    evidence.report["model_fingerprint"] = fingerprint
    for task in evidence.report["tasks"]:
        task["model_identity"] = identity
        task["model_fingerprint"] = fingerprint
    evidence.refresh_task()
    with pytest.raises(gate.EvidenceError, match="resolved model identity"):
        gate.task_evidence(evidence.api, 700, SHA, fingerprint)


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "need-input", "failed-check", "not-done", "budget", "boolean-budget"],
)
def test_passed_true_never_substitutes_for_complete_fixed_task_evidence(gate, evidence, failure):
    results = evidence.report["tasks"]
    if failure == "missing":
        results.pop()
    elif failure == "duplicate":
        results[-1] = copy.deepcopy(results[0])
    elif failure == "budget":
        evidence.report["limits"]["max_calls"] += 1
    elif failure == "boolean-budget":
        evidence.report["limits"]["endpoint_retries"] = False
    elif failure == "need-input":
        results[0]["status"] = "needs_input"
    elif failure == "failed-check":
        results[0]["verification_passed"] = False
    else:
        results[0]["passed"] = False
    evidence.refresh_task()
    with pytest.raises(gate.EvidenceError):
        gate.task_evidence(evidence.api, 700, SHA, FINGERPRINT)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_sha", OLD_SHA),
        ("tag", OLD_TAG),
        ("version", "1.2.4"),
        ("repository", "other/aster"),
        ("run_id", 42),
        ("run_attempt", 2),
    ],
)
def test_manifest_is_bound_to_explicit_tag_run_and_attempt(gate, evidence, tmp_path, field, value):
    evidence.manifest[field] = value
    evidence.refresh_candidate()
    with pytest.raises(gate.EvidenceError, match="Candidate manifest"):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


def test_inner_file_hashes_are_checked_even_when_outer_archive_digest_matches(
    gate, evidence, tmp_path
):
    evidence.files["dist/aster-1.2.3-py3-none-any.whl"] += b"modified after acceptance"
    evidence.refresh_candidate(update_hashes=False)
    with pytest.raises(gate.EvidenceError, match="exact artifact bytes"):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


@pytest.mark.parametrize(
    "failure",
    [
        "missing-package",
        "wrong-version",
        "extra-file",
        "external-manifest",
        "reinstall",
        "not-restored",
        "no-new-turn",
        "changed-native",
    ],
)
def test_sealing_requires_complete_installation_and_upgrade_evidence(gate, evidence, failure):
    if failure == "missing-package":
        del evidence.files["dist/aster-1.2.3.tar.gz"]
    elif failure == "wrong-version":
        evidence.files["dist/aster-1.2.3-py3-none-any.whl"] = wheel("aster", "1.2.2")
    elif failure == "extra-file":
        evidence.files["terminal/unverified.exe"] = b"extra"
    elif failure == "external-manifest":
        evidence.files["terminal/MANIFEST.json"] = b"{}"
    else:
        report = json.loads(evidence.files["evidence/upgrade.json"])
        if failure == "reinstall":
            report["cross_version"] = False
        elif failure == "not-restored":
            report["session_reopened_without_execution"] = False
        elif failure == "no-new-turn":
            report["continued_after_install"] = False
        else:
            report["native_sha256"] = "0" * 64
        evidence.files["evidence/upgrade.json"] = encoded(report)
    with pytest.raises(gate.EvidenceError):
        gate.check_payload(evidence.files, SHA, TAG, evidence.manifest["sources"]["baseline"])


@pytest.mark.parametrize(
    "field,value", [("draft", True), ("prerelease", True), ("published_at", None), ("assets", [])]
)
def test_upgrade_requires_a_previously_published_distribution(gate, evidence, field, value):
    evidence.api.responses[f"/releases/tags/{OLD_TAG}"][field] = value
    with pytest.raises(gate.EvidenceError):
        gate.baseline_evidence(evidence.api, OLD_TAG, "1.2.3")


@pytest.mark.parametrize("version", ["1.2.2", "1.2.1"])
def test_baseline_cannot_be_same_or_newer_version(gate, evidence, version):
    with pytest.raises(gate.EvidenceError, match="older release"):
        gate.baseline_evidence(evidence.api, OLD_TAG, version)


def test_annotated_final_tag_is_resolved(gate, evidence, tmp_path):
    evidence.api.responses[f"/git/ref/tags/{TAG}"] = {"object": {"type": "tag", "sha": "d" * 40}}
    evidence.api.responses[f"/git/tags/{'d' * 40}"] = {"object": {"type": "commit", "sha": SHA}}
    gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")


def test_tag_is_rechecked_before_materializing_publish_files(gate, evidence, tmp_path, monkeypatch):
    original = evidence.api.tag_sha
    seen = 0

    def moving_tag(tag):
        nonlocal seen
        if tag == TAG:
            seen += 1
            if seen > 1:
                return OLD_SHA
        return original(tag)

    monkeypatch.setattr(evidence.api, "tag_sha", moving_tag)
    with pytest.raises(gate.EvidenceError, match="tag moved"):
        gate.fetch(evidence.api, TAG, 800, tmp_path / "accepted")
    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "C:/escape", "a/../escape", "a//escape", "NUL.txt", "a./file"],
)
def test_archive_paths_cannot_escape(gate, name):
    with pytest.raises(gate.EvidenceError, match="archive"):
        gate.archive_files(zip_bytes({name: b"payload"}))


def test_archive_rejects_raw_backslashes_on_both_hosts(gate):
    raw = zip_bytes({"a/escape": b"payload"}).replace(b"a/escape", b"a\\escape")
    with pytest.raises(gate.EvidenceError, match="archive"):
        gate.archive_files(raw)


def test_duplicate_json_keys_are_not_silently_overridden(gate):
    with pytest.raises(gate.EvidenceError, match="Duplicate JSON"):
        gate.json_data(b'{"dirty":true,"dirty":false}')


@pytest.mark.parametrize(
    "body",
    [
        "",
        "Candidate artifact: 0",
        "Candidate artifact: 800\nCandidate artifact: 801",
        "Candidate artifact: $(bad)",
    ],
)
def test_release_requires_one_explicit_artifact_locator(gate, body):
    with pytest.raises(gate.EvidenceError):
        gate.publication_inputs(
            "release",
            {"action": "published", "release": {"draft": False, "tag_name": TAG, "body": body}},
        )


def test_only_existing_release_and_rehearsal_triggers_are_accepted(gate):
    assert gate.publication_inputs(
        "release",
        {
            "action": "published",
            "release": {"draft": False, "tag_name": TAG, "body": "Candidate artifact: 800\n"},
        },
    ) == (TAG, 800)
    assert gate.publication_inputs(
        "workflow_dispatch", {"inputs": {"release_tag": TAG, "candidate_artifact_id": "800"}}
    ) == (TAG, 800)
    with pytest.raises(gate.EvidenceError):
        gate.publication_inputs("push", {})


def test_api_auth_is_not_sent_to_signed_download_hosts(gate, monkeypatch):
    request_headers = []

    class Opener:
        def open(self, request, timeout):
            request_headers.append(request.headers)
            raise urllib.error.HTTPError(
                request.full_url,
                302,
                "redirect",
                {"Location": "https://release-assets.githubusercontent.com/signed?token=test"},
                None,
            )

    downloads = []

    def download(url, timeout):
        downloads.append(url)
        assert isinstance(url, str)
        return io.BytesIO(b"verified later")

    monkeypatch.setattr(gate.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(gate.urllib.request, "urlopen", download)
    api = gate.GitHub("owner/aster", "secret-test-token")
    assert api._request("/releases/assets/500", download=True) == b"verified later"
    assert request_headers[0]["Authorization"] == "Bearer secret-test-token"
    assert downloads == ["https://release-assets.githubusercontent.com/signed?token=test"]


def test_workflows_require_candidate_evidence_without_breaking_ordinary_ci(gate):
    ci = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    publish = yaml.load(
        (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert ci["on"]["workflow_dispatch"]["inputs"]["release_candidate"]["default"] == "false"
    for job in ("release-upgrade", "release-candidate"):
        assert (
            ci["jobs"][job]["if"]
            == "github.event_name == 'workflow_dispatch' && inputs.release_candidate"
        )
    needed = set(ci["jobs"]["release-candidate"]["needs"]) | {"release-candidate"}
    names = {ci["jobs"][job].get("name", job) for job in needed}
    assert names == gate.REQUIRED_JOBS
    candidate_steps = ci["jobs"]["release-candidate"]["steps"]
    smoke = next(step for step in candidate_steps if step.get("name", "").startswith("Smoke-test"))
    seal = next(step for step in candidate_steps if step.get("id") == "seal")
    assert candidate_steps.index(smoke) < candidate_steps.index(seal)
    assert 'uv pip install "$RELEASE_DIR"/dist/*.whl' in smoke["run"]
    assert 'cd "$RUNNER_TEMP"' in smoke["run"]
    assert "push" not in publish["on"]
    assert publish["on"]["release"]["types"] == ["published"]
    release_steps = publish["jobs"]["attach-to-release"]["steps"]
    checksum = next(
        step for step in release_steps if step.get("name") == "Generate terminal bundle checksum"
    )
    attach = next(
        step for step in release_steps if step.get("name") == "Attach artifacts to the release"
    )
    assert release_steps.index(checksum) < release_steps.index(attach)
    assert 'sha256sum "${archives[0]}" > "${archives[0]}.sha256"' in checksum["run"]
    assert "terminal/*" in attach["run"]
    for job in ("publish-testpypi", "publish-pypi"):
        assert publish["jobs"][job]["needs"] == "verify-candidate"
    controller = publish["jobs"]["verify-candidate"]["steps"][0]
    assert controller["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    commands = "\n".join(
        step.get("run", "") for job in publish["jobs"].values() for step in job["steps"]
    )
    assert "uv build" not in commands and "make_release.py" not in commands
    assert "check_release_evidence.py fetch" in commands
