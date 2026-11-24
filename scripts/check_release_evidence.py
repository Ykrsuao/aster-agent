# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seal or fetch tested release files. This command never builds, evaluates, or publishes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"aster", "aster-cli", "aster-acp", "aster-memory", "aster-bench"}
CI_WORKFLOW = ".github/workflows/ci.yml"
# Only the manual real-model producer can satisfy this provenance contract;
# neither a manual JSON upload nor ordinary CI can substitute.
TASK_WORKFLOW = ".github/workflows/terminal-acceptance.yml"
TASK_JOB = "Terminal task acceptance"
TASK_ARTIFACT = "terminal-task-acceptance"
REQUIRED_JOBS = {
    "lint",
    "test",
    "Linux native terminal",
    "Linux sandbox",
    "python-313-error-compat",
    "build",
    "frontend-build",
    "rust-dependency-audit",
    "secret-scan",
    "windows-core-smoke",
    "release-upgrade",
    "release-candidate",
}
TASK_LIMITS = {
    "timeout_seconds": 180,
    "max_calls": 8,
    "max_output_tokens": 4096,
    "endpoint_retries": 0,
}
MAX_ARCHIVE_BYTES = 1024**3
MAX_EXTRACTED_BYTES = 2 * MAX_ARCHIVE_BYTES
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(RuntimeError):
    """Missing or inconsistent evidence; publication must remain blocked."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_data(data: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(data, object_pairs_hook=unique_pairs)
    except (ValueError, UnicodeError) as exc:
        raise EvidenceError("Invalid JSON evidence") from exc
    require(isinstance(result, dict), "Evidence must be a JSON object")
    return result


def release_version(tag: str) -> str:
    require(bool(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag)), "Expected a final vX.Y.Z tag")
    version = str(Version(tag[1:]))
    require(version == tag[1:], "Release version must use canonical spelling")
    return version


def positive_id(value: object) -> int:
    require(type(value) in (int, str) and str(value).isdigit(), "Expected a positive GitHub ID")
    result = int(value)
    require(result > 0, "Expected a positive GitHub ID")
    return result


def archive_files(data: bytes) -> dict[str, bytes]:
    """Validate every member before any archive content reaches the filesystem."""
    require(len(data) <= MAX_ARCHIVE_BYTES, "Archive exceeds the size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set()
            members = []
            size = 0
            for info in archive.infolist():
                name = info.filename
                path = PurePosixPath(name)
                require(
                    name
                    and info.orig_filename == name
                    and not path.is_absolute()
                    and ".." not in path.parts
                    and "\\" not in name
                    and ":" not in name
                    and path.as_posix() == name.rstrip("/"),
                    "Unsafe archive member",
                )
                require(
                    all(
                        part.rstrip(" .") == part
                        and not re.match(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", part)
                        for part in path.parts
                    ),
                    "Unsafe Windows archive member",
                )
                require(name.casefold().rstrip("/") not in names, "Duplicate archive member")
                names.add(name.casefold().rstrip("/"))
                require(not stat.S_ISLNK(info.external_attr >> 16), "Archive contains a symlink")
                if info.is_dir():
                    continue
                size += info.file_size
                require(size <= MAX_EXTRACTED_BYTES, "Expanded archive exceeds the size limit")
                members.append(info)
            return {info.filename: archive.read(info) for info in members}
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise EvidenceError("Invalid release archive") from exc


def write_files(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def local_files(directory: Path) -> dict[str, bytes]:
    require(directory.is_dir() and not directory.is_symlink(), "Missing evidence directory")
    result = {}
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "Evidence directory contains a symlink")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = path.read_bytes()
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    """Read-only GitHub API client; credentials never follow download redirects."""

    def __init__(self, repository: str, token: str):
        require(bool(re.fullmatch(r"[\w.-]+/[\w.-]+", repository)), "Invalid repository")
        self.repository = repository
        self.token = token
        self.base = f"https://api.github.com/repos/{repository}"

    def _request(self, path: str, *, download: bool = False) -> bytes:
        require((not path or path.startswith("/")) and ".." not in path, "Invalid GitHub API path")
        headers = {
            "Accept": "application/octet-stream" if download else "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "aster-release-evidence",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.base + path, headers=headers)
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            try:
                response = opener.open(request, timeout=60)
            except urllib.error.HTTPError as exc:
                if not download or exc.code != 302:
                    raise
                location = exc.headers.get("Location", "")
                target = urllib.parse.urlsplit(location)
                require(
                    target.scheme == "https"
                    and not target.username
                    and not target.password
                    and target.hostname is not None
                    and target.hostname.endswith(
                        (".githubusercontent.com", ".blob.core.windows.net")
                    ),
                    "Untrusted artifact download redirect",
                )
                # Deliberately a new request without the API Authorization header.
                response = urllib.request.urlopen(location, timeout=60)
            with response:
                limit = MAX_ARCHIVE_BYTES if download else 16 * 1024**2
                data = response.read(limit + 1)
                require(len(data) <= limit, "GitHub response exceeds the size limit")
                return data
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EvidenceError(f"GitHub could not supply required evidence at {path}") from exc

    def get(self, path: str) -> dict:
        return json_data(self._request(path))

    def pages(self, path: str, key: str) -> list:
        result = []
        for page in range(1, 101):
            data = self.get(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
            rows = data.get(key)
            require(isinstance(rows, list), f"Missing GitHub {key}")
            result.extend(rows)
            if len(result) == data.get("total_count"):
                return result
            require(len(rows) == 100, f"Incomplete GitHub {key} listing")
        raise EvidenceError("GitHub evidence listing exceeds the page limit")

    def tag_sha(self, tag: str) -> str:
        release_version(tag)
        item = self.get(f"/git/ref/tags/{tag}").get("object", {})
        for _ in range(10):
            require(bool(SHA.fullmatch(str(item.get("sha", "")))), "Invalid tag SHA")
            if item.get("type") == "commit":
                return item["sha"]
            require(item.get("type") == "tag", "Tag does not resolve to a commit")
            item = self.get(f"/git/tags/{item['sha']}").get("object", {})
        raise EvidenceError("Too many annotated tag indirections")

    def trusted_commit(self, sha: str) -> dict:
        require(bool(SHA.fullmatch(sha)), "Expected a complete candidate SHA")
        repository = self.get("")
        require(
            repository.get("full_name", "").lower() == self.repository.lower(), "Wrong repository"
        )
        branch = urllib.parse.quote(repository["default_branch"], safe="")
        comparison = self.get(f"/compare/{sha}...{branch}")
        require(
            comparison.get("status") in {"ahead", "identical"}
            and comparison.get("base_commit", {}).get("sha") == sha,
            "Candidate is not reachable from the repository's default branch",
        )
        return repository

    def source_file(self, path: str, sha: str) -> bytes:
        source = self.get(f"/contents/{path}?ref={sha}")
        require(
            source.get("type") == "file" and source.get("encoding") == "base64",
            "Missing candidate source file",
        )
        data = base64.b64decode("".join(source.get("content", "").split()), validate=True)
        require(len(data) == source.get("size"), "Incomplete candidate source file")
        return data

    def source_run(self, run_id: int, sha: str, workflow: str, jobs: set[str]) -> dict:
        repository = self.trusted_commit(sha)
        trusted = self.get(f"/actions/workflows/{Path(workflow).name}")
        require(trusted.get("path") == workflow, "Wrong trusted workflow")
        run = self.get(f"/actions/runs/{positive_id(run_id)}")
        require(
            run.get("workflow_id") == trusted.get("id")
            and run.get("path", "").split("@", 1)[0] == workflow
            and run.get("event") == "workflow_dispatch"
            and run.get("head_sha") == sha
            and run.get("repository", {}).get("id") == repository.get("id")
            and run.get("head_repository", {}).get("id") == repository.get("id")
            and run.get("status") == "completed"
            and run.get("conclusion") == "success",
            "Evidence is not from a successful trusted workflow on this candidate",
        )
        attempt = positive_id(run.get("run_attempt"))
        rows = self.pages(f"/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")
        for name in jobs:
            matching = [row for row in rows if row.get("name") == name]
            require(
                len(matching) == 1
                and matching[0].get("status") == "completed"
                and matching[0].get("conclusion") == "success"
                and matching[0].get("head_sha") == sha,
                f"Required candidate job did not succeed: {name}",
            )
        return run

    def artifact(
        self, artifact_id: int, sha: str, workflow: str, jobs: set[str]
    ) -> tuple[dict, bytes]:
        artifact = self.get(f"/actions/artifacts/{positive_id(artifact_id)}")
        source = artifact.get("workflow_run", {})
        require(
            artifact.get("id") == artifact_id
            and artifact.get("expired") is False
            and source.get("head_sha") == sha,
            "Artifact is expired or belongs to another candidate",
        )
        run = self.source_run(positive_id(source.get("id")), sha, workflow, jobs)
        repository_id = run["repository"]["id"]
        require(
            source.get("repository_id") == repository_id
            and source.get("head_repository_id") == repository_id,
            "Artifact belongs to another repository",
        )
        digest = artifact.get("digest", "")
        require(isinstance(digest, str) and digest.startswith("sha256:"), "Artifact has no SHA256")
        data = self._request(f"/actions/artifacts/{artifact_id}/zip", download=True)
        require(
            digest == f"sha256:{sha256(data)}", "Artifact download digest does not match GitHub"
        )
        return {**artifact, "run_attempt": run["run_attempt"]}, data


def wheel_metadata(data: bytes, name: str, version: str, *, native: bool = False) -> str | None:
    files = archive_files(data)
    metadata = [path for path in files if path.endswith(".dist-info/METADATA")]
    require(len(metadata) == 1, "Wheel must contain exactly one distribution")
    parsed = BytesParser().parsebytes(files[metadata[0]])
    require(parsed["Name"] == name and parsed["Version"] == version, "Wrong wheel metadata")
    if native:
        wheel = BytesParser().parsebytes(files[metadata[0].replace("METADATA", "WHEEL")])
        require(
            wheel["Root-Is-Purelib"] == "false" and wheel.get_all("Tag") == ["py3-none-win_amd64"],
            "Terminal wheel must be native Windows x64",
        )
        binary = files.get("aster_cli/bin/aster.exe", b"")
        require(binary.startswith(b"MZ"), "Terminal executable is missing")
        return sha256(binary)
    return None


def bundle_evidence(data: bytes, sha: str, version: str) -> dict:
    files = archive_files(data)
    require("MANIFEST.json" in files and "SHA256SUMS" in files, "Missing terminal manifest")
    manifest = json_data(files["MANIFEST.json"])
    require(
        manifest.get("schema_version") == 1
        and manifest.get("platform") == "win_amd64"
        and manifest.get("commit") == sha
        and manifest.get("version") == version
        and manifest.get("dirty") is False,
        "Terminal bundle must match the clean final candidate",
    )
    hashes = manifest.get("files")
    require(isinstance(hashes, dict), "Missing bundle hashes")
    require(set(files) == set(hashes) | {"MANIFEST.json", "SHA256SUMS"}, "Unexpected bundle files")
    require(
        all(sha256(files[name]) == digest for name, digest in hashes.items()),
        "Bundle hash mismatch",
    )
    try:
        checksum_rows = [line.split("  ", 1) for line in files["SHA256SUMS"].decode().splitlines()]
        checksums = {name: digest for digest, name in checksum_rows}
    except (ValueError, UnicodeError) as exc:
        raise EvidenceError("Invalid bundle checksums") from exc
    require(
        len(checksum_rows) == len(checksums)
        and checksums == {**hashes, "MANIFEST.json": sha256(files["MANIFEST.json"])},
        "Bundle checksum inventory differs from its manifest",
    )
    packages = manifest.get("packages", {})
    require(set(packages) == PACKAGES, "Bundle must contain all five packages")
    native_sha = None
    for name, filename in packages.items():
        tag = "py3-none-win_amd64" if name == "aster-cli" else "py3-none-any"
        require(
            filename == f"{name.replace('-', '_')}-{version}-{tag}.whl", "Wrong bundled wheel name"
        )
        wheel_path = f"wheels/{filename}"
        require(wheel_path in files, "Missing bundled wheel")
        digest = wheel_metadata(files[wheel_path], name, version, native=name == "aster-cli")
        native_sha = digest or native_sha
    return {
        "versions": dict.fromkeys(PACKAGES, version),
        "bundle_commit": sha,
        "bundle_manifest_sha256": sha256(files["MANIFEST.json"]),
        "native_sha256": native_sha,
        "bundled_terminal": True,
    }


def check_distributions(files: dict[str, bytes], version: str) -> None:
    found = set()
    for path, data in files.items():
        if not path.startswith("dist/"):
            continue
        filename = path.removeprefix("dist/")
        require("/" not in filename, "Nested distribution file")
        try:
            if filename.endswith(".whl"):
                name, actual_version, _, _ = parse_wheel_filename(filename)
                kind = "wheel"
                wheel_metadata(data, str(name), version)
            else:
                name, actual_version = parse_sdist_filename(filename)
                kind = "sdist"
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                    metadata = [
                        member
                        for member in archive.getmembers()
                        if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
                    ]
                    require(len(metadata) == 1 and metadata[0].isfile(), "Invalid sdist metadata")
                    source = archive.extractfile(metadata[0])
                    require(source is not None, "Missing sdist metadata")
                    parsed = BytesParser().parsebytes(source.read())
                    require(
                        canonicalize_name(parsed["Name"]) == name and parsed["Version"] == version,
                        "Wrong sdist metadata",
                    )
        except (ValueError, tarfile.TarError) as exc:
            raise EvidenceError("Invalid Python distribution") from exc
        require(name in PACKAGES and str(actual_version) == version, "Wrong distribution version")
        require((name, kind) not in found, "Duplicate distribution")
        found.add((name, kind))
    require(
        found == {(name, kind) for name in PACKAGES for kind in ("wheel", "sdist")},
        "Expected five wheels and five source distributions",
    )


def check_task_report(
    report: dict, sha: str, fingerprint: str, run: dict, *, fixture: bytes, runner: bytes
) -> None:
    require(bool(DIGEST.fullmatch(fingerprint)), "Missing model configuration fingerprint")
    tasks = json_data(fixture)["tasks"]
    expected_ids = {task["id"] for task in tasks}
    identity = report.get("model_identity")
    require(
        isinstance(identity, dict)
        and set(identity)
        == {"client_type", "model", "endpoint_origin", "sampling", "reasoning_level"}
        and isinstance(identity["client_type"], str)
        and bool(identity["client_type"])
        and isinstance(identity["model"], str)
        and bool(identity["model"])
        and (identity["endpoint_origin"] is None or isinstance(identity["endpoint_origin"], str))
        and isinstance(identity["sampling"], dict)
        and (identity["reasoning_level"] is None or isinstance(identity["reasoning_level"], str)),
        "Missing or invalid resolved model identity",
    )
    # Same canonical encoding as terminal_acceptance.fingerprint_model(). The
    # trusted producer additionally enforces its non-secret identity whitelist.
    identity_bytes = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    require(sha256(identity_bytes) == fingerprint, "Model identity fingerprint does not match")
    require(
        type(report.get("schema_version")) is int
        and report["schema_version"] == 1
        and report.get("report_complete") is True
        and report.get("measurement_kind") == "live-model"
        and type(report.get("repeat_id")) is int
        and report["repeat_id"] >= 1
        and report.get("candidate", {}).get("commit") == sha
        and report.get("candidate", {}).get("dirty") is False
        and report.get("fixture_sha256") == sha256(fixture)
        and report.get("runner_sha256") == sha256(runner)
        and report.get("model_fingerprint") == fingerprint
        and type(report.get("source_run_id")) is int
        and report.get("source_run_id") == run["workflow_run"]["id"]
        and type(report.get("source_run_attempt")) is int
        and report.get("source_run_attempt") == run["run_attempt"],
        "Task report does not match its trusted run, candidate, runner, fixtures, and model",
    )
    require(report.get("limits") == TASK_LIMITS, "Task report uses a different predeclared budget")
    require(
        all(type(value) in (int, float) for value in report["limits"].values()),
        "Invalid task budget",
    )
    results = report.get("tasks")
    require(isinstance(results, list), "Missing task results")
    require(
        len(results) == len(expected_ids)
        and {item.get("id") for item in results} == expected_ids
        and type(report.get("total")) is int
        and report["total"] == len(expected_ids)
        and type(report.get("passed")) is int
        and report["passed"] == len(expected_ids),
        "Task report is incomplete or has duplicate tasks",
    )
    require(
        all(
            item.get("passed") is True
            and item.get("agent_completed") is True
            and item.get("verification_passed") is True
            and item.get("status") == "completed"
            and item.get("model_fingerprint") == fingerprint
            and item.get("model_identity") == identity
            for item in results
        ),
        "Every frozen task must complete and pass independent verification",
    )


def task_evidence(api: GitHub, artifact_id: int, sha: str, fingerprint: str) -> tuple[bytes, dict]:
    source, data = api.artifact(artifact_id, sha, TASK_WORKFLOW, {TASK_JOB})
    require(source.get("name") == TASK_ARTIFACT, "Wrong task evidence artifact")
    files = archive_files(data)
    require("report.json" in files, "Trusted task artifact has no report.json")
    check_task_report(
        json_data(files["report.json"]),
        sha,
        fingerprint,
        source,
        fixture=api.source_file("tests/fixtures/terminal_acceptance/tasks.json", sha),
        runner=api.source_file("scripts/terminal_acceptance.py", sha),
    )
    return files["report.json"], {
        "artifact_id": artifact_id,
        "artifact_digest": source["digest"],
        "run_id": source["workflow_run"]["id"],
        "run_attempt": source["run_attempt"],
        "model_fingerprint": fingerprint,
    }


def baseline_evidence(api: GitHub, tag: str, candidate_version: str) -> tuple[bytes, dict]:
    version = release_version(tag)
    require(
        Version(version) < Version(candidate_version), "Upgrade must start from an older release"
    )
    release = api.get(f"/releases/tags/{tag}")
    require(
        release.get("tag_name") == tag
        and release.get("draft") is False
        and release.get("prerelease") is False
        and release.get("published_at"),
        "Upgrade baseline must be a published final release",
    )
    name = f"aster-terminal-{version}-windows-x64.zip"
    assets = [asset for asset in release.get("assets", []) if asset.get("name") == name]
    require(len(assets) == 1, "Published baseline has no unique complete terminal bundle")
    asset = assets[0]
    asset_id = positive_id(asset.get("id"))
    data = api._request(f"/releases/assets/{asset_id}", download=True)
    require(
        asset.get("digest") == f"sha256:{sha256(data)}",
        "Baseline asset has no matching GitHub digest",
    )
    sha = api.tag_sha(tag)
    package = bundle_evidence(data, sha, version)
    return data, {
        "tag": tag,
        "sha": sha,
        "asset_id": asset_id,
        "digest": asset["digest"],
        "package": package,
    }


def check_smoke(report: dict, expected: dict, *, baseline: dict | None = None) -> None:
    require(
        all(report.get(key) == value for key, value in expected.items()),
        "Installed files do not match the accepted bundle",
    )
    require(
        report.get("session_reopened_without_execution") is True
        and report.get("continued_after_install") is True
        and report.get("saved_input_bytes_preserved") is True,
        "Installed terminal did not restore saved work and finish a new turn",
    )
    if baseline is not None:
        require(
            report.get("cross_version") is True
            and report.get("upgrade_verified") is True
            and report.get("seed_package") == baseline["package"],
            "Same-version reinstall is not cross-version upgrade evidence",
        )


def check_payload(files: dict[str, bytes], sha: str, tag: str, baseline: dict) -> None:
    version = release_version(tag)
    check_distributions(files, version)
    archive = f"terminal/aster-terminal-{version}-windows-x64.zip"
    require(archive in files, "Missing complete Windows terminal")
    package = bundle_evidence(files[archive], sha, version)
    bundle = archive_files(files[archive])
    for name in ("MANIFEST.json", "SHA256SUMS"):
        require(
            files.get(f"terminal/{name}") == bundle[name],
            "External bundle metadata differs from ZIP",
        )
    expected = {name for name in files if name.startswith("dist/")} | {
        archive,
        "terminal/MANIFEST.json",
        "terminal/SHA256SUMS",
        "evidence/task-report.json",
        "evidence/sources.json",
        "evidence/install.json",
        "evidence/upgrade.json",
    }
    require(set(files) == expected, "Unexpected or missing candidate payload files")
    require(
        Version(release_version(baseline["tag"])) < Version(version),
        "Upgrade baseline is not older",
    )
    for name in ("install", "upgrade"):
        path = f"evidence/{name}.json"
        require(path in files, f"Missing {name} evidence")
        check_smoke(
            json_data(files[path]), package, baseline=baseline if name == "upgrade" else None
        )


def candidate_name(tag: str, sha: str, attempt: int) -> str:
    return f"aster-release-candidate-{tag}-{sha}-{attempt}"


def clean_checkout(tag: str) -> str:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    sha = git("rev-parse", "HEAD")
    require(not git("status", "--porcelain"), "Candidate checkout is dirty")
    require(git("rev-parse", f"{tag}^{{commit}}") == sha, "Final tag does not match checkout")
    require(
        os.environ.get("GITHUB_SHA") == sha and os.environ.get("GITHUB_REF") == f"refs/tags/{tag}",
        "Candidate CI must run on the final tag",
    )
    return sha


def prepare(api: GitHub, args) -> None:
    version = release_version(args.tag)
    sha = clean_checkout(args.tag)
    require(api.tag_sha(args.tag) == sha, "Remote candidate tag changed")
    api.trusted_commit(sha)
    report, task_source = task_evidence(api, args.task_artifact_id, sha, args.model_fingerprint)
    baseline_zip, baseline = baseline_evidence(api, args.upgrade_from_tag, version)
    archives = list(args.bundle.glob("*.zip"))
    require(len(archives) == 1, "Expected one candidate terminal ZIP")
    candidate_zip = archives[0].read_bytes()
    bundle_evidence(candidate_zip, sha, version)
    args.destination.mkdir(parents=True, exist_ok=False)
    write_files(args.destination / "baseline", archive_files(baseline_zip))
    write_files(args.destination / "candidate", archive_files(candidate_zip))
    write_files(
        args.destination / "evidence",
        {
            "task-report.json": report,
            "sources.json": json.dumps(
                {"baseline": baseline, "task": task_source}, indent=2
            ).encode(),
        },
    )


def seal(api: GitHub, args) -> None:
    sha = clean_checkout(args.tag)
    require(api.tag_sha(args.tag) == sha, "Remote candidate tag changed")
    files = local_files(args.directory)
    require("release-evidence.json" not in files, "Candidate was already sealed")
    require("evidence/sources.json" in files, "Missing evidence sources")
    sources = json_data(files["evidence/sources.json"])
    report, task_source = task_evidence(api, args.task_artifact_id, sha, args.model_fingerprint)
    _, baseline = baseline_evidence(api, args.upgrade_from_tag, release_version(args.tag))
    require(
        sources == {"baseline": baseline, "task": task_source}
        and files.get("evidence/task-report.json") == report,
        "Evidence source changed before sealing",
    )
    check_payload(files, sha, args.tag, baseline)
    run_id = positive_id(os.environ.get("GITHUB_RUN_ID"))
    attempt = positive_id(os.environ.get("GITHUB_RUN_ATTEMPT"))
    manifest = {
        "schema_version": 1,
        "repository": api.repository,
        "candidate_sha": sha,
        "tag": args.tag,
        "version": release_version(args.tag),
        "run_id": run_id,
        "run_attempt": attempt,
        "sources": sources,
        "files": {name: sha256(data) for name, data in sorted(files.items())},
    }
    (args.directory / "release-evidence.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"artifact_name={candidate_name(args.tag, sha, attempt)}\n")


def publication_inputs(event_name: str, event: dict) -> tuple[str, int]:
    if event_name == "release":
        release = event.get("release", {})
        require(
            event.get("action") == "published" and release.get("draft") is False,
            "Publication requires a published release",
        )
        matches = re.findall(
            r"^Candidate artifact: ([1-9][0-9]*)\s*$", release.get("body") or "", re.MULTILINE
        )
        require(
            len(matches) == 1, "Release notes must identify exactly one 'Candidate artifact: ID'"
        )
        tag, artifact_id = release.get("tag_name", ""), matches[0]
    else:
        require(event_name == "workflow_dispatch", "Unsupported publication trigger")
        inputs = event.get("inputs", {})
        tag, artifact_id = inputs.get("release_tag", ""), inputs.get("candidate_artifact_id")
    release_version(tag)
    return tag, positive_id(artifact_id)


def fetch(api: GitHub, tag: str, artifact_id: int, destination: Path) -> None:
    sha = api.tag_sha(tag)
    source, data = api.artifact(artifact_id, sha, CI_WORKFLOW, REQUIRED_JOBS)
    require(
        source.get("name") == candidate_name(tag, sha, source["run_attempt"]),
        "Wrong candidate artifact or rerun attempt",
    )
    files = archive_files(data)
    require("release-evidence.json" in files, "Artifact is not a sealed release candidate")
    manifest_bytes = files.pop("release-evidence.json")
    manifest = json_data(manifest_bytes)
    require(
        manifest.get("schema_version") == 1
        and manifest.get("repository") == api.repository
        and manifest.get("candidate_sha") == sha
        and manifest.get("tag") == tag
        and manifest.get("version") == release_version(tag)
        and manifest.get("run_id") == source["workflow_run"]["id"]
        and manifest.get("run_attempt") == source["run_attempt"]
        and manifest.get("files") == {name: sha256(content) for name, content in files.items()},
        "Candidate manifest does not match the trusted run and exact artifact bytes",
    )
    sources = manifest.get("sources", {})
    require(
        isinstance(sources.get("task"), dict) and isinstance(sources.get("baseline"), dict),
        "Missing candidate provenance",
    )
    task = sources["task"]
    report, task_source = task_evidence(
        api, positive_id(task.get("artifact_id")), sha, task.get("model_fingerprint", "")
    )
    _, baseline = baseline_evidence(api, sources["baseline"].get("tag", ""), release_version(tag))
    require(
        sources == {"task": task_source, "baseline": baseline}
        and files.get("evidence/task-report.json") == report,
        "Candidate evidence no longer matches its trusted sources",
    )
    check_payload(files, sha, tag, baseline)
    require(api.tag_sha(tag) == sha, "Release tag moved during evidence validation")
    files["release-evidence.json"] = manifest_bytes
    write_files(destination, files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "seal"):
        child = commands.add_parser(command)
        child.add_argument("--tag", required=True)
        child.add_argument("--upgrade-from-tag", required=True)
        child.add_argument("--task-artifact-id", required=True, type=positive_id)
        child.add_argument("--model-fingerprint", required=True)
        if command == "prepare":
            child.add_argument("--bundle", required=True, type=Path)
            child.add_argument("--destination", required=True, type=Path)
        else:
            child.add_argument("--directory", required=True, type=Path)
    commands.add_parser("fetch").add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        api = GitHub(os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GH_TOKEN", ""))
        if args.command == "fetch":
            event = json_data(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
            tag, artifact_id = publication_inputs(os.environ.get("GITHUB_EVENT_NAME", ""), event)
            fetch(api, tag, artifact_id, args.destination)
        elif args.command == "prepare":
            prepare(api, args)
        else:
            seal(api, args)
    except (
        EvidenceError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Release blocked: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
