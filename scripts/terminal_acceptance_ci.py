# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Produce the frozen task report only for an explicitly approved manual CI run."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import check_release_evidence as evidence
from scripts import terminal_acceptance as acceptance

MODEL_ALIAS = "terminal-acceptance"
KEY_ENV = "TERMINAL_ACCEPTANCE_API_KEY"
CONFIG_ENV = "TERMINAL_ACCEPTANCE_MODEL_CONFIG"
MODEL_FIELDS = {
    "model_name",
    "client_type",
    "api_base",
    "temperature",
    "top_p",
    "reasoning_effort",
    "context_window",
}


def model_config(raw: str) -> dict:
    """Accept one non-secret route; retries, credentials and budget stay fixed."""
    config = evidence.json_data(raw.encode("utf-8"))
    evidence.require(
        isinstance(config, dict)
        and {"model_name", "client_type", "api_base"} <= config.keys()
        and not config.keys() - MODEL_FIELDS,
        "Model configuration has missing or unsupported fields",
    )
    evidence.require(
        config["client_type"] in {"completion", "responses"}, "Unsupported model interface"
    )
    model = config["model_name"]
    evidence.require(
        isinstance(model, str)
        and re.fullmatch(r"(?:openai|anthropic)/[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,240}", model)
        and (config["client_type"] != "responses" or model.startswith("openai/")),
        "Use an explicit OpenAI-compatible or Anthropic model route",
    )
    endpoint = config["api_base"]
    evidence.require(
        isinstance(endpoint, str)
        and not any(ord(char) <= 32 or ord(char) == 127 for char in endpoint),
        "Invalid model endpoint",
    )
    parts = urlsplit(endpoint)
    evidence.require(
        parts.scheme == "https"
        and parts.hostname
        and "@" not in parts.netloc
        and not parts.query
        and not parts.fragment
        and "\\" not in endpoint,
        "Model endpoint must be HTTPS without credentials, query or fragment",
    )
    acceptance._endpoint_origin(endpoint)
    for name, lower, upper in (("temperature", 0, 2), ("top_p", 0, 1)):
        if name in config:
            value = config[name]
            evidence.require(
                type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper,
                "Invalid sampling value",
            )
    if "reasoning_effort" in config:
        evidence.require(
            isinstance(config["reasoning_effort"], str)
            and re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", config["reasoning_effort"]),
            "Invalid reasoning effort",
        )
    if "context_window" in config:
        evidence.require(
            type(config["context_window"]) is int and config["context_window"] > 4096,
            "Context window must exceed the fixed reply limit",
        )
    return {
        **config,
        "api_key_env": KEY_ENV,
        "max_tokens": evidence.TASK_LIMITS["max_output_tokens"],
        "retry_config": False,
        "media_inputs": [],
        **(
            {"store": False, "include": ["reasoning.encrypted_content"]}
            if config["client_type"] == "responses"
            else {}
        ),
    }


@contextlib.contextmanager
def isolated_environment(directory: Path, config: dict):
    """Do not discover workstation aliases, settings or layered secrets in CI."""
    registry = directory / "llm_config.yaml"
    registry.write_text(json.dumps({"models": {MODEL_ALIAS: config}}), encoding="utf-8")
    overrides = {
        "ASTER_USER_DIR": str(directory / "user"),
        "ASTER_PROJECT_DIR": str(directory / "project"),
        "ASTER_LLM_CONFIG": str(registry),
        "ASTER_SETTINGS": str(directory / "settings.yaml"),
        "ASTER_SECRETS": str(directory / "secrets.yaml"),
        "ASTER_MODEL_METADATA_REFRESH": "False",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


async def resolved_identity(config: dict) -> dict:
    from aster.unifiedllm.registry import client_from_config

    # Construction resolves settings only. No inference or credential probe occurs.
    llm = client_from_config(MODEL_ALIAS, config, api_key="fingerprint-only")
    try:
        return acceptance.model_identity(llm)
    finally:
        await llm.aclose()


def dispatch_context(candidate_sha: str, fingerprint: str, output: Path) -> dict:
    evidence.require(
        os.name == "nt"
        and os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch",
        "Live acceptance requires a native Windows manual GitHub Actions run",
    )
    evidence.require(
        evidence.SHA.fullmatch(candidate_sha) and evidence.DIGEST.fullmatch(fingerprint),
        "A full candidate SHA and model fingerprint are required",
    )
    event = evidence.json_data(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    inputs = event.get("inputs", {})
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    branch = os.environ.get("DEFAULT_BRANCH", "")
    evidence.require(
        (inputs.get("confirm_live") is True or inputs.get("confirm_live") == "true")
        and inputs.get("candidate_sha") == candidate_sha
        and inputs.get("model_fingerprint") == fingerprint
        and event.get("repository", {}).get("full_name") == repo
        and event["repository"].get("default_branch") == branch
        and branch
        and repo
        and os.environ.get("GITHUB_SHA") == candidate_sha
        and os.environ.get("GITHUB_WORKFLOW_REF")
        == f"{repo}/{evidence.TASK_WORKFLOW}@{os.environ.get('GITHUB_REF')}",
        "Dispatch authorization, source or model does not match",
    )
    for command in (
        ["git", "check-ref-format", f"refs/heads/{branch}"],
        ["git", "merge-base", "--is-ancestor", candidate_sha, f"refs/remotes/origin/{branch}"],
    ):
        result = subprocess.run(command, cwd=acceptance.ROOT, capture_output=True)
        evidence.require(
            result.returncode == 0, "Candidate is not reachable from the default branch"
        )
    current = acceptance.candidate()
    evidence.require(
        current["commit"] == candidate_sha and current["dirty"] is False,
        "Acceptance requires the exact clean candidate checkout",
    )
    root = acceptance.ROOT.resolve()
    temp_root = Path(os.environ["RUNNER_TEMP"]).resolve()
    output = output.resolve()
    evidence.require(
        not output.is_relative_to(root)
        and output.is_relative_to(temp_root)
        and output != temp_root
        and not output.exists(),
        "Use a new output directory below RUNNER_TEMP and outside the checkout",
    )
    return {
        "workflow_run": {"id": evidence.positive_id(os.environ["GITHUB_RUN_ID"])},
        "run_attempt": evidence.positive_id(os.environ["GITHUB_RUN_ATTEMPT"]),
    }


def run(args) -> int:
    run_info = dispatch_context(args.candidate_sha, args.model_fingerprint, args.output)
    evidence.require(bool(os.environ.get(KEY_ENV, "").strip()), "Missing acceptance API credential")
    config = model_config(os.environ.get(CONFIG_ENV, ""))
    with tempfile.TemporaryDirectory(
        prefix="aster-acceptance-", dir=os.environ["RUNNER_TEMP"]
    ) as temp:
        with isolated_environment(Path(temp), config):
            identity = asyncio.run(resolved_identity(config))
            evidence.require(
                acceptance.fingerprint_model(identity) == args.model_fingerprint,
                "Resolved model differs from the approved fingerprint",
            )
            tasks, fixture_hash = acceptance.load_tasks()
            result = acceptance.run_suite(
                argparse.Namespace(
                    output=args.output,
                    model=MODEL_ALIAS,
                    timeout=evidence.TASK_LIMITS["timeout_seconds"],
                    max_calls=evidence.TASK_LIMITS["max_calls"],
                    max_output_tokens=evidence.TASK_LIMITS["max_output_tokens"],
                    expected_model_fingerprint=args.model_fingerprint,
                    require_live_model=True,
                    serial_calls=True,
                ),
                tasks,
                fixture_hash,
            )
    report = evidence.json_data((args.output / "report.json").read_bytes())
    evidence.require(
        result == 0 and acceptance.candidate() == report["candidate"],
        "Tasks failed or the candidate changed during acceptance",
    )
    evidence.check_task_report(
        report,
        args.candidate_sha,
        args.model_fingerprint,
        run_info,
        fixture=acceptance.FIXTURES.read_bytes(),
        runner=Path(acceptance.__file__).read_bytes(),
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sha")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--describe-model", action="store_true", help="resolve identity offline; no key required"
    )
    args = parser.parse_args(argv)
    if not args.describe_model and not (
        args.candidate_sha and args.model_fingerprint and args.output
    ):
        parser.error("live CI requires candidate SHA, model fingerprint and output")
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        # Native tasks may print arbitrary text; only this driver's fixed status reaches CI logs.
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                if args.describe_model:
                    config = model_config(os.environ.get(CONFIG_ENV, ""))
                    with tempfile.TemporaryDirectory(prefix="aster-identity-") as temp:
                        with isolated_environment(Path(temp), config):
                            identity = asyncio.run(resolved_identity(config))
                    result = 0
                else:
                    result = run(args)
        if args.describe_model:
            print(
                json.dumps(
                    {
                        "model_identity": identity,
                        "model_fingerprint": acceptance.fingerprint_model(identity),
                    }
                )
            )
        else:
            print("Terminal acceptance completed: all frozen tasks verified.")
        return result
    except Exception:
        print(
            "Terminal acceptance failed; check approved inputs and the sanitized report if present.",
            file=sys.stderr,
        )
        return 1
    finally:
        logging.disable(previous_logging)


if __name__ == "__main__":
    raise SystemExit(main())
