# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen coding-workspace checks; live runs require an explicitly selected model alias."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/terminal_acceptance/tasks.json"
SAMPLING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
    }
)


def _endpoint_origin(value: str | None) -> str | None:
    if value is None:
        return None
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("model endpoint must have an HTTP(S) origin")
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port is not None else "")


def fingerprint_model(identity: Mapping) -> str:
    """Hash only the documented non-secret model identity, rejecting arbitrary config."""
    if not isinstance(identity, Mapping) or set(identity) != {
        "client_type",
        "model",
        "endpoint_origin",
        "sampling",
        "reasoning_level",
    }:
        raise ValueError("unexpected model identity fields")
    for name, pattern in (
        ("client_type", r"[A-Za-z_][A-Za-z0-9_.]{0,255}"),
        ("model", r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}"),
        ("reasoning_level", r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}"),
    ):
        value = identity[name]
        if name == "reasoning_level" and value is None:
            continue
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError(f"invalid model identity {name}")
    origin = identity["endpoint_origin"]
    if origin is not None and (not isinstance(origin, str) or _endpoint_origin(origin) != origin):
        raise ValueError("endpoint_origin must exclude credentials, path, query and fragment")
    sampling = identity["sampling"]
    if not isinstance(sampling, dict) or sampling.keys() - SAMPLING_FIELDS:
        raise ValueError("unexpected model sampling fields")
    for name, value in sampling.items():
        if name == "reasoning_effort":
            valid = isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.\-]{1,64}", value)
        elif name in {"seed", "max_tokens", "max_completion_tokens", "max_output_tokens"}:
            valid = type(value) is int and (name == "seed" or value > 0)
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"invalid model sampling {name}")
    encoded = json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_identity(llm, *, overrides: Mapping | None = None) -> dict:
    """Capture the resolved client, never its mutable alias or credential-bearing config."""
    overrides = dict(overrides or {})
    config = llm._prepare_call_config(overrides)
    extra = config.get("extra_body") or {}
    sampling = {}
    for name in SAMPLING_FIELDS:
        top, nested = config.get(name), extra.get(name)
        if top is not None and nested is not None and top != nested:
            raise ValueError("ambiguous sampling configuration")
        value = top if top is not None else nested
        if value is not None:
            sampling[name] = value
    identity = {
        "client_type": f"{type(llm).__module__}.{type(llm).__qualname__}",
        "model": config.get("model", llm.model),
        "endpoint_origin": _endpoint_origin(config.get("base_url") or config.get("api_base")),
        "sampling": sampling,
        "reasoning_level": overrides.get("reasoning_level", llm.reasoning_level),
    }
    fingerprint_model(identity)
    return identity


def load_tasks(path: Path = FIXTURES) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    tasks = data["tasks"]
    if data["schema_version"] != 1 or len(tasks) != 10:
        raise ValueError("expected the ten frozen version-1 tasks")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate task id")
    for task in tasks:
        for name in [task["id"], *task["files"], *task["solution"]]:
            if (
                not name
                or any(c in name for c in ("\\", ":", "\x00"))
                or any(part in ("", ".", "..") for part in name.split("/"))
            ):
                raise ValueError("fixture paths must be relative")
    return tasks, hashlib.sha256(raw).hexdigest()


def write_files(workspace: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def run_process(args: list[str], *, cwd: Path, text: str, timeout: float, env=None):
    """Bound Windows process trees (POSIX process groups); retain 64 KiB of stdout."""
    import threading

    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=os.name != "nt",
    )
    output = [""]

    def drain():
        for chunk in iter(lambda: process.stdout.read(4096), ""):
            output[0] = (output[0] + chunk)[-65536:]

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    job = None
    timed_out = False
    try:
        if os.name == "nt":
            from aster.tools._windows_process import WindowsJob

            job = WindowsJob(process.pid)
        process.stdin.write(text)
        process.stdin.close()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if job is not None:
                job.terminate(1)
    finally:
        if job is not None:
            job.close()
        elif os.name != "nt":
            import signal

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        if process.poll() is None:
            process.wait(timeout=10)
        reader.join(timeout=10)
        process.stdout.close()
        process.stdin.close()
    return process.returncode, output[0], timed_out


def verify(task: dict, workspace: Path) -> tuple[bool, str | None]:
    for name in task.get("protected", []):
        path = workspace / name
        try:
            unchanged = path.is_file() and path.read_text(encoding="utf-8") == task["files"][name]
        except UnicodeError:
            return False, "protected_file_changed"
        except OSError:
            return False, "protected_file_unreadable"
        if not unchanged:
            return False, "protected_file_changed"
    token = "ACCEPTANCE-" + uuid.uuid4().hex
    # Verification is supplied only after the agent process tree has stopped.
    source = (
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd()))\n"
        "try:\n"
        f"    exec(compile({task['verify']!r}, '<acceptance-verifier>', 'exec'))\n"
        "except BaseException as error:\n"
        "    print('VERIFY_ERROR:' + type(error).__name__)\n"
        "    raise SystemExit(1)\n"
        f"print({token!r})\n"
    )
    code, output, timed_out = run_process(
        [sys.executable, "-I", "-c", "import sys; exec(sys.stdin.buffer.read().decode('utf-8'))"],
        cwd=workspace,
        text=source,
        timeout=15,
    )
    passed = code == 0 and token in output.splitlines()
    reason = "verification_timeout" if timed_out else "verification_failed"
    for line in output.splitlines():
        if line.startswith("VERIFY_ERROR:") and line.removeprefix("VERIFY_ERROR:").isidentifier():
            reason += ":" + line.removeprefix("VERIFY_ERROR:")
    return passed, None if passed else reason


def self_check(tasks: list[dict]) -> None:
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="aster-acceptance-check-") as temp:
            workspace = Path(temp)
            write_files(workspace, task["files"])
            if verify(task, workspace)[0]:
                raise AssertionError(f"{task['id']}: broken seed unexpectedly passed")
            write_files(workspace, task["solution"])
            passed, reason = verify(task, workspace)
            if not passed:
                raise AssertionError(f"{task['id']}: reference solution failed ({reason})")
    print(
        f"Self-check: {len(tasks)} broken seeds rejected; {len(tasks)} reference solutions passed."
    )


def usage_totals(usages: list[dict | None], *, attempted_calls: int | None = None) -> dict:
    attempted = len(usages) if attempted_calls is None else attempted_calls
    if type(attempted) is not int or attempted < len(usages):
        raise ValueError("attempted calls cannot be fewer than completed responses")
    all_responded = bool(usages) and attempted == len(usages)
    result = {"completed_calls": len(usages), "attempted_calls": attempted}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        known = []
        for usage in usages:
            if not usage:
                continue
            if name == "total_tokens":
                # Cache/read/write and reasoning counts are subsets, never additional tokens.
                values = [usage.get(key) for key in ("input_tokens", "output_tokens")]
                reported = all(
                    usage.get(f"{key}_reported") is True
                    for key in ("input_tokens", "output_tokens")
                )
                valid = reported and all(type(value) is int and value >= 0 for value in values)
                value = sum(values) if valid else None
            else:
                value = usage.get(name)
                valid = usage.get(f"{name}_reported") is True and type(value) is int and value >= 0
            if valid:
                known.append(value)
        complete = all_responded and len(known) == len(usages)
        result[f"known_{name}"] = sum(known)
        result[f"{name}_complete"] = complete
        result[name] = sum(known) if complete else None
    known_costs = [
        usage["cost_usd"]
        for usage in usages
        if usage
        and usage.get("cost_known") is True
        and type(usage.get("cost_usd")) in (int, float)
        and math.isfinite(usage["cost_usd"])
        and usage["cost_usd"] >= 0
    ]
    result["cost_complete"] = all_responded and len(known_costs) == len(usages)
    result["cost_usd"] = sum(known_costs) if result["cost_complete"] else None
    result["known_cost_subtotal_usd"] = sum(known_costs) if known_costs else None
    result["usage_scope"] = "all responses through the shared client, including delegated calls"
    return result


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def worker(payload: dict) -> None:
    from aster_cli.coding.agent import CodingAgent

    from aster.interactive import Done, SummarizationConfig
    from aster.llm_config import llm_config_chain
    from aster.secrets import load_secrets_into_env
    from aster.tracing import enable_tracing
    from aster.unifiedllm import FakeLLMClient, RetryConfig, get_llm_client
    from aster.unifiedllm.registry import load_model_entries

    class CallBudgetExceeded(BaseException):
        pass

    class ConcurrentCallRejected(BaseException):
        pass

    meter = Path(payload["meter"])
    usages = []
    calls = 0
    status = "starting"
    error = None
    agent = None
    llm = None
    identity = None
    fingerprint = None
    measurement_kind = None

    def save():
        write_json(
            meter,
            {
                "status": status,
                "error": error,
                "model_identity": identity,
                "model_fingerprint": fingerprint,
                "measurement_kind": measurement_kind,
                **usage_totals(usages, attempted_calls=calls),
            },
        )

    save()
    try:
        # Resolve credentials and alias while cwd is still the source checkout.
        load_secrets_into_env()
        if payload["model"] not in load_model_entries(*llm_config_chain()):
            raise ValueError("model must be a saved alias")
        llm = get_llm_client(
            payload["model"],
            max_tokens=payload["max_output_tokens"],
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        )
        identity = model_identity(llm)
        fingerprint = fingerprint_model(identity)
        measurement_kind = "offline-scripted" if isinstance(llm, FakeLLMClient) else "live-model"
        expected_fingerprint = payload.get("expected_model_fingerprint")
        if payload.get("require_live_model") and measurement_kind != "live-model":
            raise ValueError("live acceptance requires a real model client")
        if expected_fingerprint and fingerprint != expected_fingerprint:
            raise ValueError("resolved model does not match the approved fingerprint")
        original_acall = llm.acall
        original_call = llm.call
        # Both entry points share one gate; generated helpers can use either one.
        call_lock = threading.Lock() if payload.get("serial_calls") else None
        async_gate = asyncio.Lock()

        def admit(request):
            nonlocal calls, identity, fingerprint
            if calls >= payload["max_calls"]:
                raise CallBudgetExceeded()
            actual_identity = model_identity(llm, overrides=request)
            if expected_fingerprint and fingerprint_model(actual_identity) != expected_fingerprint:
                raise ValueError("request does not match the approved model fingerprint")
            if calls and actual_identity != identity:
                raise ValueError("model settings changed within a measured task")
            identity = actual_identity
            fingerprint = fingerprint_model(identity)
            calls += 1
            save()

        def record(response):
            usage = response.usage
            usages.append(usage.model_dump() if usage is not None else None)
            save()
            return response

        async def acall(*args, **kwargs):
            if call_lock is None:
                admit(kwargs)
                return record(await original_acall(*args, **kwargs))
            async with async_gate:
                # A sync call during this await must fail instead of blocking the loop.
                if not call_lock.acquire(blocking=False):
                    raise ConcurrentCallRejected()

                async def measured_call():
                    admit(kwargs)
                    return record(await original_acall(*args, **kwargs))

                request = asyncio.create_task(measured_call())
                try:
                    return await asyncio.shield(request)
                except asyncio.CancelledError:
                    # Providers shield their own tasks. Keep capacity until the
                    # owned call ends, even if its caller is cancelled repeatedly.
                    while not request.done():
                        try:
                            await asyncio.shield(request)
                        except asyncio.CancelledError:
                            continue
                        except BaseException:
                            break
                    if not request.cancelled():
                        request.exception()
                    raise
                finally:
                    call_lock.release()

        def call(*args, **kwargs):
            if call_lock is None:
                admit(kwargs)
                return record(original_call(*args, **kwargs))
            if not call_lock.acquire(blocking=False):
                raise ConcurrentCallRejected()
            try:
                admit(kwargs)
                return record(original_call(*args, **kwargs))
            finally:
                call_lock.release()

        llm.acall = acall
        llm.call = call
        workspace = Path(payload["workspace"])
        os.chdir(workspace)
        os.environ["ASTER_PROJECT_DIR"] = str(workspace / ".aster")
        os.environ["ASTER_USER_DIR"] = str(workspace / ".aster-user")
        os.environ.pop("ASTER_SETTINGS", None)
        enable_tracing(exporters=[])
        agent = CodingAgent(
            llm=llm,
            cwd=workspace,
            libs_dir=workspace / ".aster/libs",
            skills_dirs=[],
            summarization=SummarizationConfig(policy="none"),
        )
        status = "running"
        save()
        result = await agent.handle({"user_messages": [payload["prompt"]]})
        status = (
            "completed"
            if isinstance(result, Done) and getattr(result, "complete_task", True)
            else "needs_input"
        )
    except CallBudgetExceeded:
        status = "call_limit"
    except ConcurrentCallRejected:
        status = "concurrent_call_rejected"
    except BaseException as exc:
        status = "worker_error"
        error = type(exc).__name__
    finally:
        save()
        if agent is not None:
            await agent.close()
        elif llm is not None:
            await llm.aclose()


def candidate(*, artifact_dir: Path | None = None) -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
        return result.stdout

    excluded = []
    if artifact_dir is not None:
        try:
            relative = artifact_dir.resolve().relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            if relative == Path("."):
                raise ValueError("artifact directory cannot be the source root")
            excluded = [f":(top,literal,exclude){relative.as_posix()}"]

    untracked = {}
    for raw in git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "src",
        "packages",
        "scripts",
        "util",
        "experiments",
        *excluded,
    ).split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = ROOT / name
        if path.suffix in {".py", ".rs", ".toml"} and path.is_file() and not path.is_symlink():
            untracked[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain", "--", ".", *excluded)),
        "diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--binary", "--", ".", *excluded)
        ).hexdigest(),
        "untracked_source_sha256": untracked,
    }


def run_suite(args, tasks: list[dict], fixture_hash: str) -> int:
    if not tasks:
        raise ValueError("a report requires at least one task")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def source_number(name):
        value = os.environ.get(name, "")
        return int(value) if value.isascii() and value.isdigit() and int(value) > 0 else None

    report = {
        "schema_version": 1,
        "fixture_sha256": fixture_hash,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate": candidate(artifact_dir=getattr(args, "artifact_dir", output)),
        "model_alias": args.model,
        "model_identity": None,
        "model_fingerprint": None,
        "measurement_kind": None,
        "repeat_id": getattr(args, "repeat_id", 1),
        "report_complete": False,
        "source_run_id": source_number("GITHUB_RUN_ID"),
        "source_run_attempt": source_number("GITHUB_RUN_ATTEMPT"),
        "limits": {
            "timeout_seconds": args.timeout,
            "max_calls": args.max_calls,
            "max_output_tokens": args.max_output_tokens,
            "endpoint_retries": 0,
        },
        "scope": "native CodingAgent on small coding fixtures; not terminal UI, sandbox containment or real-repository acceptance",
        "elapsed_scope": "agent worker wall time, excluding independent verification",
        "tasks": [],
    }
    write_json(output / "report.json", report)
    env = os.environ.copy()
    env["ASTER_MODEL_METADATA_REFRESH"] = "False"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    frozen_ids = {task["id"] for task in load_tasks()[0]}
    for task in tasks:
        task_dir = output / task["id"]
        workspace = task_dir / "workspace"
        workspace.mkdir(parents=True)
        write_files(workspace, task["files"])
        subprocess.run(["git", "init", "-q", str(workspace)], check=True, capture_output=True)
        meter = task_dir / "usage.json"
        payload = {
            "model": args.model,
            "workspace": str(workspace),
            "meter": str(meter),
            "prompt": task["prompt"] + f"\nHard budget: {args.max_calls} model calls total. "
            "Batch related inspection and edits; implement the smallest fix and run focused checks once. "
            "Stop when the checks pass; use the last call to submit return_result(Done(...)). "
            "Work only inside the supplied workspace. Use standard-library code; no package installation or internet is needed.",
            "max_calls": args.max_calls,
            "max_output_tokens": args.max_output_tokens,
            "expected_model_fingerprint": getattr(args, "expected_model_fingerprint", None),
            "require_live_model": getattr(args, "require_live_model", False),
            "serial_calls": getattr(args, "serial_calls", False),
        }
        start = time.monotonic()
        code, _, timed_out = run_process(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT,
            text=json.dumps(payload),
            timeout=args.timeout,
            env=env,
        )
        elapsed = time.monotonic() - start
        metering = {
            **usage_totals([]),
            **(
                json.loads(meter.read_text(encoding="utf-8"))
                if meter.exists()
                else usage_totals([])
            ),
        }
        if report["model_identity"] is None and metering.get("model_identity") is not None:
            for name in ("model_identity", "model_fingerprint", "measurement_kind"):
                report[name] = metering.get(name)
        verification_passed, verification_reason = verify(task, workspace)
        passed, reason = verification_passed, verification_reason
        status = "timeout" if timed_out else metering.get("status", "worker_error")
        agent_completed = status == "completed" and code == 0
        if not agent_completed:
            passed = False
            reason = status if code == 0 or timed_out else "worker_exit"
        result = {
            **metering,
            "id": task["id"],
            "category": task["category"],
            "passed": passed,
            "reason": reason,
            "status": status,
            "agent_completed": agent_completed,
            "verification_passed": verification_passed,
            "verification_reason": verification_reason,
            "elapsed_seconds": round(elapsed, 3),
            "manual_interventions": 0,
            "recovery_scope": "not_exercised",
            "workspace": str(workspace),
        }
        # A request killed or failed before its response may still have incurred cost.
        if timed_out or metering.get("attempted_calls", 0) != metering.get("completed_calls", 0):
            result.update(cost_usd=None, cost_complete=False)
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                result[name] = None
                result[f"{name}_complete"] = False
        report["tasks"].append(result)
        report["passed"] = sum(item["passed"] for item in report["tasks"])
        report["total"] = len(tasks)
        report["report_complete"] = (
            len(report["tasks"]) == len(frozen_ids)
            and {item["id"] for item in report["tasks"]} == frozen_ids
            and candidate(artifact_dir=getattr(args, "artifact_dir", output)) == report["candidate"]
        )
        write_json(output / "report.json", report)
        lines = [
            "# Terminal coding acceptance",
            "",
            report["scope"],
            "",
            f"Model: {args.model}",
            f"Fixture SHA256: {fixture_hash}",
            "",
            "| Task | Result | Seconds | Cost USD | Reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for item in report["tasks"]:
            cost = item.get("cost_usd")
            lines.append(
                f"| {item['id']} | {'PASS' if item['passed'] else 'FAIL'} | {item['elapsed_seconds']} | {cost if cost is not None else 'unknown'} | {item['reason'] or ''} |"
            )
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            f"{task['id']}: {'PASS' if passed else 'FAIL'} ({reason or 'verified'}, {elapsed:.1f}s)",
            flush=True,
        )
    print(f"{report['passed']}/{len(tasks)} passed. Evidence: {output / 'report.json'}")
    return 0 if report["passed"] == len(tasks) else 1


def aggregate_reports(
    reports: Sequence[Mapping], *, expected_task_ids: Sequence[str], expected_repeats: int
) -> dict:
    """Summarize only complete, comparable repeats; unknown usage remains a lower bound."""
    task_ids = set(expected_task_ids)
    if not task_ids or len(task_ids) != len(expected_task_ids):
        raise ValueError("expected task ids must be nonempty and unique")
    if (
        type(expected_repeats) is not int
        or expected_repeats < 1
        or len(reports) != expected_repeats
    ):
        raise ValueError("missing or extra repeat reports")
    if any(not isinstance(report, Mapping) for report in reports):
        raise ValueError("reports must be JSON objects")
    reference = reports[0]
    comparable = (
        "candidate",
        "fixture_sha256",
        "runner_sha256",
        "model_identity",
        "model_fingerprint",
        "measurement_kind",
        "limits",
    )
    if any(reference.get(name) is None for name in comparable):
        raise ValueError("missing comparison metadata")
    if reference["measurement_kind"] not in ("live-model", "offline-scripted"):
        raise ValueError("unknown measurement kind")
    if fingerprint_model(reference["model_identity"]) != reference["model_fingerprint"]:
        raise ValueError("model fingerprint does not match resolved identity")
    if (
        reference["model_identity"]["client_type"] == "aster.unifiedllm.fake.FakeLLMClient"
        and reference["measurement_kind"] != "offline-scripted"
    ):
        raise ValueError("scripted clients are not live model evidence")
    source = reference["candidate"]
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("commit"), str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source["commit"]) is None
        or type(source.get("dirty")) is not bool
    ):
        raise ValueError("candidate requires a full commit id and explicit dirty state")
    for digest in (
        reference["fixture_sha256"],
        reference["runner_sha256"],
        source.get("diff_sha256"),
    ):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("missing or invalid source digest")
    limits = reference["limits"]
    if not isinstance(limits, Mapping) or set(limits) != {
        "timeout_seconds",
        "max_calls",
        "max_output_tokens",
        "endpoint_retries",
    }:
        raise ValueError("missing evaluation limits")
    if any(
        type(limits[name]) is not int or limits[name] <= 0
        for name in ("max_calls", "max_output_tokens")
    ) or (
        type(limits["timeout_seconds"]) not in (int, float)
        or not math.isfinite(limits["timeout_seconds"])
        or limits["timeout_seconds"] <= 0
        or type(limits["endpoint_retries"]) is not int
        or limits["endpoint_retries"] != 0
    ):
        raise ValueError("invalid evaluation limits")
    repeats = set()
    rows = []
    for report in reports:
        if report.get("schema_version") != 1 or report.get("report_complete") is not True:
            raise ValueError("incomplete or unsupported report")
        if any(report.get(name) != reference[name] for name in comparable):
            raise ValueError(
                "cannot combine different candidates, models, fixtures, runners or limits"
            )
        repeat = report.get("repeat_id")
        if type(repeat) is not int or repeat < 1 or repeat in repeats:
            raise ValueError("invalid or duplicate repeat id")
        repeats.add(repeat)
        tasks = report.get("tasks")
        if (
            not isinstance(tasks, list)
            or len(tasks) != len(task_ids)
            or any(not isinstance(task, Mapping) for task in tasks)
            or {task.get("id") for task in tasks} != task_ids
        ):
            raise ValueError("missing, duplicate or unexpected task")
        for task in tasks:
            if any(
                task.get(name) != reference[name]
                for name in (
                    "model_identity",
                    "model_fingerprint",
                    "measurement_kind",
                )
            ):
                raise ValueError("task model differs from report identity")
            if task.get("status") not in {
                "completed",
                "call_limit",
                "concurrent_call_rejected",
                "needs_input",
                "worker_error",
                "timeout",
            }:
                raise ValueError("task has not reached a final status")
            if any(
                type(task.get(name)) is not bool
                for name in (
                    "agent_completed",
                    "verification_passed",
                    "passed",
                )
            ) or task["passed"] != (task["agent_completed"] and task["verification_passed"]):
                raise ValueError("task outcome must distinguish completion and verification")
            if task["agent_completed"] and task["status"] != "completed":
                raise ValueError("agent completion contradicts final status")
            attempted, completed = task.get("attempted_calls"), task.get("completed_calls")
            if (
                type(attempted) is not int
                or type(completed) is not int
                or not 0 <= completed <= attempted
            ):
                raise ValueError("missing or invalid call counts")
            elapsed = task.get("elapsed_seconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("missing or invalid elapsed time")
            interventions = task.get("manual_interventions")
            if (
                type(interventions) is not int
                or interventions < 0
                or not task.get("recovery_scope")
            ):
                raise ValueError("missing intervention or recovery evidence")
            rows.append(task)
    if repeats != set(range(1, expected_repeats + 1)):
        raise ValueError("missing expected repeat id")
    summary = {
        "schema_version": 1,
        **{name: reference[name] for name in comparable},
        "report_complete": True,
        "expected_repeats": expected_repeats,
        "repeat_ids": sorted(repeats),
        "task_count": len(task_ids),
        "observations": len(rows),
        "agent_completed": sum(row["agent_completed"] for row in rows),
        "verification_passed": sum(row["verification_passed"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "manual_interventions": sum(row["manual_interventions"] for row in rows),
        "attempted_calls": sum(row["attempted_calls"] for row in rows),
        "completed_calls": sum(row["completed_calls"] for row in rows),
        "recovery_scopes": sorted({row["recovery_scope"] for row in rows}),
        "scope": "observed fixed-fixture outcomes; offline scripts do not measure live model reliability",
    }
    for name in ("agent_completed", "verification_passed", "passed"):
        summary[f"{name}_rate"] = summary[name] / len(rows)
    for name in ("input_tokens", "output_tokens", "total_tokens", "cost_usd"):
        subtotal = "known_cost_subtotal_usd" if name == "cost_usd" else f"known_{name}"
        complete = "cost_complete" if name == "cost_usd" else f"{name}_complete"
        values = [row.get(subtotal) for row in rows]
        known = [value for value in values if value is not None]
        if any(
            type(value) not in ((int, float) if name == "cost_usd" else (int,))
            or not math.isfinite(value)
            or value < 0
            for value in known
        ):
            raise ValueError("invalid usage subtotal")
        is_complete = len(known) == len(rows) and all(row.get(complete) is True for row in rows)
        for row in rows:
            if row.get(complete) is True and (
                row.get(subtotal) is None
                or row.get(name) != row[subtotal]
                or row.get("attempted_calls") != row.get("completed_calls")
            ):
                raise ValueError("complete usage contradicts call or subtotal evidence")
        summary[subtotal] = sum(known) if known else None
        summary[complete] = is_complete
        summary[name] = sum(known) if is_complete else None
    summary["tasks"] = [
        {
            "id": task_id,
            "observations": expected_repeats,
            **{
                name: sum(row[name] for row in rows if row["id"] == task_id)
                for name in (
                    "agent_completed",
                    "verification_passed",
                    "passed",
                )
            },
        }
        for task_id in expected_task_ids
    ]
    return summary


def run_repeats(args, tasks: list[dict], fixture_hash: str) -> int:
    if args.repeats == 1:
        return run_suite(args, tasks, fixture_hash)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for repeat in range(1, args.repeats + 1):
        repeat_args = argparse.Namespace(**vars(args))
        repeat_args.output = output / f"repeat-{repeat}"
        repeat_args.repeat_id = repeat
        repeat_args.artifact_dir = output
        run_suite(repeat_args, tasks, fixture_hash)
        reports.append(json.loads((repeat_args.output / "report.json").read_text(encoding="utf-8")))
    aggregate = aggregate_reports(
        reports, expected_task_ids=[task["id"] for task in tasks], expected_repeats=args.repeats
    )
    write_json(output / "aggregate.json", aggregate)
    return 0 if aggregate["passed"] == aggregate["observations"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--model", help="saved model alias; live run sends requests")
    parser.add_argument("--task", action="append", help="run only this frozen task id (repeatable)")
    parser.add_argument(
        "--output", type=Path, help="new directory for reports and retained workspaces"
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--expected-model-fingerprint", help="reject unapproved resolved settings")
    parser.add_argument("--require-live-model", action="store_true")
    parser.add_argument(
        "--serial-calls", action="store_true", help="allow one shared-client call at a time"
    )
    parser.add_argument("--repeat-id", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--aggregate", nargs="+", type=Path, help="summarize existing reports without model calls"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        asyncio.run(worker(json.loads(sys.stdin.buffer.read().decode("utf-8"))))
        return 0
    tasks, fixture_hash = load_tasks()
    if args.task:
        selected = set(args.task)
        if selected - {task["id"] for task in tasks}:
            parser.error("unknown task id")
        tasks = [task for task in tasks if task["id"] in selected]
    if args.list:
        for task in tasks:
            print(f"{task['id']} ({task['category']}): {task['prompt']}")
        return 0
    if args.self_check:
        self_check(tasks)
        return 0
    if args.repeats < 1 or args.repeat_id < 1 or (args.repeats > 1 and args.repeat_id != 1):
        parser.error("repeats and repeat-id must be positive; multiple repeats start at 1")
    if args.repeats > 1 and args.task:
        parser.error("repeated evidence requires the complete frozen task suite")
    if args.aggregate:
        if args.output is None:
            parser.error("aggregation requires a new --output directory")
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.aggregate]
        aggregate = aggregate_reports(
            reports, expected_task_ids=[task["id"] for task in tasks], expected_repeats=args.repeats
        )
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "aggregate.json", aggregate)
        return 0
    if not args.model or not args.output:
        parser.error("live runs require --model and --output")
    if (
        not math.isfinite(args.timeout)
        or args.timeout <= 0
        or args.max_calls <= 0
        or args.max_output_tokens <= 0
    ):
        parser.error("all limits must be finite and positive")
    if args.expected_model_fingerprint and not re.fullmatch(
        r"[0-9a-f]{64}", args.expected_model_fingerprint
    ):
        parser.error("model fingerprint must be a lowercase SHA256")
    if os.name != "nt":
        parser.error("live acceptance currently requires native Windows process-tree containment")
    return run_repeats(args, tasks, fixture_hash)


if __name__ == "__main__":
    raise SystemExit(main())
