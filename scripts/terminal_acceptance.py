# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen coding-workspace checks; live runs require an explicitly selected model alias."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/terminal_acceptance/tasks.json"


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


def usage_totals(usages: list[dict | None]) -> dict:
    result = {"completed_calls": len(usages)}
    for name in ("input_tokens", "output_tokens"):
        known = [usage[name] for usage in usages if usage and usage.get(f"{name}_reported")]
        result[name] = sum(known) if len(known) == len(usages) and usages else None
    known_costs = [usage["cost_usd"] for usage in usages if usage and usage.get("cost_known")]
    result["cost_usd"] = sum(known_costs) if len(known_costs) == len(usages) and usages else None
    result["known_cost_subtotal_usd"] = sum(known_costs) if known_costs else None
    result["usage_scope"] = "all responses through the shared client, including delegated calls"
    return result


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def worker(payload: dict) -> None:
    from aster_cli.coding.agent import CodingAgent

    from aster.interactive import SummarizationConfig
    from aster.llm_config import llm_config_chain
    from aster.secrets import load_secrets_into_env
    from aster.tracing import enable_tracing
    from aster.unifiedllm import RetryConfig, get_llm_client
    from aster.unifiedllm.registry import load_model_entries

    class CallBudgetExceeded(BaseException):
        pass

    meter = Path(payload["meter"])
    usages = []
    calls = 0
    status = "starting"
    error = None
    agent = None
    llm = None

    def save():
        write_json(
            meter,
            {"status": status, "error": error, "attempted_calls": calls, **usage_totals(usages)},
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
        original_acall = llm.acall
        original_call = llm.call

        def admit():
            nonlocal calls
            if calls >= payload["max_calls"]:
                raise CallBudgetExceeded()
            calls += 1
            save()

        def record(response):
            usage = response.usage
            usages.append(usage.model_dump() if usage is not None else None)
            save()
            return response

        async def acall(*args, **kwargs):
            admit()
            return record(await original_acall(*args, **kwargs))

        def call(*args, **kwargs):
            admit()
            return record(original_call(*args, **kwargs))

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
        status = "completed" if type(result).__name__ == "Done" else "needs_input"
    except CallBudgetExceeded:
        status = "call_limit"
    except BaseException as exc:
        status = "worker_error"
        error = type(exc).__name__
    finally:
        save()
        if agent is not None:
            await agent.close()
        elif llm is not None:
            await llm.aclose()


def candidate() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
        return result.stdout

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
    ).split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = ROOT / name
        if path.suffix in {".py", ".rs", ".toml"} and path.is_file() and not path.is_symlink():
            untracked[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain")),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
        "untracked_source_sha256": untracked,
    }


def run_suite(args, tasks: list[dict], fixture_hash: str) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "fixture_sha256": fixture_hash,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate": candidate(),
        "model_alias": args.model,
        "limits": {
            "timeout_seconds": args.timeout,
            "max_calls": args.max_calls,
            "max_output_tokens": args.max_output_tokens,
            "endpoint_retries": 0,
        },
        "scope": "native CodingAgent on small coding fixtures; not terminal UI, sandbox containment or real-repository acceptance",
        "tasks": [],
    }
    env = os.environ.copy()
    env["ASTER_MODEL_METADATA_REFRESH"] = "False"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
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
            "prompt": task["prompt"]
            + "\nWork only inside the supplied workspace. Use standard-library code; no package installation or internet is needed. Run your own checks before reporting completion.",
            "max_calls": args.max_calls,
            "max_output_tokens": args.max_output_tokens,
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
        metering = (
            json.loads(meter.read_text(encoding="utf-8")) if meter.exists() else usage_totals([])
        )
        verification_passed, verification_reason = verify(task, workspace)
        passed, reason = verification_passed, verification_reason
        status = "timeout" if timed_out else metering.get("status", "worker_error")
        if status != "completed" or code != 0:
            passed = False
            reason = status if code == 0 or timed_out else "worker_exit"
        result = {
            "id": task["id"],
            "category": task["category"],
            "passed": passed,
            "reason": reason,
            "verification_passed": verification_passed,
            "verification_reason": verification_reason,
            "elapsed_seconds": round(elapsed, 3),
            "manual_interventions": 0,
            "workspace": str(workspace),
            **metering,
        }
        # A request killed or failed before its response may still have incurred cost.
        if timed_out or metering.get("attempted_calls", 0) != metering.get("completed_calls", 0):
            result.update(cost_usd=None, input_tokens=None, output_tokens=None)
        report["tasks"].append(result)
        report["passed"] = sum(item["passed"] for item in report["tasks"])
        report["total"] = len(tasks)
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
    if not args.model or not args.output:
        parser.error("live runs require --model and --output")
    if os.name != "nt":
        parser.error("live acceptance currently requires native Windows process-tree containment")
    if args.timeout <= 0 or args.max_calls <= 0 or args.max_output_tokens <= 0:
        parser.error("all limits must be positive")
    return run_suite(args, tasks, fixture_hash)


if __name__ == "__main__":
    raise SystemExit(main())
