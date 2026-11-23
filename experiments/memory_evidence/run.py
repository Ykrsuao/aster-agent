# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run reproducible memory controls with real storage and a scripted evidence reader."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aster_memory import MemoryConfig, MemoryManager
from aster_memory.memory_skill import MemorySkill

from aster import Agent
from aster.unifiedllm import FakeLLMClient
from experiments.memory_evidence.responder import (
    CELL_TIMEOUT_SECONDS,
    MAX_ITERATIONS,
    READ_EVIDENCE,
    EvidenceReader,
    offline_client,
)
from experiments.memory_evidence.scoring import score_case, summarize_arm
from scripts.terminal_acceptance import (
    candidate,
    fingerprint_model,
    model_identity,
    usage_totals,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXTURES = HERE / "fixtures.json"
ARMS = ("off", "current", "stale", "corrected")


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_fixture(path: Path = FIXTURES) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("A version-1 memory fixture must be a JSON object.")
    cases = document.get("cases")
    if document.get("schema_version") != 1 or not isinstance(cases, list) or not cases:
        raise ValueError("A nonempty version-1 synthetic memory fixture is required.")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip()
            for key in ("id", "subject", "question", "primary_key")
        ):
            raise ValueError("Every case needs an ID, subject, question and primary key.")
        if case["id"] in seen:
            raise ValueError("Fixture case IDs must be unique.")
        seen.add(case["id"])
        current, obsolete = case.get("current"), case.get("obsolete")
        if (
            not isinstance(current, dict)
            or not isinstance(obsolete, dict)
            or len(current) < 2
            or current.keys() != obsolete.keys()
            or case["primary_key"] not in current
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(current[key], str)
                or not current[key]
                or not isinstance(obsolete[key], str)
                or not obsolete[key]
                or current[key] == obsolete[key]
                for key in current
            )
        ):
            raise ValueError("Each case needs at least two distinct current/obsolete fact pairs.")
    return cases, _sha(raw)


def memory_config(path: Path) -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        path=str(path),
        owner="EvidenceProject",
        embedding={"backend": "hashing", "dim": 256},
        vector={"backend": "numpy"},
        spontaneous={"enabled": False},
        write={"on_events": (), "write_episodic": False},
        reflection={"enabled": False},
        instruct=False,
    )


def _note(case: dict[str, Any], key: str, *, obsolete: bool) -> str:
    return _canonical(
        {
            "subject": case["subject"],
            "key": key,
            "value": case["obsolete" if obsolete else "current"][key],
            "revision": "obsolete" if obsolete else "current",
        }
    )


async def seed_store(path: Path, cases: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Import facts, close the writer, and use another writer for explicit correction."""
    client = FakeLLMClient(strict_exhaustion=True)
    writer = Agent(llm=client)
    manager = MemoryManager.install(writer, config=memory_config(path))
    memory_ids: dict[tuple[str, str], str] = {}
    try:
        manager.session_ref = "fixture-authoring-session"
        for case in cases:
            for key in case["current"]:
                memory_ids[case["id"], key] = manager.remember(
                    _note(case, key, obsolete=arm in {"stale", "corrected"}), dedup=False
                )
    finally:
        manager.uninstall()
        await writer.aclose()
        await client.aclose()
    corrections = 0
    if arm == "corrected":
        correction_client = FakeLLMClient(strict_exhaustion=True)
        correction_agent = Agent(llm=correction_client)
        correction = MemoryManager.install(correction_agent, config=memory_config(path))
        try:
            correction.session_ref = "explicit-correction-session"
            for case in cases:
                for key in case["current"]:
                    if not correction.update(
                        memory_ids[case["id"], key], content=_note(case, key, obsolete=False)
                    ):
                        raise RuntimeError("A seeded fact disappeared before explicit correction.")
                    corrections += 1
        finally:
            correction.uninstall()
            await correction_agent.aclose()
            await correction_client.aclose()
    return {"imported_facts": len(memory_ids), "explicit_corrections": corrections}


def _redacted_answer(answer: dict[str, Any] | None) -> dict[str, Any] | None:
    if answer is None:
        return None
    return {
        "answer_sha256": _sha(answer["answer"]) if answer["answer"] is not None else None,
        "task_fields_sha256": {key: _sha(value) for key, value in answer["task_fields"].items()},
        "citations_sha256": {key: _sha(value) for key, value in answer["citations"].items()},
        "claimed_complete": answer["claimed_complete"],
    }


def _redacted_retrieval(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "query_sha256": _sha(query["query"]),
            "complete": query["complete"],
            "facts": [
                {
                    "memory_id_sha256": _sha(note["memory_id"]),
                    "key": note["key"],
                    "value_sha256": _sha(note["value"]),
                    "revision": note["revision"],
                }
                for note in query["notes"]
            ],
        }
        for query in log
    ]


async def run_case(
    path: Path,
    case: dict[str, Any],
    arm: str,
    repeat_id: int,
    *,
    top_k: int,
    timeout: float,
    client_factory: Callable[[], FakeLLMClient] = offline_client,
) -> tuple[dict[str, Any], list[dict[str, Any] | None]]:
    client = client_factory()
    if type(client) is not FakeLLMClient:
        raise TypeError("This harness accepts only an in-process FakeLLMClient; no live aliases.")
    reader = EvidenceReader(llm=client, top_k=top_k)
    identity = model_identity(client)
    skill = None
    result, error = None, None
    started = time.perf_counter()
    try:
        if arm != "off":
            skill = MemorySkill(memory_config(path))
            skill.attach(reader)
            reader._memory_skill = skill
            reader._memory.session_ref = f"reader-{repeat_id}-{arm}-{case['id']}"
        async with asyncio.timeout(timeout):
            returned = await reader.respond(
                case["question"], case["subject"], case["primary_key"], list(case["current"])
            )
            result = returned.model_dump()
    except Exception as exc:
        error = type(exc).__name__
    finally:
        if skill is not None:
            skill.detach()
        await reader.aclose()
        await client.aclose()
    usages = [
        call.response.usage.model_dump() if call.response and call.response.usage else None
        for call in client.calls
        if call.response is not None
    ]
    scores = score_case(case, result, reader._retrieval_log)
    return {
        "case_id": case["id"],
        "arm": arm,
        "repeat_id": repeat_id,
        "measurement_kind": "offline-scripted",
        "model_identity": identity,
        "model_fingerprint": fingerprint_model(identity),
        "response_returned": result is not None,
        "agent_completed": result["claimed_complete"] if result is not None else False,
        "verification_passed": scores["task_completed"],
        "manual_interventions": 0,
        "recovery_scope": "new-agent-existing-memory-store",
        "elapsed_seconds": time.perf_counter() - started,
        "error_type": error,
        "model_calls": client.call_count,
        "model_calls_complete": True,
        "external_provider_calls": 0,
        "usage": usage_totals(usages, attempted_calls=client.call_count),
        "scores": scores,
        "answer_evidence": _redacted_answer(result),
        "retrieval_evidence": _redacted_retrieval(reader._retrieval_log),
    }, usages


async def run_experiment(
    *,
    fixture: Path = FIXTURES,
    repeats: int = 2,
    top_k: int = 3,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if type(repeats) is not int or repeats < 1 or type(top_k) is not int or top_k < 1:
        raise ValueError("repeats and top_k must be positive integers.")
    if not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive.")
    cases, fixture_hash = load_fixture(fixture)
    source_files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path.read_bytes())
        for path in (
            HERE / "run.py",
            HERE / "responder.py",
            HERE / "scoring.py",
            ROOT / "scripts" / "terminal_acceptance.py",
        )
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "measurement_kind": "offline-scripted",
        "evidence_scope": "storage-reopen-retrieval-scripted-answer-independent-scoring",
        "real_model_benefit_verified": False,
        "created_at": datetime.now(UTC).isoformat(),
        "platform": {"system": platform.system(), "python": platform.python_version()},
        "candidate": candidate(),
        "fixture_sha256": fixture_hash,
        "runner_sha256": _sha(_canonical(source_files)),
        "source_files_sha256": source_files,
        "scripted_policy_sha256": _sha(READ_EVIDENCE),
        "source": "Local fixture identified by fixture_sha256; fresh temporary SQLite databases per arm/repeat.",
        "limits": {
            "repeats": repeats,
            "cases_per_arm_per_repeat": len(cases),
            "top_k_per_query": top_k,
            "task_timeout_seconds": timeout,
            "codeact_max_iterations": MAX_ITERATIONS,
            "cell_timeout_seconds": CELL_TIMEOUT_SECONDS,
            "embedding_backend": "hashing",
            "embedding_dimensions": 256,
            "vector_backend": "numpy",
            "external_provider_calls_allowed": 0,
            "token_limit": None,
            "cost_limit_usd": None,
        },
        "report_complete": False,
        "arms": {},
    }
    with tempfile.TemporaryDirectory(prefix="aster-memory-evidence-") as temporary:
        for arm in ARMS:
            rows, all_usages, stores = [], [], []
            for repeat_id in range(1, repeats + 1):
                path = Path(temporary) / f"{arm}-{repeat_id}.sqlite"
                stores.append(await seed_store(path, cases, arm))
                for case in cases:
                    row, usages = await run_case(
                        path,
                        case,
                        arm,
                        repeat_id,
                        top_k=top_k,
                        timeout=timeout,
                    )
                    rows.append(row)
                    all_usages.extend(usages)
            calls = sum(row["model_calls"] for row in rows)
            report["arms"][arm] = {
                "stores": stores,
                "rows": rows,
                "scores": summarize_arm(rows),
                "model_calls": calls,
                "external_provider_calls": 0,
                "usage": usage_totals(all_usages, attempted_calls=calls),
            }
    report["report_complete"] = all(
        len(report["arms"][arm]["rows"]) == len(cases) * repeats for arm in ARMS
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output path; existing evidence is not overwritten.")
    report = asyncio.run(
        run_experiment(
            fixture=args.fixture, repeats=args.repeats, top_k=args.top_k, timeout=args.timeout
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {arm: data["scores"] for arm, data in report["arms"].items()},
            indent=2,
            ensure_ascii=False,
        )
    )
    return int(any(row["error_type"] for arm in report["arms"].values() for row in arm["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
