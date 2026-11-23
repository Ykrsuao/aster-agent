# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent exact-answer scoring; no agent, memory, model, or network calls."""

from __future__ import annotations

from typing import Any


def score_case(
    case: dict[str, Any],
    answer: dict[str, Any] | None,
    retrieval_log: list[dict[str, Any]],
) -> dict[str, Any]:
    expected, obsolete = case["current"], case["obsolete"]
    subject, primary = case["subject"], case["primary_key"]
    observed = [note for query in retrieval_log for note in query["notes"]]
    relevant = [
        note for note in observed if note.get("subject") == subject and note.get("key") in expected
    ]
    retrieval_complete = answer is not None and all(query["complete"] for query in retrieval_log)
    retrieved_keys = {note["key"] for note in relevant}
    current_keys = {note["key"] for note in relevant if note.get("value") == expected[note["key"]]}
    metrics: dict[str, Any] = {
        "retrieval_covered_facts": len(retrieved_keys) if retrieval_complete else None,
        "retrieval_current_facts": len(current_keys) if retrieval_complete else None,
        "required_facts": len(expected),
        "answer_correct": None,
        "stale_adoption": None,
        "task_completed": False if answer is None else None,
    }
    if answer is None:
        return metrics
    fields, citations = answer["task_fields"], answer["citations"]
    metrics["answer_correct"] = answer["answer"] == expected[primary]
    metrics["stale_adoption"] = answer["answer"] == obsolete[primary] or any(
        fields.get(key) == obsolete[key] for key in expected
    )
    actual = {note["memory_id"]: note for note in relevant}
    supported = all(
        (note := actual.get(citations.get(key))) is not None
        and note.get("key") == key
        and note.get("value") == value
        for key, value in expected.items()
    )
    metrics["task_completed"] = metrics["answer_correct"] and fields == expected and supported
    return metrics


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep unknown observations out of numerators and denominators, with a count."""
    summary: dict[str, Any] = {"cases": len(rows)}
    for key in ("answer_correct", "stale_adoption", "task_completed"):
        known = [row["scores"][key] for row in rows if row["scores"][key] is not None]
        summary[key] = {
            "true_count": sum(known),
            "observed_cases": len(known),
            "unknown_cases": len(rows) - len(known),
            "rate": sum(known) / len(known) if known else None,
        }
    for key in ("retrieval_covered_facts", "retrieval_current_facts"):
        measured = [row["scores"] for row in rows if row["scores"][key] is not None]
        numerator = sum(score[key] for score in measured)
        denominator = sum(score["required_facts"] for score in measured)
        summary[key] = {
            "retrieved_facts": numerator,
            "required_facts_observed": denominator,
            "unknown_cases": len(rows) - len(measured),
            "rate": numerator / denominator if denominator else None,
        }
    return summary
