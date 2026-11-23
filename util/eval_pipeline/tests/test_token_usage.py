# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Totals require evidence from every source, including missing traces."""

import pytest
from pydantic import ValidationError

from eval_pipeline.eval_parser import EvalFileParser
from eval_pipeline.eval_types import EvalTestResult
from eval_pipeline.experiment_writer import ExperimentWriter
from eval_pipeline.usage import TokenUsageSummary


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_counts_do_not_become_known_usage(value):
    usage = TokenUsageSummary.from_counts(input_tokens=value, output_tokens=0)
    assert usage.totals() == {"input_tokens": None, "output_tokens": 0, "total_tokens": None}
    assert usage.known_input_tokens == 0
    with pytest.raises(ValidationError):
        TokenUsageSummary(known_input_tokens=value)


def test_missing_trace_keeps_aggregate_incomplete_even_with_zero_records(tmp_path):
    known = TokenUsageSummary.from_counts(input_tokens=8, output_tokens=0, total_tokens=8)
    missing_trace = TokenUsageSummary()
    writer = ExperimentWriter(tmp_path, "missing-trace")
    writer.start()
    for index, usage in enumerate((known, missing_trace)):
        writer.append_result(
            EvalTestResult(
                test_id=str(index),
                agent_class="Agent",
                method="run",
                model="offline",
                variant="test",
                passed=True,
                scores={},
                input=None,
                output=None,
                expected=None,
                # Conflicting flat values must not override the completeness evidence.
                input_tokens=999,
                token_usage=usage,
            )
        )
    writer.finalize()
    _, rows, completion, _ = EvalFileParser().parse_file(writer.file_path)
    assert rows[0].input_tokens == 8 and rows[1].input_tokens is None
    assert completion.token_usage.records == 1
    assert completion.token_usage.known_input_tokens == 8
    assert completion.token_usage.known_output_tokens == 0
    assert completion.token_usage.input_tokens_complete is False
    assert completion.total_input_tokens is None and completion.total_output_tokens is None


def test_nested_sessions_preserve_partial_usage():
    from aster.trace_explorer.explorer import AgentSession, LLMTurn, TraceExplorer
    from eval_pipeline.models import ExecutionResult
    from eval_pipeline.scoring import build_scoring_context

    parent = AgentSession(
        session_id="parent", agent_name="A", method_name="run", parent_session_id=None
    )
    child = AgentSession(
        session_id="child", agent_name="B", method_name="run", parent_session_id="parent"
    )
    parent.children.append(child)
    parent.turns.append(
        LLMTurn(
            "parent", [], "", "offline", token_counts={"prompt": 8, "completion": 0, "total": 8}
        )
    )
    child.turns.append(LLMTurn("child", [], "", "offline", token_counts={"prompt": 2}))
    result = ExecutionResult("task", ((), {}), None, None, None, 0)
    ctx = build_scoring_context(result, trace=TraceExplorer([parent], "test://nested"))
    assert ctx.input_tokens == 10 and ctx.output_tokens is None and ctx.total_tokens is None
    assert ctx.token_usage.records == 2 and ctx.token_usage.known_total_tokens == 8
