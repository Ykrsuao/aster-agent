# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An evidence reader with no fixture or scoring answers in its module namespace."""

from __future__ import annotations

import json
from typing import Annotated, Any

from aster_memory.memory_skill import MemorySkill
from aster_memory.schema import Memory
from pydantic import BaseModel

from aster import Agent, hidden, strategy
from aster.config.strategy_config import CodeActConfig
from aster.strategies import CodeActStrategy
from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

MAX_ITERATIONS = 2
CELL_TIMEOUT_SECONDS = 10.0

# This identical script reads actual tool results in every arm. It knows no answers.
READ_EVIDENCE = """\
fields = {}
citations = {}
for key in keys:
    for note in self.lookup(subject + ' ' + key):
        fact = json.loads(note.content)
        if fact.get('subject') == subject and fact.get('key') == key:
            fields[key] = fact['value']
            citations[key] = note.id
            break
return_result(Answer(
    answer=fields.get(primary_key),
    task_fields=fields,
    citations=citations,
    claimed_complete=all(key in fields for key in keys),
))
"""


class Answer(BaseModel):
    answer: str | None
    task_fields: dict[str, str]
    citations: dict[str, str]
    claimed_complete: bool


class EvidenceReader(Agent):
    _memory_skill: Annotated[MemorySkill | None, hidden]
    _retrieval_log: Annotated[list[dict[str, Any]], hidden]
    _top_k: Annotated[int, hidden]

    def __init__(self, *, llm: FakeLLMClient, top_k: int):
        super().__init__(llm=llm)
        self._memory_skill = None
        self._retrieval_log = []
        self._top_k = top_k

    def lookup(self, query: str) -> list[Memory]:
        """Retrieve saved evidence; an unavailable memory tool returns no evidence."""
        observation: dict[str, Any] = {"query": query, "complete": False, "notes": []}
        self._retrieval_log.append(observation)
        notes = self._memory_skill.recall(query, k=self._top_k) if self._memory_skill else []
        observation["notes"] = [
            {"memory_id": note.id, **json.loads(note.content)} for note in notes
        ]
        observation["complete"] = True
        return notes

    @strategy(
        CodeActStrategy(
            config=CodeActConfig(max_iterations=MAX_ITERATIONS, cell_timeout=CELL_TIMEOUT_SECONDS)
        )
    )
    async def respond(
        self, question: str, subject: str, primary_key: str, keys: list[str]
    ) -> Answer:
        """Answer the question and prepare the requested dispatch fields.

        Retrieve each subject/key using self.lookup(). Use only retrieved facts;
        report missing evidence with a null answer and omit unsupported fields.
        Cite the actual memory ID for every dispatch field. Mark the task complete
        only if every requested key has a field and a citation.
        """
        ...


def offline_client() -> FakeLLMClient:
    """One fixed code cell; unexpected retries fail instead of inventing an answer."""
    return FakeLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="read-evidence",
                        name="execute_python",
                        arguments=json.dumps({"code": READ_EVIDENCE}),
                    )
                ],
                finish_reason="tool_calls",
                usage=None,
            )
        ],
        strict_exhaustion=True,
    )
