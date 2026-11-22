# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline model script for task acceptance across an ACP process restart."""

import asyncio
import json
import sys

from aster_acp.server import serve

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def llm_factory():
    if sys.argv[1] == "edit":
        calls = [
            ("execute_python", {"code": "await self.shell.write_file('code.txt', 'fixed')"}),
            ("return_result", {"result": {"question": "Choose the next step"}}),
            (
                "return_result",
                {"result": {"message": "Paris", "explanation": "Answered", "complete_task": False}},
            ),
        ]
    else:
        calls = [
            ("return_result", {"result": {"message": "premature", "explanation": "done"}}),
            ("execute_python", {"code": "await self.verify('tests')"}),
            (
                "return_result",
                {"result": {"message": "verified-after-reopen", "explanation": "done"}},
            ),
        ]
    return FakeLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"verification-wire-{index}", name=name, arguments=json.dumps(arguments)
                    )
                ],
                finish_reason="tool_calls",
            )
            for index, (name, arguments) in enumerate(calls)
        ],
        strict_exhaustion=True,
    )


asyncio.run(serve(llm_factory, sandbox=False))
