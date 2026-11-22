# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline model script for project memory across real ACP process restarts."""

import asyncio
import json
import sys

from aster_acp.server import serve

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def llm_factory():
    mode = sys.argv[1]
    code = {
        "write": "self.memory.remember('The hummingbird project uses the cobalt backend.')",
        "read": (
            "notes = self.memory.recall('hummingbird cobalt')\n"
            "assert len(notes) == 1\n"
            "assert 'cobalt backend' in notes[0].content"
        ),
        "off": "assert not hasattr(self, 'memory')",
    }[mode]
    calls = [
        ("execute_python", {"code": code}),
        (
            "return_result",
            {
                "result": {
                    "message": f"memory-{mode}-verified",
                    "explanation": "offline verification",
                }
            },
        ),
    ]
    return FakeLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"memory-wire-{mode}-{index}", name=name, arguments=json.dumps(arguments)
                    )
                ],
                finish_reason="tool_calls",
            )
            for index, (name, arguments) in enumerate(calls)
        ],
        strict_exhaustion=True,
    )


asyncio.run(serve(llm_factory, sandbox=False))
