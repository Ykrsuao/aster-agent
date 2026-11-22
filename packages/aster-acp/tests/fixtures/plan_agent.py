# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline model script for inspecting the actual checkout through ACP."""

import asyncio
import json
import sys

from aster_acp.server import serve

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def llm_factory():
    directory = "packages/aster-acp/src/aster_acp"
    calls = [
        ("execute_python", {"code": f"await self.files.list({directory!r})"}),
        (
            "execute_python",
            {
                "code": f"await self.files.search('_has_unreviewable_characters', path={directory!r}, glob='runtime_controls.py')"
            },
        ),
        (
            "execute_python",
            {
                "code": f"await self.files.read('{directory}/runtime_controls.py', start_line=30, end_line=60)"
            },
        ),
        (
            "execute_python",
            {"code": f"await self.shell.write_file({sys.argv[1]!r}, 'must not be written')"},
        ),
        (
            "return_result",
            {
                "result": {
                    "message": "Plan inspection finished.",
                    "explanation": "Inspected files; write refused.",
                }
            },
        ),
    ]
    return FakeLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"plan-wire-{index}", name=name, arguments=json.dumps(arguments))
                ],
                finish_reason="tool_calls",
            )
            for index, (name, arguments) in enumerate(calls)
        ],
        strict_exhaustion=True,
    )


asyncio.run(serve(llm_factory, sandbox=False))
