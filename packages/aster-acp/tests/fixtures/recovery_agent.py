# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pause after a real append and before saved progress, for the crash test."""

import asyncio
import json
import os
import shlex
import sys
import threading
from pathlib import Path

from aster_acp.server import serve
from aster_cli.sessions.actions import SessionActions

from aster.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

mode, phase, signal_path, operation = sys.argv[1:5]
target = "'append.txt'" if operation == "shell" else "self.cwd / 'append.txt'"
program = (
    f"import os; f=open({target},'a',encoding='utf-8'); "
    "f.write('once\\n'); f.flush(); os.fsync(f.fileno()); f.close()"
)
args = [sys.executable, "-c", program]
command = (
    "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in args)
    if sys.platform == "win32"
    else shlex.join(args)
)
append = (
    "execute_python",
    {"code": f"await self.shell.run({command!r})" if operation == "shell" else program},
)

if mode == "crash":
    original_finish = SessionActions.finish

    def pause_after_effect(self, action_id, *, outcome):
        if phase == "outcome":
            original_finish(self, action_id, outcome=outcome)
        signal = Path(signal_path)
        temporary = signal.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump({"action_id": action_id}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(signal)
        # The parent terminates this process only after seeing the fsynced
        # signal. No checkpoint or finally block can complete during this gap.
        threading.Event().wait()

    SessionActions.finish = pause_after_effect


def llm_factory():
    calls = (
        [append]
        if mode == "crash"
        else [
            append,
            ("return_result", {"result": {"message": "premature", "explanation": "done"}}),
            ("execute_python", {"code": "await self.files.read('append.txt')"}),
            (
                "return_result",
                {"result": {"question": "Confirm the appended line with /recovery."}},
            ),
            (
                "return_result",
                {"result": {"message": "recovered-without-repeating", "explanation": "Reviewed"}},
            ),
        ]
    )
    return FakeLLMClient(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"{mode}-{index}", name=name, arguments=json.dumps(arguments))
                ],
                finish_reason="tool_calls",
            )
            for index, (name, arguments) in enumerate(calls)
        ],
        strict_exhaustion=True,
    )


asyncio.run(serve(llm_factory, sandbox=False))
