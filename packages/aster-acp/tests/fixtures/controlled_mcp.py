# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local stdio MCP peer with observable startup, calls and shutdown."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

control_path, events_path = (Path(argument) for argument in sys.argv[1:])


def control():
    return json.loads(control_path.read_text(encoding="utf-8"))


def record(event):
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": event,
                    "pid": os.getpid(),
                    "cwd": str(Path.cwd()),
                    "label": control()["label"],
                    "literal": os.environ.get("ASTER_MCP_TEST_LITERAL"),
                }
            )
            + "\n"
        )


server = FastMCP("protocol-probe", log_level="ERROR")


@server.tool()
async def echo(value: str) -> str:
    """Return a deterministic response and expose a cancellable blocked call."""
    record("call")
    if control().get("mode") == "block_call":
        await asyncio.Event().wait()
    return f"{control()['label']}: {value}"


record("started")
try:
    if control().get("mode") == "offline":
        raise SystemExit(23)
    if control().get("mode") == "block_start":
        # The peer never answers initialize. Closing/retrying must cancel it.
        time.sleep(120)
    server.run(transport="stdio")
finally:
    record("stopped")
