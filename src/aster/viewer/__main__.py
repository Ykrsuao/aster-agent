# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allow running the viewer with `python -m aster.viewer`."""

import logging
import os

import uvicorn

from .main import app

log = logging.getLogger(__name__)

port = int(os.environ.get("ASTER_TRACE_VIEWER_PORT", "5001"))
# Loopback unless the operator opts in to other machines reaching the viewer.
host = os.environ.get("ASTER_TRACE_VIEWER_HOST", "127.0.0.1")
log.info("Starting viewer on %s:%d", host, port)
uvicorn.run(app, host=host, port=port)
