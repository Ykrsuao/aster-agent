# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace binaries must remain executable through the microVM mount."""

from __future__ import annotations

import pytest

from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.executor import SandboxedExecutor

pytestmark = pytest.mark.sandbox


async def test_workspace_elf_is_executable(tmp_path):
    executor = SandboxedExecutor(object(), SandboxConfig(workspace=str(tmp_path)), cell_timeout=15)
    try:
        result = await executor.run_cell(
            "import shutil\nimport subprocess\nfrom pathlib import Path\n"
            "source = Path('/usr/bin/true')\n"
            "assert subprocess.run([str(source)]).returncode == 0\n"
            "target = Path.cwd() / 'workspace-elf'\n"
            "shutil.copyfile(source, target)\n"
            "target.chmod(0o755)\n"
            "assert target.read_bytes() == source.read_bytes()\n"
            "assert subprocess.run([str(target)]).returncode == 0"
        )
        assert result.error is None, result.error
    finally:
        await executor.aclose()
