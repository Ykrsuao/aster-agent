# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File tools use UTF-8 and reject invalid text before overwriting files."""

import os
import subprocess
import sys
from textwrap import dedent

import pytest

from aster.tools.shell_tools import ShellTools


@pytest.mark.parametrize("operation", ["read", "read_region", "write_file", "path", "match"])
def test_utf8_file_operations_with_utf8_mode_disabled(tmp_path, operation):
    script = dedent(
        r"""
        import asyncio
        import os
        import stat
        import sys
        from pathlib import Path
        from aster.tools.shell_tools import ShellTools

        async def main():
            root, operation = Path(sys.argv[1]), sys.argv[2]
            path = root / "source.txt"
            original = "\u4f60\u597d \U0001f600\nsecond\n"
            replacement = "\u518d\u89c1 \U0001f642"
            path.write_bytes(original.encode("utf-8"))
            before = path.stat()
            shell = ShellTools(cwd=str(root))
            try:
                if operation == "read":
                    assert (await shell.read("source.txt")).text == original
                elif operation == "read_region":
                    assert (await shell.read("source.txt", (1, 1))).text == original.splitlines(True)[0]
                else:
                    expected = replacement + "\nsecond\n"
                    if operation == "write_file":
                        await shell.write_file("source.txt", expected)
                    elif operation == "path":
                        await shell.replace("source.txt", original.splitlines()[0], replacement)
                    else:
                        anchor = await shell.read("source.txt", (1, 1))
                        await shell.replace(anchor, replacement)
                    assert path.read_bytes() == expected.replace("\n", os.linesep).encode("utf-8")
                    after = path.stat()
                    assert after.st_ino == before.st_ino
                    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
            finally:
                await shell.close()

        asyncio.run(main())
        """
    )
    # Linux uses the C locale; Windows keeps its native legacy code page.
    # A UTF-8 developer environment must not mask the Windows failure.
    environment = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C"}
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", script, str(tmp_path), operation],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("operation", ["write_file", "path", "match"])
async def test_encoding_failure_preserves_existing_file_bytes(tmp_path, operation):
    path = tmp_path / "source.txt"
    original = b"first\r\nsecond\n"
    path.write_bytes(original)
    shell = ShellTools(cwd=str(tmp_path))
    try:
        with pytest.raises(UnicodeEncodeError):
            if operation == "write_file":
                await shell.write_file("source.txt", "\ud800")
            elif operation == "path":
                await shell.replace("source.txt", "first", "\ud800")
            else:
                await shell.replace(await shell.read("source.txt", (1, 1)), "\ud800")
        assert path.read_bytes() == original
    finally:
        await shell.close()


async def test_encoding_failure_does_not_create_a_new_file(tmp_path):
    shell = ShellTools(cwd=str(tmp_path))
    try:
        with pytest.raises(UnicodeEncodeError):
            await shell.write_file("nested/source.txt", "\ud800")
        assert not (tmp_path / "nested").exists()
    finally:
        await shell.close()
