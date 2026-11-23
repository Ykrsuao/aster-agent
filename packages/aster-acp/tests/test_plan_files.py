# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan's literal host requests never authorize their Python spelling."""

import pytest
from aster_acp._plan_files import plan_file_executor
from aster_cli.tools.workspace_files import WorkspaceFiles


def _prepare(code, root, *, allowed=lambda: True, host_path=None):
    return plan_file_executor(
        code, files=WorkspaceFiles(root), allowed=allowed, timeout=2, host_path=host_path
    )


@pytest.mark.parametrize(
    "code",
    [
        "await self.files.write('x', 'bad')",
        "await self.files.read.__call__('x')",
        "await self.files.read(path_variable)",
        "await self.files.read(f'{value}')",
        "await self.files.read(**{'path': 'x'})",
        "await self.files.read(*['x'])",
        "await self.files.read('x', path='y')",
        "await self.files.read('x', unknown='y')",
        "result = await self.files.read('x')",
        "print(await self.files.read('x'))",
        "await self.files.read('x'); open('created', 'w')",
        "await self.files.read((open('created', 'w'), 'x')[1])",
        "await self.files.list(); await self.files.list()",
        "await self.shell.run('ls')",
        "#" * 8193,
    ],
)
def test_computed_mixed_and_unknown_requests_never_get_an_executor(tmp_path, code):
    assert _prepare(code, tmp_path) is None
    assert not (tmp_path / "created").exists()


async def test_changed_code_and_revoked_permission_are_refused_before_io(tmp_path):
    (tmp_path / "note.txt").write_text("original", encoding="utf-8")
    code = "await self.files.read('note.txt')"
    executor = _prepare(code, tmp_path)
    result = await executor.run_cell("await self.files.read('other.txt')")
    assert isinstance(result.error, PermissionError)
    refused = _prepare(code, tmp_path, allowed=lambda: False)
    result = await refused.run_cell(code)
    assert isinstance(result.error, PermissionError)
    assert not result.has_return


async def test_sandbox_translation_maps_only_the_path_and_stays_inside_workspace(tmp_path):
    (tmp_path / "note.txt").write_text("literal /workspace value", encoding="utf-8")
    mapped = []

    def host_path(path):
        mapped.append(path)
        return str(tmp_path)

    code = "await self.files.search('/workspace', path='/workspace')"
    result = await _prepare(code, tmp_path, host_path=host_path).run_cell(code)
    assert result.success
    assert result.returned_value["matches"][0]["text"] == "literal /workspace value"
    assert mapped == ["/workspace"]

    outside = tmp_path.parent / "external.txt"
    outside.write_text("outside canary", encoding="utf-8")
    code = "await self.files.read('/allowed/external.txt')"
    result = await _prepare(code, tmp_path, host_path=lambda path: str(outside)).run_cell(code)
    assert isinstance(result.error, PermissionError)
    assert "outside canary" not in repr(result)
