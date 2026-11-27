# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real installer PATH helper without reading or writing the registry."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows installer PATH handling")
INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"

HARNESS = r"""
param([string]$ConfigFile)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$config = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($config.installer, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw "Installer parse errors: $parseErrors" }
$functions = @($ast.FindAll({
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Add-UserPath'
}, $true))
if ($functions.Count -ne 1) { throw 'Expected exactly one Add-UserPath function' }
$body = $functions[0].Extent.Text
$userGetter = "[Environment]::GetEnvironmentVariable('Path', 'User')"
$machineGetter = "[Environment]::GetEnvironmentVariable('Path', 'Machine')"
$setter = '[Environment]::SetEnvironmentVariable(''Path'', $newPath, ''User'')'
foreach ($operation in @($userGetter, $machineGetter, $setter)) {
    if (-not $body.Contains($operation)) { throw "Unexpected registry access: $operation" }
}
$body = $body.Replace($userGetter, '$script:userPath').Replace($machineGetter, '$script:machinePath')
$body = $body.Replace($setter, '$script:userPath = $newPath; $script:writes++')
if ($body -match '\[Environment\]::(?:Get|Set)EnvironmentVariable') {
    throw 'A registry operation was not replaced'
}
. ([scriptblock]::Create($body))
$script:userPath = $config.user_path
$script:machinePath = $config.machine_path
$script:writes = 0
$env:ASTER_PATH_TEST_TARGET = $config.target
$env:ASTER_PATH_TEST_OTHER = $config.other
$env:PATH = $config.process_path
$runs = @()
for ($index = 0; $index -lt 2; $index++) {
    $warnings = @(Add-UserPath $config.target 3>&1 | ForEach-Object { $_.ToString() })
    $processPath = $env:PATH
    $current = (Get-Command aster -CommandType Application | Select-Object -First 1).Source
    try {
        $env:PATH = [Environment]::ExpandEnvironmentVariables("$script:machinePath;$script:userPath")
        $fresh = (Get-Command aster -CommandType Application | Select-Object -First 1).Source
    } finally {
        $env:PATH = $processPath
    }
    $runs += @{
        user_path=$script:userPath; process_path=$processPath; writes=$script:writes
        current=$current; fresh=$fresh; warnings=$warnings; machine_path=$script:machinePath
    }
}
@{runs=$runs} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $config.result -Encoding UTF8
"""


@pytest.mark.parametrize("shell", ["powershell.exe", "pwsh.exe"])
@pytest.mark.parametrize("existing_equivalents", [False, True])
@pytest.mark.parametrize("machine_conflict", [None, "aster.exe", "aster.cmd"])
def test_installer_path_precedence_and_repeated_updates(
    tmp_path, shell, existing_equivalents, machine_conflict
):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is not installed")
    old, target, other, machine = (
        tmp_path / name for name in ("old", "new commands", "other", "machine")
    )
    for directory in (old, target, other, machine):
        directory.mkdir()
    for directory in (old, target):
        (directory / "aster.exe").write_bytes(b"Command discovery sentinel; never executed")
    if machine_conflict:
        (machine / machine_conflict).write_bytes(b"Command discovery sentinel; never executed")
    preserved = [str(old), "%ASTER_PATH_TEST_OTHER%", str(other / "missing")]
    entries = preserved.copy()
    if existing_equivalents:
        entries += [
            str(target),
            str(target).upper(),
            str(target) + "\\",
            "%ASTER_PATH_TEST_TARGET%",
        ]
    config = {
        "installer": str(INSTALLER),
        "target": str(target),
        "other": str(other),
        "user_path": ";".join(entries),
        "process_path": ";".join(entries),
        "machine_path": str(machine),
        "result": str(tmp_path / "result.json"),
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    harness = tmp_path / "harness.ps1"
    harness.write_text(HARNESS, encoding="utf-8")
    result = subprocess.run(
        [
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-ConfigFile",
            str(config_file),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    first, second = json.loads(Path(config["result"]).read_text(encoding="utf-8-sig"))["runs"]
    assert first == second
    assert first["writes"] == 1
    assert first["user_path"].split(";") == [str(target), *preserved]
    assert first["process_path"].split(";")[0] == str(target)
    assert first["current"] == str(target / "aster.exe")
    assert first["machine_path"] == str(machine)
    if machine_conflict:
        assert first["fresh"] == str(machine / machine_conflict)
        assert len(first["warnings"]) == 1
        assert str(machine / machine_conflict) in first["warnings"][0]
    else:
        assert first["fresh"] == str(target / "aster.exe")
        assert first["warnings"] == []
