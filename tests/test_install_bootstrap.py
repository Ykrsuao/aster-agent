# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Windows bootstrap without network, installations, or registry writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows PowerShell installer",
)

INSTALLER = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"

HARNESS = r"""
param([string]$ConfigFile)
$ErrorActionPreference = 'Stop'
# Load Windows PowerShell's modules before replacing command discovery.
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1')
Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Archive\Microsoft.PowerShell.Archive.psd1')
$script:config = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
$script:calls = [Collections.Generic.List[object]]::new()
$script:uvAvailable = $config.existing_uv
$script:privateUvAvailable = $config.existing_private_uv
$env:UV_INSTALL_DIR = $config.original_install_dir
$env:UV_NO_MODIFY_PATH = $config.original_no_modify_path
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::SystemDefault
$global:LASTEXITCODE = 0
$privateUv = Join-Path $env:LOCALAPPDATA 'Aster\uv\uv.exe'
New-Item -ItemType Directory -Path (Split-Path $privateUv) -Force | Out-Null
Set-Alias -Name $privateUv -Value Invoke-TestPrivateUv -Scope Global
if ($script:privateUvAvailable) { Set-Content -LiteralPath $privateUv -Value 'test sentinel' }

function Invoke-WebRequest {
    param($Uri, $OutFile, [switch]$UseBasicParsing, $TimeoutSec, $Method, $MaximumRedirection, $ErrorAction)
    if ($Uri -eq 'https://github.com/Ykrsuao/aster-agent/releases/latest') {
        $script:calls.Add(@{kind='latest'; uri=[string]$Uri; method=$Method; redirects=$MaximumRedirection})
        if ($config.failure -eq 'network') {
            throw [Net.WebException]::new('Simulated connection failure')
        }
        $status = if ($config.failure -eq 'http') { 503 } else { 302 }
        $location = switch ($config.failure) {
            'no_release' { 'https://github.com/Ykrsuao/aster-agent/releases' }
            'wrong_host' { 'https://example.invalid/Ykrsuao/aster-agent/releases/tag/v1.2.3' }
            'unstable_tag' { 'https://github.com/Ykrsuao/aster-agent/releases/tag/v1.2.3-rc.1' }
            default { 'https://github.com/Ykrsuao/aster-agent/releases/tag/v1.2.3' }
        }
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            $response = [Net.Http.HttpResponseMessage]::new([Net.HttpStatusCode]$status)
            $response.Headers.Location = [Uri]$location
            throw [Microsoft.PowerShell.Commands.HttpResponseException]::new('Simulated response', $response)
        }
        return [pscustomobject]@{StatusCode=$status; Headers=@{Location=$location}}
    }
    $script:calls.Add(@{kind='download'; uri=[string]$Uri})
    $asset = 'https://github.com/Ykrsuao/aster-agent/releases/download/v1.2.3/aster-terminal-1.2.3-windows-x64.zip'
    if ($Uri -eq $asset) {
        if ($config.failure -eq 'download') { throw 'Simulated bundle download failure' }
        Copy-Item -LiteralPath $config.archive -Destination $OutFile
    } elseif ($Uri -eq "$asset.sha256") {
        $digest = if ($config.failure -eq 'checksum') { '0' * 64 } else { $config.digest }
        $filename = if ($config.failure -eq 'checksum_filename') { 'wrong.zip' } else { 'aster-terminal-1.2.3-windows-x64.zip' }
        Set-Content -LiteralPath $OutFile -Value "$digest $($config.checksum_marker)$filename" -Encoding ASCII
    } elseif ($Uri -eq 'https://astral.sh/uv/0.11.4/install.ps1') {
        return [pscustomobject]@{Content='Invoke-TestUvInstall'}
    } else {
        throw "Unexpected download: $Uri"
    }
}

function Invoke-TestUvInstall {
    $script:calls.Add(@{
        kind='uv_install'; directory=$env:UV_INSTALL_DIR; no_modify_path=$env:UV_NO_MODIFY_PATH
    })
    if ($config.failure -eq 'uv_install') { throw 'Simulated uv installation failure' }
    $script:privateUvAvailable = $true
    Set-Content -LiteralPath $privateUv -Value 'test sentinel'
}

function Get-Command {
    param($Name, $CommandType, $ErrorAction)
    if ($Name -eq 'uv') {
        if ($script:privateUvAvailable -and $env:PATH.StartsWith((Split-Path $privateUv) + ';')) {
            return [pscustomobject]@{Source=$privateUv}
        }
        if ($script:uvAvailable) { return [pscustomobject]@{Source='Invoke-TestUv'} }
        return
    }
    Microsoft.PowerShell.Core\Get-Command @PSBoundParameters
}

function Invoke-TestPrivateUv {
    if ($args[0] -eq '--version') {
        $script:calls.Add(@{kind='uv_version'; source='private'})
        $global:LASTEXITCODE = 0
        return "uv $($config.private_uv_version)"
    }
    Invoke-TestUv @args
}

function Invoke-TestUv {
    if ($args[0] -eq '--version') {
        $script:calls.Add(@{kind='uv_version'; source='existing'})
        $global:LASTEXITCODE = 0
        return "uv $($config.uv_version)"
    }
    $script:calls.Add(@{kind='uv'; arguments=@($args)})
    $global:LASTEXITCODE = 0
    return $config.bin_directory
}

function powershell.exe {
    $arguments = @($args)
    $script:calls.Add(@{kind='bundle_install'; arguments=$arguments})
    $bundle = $arguments[[Array]::IndexOf($arguments, '-File') + 1]
    if (-not (Test-Path -LiteralPath $bundle -PathType Leaf)) { throw 'Installer is missing' }
    if ($config.failure -eq 'bundle_install') { $global:LASTEXITCODE = 19; return }
    $bin = $arguments[[Array]::IndexOf($arguments, '-BinDirectory') + 1]
    New-Item -ItemType Directory -Path $bin -Force | Out-Null
    $launcher = Join-Path $bin 'aster.exe'
    Set-Content -LiteralPath $launcher -Value 'test sentinel, not an executable'
    Set-Alias -Name $launcher -Value Invoke-TestAster -Scope Global
    $global:LASTEXITCODE = 0
}

function Invoke-TestAster {
    $script:calls.Add(@{kind='version'; arguments=@($args)})
    $global:LASTEXITCODE = if ($config.failure -eq 'version') { 7 } else { 0 }
    'Aster 1.2.3'
}

# Replace only the registry-writing function, keeping the real install control flow.
$source = Get-Content -LiteralPath $config.installer -Raw
$tokens = $null
$parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw "Installer parse errors: $parseErrors" }
$functions = @($ast.FindAll({
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Add-UserPath'
}, $true))
if ($functions.Count -ne 1) { throw 'Expected exactly one Add-UserPath function' }
$extent = $functions[0].Extent
$replacement = @'
function Add-UserPath([string]$Directory) {
    $script:calls.Add(@{kind='user_path'; directory=$Directory})
    $env:PATH = "$Directory;$env:PATH"
}
'@
$source = $source.Remove($extent.StartOffset, $extent.EndOffset - $extent.StartOffset).Insert($extent.StartOffset, $replacement)
$runs = @()
for ($index = 0; $index -lt $config.repeat; $index++) {
    $failure = $null
    $protocolBefore = [int][Net.ServicePointManager]::SecurityProtocol
    try { & ([scriptblock]::Create($source)) } catch { $failure = $_.Exception.Message }
    $runs += @{
        error=$failure; install_dir=$env:UV_INSTALL_DIR; no_modify_path=$env:UV_NO_MODIFY_PATH
        protocol_before=$protocolBefore; protocol_after=[int][Net.ServicePointManager]::SecurityProtocol
    }
}
@{
    calls=@($script:calls.ToArray()); runs=$runs
    temporary_directories=@(Get-ChildItem -LiteralPath $env:TEMP -Directory -Filter 'aster-install-*' | ForEach-Object Name)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $config.result -Encoding UTF8
"""


@pytest.fixture(params=["powershell.exe", "pwsh.exe"], ids=["powershell5", "powershell7"])
def powershell(request):
    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not available")
    return executable


def run_installer(
    tmp_path,
    *,
    shell,
    existing_uv=True,
    failure="",
    repeat=1,
    existing_environment=True,
    checksum_marker=" ",
    uv_version="0.11.4",
    private_uv_version="0.11.4",
    existing_private_uv=False,
):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("install.ps1", "throw 'The test must intercept this installer'\n")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    local_app_data = tmp_path / "local app data"
    local_app_data.mkdir()
    config = {
        "installer": str(INSTALLER),
        "archive": str(archive),
        "digest": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "bin_directory": str(tmp_path / "existing uv commands"),
        "result": str(tmp_path / "result.json"),
        "existing_uv": existing_uv,
        "existing_private_uv": existing_private_uv,
        "uv_version": uv_version,
        "private_uv_version": private_uv_version,
        "failure": failure,
        "repeat": repeat,
        "checksum_marker": checksum_marker,
        "original_install_dir": "original-install-dir" if existing_environment else None,
        "original_no_modify_path": "original-no-modify-path" if existing_environment else None,
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    harness = tmp_path / "harness.ps1"
    harness.write_text(HARNESS, encoding="utf-8")
    process = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-ConfigFile",
            str(config_file),
        ],
        env=os.environ
        | {"TEMP": str(temporary), "TMP": str(temporary), "LOCALAPPDATA": str(local_app_data)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(Path(config["result"]).read_text(encoding="utf-8-sig"))
    assert result["temporary_directories"] == []
    for run in result["runs"]:
        assert run["install_dir"] == config["original_install_dir"]
        assert run["no_modify_path"] == config["original_no_modify_path"]
        assert run["protocol_before"] == run["protocol_after"]
    return result, config


@pytest.mark.parametrize("existing_uv", [False, True])
@pytest.mark.parametrize("existing_environment", [False, True])
def test_installer_downloads_verifies_and_installs_offline(
    tmp_path, powershell, existing_uv, existing_environment
):
    result, config = run_installer(
        tmp_path,
        shell=powershell,
        existing_uv=existing_uv,
        existing_environment=existing_environment,
    )
    assert result["runs"][0]["error"] is None
    calls = result["calls"]
    assert calls[0]["method"] == "Head"
    assert calls[0]["redirects"] == 0
    kinds = [call["kind"] for call in calls]
    assert kinds.count("uv_install") == (0 if existing_uv else 1)
    assert kinds[-4:] == ["uv", "bundle_install", "version", "user_path"]
    assert next(call for call in calls if call["kind"] == "uv")["arguments"] == [
        "tool",
        "dir",
        "--bin",
    ]
    install = next(call for call in calls if call["kind"] == "bundle_install")
    assert install["arguments"][-2:] == ["-BinDirectory", config["bin_directory"]]
    assert calls[-1]["directory"] == config["bin_directory"]
    if not existing_uv:
        uv_install = next(call for call in calls if call["kind"] == "uv_install")
        assert uv_install["no_modify_path"] == "1"
        assert uv_install["directory"] == str(tmp_path / "local app data" / "Aster" / "uv")


@pytest.mark.parametrize(
    ("failure", "message", "forbidden"),
    [
        ("no_release", "No stable Aster release", {"download", "uv_install", "bundle_install"}),
        ("wrong_host", "No stable Aster release", {"download", "uv_install", "bundle_install"}),
        ("unstable_tag", "No stable Aster release", {"download", "uv_install", "bundle_install"}),
        ("http", "HTTP 503", {"download", "uv_install", "bundle_install"}),
        ("network", "Simulated connection failure", {"download", "uv_install", "bundle_install"}),
        ("download", "Simulated bundle download failure", {"uv_install", "bundle_install"}),
        ("checksum", "checksum does not match", {"uv_install", "bundle_install"}),
        ("checksum_filename", "checksum is missing or invalid", {"uv_install", "bundle_install"}),
        ("uv_install", "Simulated uv installation failure", {"bundle_install"}),
        ("bundle_install", "installation failed (exit 19)", {"version", "user_path"}),
        ("version", "did not pass its version check", {"user_path"}),
    ],
)
def test_installer_failure_stops_before_next_side_effect_and_cleans_up(
    tmp_path, powershell, failure, message, forbidden
):
    result, _ = run_installer(tmp_path, shell=powershell, existing_uv=False, failure=failure)
    assert message in result["runs"][0]["error"]
    assert forbidden.isdisjoint(call["kind"] for call in result["calls"])
    assert (tmp_path / "existing uv commands" / "aster.exe").exists() == (failure == "version")


def test_installer_reuses_uv_command_directory_when_run_again(tmp_path, powershell):
    result, config = run_installer(tmp_path, shell=powershell, existing_uv=False, repeat=2)
    assert all(run["error"] is None for run in result["runs"])
    assert sum(call["kind"] == "uv_install" for call in result["calls"]) == 1
    for kind in ("uv", "bundle_install", "version", "user_path"):
        assert sum(call["kind"] == kind for call in result["calls"]) == 2
    paths = [call["directory"] for call in result["calls"] if call["kind"] == "user_path"]
    assert paths == [config["bin_directory"]] * 2


@pytest.mark.parametrize(
    ("version", "install_private"),
    [("0.8.4", True), ("unknown", True), ("0.11.4", False), ("0.12.0", False)],
)
def test_installer_requires_compatible_uv(tmp_path, powershell, version, install_private):
    result, _ = run_installer(tmp_path, shell=powershell, uv_version=version)
    assert result["runs"][0]["error"] is None
    installs = [call for call in result["calls"] if call["kind"] == "uv_install"]
    assert len(installs) == int(install_private)
    if installs:
        assert installs[0]["directory"] == str(tmp_path / "local app data" / "Aster" / "uv")
    versions = [call["source"] for call in result["calls"] if call["kind"] == "uv_version"]
    assert versions == (["existing", "private"] if install_private else ["existing"])


def test_installer_reuses_compatible_private_uv(tmp_path, powershell):
    result, _ = run_installer(
        tmp_path, shell=powershell, uv_version="0.8.4", existing_private_uv=True
    )
    assert result["runs"][0]["error"] is None
    assert not any(call["kind"] == "uv_install" for call in result["calls"])
    assert [call["source"] for call in result["calls"] if call["kind"] == "uv_version"] == [
        "existing",
        "private",
    ]


def test_installer_rejects_incompatible_uv_after_bootstrap(tmp_path, powershell):
    result, _ = run_installer(
        tmp_path, shell=powershell, uv_version="0.8.4", private_uv_version="0.8.4"
    )
    assert "did not provide uv 0.11.4 or newer" in result["runs"][0]["error"]
    assert not any(call["kind"] in {"uv", "bundle_install"} for call in result["calls"])


def test_installer_accepts_binary_checksum_marker(tmp_path, powershell):
    result, _ = run_installer(tmp_path, shell=powershell, checksum_marker="*")
    assert result["runs"][0]["error"] is None
    assert any(call["kind"] == "bundle_install" for call in result["calls"])
