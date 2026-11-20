# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$Python = '3.12',
    [string]$ToolsDirectory,
    [string]$BinDirectory,
    [switch]$Offline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or
    [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') {
    throw 'This bundle requires native Windows x64 PowerShell.'
}
$bundle = $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $bundle 'MANIFEST.json') -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or $manifest.platform -ne 'win_amd64' -or $manifest.version -notmatch '^[0-9][A-Za-z0-9.+]*$') {
    throw 'Unsupported terminal bundle manifest.'
}
$packages = @('aster', 'aster-cli', 'aster-acp', 'aster-memory', 'aster-bench')
if (@($manifest.packages.PSObject.Properties).Count -ne $packages.Count) {
    throw 'The bundle must contain exactly five Aster packages.'
}
$required = @('install.ps1', 'constraints.txt', 'README.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md')
$wheels = @{}
foreach ($package in $packages) {
    $tag = if ($package -eq 'aster-cli') { 'py3-none-win_amd64' } else { 'py3-none-any' }
    $expected = "$($package.Replace('-', '_'))-$($manifest.version)-$tag.whl"
    if ($manifest.packages.$package -cne $expected) {
        throw "Invalid package version or platform: $package"
    }
    $relative = "wheels/$expected"
    $required += $relative
    $wheels[$package] = Join-Path $bundle $relative
}
if (@($manifest.files.PSObject.Properties).Count -ne $required.Count) {
    throw 'Unexpected bundle file list.'
}
foreach ($relative in $required) {
    $digest = $manifest.files.$relative
    $path = Join-Path $bundle $relative
    if ($digest -notmatch '^[a-f0-9]{64}$' -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing or invalid bundle file: $relative"
    }
    $stream = [IO.File]::OpenRead($path)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $actual = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $hasher.Dispose()
        $stream.Dispose()
    }
    if ($actual -cne $digest) {
        throw "Bundle checksum mismatch: $relative"
    }
}
$uv = (Get-Command uv -CommandType Application -ErrorAction Stop).Source
$previousTools = $env:UV_TOOL_DIR
$previousBin = $env:UV_TOOL_BIN_DIR
try {
    if ($ToolsDirectory) { $env:UV_TOOL_DIR = [IO.Path]::GetFullPath($ToolsDirectory) }
    if ($BinDirectory) { $env:UV_TOOL_BIN_DIR = [IO.Path]::GetFullPath($BinDirectory) }
    $arguments = @('tool', 'install', '--no-config', '--python', $Python,
        '--constraints', (Join-Path $bundle 'constraints.txt'), '--with-executables-from', 'aster-acp',
        $wheels['aster-cli'])
    foreach ($package in $packages) {
        $arguments += @('--reinstall-package', $package)
        if ($package -ne 'aster-cli') { $arguments += @('--with', $wheels[$package]) }
    }
    if ($Offline) { $arguments += '--offline' }
    & $uv @arguments
    if ($LASTEXITCODE -ne 0) { throw "uv tool install failed (exit $LASTEXITCODE)." }
    $commandDirectory = & $uv tool dir --bin
    if ($LASTEXITCODE -ne 0) { throw 'Could not locate the installed command directory.' }
    Write-Host "Aster $($manifest.version) installed. Run: $(Join-Path $commandDirectory 'aster.exe')"
    Write-Host 'Model configuration, workspaces, conversations and drafts were not modified.'
    Write-Host 'If the command directory is not on PATH, run: uv tool update-shell'
} finally {
    $env:UV_TOOL_DIR = $previousTools
    $env:UV_TOOL_BIN_DIR = $previousBin
}
