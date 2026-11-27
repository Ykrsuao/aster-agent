# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
& {

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$uvVersion = '0.11.4' # Keep in sync with .github/workflows/publish.yml.
$repository = 'Ykrsuao/aster-agent'

if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or
    [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') {
    throw 'The one-line installer currently supports native Windows x64.'
}

$previousProtocol = [Net.ServicePointManager]::SecurityProtocol
try {
# Older Windows PowerShell may otherwise negotiate TLS 1.0 for these downloads.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Get-LatestStableVersion {
    $latestUrl = "https://github.com/$repository/releases/latest"
    try {
        $response = Invoke-WebRequest -Uri $latestUrl -Method Head -MaximumRedirection 0 `
            -UseBasicParsing -TimeoutSec 30 -ErrorAction SilentlyContinue
    } catch {
        # PowerShell 7 throws on redirects; Windows PowerShell 5 returns the response.
        $response = $_.Exception.Response
        if ($null -eq $response) { throw }
    }
    $status = [int]$response.StatusCode
    if ($status -notin @(301, 302, 303, 307, 308)) {
        throw "Could not find a stable Aster release (HTTP $status). Check $latestUrl. No installation was changed."
    }
    $location = if ($response.Headers -is [Collections.IDictionary]) {
        [string]$response.Headers['Location']
    } else {
        [string]$response.Headers.Location
    }
    if ([string]::IsNullOrWhiteSpace($location)) { throw 'GitHub returned an invalid release redirect.' }
    $releaseUri = [Uri]::new([Uri]$latestUrl, $location)
    if ($releaseUri.Scheme -ne 'https' -or $releaseUri.Host -ne 'github.com' -or
        $releaseUri.AbsolutePath -cnotmatch '^/Ykrsuao/aster-agent/releases/tag/v(?<version>[0-9]+\.[0-9]+\.[0-9]+)$') {
        throw 'No stable Aster release is published yet. Publish a vX.Y.Z GitHub Release before using this installer.'
    }
    return $Matches.version
}

function Get-CompatibleUv([string]$Executable) {
    try {
        $reportedVersion = & $Executable --version 2>$null
        if ($LASTEXITCODE -eq 0 -and
            $reportedVersion -match '^uv (?<version>[0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)' -and
            [version]$Matches.version -ge [version]$uvVersion) {
            return $Executable
        }
    } catch {
        # A broken or unrecognized uv must not block the private bootstrap.
    }
}

function Add-UserPath([string]$Directory) {
    $fullDirectory = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    $remaining = @($entries | Where-Object {
        try {
            $candidate = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($_)).TrimEnd('\')
            -not [string]::Equals($candidate, $fullDirectory, [StringComparison]::OrdinalIgnoreCase)
        } catch {
            # Preserve existing PATH entries we cannot normalize.
            $true
        }
    })
    $newPath = (@($fullDirectory) + $remaining) -join ';'
    if ($newPath -cne $userPath) {
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    }
    $currentPath = $env:PATH
    try {
        $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $env:PATH = [Environment]::ExpandEnvironmentVariables("$machinePath;$newPath")
        $nextCommand = Get-Command aster -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($nextCommand -and $nextCommand.Source -ine (Join-Path $fullDirectory 'aster.exe')) {
            Write-Warning "Another Aster command takes priority in the system PATH: $($nextCommand.Source). For new terminals, run $(Join-Path $fullDirectory 'aster.exe') or remove the old entry from PATH."
        }
    } finally {
        $currentEntries = @($currentPath -split ';' | Where-Object { $_ -ine $fullDirectory })
        $env:PATH = (@($fullDirectory) + $currentEntries) -join ';'
    }
}

$asterInstallVersion = Get-LatestStableVersion
$tag = "v$asterInstallVersion"
$assetName = "aster-terminal-$asterInstallVersion-windows-x64.zip"
$assetUrl = "https://github.com/$repository/releases/download/$tag/$assetName"
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
$tempDirectory = [IO.Path]::GetFullPath((Join-Path $tempRoot ("aster-install-" + [Guid]::NewGuid().ToString('N'))))
if (-not $tempDirectory.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Temporary install directory escaped the system temporary directory.'
}

$uvDirectory = Join-Path $env:LOCALAPPDATA 'Aster\uv'
$previousUvInstallDir = $env:UV_INSTALL_DIR
$previousUvNoModifyPath = $env:UV_NO_MODIFY_PATH

try {
    New-Item -ItemType Directory -Path $tempDirectory | Out-Null
    $archive = Join-Path $tempDirectory $assetName
    $extractDirectory = Join-Path $tempDirectory 'bundle'
    Write-Host "Downloading Aster $asterInstallVersion..."
    Invoke-WebRequest -Uri $assetUrl -OutFile $archive -UseBasicParsing -TimeoutSec 600
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf) -or (Get-Item -LiteralPath $archive).Length -eq 0) {
        throw 'GitHub returned an empty or invalid terminal bundle.'
    }
    $checksumFile = "$archive.sha256"
    Invoke-WebRequest -Uri "$assetUrl.sha256" -OutFile $checksumFile -UseBasicParsing -TimeoutSec 120
    $checksumLine = (Get-Content -LiteralPath $checksumFile -Raw).Trim()
    $checksumPattern = '^(?<hash>[0-9a-fA-F]{64}) [ *]' + [regex]::Escape($assetName) + '$'
    $checksumMatch = [regex]::Match($checksumLine, $checksumPattern)
    if (-not $checksumMatch.Success) {
        throw 'The release archive checksum is missing or invalid.'
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if (-not [string]::Equals($checksumMatch.Groups['hash'].Value, $actualHash, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The release archive SHA-256 checksum does not match.'
    }

    Expand-Archive -LiteralPath $archive -DestinationPath $extractDirectory
    $bundleInstaller = Join-Path $extractDirectory 'install.ps1'
    if (-not (Test-Path -LiteralPath $bundleInstaller -PathType Leaf)) {
        throw 'The release archive does not contain its installer.'
    }

    $uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $uv = if ($null -ne $uvCommand) { Get-CompatibleUv $uvCommand.Source }
    $privateUv = Join-Path $uvDirectory 'uv.exe'
    if (-not $uv -and (Test-Path -LiteralPath $privateUv -PathType Leaf)) {
        $uv = Get-CompatibleUv $privateUv
    }
    if (-not $uv) {
        Write-Host "Installing uv $uvVersion..."
        $uvInstallerUrl = "https://astral.sh/uv/$uvVersion/install.ps1"
        $uvResponse = Invoke-WebRequest -Uri $uvInstallerUrl -UseBasicParsing -TimeoutSec 120
        $uvInstaller = if ($uvResponse.Content -is [byte[]]) {
            [Text.Encoding]::UTF8.GetString($uvResponse.Content)
        } else {
            [string]$uvResponse.Content
        }
        $env:UV_INSTALL_DIR = $uvDirectory
        $env:UV_NO_MODIFY_PATH = '1'
        Invoke-Expression $uvInstaller
        $uv = Get-CompatibleUv $privateUv
        if (-not $uv) {
            throw "The pinned uv installer did not provide uv $uvVersion or newer."
        }
    }
    if ($uv -ieq $privateUv) { $env:PATH = "$uvDirectory;$env:PATH" }
    $binDirectory = & $uv tool dir --bin
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($binDirectory)) {
        throw 'Could not locate the Aster command directory.'
    }
    $binDirectory = $binDirectory.Trim()

    # The embedded script carries Mark-of-the-Web on some Windows setups.
    # Run its verifier with policy bypass scoped to this child process only.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $bundleInstaller -BinDirectory $binDirectory
    if ($LASTEXITCODE -ne 0) { throw "Aster installation failed (exit $LASTEXITCODE)." }

    $aster = Join-Path $binDirectory 'aster.exe'
    if (-not (Test-Path -LiteralPath $aster -PathType Leaf)) {
        throw "Aster was installed, but its command was not found at $aster."
    }
    & $aster --version
    if ($LASTEXITCODE -ne 0) { throw 'Aster did not pass its version check.' }
    try {
        Add-UserPath $binDirectory
    } catch {
        Write-Warning "Aster installed, but the user PATH could not be updated: $_"
        $env:PATH = "$binDirectory;$env:PATH"
        Write-Warning "For future terminals, add this directory to PATH: $binDirectory"
    }
    Write-Host 'Run: aster connect (first setup), then: aster'
    Write-Host 'To update, close Aster and run this installation command again.'
} finally {
    $env:UV_INSTALL_DIR = $previousUvInstallDir
    $env:UV_NO_MODIFY_PATH = $previousUvNoModifyPath
    if (Test-Path -LiteralPath $tempDirectory -PathType Container) {
        Remove-Item -LiteralPath $tempDirectory -Recurse -Force
    }
}
} finally {
    [Net.ServicePointManager]::SecurityProtocol = $previousProtocol
}
}
