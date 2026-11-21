# sync_release_version.ps1 -- Align src/version.py and msix/AppxManifest.xml with a release tag.
# Usage: .\scripts\sync_release_version.ps1 -Tag "v2.0.3"
# CI sets Tag from GITHUB_REF_NAME (e.g. refs/tags/v2.0.3 -> pass v2.0.3 only).

param(
    [Parameter(Mandatory = $true)]
    [string]$Tag
)

$ErrorActionPreference = "Stop"

# Set-Content -Encoding utf8NoBOM needs PowerShell 7; the documented local
# run is often Windows PowerShell 5.1, where its -Encoding UTF8 adds a BOM.
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text + "`n", $utf8NoBom)
}

$trimmed = $Tag.Trim()
if ($trimmed.StartsWith("refs/tags/")) {
    $trimmed = $trimmed.Substring("refs/tags/".Length)
}

if ($trimmed -cnotmatch '^v(\d+)\.(\d+)\.(\d+)(?:-(alpha|beta|rc)\.([1-9]\d?))?$') {
    throw "Tag must be vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-(alpha|beta|rc).N (N in 1..99), got: $Tag"
}

$major = [int]$Matches[1]
$minor = [int]$Matches[2]
$patch = [int]$Matches[3]
$semver = "$major.$minor.$patch"
# Microsoft Store Partner Center reserves the fourth MSIX part and requires
# .0 on Store submissions. Prerelease revisions (100+N/200+N/300+N) are
# sideload-only and must be uninstalled before installing stable X.Y.Z.0.
$revision = 0
if ($Matches[4]) {
    $stageOffset = switch ($Matches[4]) {
        "alpha" { 100 }
        "beta" { 200 }
        "rc" { 300 }
    }
    $revision = $stageOffset + [int]$Matches[5]
}
$msixFourPart = "$semver.$revision"

$repoRoot = Split-Path $PSScriptRoot -Parent

$versionFile = Join-Path $repoRoot "src\version.py"
$manifestFile = Join-Path $repoRoot "msix\AppxManifest.xml"

if (!(Test-Path $versionFile)) { throw "Not found: $versionFile" }
if (!(Test-Path $manifestFile)) { throw "Not found: $manifestFile" }

$versionContent = @"
"""stemma version string."""

__version__ = "$semver"
"@
Write-Utf8NoBom $versionFile ($versionContent -replace "`r`n", "`n")

$manifestRaw = Get-Content -Path $manifestFile -Raw -Encoding UTF8
# Word boundary so MinVersion="..." is not matched (it contains the substring Version=").
$pattern = '\bVersion="\d+\.\d+\.\d+\.\d+"'
$replacement = "Version=`"$msixFourPart`""
if ($manifestRaw -notmatch $pattern) {
    throw "AppxManifest.xml: could not find Identity Version attribute to update"
}
$updatedManifest = $manifestRaw -replace $pattern, $replacement
Write-Utf8NoBom $manifestFile ($updatedManifest.TrimEnd() -replace "`r`n", "`n")

Write-Output "Synced release version: app $semver, MSIX $msixFourPart (from tag $trimmed)"
