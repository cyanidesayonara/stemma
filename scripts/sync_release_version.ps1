# sync_release_version.ps1 -- Align src/version.py and msix/AppxManifest.xml with a release tag.
# Usage: .\scripts\sync_release_version.ps1 -Tag "v2.0.3"
# CI sets Tag from GITHUB_REF_NAME (e.g. refs/tags/v2.0.3 -> pass v2.0.3 only).

param(
    [Parameter(Mandatory = $true)]
    [string]$Tag
)

$trimmed = $Tag.Trim()
if ($trimmed.StartsWith("refs/tags/")) {
    $trimmed = $trimmed.Substring("refs/tags/".Length)
}

if ($trimmed -notmatch '^v(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$') {
    throw "Tag must be vMAJOR.MINOR.PATCH (optional prerelease suffix after '-' is ignored for MSIX), got: $Tag"
}

$major = [int]$Matches[1]
$minor = [int]$Matches[2]
$patch = [int]$Matches[3]
$semver = "$major.$minor.$patch"
$msixFourPart = "$semver.0"

$repoRoot = Split-Path $PSScriptRoot -Parent

$versionFile = Join-Path $repoRoot "src\version.py"
$manifestFile = Join-Path $repoRoot "msix\AppxManifest.xml"

if (!(Test-Path $versionFile)) { throw "Not found: $versionFile" }
if (!(Test-Path $manifestFile)) { throw "Not found: $manifestFile" }

$versionContent = @"
"""stemma version string."""

__version__ = "$semver"
"@
Set-Content -Path $versionFile -Value $versionContent -Encoding utf8NoBOM

$manifestRaw = Get-Content -Path $manifestFile -Raw
# Word boundary so MinVersion="..." is not matched (it contains the substring Version=").
$pattern = '\bVersion="\d+\.\d+\.\d+\.\d+"'
$replacement = "Version=`"$msixFourPart`""
$updatedManifest = $manifestRaw -replace $pattern, $replacement
if ($updatedManifest -eq $manifestRaw) {
    throw "AppxManifest.xml: could not find Identity Version attribute to update"
}
Set-Content -Path $manifestFile -Value $updatedManifest.TrimEnd() -Encoding utf8NoBOM

Write-Output "Synced release version: app $semver, MSIX $msixFourPart (from tag $trimmed)"
