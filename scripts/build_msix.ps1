# build_msix.ps1 -- Package the PyInstaller output into an MSIX.
#
# Prerequisites:
#   1. Run `pyinstaller stemma.spec` first (creates dist/stemma/).
#   2. Windows SDK installed (provides makeappx.exe).
#
# Usage:
#   .\scripts\build_msix.ps1            # defaults to dist/stemma.msix
#   .\scripts\build_msix.ps1 -Output C:\out\stemma.msix

param(
    [string]$Output = "dist\stemma.msix",
    [string]$LayoutDir = "dist\stemma"
)

$ErrorActionPreference = "Stop"

# -- Verify PyInstaller output exists --
if (-not (Test-Path "$LayoutDir\stemma.exe")) {
    throw "PyInstaller output not found at $LayoutDir\stemma.exe. Run 'pyinstaller stemma.spec' first."
}

# -- Copy manifest into layout --
Copy-Item "msix\AppxManifest.xml" "$LayoutDir\AppxManifest.xml" -Force
Write-Output "Copied AppxManifest.xml"

# -- Copy MSIX visual assets --
$imagesDir = "$LayoutDir\Images"
New-Item -ItemType Directory -Path $imagesDir -Force | Out-Null

# Map manifest asset names to generated filenames
$assetMap = @{
    "StoreLogo.png"          = "assets\msix\StoreLogo.scale-100.png"
    "Square44x44Logo.png"    = "assets\msix\Square44x44Logo.scale-100.png"
    "Square150x150Logo.png"  = "assets\msix\Square150x150Logo.scale-100.png"
    "Wide310x150Logo.png"    = "assets\msix\Wide310x150Logo.scale-100.png"
}

# Also copy scaled variants so Windows picks the best resolution
$scaledAssets = @(
    "StoreLogo.scale-100.png",
    "StoreLogo.scale-200.png",
    "Square44x44Logo.scale-100.png",
    "Square44x44Logo.scale-200.png",
    "Square44x44Logo.targetsize-24.png",
    "Square44x44Logo.targetsize-32.png",
    "Square44x44Logo.targetsize-48.png",
    "Square150x150Logo.scale-100.png",
    "Square150x150Logo.scale-200.png",
    "Wide310x150Logo.scale-100.png"
)

foreach ($entry in $assetMap.GetEnumerator()) {
    Copy-Item $entry.Value "$imagesDir\$($entry.Key)" -Force
}

foreach ($asset in $scaledAssets) {
    $src = "assets\msix\$asset"
    if (Test-Path $src) {
        Copy-Item $src "$imagesDir\$asset" -Force
    }
}
Write-Output "Copied MSIX visual assets to $imagesDir"

# -- Find makeappx.exe --
$makeappx = $null
$sdkPaths = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\bin",
    "$env:ProgramFiles\Windows Kits\10\bin"
)
foreach ($sdkBase in $sdkPaths) {
    if (Test-Path $sdkBase) {
        $candidates = Get-ChildItem "$sdkBase\*\x64\makeappx.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending
        if ($candidates) {
            $makeappx = $candidates[0].FullName
            break
        }
    }
}

if (-not $makeappx) {
    throw "makeappx.exe not found. Install the Windows 10/11 SDK."
}
Write-Output "Using: $makeappx"

# -- Remove old package if present --
if (Test-Path $Output) {
    Remove-Item $Output -Force
}

# -- Pack --
& $makeappx pack /d $LayoutDir /p $Output /o
if ($LASTEXITCODE -ne 0) {
    throw "makeappx pack failed with exit code $LASTEXITCODE"
}

$size = [math]::Round((Get-Item $Output).Length / 1MB, 1)
Write-Output ""
Write-Output "MSIX package created: $Output ($size MB)"
Write-Output "Upload this file to Partner Center to submit to the Microsoft Store."
