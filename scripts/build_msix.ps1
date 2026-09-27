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
# scripts/generate_app_icons.py writes every scale and target size, each
# rendered from the drawing made for that size. They are only used through
# resources.pri (built below); the manifest names the unqualified files.
$imagesDir = "$LayoutDir\Images"
if (Test-Path $imagesDir) {
    Remove-Item $imagesDir -Recurse -Force
}
New-Item -ItemType Directory -Path $imagesDir -Force | Out-Null
$images = Get-ChildItem "assets\msix\*.png"
if ($images.Count -eq 0) {
    throw "No MSIX images in assets\msix. Run scripts\generate_app_icons.py."
}
Copy-Item $images.FullName $imagesDir -Force
Write-Output "Copied $($images.Count) MSIX visual assets to $imagesDir"

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

# -- Build resources.pri so Windows picks the per-size icons --
$makepri = Join-Path (Split-Path $makeappx) "makepri.exe"
if (-not (Test-Path $makepri)) {
    throw "makepri.exe not found next to $makeappx."
}
# Index a staging copy holding only the manifest and Images: the PRI stores
# Images\... paths, which are the same in the real layout.
$priStage = Join-Path ([System.IO.Path]::GetTempPath()) "stemma-pri-$PID"
if (Test-Path $priStage) {
    Remove-Item $priStage -Recurse -Force
}
New-Item -ItemType Directory -Path $priStage | Out-Null
try {
    Copy-Item "$LayoutDir\AppxManifest.xml" $priStage
    Copy-Item $imagesDir $priStage -Recurse
    & $makepri new /pr $priStage /cf "msix\priconfig.xml" `
        /mn "$priStage\AppxManifest.xml" /of "$priStage\resources.pri" /o
    if ($LASTEXITCODE -ne 0) {
        throw "makepri failed with exit code $LASTEXITCODE"
    }
    Copy-Item "$priStage\resources.pri" "$LayoutDir\resources.pri" -Force
} finally {
    Remove-Item $priStage -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Output "Built resources.pri"

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
