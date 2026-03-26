# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for stemma -- one-folder Windows build.

One-folder mode eliminates the 3-5 second extraction delay that one-file
mode incurs on every launch.  The output folder is zipped for distribution
and will be bundled into an MSIX package for v2.0.
"""

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# -- Collect runtime data, binaries, and hidden imports for key packages ------

ort_datas, ort_binaries, ort_hiddenimports = collect_all("onnxruntime")
sd_datas, sd_binaries, sd_hiddenimports = collect_all("sounddevice")
sd2_datas, sd2_binaries, sd2_hiddenimports = collect_all("_sounddevice_data")
ffmpeg_datas, ffmpeg_binaries, ffmpeg_hiddenimports = collect_all(
    "imageio_ffmpeg"
)
svg_datas, svg_binaries, svg_hiddenimports = collect_all("PySide6.QtSvg")

# -- Analysis -----------------------------------------------------------------

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=(
        ort_binaries
        + sd_binaries
        + sd2_binaries
        + ffmpeg_binaries
        + svg_binaries
    ),
    datas=[
        ("assets/icons", "assets/icons"),
        ("assets/audio", "assets/audio"),
    ]
    + ort_datas
    + sd_datas
    + sd2_datas
    + ffmpeg_datas
    + svg_datas,
    hiddenimports=[
        "onnxruntime",
        "lameenc",
        "sounddevice",
        "_sounddevice_data",
        "imageio_ffmpeg",
        "PySide6.QtSvg",
        "sklearn.utils._typedefs",
        "sklearn.neighbors._typedefs",
        "sklearn.neighbors._partition_nodes",
        "scipy.signal",
        "scipy._lib.messagestream",
    ]
    + ort_hiddenimports
    + sd_hiddenimports
    + sd2_hiddenimports
    + ffmpeg_hiddenimports
    + svg_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
        "PySide6.QtMultimedia",
        "PySide6.QtNetwork",
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="stemma",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/icons/stemma.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="stemma",
)
