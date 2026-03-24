# stemma

A Windows desktop music player with AI stem separation. Import a song, separate it into stems, mute/solo individual instruments, play along.

**Personal-use tool. No cloud, no subscriptions, no command line needed.**

---

## Concept

1. Import a song (MP3/WAV/FLAC)
2. AI separates it into stems (vocals, drums, bass, guitar, piano, other)
3. Mute/solo any stem — play along with your instrument
4. Export stems or custom mixes

---

## Decisions

| Decision | Choice | Notes |
|---|---|---|
| **Name** | stemma (always lowercase) | |
| **Platform** | Windows desktop only | No mobile/web for now |
| **GPU** | NVIDIA RTX 4070 Ti | DirectML acceleration via ONNX Runtime |
| **Stems** | 4-stem and 6-stem models | Both available per song |
| **Distribution** | Single `.exe` via GitHub Releases | Models download on first run |
| **Python** | 3.14 | All deps confirmed compatible (March 2026) |

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| **Language** | Python 3.14 | Rich audio ecosystem |
| **GUI** | PySide6 (Qt 6) | Modern look, powerful widgets, LGPL |
| **Inference** | ONNX Runtime + DirectML | Lean (~50MB), GPU-accelerated, no PyTorch bloat |
| **Stem Models** | HTDemucs v4 ONNX | 4-stem + 6-stem, downloaded on first run (~80-300MB) |
| **Audio Playback** | `sounddevice` + `soundfile` | Low-latency multi-track mixing via NumPy |
| **Audio Processing** | `numpy` | Efficient buffer manipulation |
| **Export** | `soundfile` (WAV), `lameenc` (MP3) | Individual stems or custom mix |
| **YouTube Import** | `yt-dlp` + `ffmpeg` | Download audio from YouTube URLs |
| **Packaging** | PyInstaller | Single `.exe` (~150-250MB without models) |
| **Future** | `librosa` / other | Key transposition (pitch-shift); tempo stretch is implemented via librosa |

### Why HTDemucs v4?

- **Developer**: Meta AI Research — MIT license (fully open, free)
- **Quality**: State-of-the-art SDR (Signal-to-Distortion Ratio)
- **4-stem**: Vocals, Drums, Bass, Other
- **6-stem variant**: Adds Guitar + Piano (exactly what we need)
- **Speed**: ~30-60 seconds for a 4-min song on GPU; ~5-15 min on CPU
- **Model size**: ~80MB (quantized) to ~300MB (full)

### Why ONNX Runtime over PyTorch?

- PyTorch adds ~2GB+ to the installer — ONNX Runtime is ~50MB
- DirectML provider gives native GPU acceleration on Windows (no CUDA toolkit needed)
- Proven approach: Intel's OpenVINO Audacity plugin and others use the same pattern
- Models are converted once from PyTorch → ONNX format and hosted on HuggingFace

---

## Project Structure

```
stemma/
├── main.py                    # App entry point
├── stemma.spec                # PyInstaller one-file build spec
├── requirements.txt
├── requirements-dev.txt       # Dev/build deps (pyinstaller)
├── pyproject.toml             # pytest config
├── README.md
├── LICENSE                    # MIT
├── .gitignore
├── .github/workflows/ci.yml      # CI: fast tests on every push
├── .github/workflows/release.yml # Build .exe + GitHub Release on v* tags
├── src/
│   ├── __init__.py
│   ├── app.py                 # QApplication setup
│   ├── app_settings.py        # Typed QSettings (audio device, import/export defaults)
│   ├── data_paths.py          # Per-user data directory resolution
│   ├── import_messages.py     # User-facing text for import/download failures
│   ├── paths.py               # app_root(): frozen-build-aware root dir
│   ├── version.py             # __version__ string
│   ├── separator.py           # ONNX Runtime stem separation
│   ├── model_manager.py       # Download/cache ONNX models on first run
│   ├── player.py              # Multi-track audio player (sounddevice)
│   ├── library.py             # Song library (JSON-based)
│   ├── exporter.py            # Export stems as WAV/MP3
│   ├── downloader.py          # YouTube audio download (yt-dlp)
│   ├── post_processing.py     # Wiener filter + soft gate
│   ├── waveform.py            # Waveform peak computation (numpy)
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py     # Main window layout, menus, drag-and-drop import
│       ├── player_controls.py # Transport + waveform + stem mixer
│       ├── waveform_widget.py # Waveform display (QPainter)
│       ├── library_panel.py   # Song list with remove
│       ├── import_dialog.py   # Import songs + YouTube URL + model download
│       ├── preferences_dialog.py  # Data dir, audio device, defaults
│       └── styles.py          # Dark / light themes
├── tests/
│   ├── conftest.py            # Shared fixtures
│   ├── test_separator.py      # 22 tests
│   ├── test_model_manager.py  # 9 tests
│   ├── test_player.py         # 25 tests
│   ├── test_library.py        # 22 tests
│   ├── test_downloader.py     # 26 tests
│   ├── test_exporter.py       # 18 tests
│   ├── test_post_processing.py # 17 tests
│   ├── test_waveform.py       # 9 tests
│   ├── test_waveform_widget.py # 7 tests
│   ├── test_import_dialog.py
│   ├── test_import_messages.py
│   ├── test_data_paths.py
│   ├── test_app_settings.py
│   ├── test_theme.py
│   └── test_integration.py    # includes slow + hardware markers
└── data/                      # Created at runtime
    ├── library.json
    ├── models/                # Downloaded ONNX models cached here
    └── songs/{song-id}/
        ├── original.mp3
        ├── vocals.wav
        ├── drums.wav
        ├── bass.wav
        ├── guitar.wav         # (6-stem only)
        ├── piano.wav          # (6-stem only)
        └── other.wav
```

---

## Module Specifications

### `separator.py` — Stem Separation Engine
- Loads HTDemucs ONNX model via `onnxruntime.InferenceSession`
- Uses DirectML execution provider for GPU acceleration
- Handles STFT/iSTFT pre/post-processing in NumPy (stripped from ONNX model)
- Runs in background `QThread`, emits progress signals
- Supports both `htdemucs` (4-stem) and `htdemucs_6s` (6-stem)

### `model_manager.py` — Model Download & Cache
- Checks if ONNX model files exist under the app data directory (`models/`)
- `ModelDownloader` (`QThread`): downloads from HuggingFace on first run (~80-300MB per model)
- Signals: `progress`, **`download_complete(str)`** (model path; not named `finished`, to avoid shadowing `QThread.finished`), `error`
- Manages both 4-stem and 6-stem model files

### `import_messages.py` — Import Error Text
- `format_import_error(message)` maps raw exceptions to short, readable strings (disk full, permission, network, SSL, HTTP/404, timeout, cancel, truncation)

### `player.py` — Multi-Track Audio Player
- Loads stem WAVs as NumPy arrays
- `sounddevice.OutputStream` callback: reads buffers per stem, applies gain, sums to output
- API: `play()`, `pause()`, `stop()`, `seek()`, `set_mute()`, `set_solo()`, `set_volume()`
- Per-stem volume control (0.0-2.0)
- A-B loop: `set_loop_a()`, `set_loop_b()`, `set_looping()`, `clear_loop()`
- Tracks playback position for UI sync
- PortAudioError on open/start: stream cleanup and **`playback_failed`** signal (user-facing message for UI dialogs)

### `library.py` — Song Library
- JSON song index: `{id, title, artist, stems_path, model_used, date_added}`
- CRUD operations on the song list
- `add_song`: on `OSError` during file copy or index save, removes the partial per-song directory and does not leave a half-added entry
- Atomic writes via `os.replace()` to prevent corruption
- Graceful recovery from corrupted JSON

### `exporter.py` — Stem Export
- Export individual stems or custom mix (with current mute/solo state) as WAV or MP3
- MP3 encoding via `lameenc` (320kbps CBR, no ffmpeg needed)
- Peak normalization instead of hard clipping
- Background export via `ExportWorker` QThread

### `post_processing.py` — Audio Post-Processing
- Wiener filtering: magnitude-based soft masks reduce inter-stem bleed
- Soft gating: RMS-envelope-driven gate suppresses faint ghost artifacts
- Chunked processing (~10s windows) to bound memory usage

### `waveform.py` — Waveform Peak Computation
- Pure numpy, no Qt dependency
- `compute_peaks()`: sums active stems weighted by volume, computes per-bin peak amplitude
- Respects mute/solo state (same logic as audio callback)

### UI Modules
- **`main_window.py`** — Left panel: song library list. Center: player controls + stem mixer. Menu: File / Edit (Preferences) / Help. Keyboard shortcuts. Window state persistence via QSettings. Drag-and-drop audio import. Connects **`playback_failed`** and startup warning when no audio output devices exist.
- **`player_controls.py`** — Transport (Play/Pause/Stop + time display). Waveform display with click-to-seek, playback cursor, and A-B loop markers. A-B loop controls (Set A/Set B/Loop toggle/Clear). Per-stem row: label + Mute + Solo + volume slider. Color-coded stems (vocals=purple, drums=orange, bass=blue, guitar=red, piano=green, other=gray). Waveform recomputes on mute/solo/volume changes. Playback speed presets.
- **`waveform_widget.py`** — Custom QPainter widget: mirrored waveform bars, playback cursor, loop region shading, loop marker lines. Click/drag-to-seek. Catppuccin Mocha colors.
- **`library_panel.py`** — Song list with search/filter, selection, remove (with confirmation), metadata edit (double-click / context menu)
- **`import_dialog.py`** — File browser or YouTube URL, metadata fields, model variant (4/6 stem). If ONNX is missing, downloads model with progress; on any failure before successful import completion, removes the new library row. Large file (100 MiB+) confirmation. **Retry import** after errors. Workers cancelled and rolled back on dialog reject.
- **`preferences_dialog.py`** — Data directory, output device, default import model, export format and MP3 bitrate

### `downloader.py` — YouTube Audio Download
- URL validation for youtube.com, youtu.be, music.youtube.com
- Metadata extraction (title, artist) via yt-dlp without downloading
- Audio download as MP3 (bestaudio + FFmpegExtractAudio, 320kbps)
- Prefers bundled ffmpeg via imageio-ffmpeg when available; falls back to ffmpeg on PATH
- Progress callback support for UI integration
- **`styles.py`** — Dark and light themes (Catppuccin-inspired), good contrast

---

## Implementation Phases

### Phase 1 — MVP (complete)
- [x] Project setup (GitHub repo, deps, structure)
- [x] Model manager (download ONNX models on first run)
- [x] Stem separation engine (ONNX Runtime + DirectML)
- [x] Multi-track player with mute/solo
- [x] Song library (import, list, remove)
- [x] Export stems as WAV
- [x] UI: main window, player controls, library panel, import dialog
- [x] Dark theme styling
- [x] Integration test suite (13 tests including hardware playback)
- [x] Overlap-add Hann windowing (click-free segment boundaries)

### Phase 2 — Polish (complete)
- [x] MP3 export support (lameenc, 320kbps)
- [x] Separation progress bar (in import dialog)
- [x] Keyboard shortcuts (Space=play/pause, S=stop, arrows=seek, 1-6=mute stems, A/B/L=loop)
- [x] Per-stem volume sliders (0-200%)
- [x] Window state persistence (QSettings)
- [x] Audio post-processing (Wiener filter + soft gating)
- [x] Error handling & edge cases (JSON recovery, thread cleanup, stream safety)
- [x] CI pipeline (GitHub Actions, fast tests on every push)

### Phase 3 — Advanced (complete)
- [x] YouTube URL import (yt-dlp)
- [x] Tempo change (time-stretch) in player via librosa
- [x] Waveform visualization
- [x] A-B loop repeat
- [x] Error handling: model download in import, friendly messages, library rollback, playback / no-device warnings (#73)
- [x] PyInstaller packaging + GitHub Release workflow (#56)

### Post-1.0 Backlog
Tickets ship as incremental 1.x releases (semver).
- [ ] Session persistence (#55)
- [ ] Metronome with BPM entry (#57)
- [ ] Count-in before playback/loop start (#78)
- [ ] Record audio track (#79)
- [ ] Key transposition / tempo manipulation (#42)
- [ ] Animated startup logo (#76)
- [ ] MSIX packaging for Microsoft Store (#74)
- [ ] Real-time streaming stem separation (#13)
- [ ] Experimental DSP (#28)

---

## Reference Projects

| Project | Relevance |
|---|---|
| **OpenVINO Audacity Plugin** | Proves HTDemucs ONNX works; source code reference |
| **MISST** | Desktop stem separation + player GUI — closest to our vision |
| **Ultimate Vocal Remover (UVR5)** | Full-featured separator — complex but good reference |
| **Demucs Web** | Runs Demucs in browser via ONNX Runtime Web + WebGPU |
| **audio-separator** | Python CLI/library for stem separation |
| **deanturpin/stems** | C++ HTDemucs via ONNX Runtime — high-performance reference |

---

## Performance Expectations

| Scenario | Expected Speed (4-min song) |
|---|---|
| GPU (DirectML, RTX 4070 Ti) | ~30-60 seconds |
| CPU fallback | ~5-15 minutes |

## Audio Quality Notes

- Bass and drums separate cleanly (primary use case supported)
- Guitar separation (6-stem) is good but not perfect — some bleed is normal
- For practice/busking purposes, quality is more than sufficient
