# stemma

A Windows desktop music player with AI stem separation. Import a song, separate it into stems, mute/solo individual instruments, play along.

**Personal-use tool. No cloud, no subscriptions, no command line needed.**

Latest stable release: **v2.5.0**. The current source tree targets
**v2.6.0**, which is not released. Release notes live in `CHANGELOG.md`;
future scope lives in `docs/ROADMAP.md`.

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
| **Acceleration** | DirectML when supported | MDX two-stem selects DirectML or reports CPU fallback; HTDemucs four/six-stem remains CPU-only |
| **Stems** | 2-stem, 4-stem, and 6-stem models | Selected per import |
| **Distribution** | Microsoft Store (primary); portable `stemma.zip` + `stemma.msix` on GitHub Releases | Models download on first run |
| **Python** | 3.14 local, 3.12 CI/release | See `docs/DEVELOPMENT.md` |

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| **Language** | Python 3.14 | Rich audio ecosystem |
| **GUI** | PySide6 (Qt 6) | Modern look, powerful widgets, LGPL |
| **Inference** | ONNX Runtime DirectML | No PyTorch; shared DirectML-first session setup with CPU fallback |
| **Stem Models** | HTDemucs v4 and MDX-Net ONNX | Downloaded and checksum-verified on first use |
| **Audio Playback** | `sounddevice` + `soundfile` | Callback-based multi-track mixing via NumPy |
| **Audio Processing** | `numpy` | Efficient buffer manipulation |
| **Export** | `soundfile` (WAV), `lameenc` (MP3) | Individual stems or custom mix |
| **YouTube Import** | `yt-dlp` + `ffmpeg` | Download audio from YouTube URLs |
| **Packaging** | PyInstaller | One-folder (COLLECT) build shipped as `stemma.zip` + `stemma.msix` |
| **Time/pitch processing** | `librosa` | Offline render from original buffers avoids compounding transformations |

### Why HTDemucs v4?

- **Developer**: Meta AI Research — MIT license (fully open, free)
- **4-stem**: Vocals, Drums, Bass, Other
- **6-stem variant**: Adds Guitar + Piano (exactly what we need)
- **Runtime constraint**: the current exports execute on CPU; a
  DirectML-compatible re-export is tracked in
  [issue #125](https://github.com/cyanidesayonara/stemma/issues/125)

### Why ONNX Runtime over PyTorch?

- ONNX Runtime avoids shipping the PyTorch training/runtime stack
- DirectML gives MDX a Windows GPU path without a CUDA toolkit and preserves
  a CPU fallback on unsupported systems
- Proven approach: Intel's OpenVINO Audacity plugin and others use the same pattern
- Models are converted once from PyTorch → ONNX format and hosted on HuggingFace

---

## Project Structure

```
stemma/
├── main.py                 # diagnostics dispatch or Qt startup
├── src/
│   ├── separator.py        # HTDemucs four/six-stem engine
│   ├── mdx_separator.py    # MDX two-stem engine
│   ├── onnx_session.py     # shared DirectML-first session policy
│   ├── separation_queue.py # serialized background jobs
│   ├── player.py           # multi-track audio engine
│   ├── library.py          # persistent song index and recovery
│   ├── beat_detector.py    # beat, tempo, key, and chord analysis
│   └── ui/                 # Qt presentation and interaction
├── tests/                  # fast, slow-model, and hardware-marked tests
├── scripts/                # assets, version sync, model cache, MSIX build
├── docs/                   # development, roadmap, Store, policy, history
├── stemma.spec             # PyInstaller one-folder build
└── msix/AppxManifest.xml   # Desktop Bridge package identity
```

---

## Module Specifications

### `separator.py` — Stem Separation Engine
- Loads HTDemucs ONNX through the shared session factory
- Uses CPU inference because the current four/six-stem exports do not compile
  for DirectML
- Handles STFT/iSTFT pre/post-processing in NumPy (stripped from ONNX model)
- Runs in background `QThread`, emits progress signals
- Supports both `htdemucs` (4-stem) and `htdemucs_6s` (6-stem)

### `mdx_separator.py` — Two-Stem Separation Engine
- Produces vocals and backing (`other`) stems
- Requests DirectML through the shared session factory, falls back to CPU
  on session-init failure, and reports the selected provider in progress
- Handles STFT packing, windowing, context trim, and reconstruction without
  PyTorch

### `onnx_session.py` — Execution-Provider Policy
- Keeps the heavy ONNX Runtime import deferred until inference/diagnostics
- Configures DirectML for sequential execution with memory patterns disabled
- Retries with CPU when DirectML session creation fails
- Supplies a stable user-facing provider label

### `beat_detector.py` — Musical Analysis
- BPM detection and beat/downbeat tracking via the `beat_this` ONNX model (chunked inference)
- Key detection using Krumhansl-Schmuckler profiles
- Chord detection (major/minor) with Viterbi smoothing
- `transpose_key()` helper for shifting a detected key by semitones (used by pitch shift)

### `model_manager.py` — Model Download & Cache
- Checks if ONNX model files exist under the app data directory (`models/`)
- `ModelDownloader` (`QThread`): downloads immutable model artifacts on first run
- Verifies reviewed SHA-256 values before atomically publishing cached files
- Signals: `progress`, **`download_complete(str)`** (model path; not named `finished`, to avoid shadowing `QThread.finished`), `error`
- Manages HTDemucs, MDX, and beat_this model files

### `import_messages.py` — Import Error Text
- `format_import_error(message)` maps raw exceptions to short, readable strings (disk full, permission, network, SSL, HTTP/404, timeout, cancel, truncation)

### `player.py` — Multi-Track Audio Player
- Keeps a synchronous stem-loading API while exposing separate read/apply
  steps for background UI loading
- `sounddevice.OutputStream` callback: reads buffers per stem, applies gain, sums to output; optional metronome click mix; optional count-in pre-roll before advancing `_current_frame`
- API: `play()`, `pause()`, `stop()`, `seek()`, `set_mute()`, `set_solo()`, `set_volume()`
- Per-stem volume control (0.0-2.0)
- A-B loop: `set_loop_a()`, `set_loop_b()`, `set_looping()`, `clear_loop()`. While looping is on and the region is valid (`B > A`), **Stop** seeks to loop A (not track start); **seek** clamps into `[A, B)` (outside snaps to A)
- Metronome and count-in settings (BPM, volume, beats, loop-repeat count-in)
- Recording: full-duplex `sd.Stream` captures input at the playback frame position; position-indexed buffer auto-handles loop wraps; saves as `recording_takeN.wav` with optional latency offset via `np.roll`; `recording_saved` signal
- Tracks playback position for UI sync
- PortAudioError on open/start: stream cleanup and **`playback_failed`** signal (user-facing message for UI dialogs)

### `library.py` — Song Library
- JSON song index: `{id, title, artist, stems_path, model_used, date_added}`
- Restricts destructive operations to this library's `songs/` root and
  rejects unsafe persisted paths
- Rolls back memory and disk state when persistence fails
- Uses atomic writes, preserves a corrupt index, and rebuilds from safe
  on-disk song directories
- Treats a model-specific atomic completion marker plus the expected stem
  set as authoritative for new separation jobs; legacy complete songs remain
  compatible

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
- **`main_window.py`** — Coordinates the library, player, separation queue,
  session restore, navigation, and generation-safe asynchronous stem reads.
  Only the current `(song, generation)` may update player/UI state or surface
  an error.
- **`player_controls.py`** — Transport, waveform, mixer, loops, rendered
  speed/pitch, metronome/count-in, recording, and musical-analysis controls.
  Waveform-peak work also uses generations so stale futures cannot overwrite
  current state.
- **`waveform_widget.py`** — Custom QPainter widget: mirrored waveform bars, playback cursor, loop region shading, loop marker lines. Click/drag-to-seek. Catppuccin Mocha colors.
- **`library_panel.py`** — Song list with search/filter, selection, remove (with confirmation), metadata edit (double-click / context menu)
- **`import_dialog.py`** — File browser or YouTube URL, metadata fields, and
  2/4/6-stem model selection. Missing models download with progress; failures
  roll back the library row. Demucs imports perform preflight memory
  confirmation before persistence. Workers cancel and drain on rejection.
- **`preferences_dialog.py`** — Data directory, output/input devices,
  manual recording timing offset, default import model, export format, and
  MP3 bitrate

### `downloader.py` — YouTube Audio Download
- URL validation for youtube.com, youtu.be, music.youtube.com
- Metadata extraction (title, artist) via yt-dlp without downloading
- Audio download as MP3 (bestaudio + FFmpegExtractAudio, 320kbps)
- Prefers bundled ffmpeg via imageio-ffmpeg when available; falls back to ffmpeg on PATH
- Progress callback support for UI integration
- **`styles.py`** — Shared dark/light theme tokens and stylesheets

---

## Cross-Cutting Design Constraints

### Threading and lifecycle

- The PortAudio callback must not perform disk I/O, allocate large buffers,
  or invoke Qt.
- QThreads and futures are disconnected, cancelled where supported, retained
  while running, and drained before Qt teardown.
- Song loading, detection, and waveform peaks use monotonic generations.
  Results are applied only when their generation and song identity still
  match current state.
- Separation jobs run serially so concurrent imports do not multiply model
  memory usage.

### Persistence and recovery

- User data and model caches live in a writable per-user directory.
- JSON and completion-marker writes use temporary files plus atomic replace.
- Corrupt metadata is preserved for diagnosis; recovery only trusts paths
  below the library root.
- QSettings stores preferences and per-song session state. Loading applies
  that state only after the matching asynchronous stem load completes.

### Release integrity

- Source and manifest versions describe the current branch build. Tag builds
  still run `scripts/sync_release_version.ps1` before packaging.
- Runtime/build dependencies are locked with hashes for Windows/Python 3.12.
- Model downloads are pinned and checksum-verified.
- Release builds run Ruff, fast tests, PyInstaller, frozen diagnostics, MSIX
  packaging, and checksum generation.
- A GitHub Release is the authority for what shipped; code present on `main`
  is not by itself a release claim.

Development and release commands are in `docs/DEVELOPMENT.md` and
`docs/store-release-pipeline.md`.

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

## Current Limitations

- Separation time depends on track length, model, hardware, drivers, and
  execution provider; no fixed timing is promised.
- MDX two-stem can use DirectML. HTDemucs four/six-stem is CPU-only with the
  current model exports.
- Stem isolation is model-dependent and can contain bleed or artifacts; the
  post-processing stage reduces some artifacts but does not guarantee a
  quality level.
- Recording alignment depends on the user's audio hardware and configured
  manual offset; the application does not promise automatic latency
  calibration.
