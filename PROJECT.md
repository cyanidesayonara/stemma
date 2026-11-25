# stemma

A Windows desktop music player with AI stem separation. Import a song, separate it into stems, mute/solo individual instruments, play along.

**Local-only. No cloud, no subscriptions, no command line needed.** Distributed through the Microsoft Store and GitHub Releases.

Latest stable release: **v3.2.0**. Release notes live in `CHANGELOG.md`;
future scope lives in `docs/ROADMAP.md`.

---

## Concept

1. Import a song (MP3/WAV/FLAC, or a YouTube URL)
2. AI separates it into stems (vocals, drums, bass, guitar, piano, other)
3. Mute/solo any stem, loop, slow down, or transpose, and play along
4. Record takes over the stems; export stems, mixes, or loop regions

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
| **Language** | Python 3.12 for CI and releases (3.14 works for local development) | Rich audio ecosystem |
| **GUI** | PySide6 (Qt 6) | Modern look, powerful widgets, LGPL |
| **Inference** | ONNX Runtime DirectML | No PyTorch; shared DirectML-first session setup with CPU fallback |
| **Stem Models** | HTDemucs v4 and MDX-Net ONNX | Downloaded and checksum-verified on first use |
| **Audio Playback** | `sounddevice` + `soundfile` | Callback-based multi-track mixing via NumPy |
| **Audio Processing** | `numpy` | Efficient buffer manipulation |
| **Export** | `soundfile` (WAV), `lameenc` (MP3) | Individual stems or custom mix |
| **YouTube Import** | `yt-dlp` + `ffmpeg` | Download audio from YouTube URLs |
| **Packaging** | PyInstaller | One-folder (COLLECT) build shipped as `stemma.zip` + `stemma.msix` |
| **Time/pitch processing** | `src/stretch.py` (numpy phase vocoder + `soxr`) | Real-time on the mix, like DAWs and practice apps: changes apply within a frame, nothing is pre-rendered; librosa's vocoder plus identity phase locking, so the mix keeps its level (#227); speed and pitch in one pass |

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
│   ├── stretch.py          # real-time speed and pitch (phase vocoder)
│   ├── library.py          # persistent song index and recovery
│   ├── play_queue.py       # play queue and setlist playback order
│   ├── practice_state.py   # loop, speed, and pitch per song
│   ├── exporter.py         # stem and mix export
│   ├── batch_export.py     # export of several songs
│   ├── beat_detector.py    # beat, tempo, key, and chord analysis
│   ├── song_sections.py    # experimental song-section detection
│   └── ui/                 # Qt presentation and interaction
├── tests/                  # fast, slow-model, and hardware-marked tests
├── scripts/                # assets, screenshots, version sync, model cache,
│                           # MSIX build, Store listing and Partner Center
├── store/                  # Store listing source of truth (listing.yaml)
├── assets/                 # icons, brand art, startup audio, Store screenshots
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
- Callers pass `use_gpu=False` to go straight to the CPU; HTDemucs does,
  because its export never compiles on DirectML
- Supplies a stable user-facing provider label

### `beat_detector.py` — Musical Analysis
- BPM detection and beat/downbeat tracking via the `beat_this` ONNX model (chunked inference)
- Key detection using Krumhansl-Schmuckler profiles
- Chord detection (major/minor) with Viterbi smoothing
- On whole-song detection only, experimental song sections
  (`song_sections.py`) from the same chroma the chords use
- `transpose_key()` helper for shifting a detected key by semitones (used by pitch shift)
- `DETECTION_VERSION` gates saved beats, chords, and sections: a song saved
  by an older detector is analysed again when opened

### `song_sections.py` — Experimental Song Sections
- numpy and librosa only. Per-bar chroma (shared with chord detection),
  MFCC timbre, and loudness are averaged over four bars around each bar
  (so a section cycling through its progression reads as one steady
  thing) and scaled by single-bar spread; Foote checkerboard novelty on
  the bars' Gaussian self-similarity matrix proposes changes; each must
  also move the bars on its two sides apart far more than chance, judged
  against how closely bars match the bars they repeat, so random
  bar-to-bar variation (white noise) gives no sections. Changes are kept
  at least four bars apart and above an absolute floor, so a song that
  never changes gets no sections rather than arbitrary ones
- Boundaries are detected downbeats (every fourth beat when there are
  none), so a section loop starts and ends on bar lines
- Labels are letters (A, B, A, B, C); sections whose mean features are
  alike share one. Verse or chorus names are not guessed
- Sections are `(start_s, end_s, label)`, kept on the player and saved
  with the song's detection (`detection/<id>/sections`)

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
- Speed and pitch are applied live in the callback: `StreamingStretcher` paths (src/stretch.py) stretch the mix as it plays, primed off the audio thread when built or seeked. Positions, loop points, beats, and chords stay in song frames at any speed; clicks are mixed after stretching. Recording takes get their own unpitched path unless they follow the pitch
- A-B loop: `set_loop_a()`, `set_loop_b()`, `set_looping()`, `clear_loop()`. While looping is on and the region is valid (`B > A`), **Stop** seeks to loop A (not track start); **seek** clamps into `[A, B)` (outside snaps to A)
- Metronome and count-in settings (BPM, volume, beats, loop-repeat count-in)
- Recording: full-duplex `sd.Stream` captures input at the playback frame position; position-indexed buffer auto-handles loop wraps; saves as `recording_takeN.wav` with optional latency offset via `np.roll`; `recording_saved` signal; a take that is digital silence (peak below -80 dBFS, as when Windows denies microphone access) is kept and followed by `recording_silent`, which the window shows as a warning pointing to the microphone privacy settings
- Tracks playback position for UI sync
- PortAudioError on open/start and mid-playback stream loss (`finished_callback` or inactive stream): stream cleanup, pause, and **`playback_failed`** signal (user-facing message for UI dialogs)

### `library.py` — Song Library
- Versioned JSON index (schema 2): songs `{id, title, artist,
  original_path, stems_path, model_used, date_added, tags,
  practice_status}` and ordered setlists `{id, name, song_ids}`. A v1 list
  is migrated on load
- Restricts destructive operations to this library's `songs/` root and
  rejects unsafe persisted paths
- Rolls back memory and disk state when persistence fails
- Uses atomic writes, preserves a corrupt index, and rebuilds from safe
  on-disk song directories
- Treats a model-specific atomic completion marker plus the expected stem
  set as authoritative for new separation jobs; legacy complete songs remain
  compatible

### `play_queue.py` — Play Queue
- Up Next queue, active setlist, shuffle, and history, kept apart from the
  library view: browsing a setlist or tag does not change what plays next

### `practice_state.py` — Per-Song Practice State
- Loop region, speed, and pitch per song, stored in settings
  (`practice/<id>`) rather than the library index, because they change on
  every adjustment
- A song loaded by autoplay gets its loop points back without looping, so
  a setlist can move on

### `exporter.py` — Stem Export
- Export individual stems or custom mix (the mix you hear: current mute/solo, stem volumes, and take nudge, at the original speed and pitch; the master volume is the listening level and is not applied) as WAV or MP3
- MP3 encoding via `lameenc` (320kbps CBR, no ffmpeg needed)
- Peak normalization instead of hard clipping
- Background export via `ExportWorker` QThread
- `batch_export.py` exports several songs to a folder, one mix per song or
  separate stem files, without overwriting existing files

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
- **`player_controls.py`** — Composition facade and coordinator over the
  practice cockpit components. It owns the layout: a scrolling column of
  waveform, song readout, practice cards, and stem mixer, with the transport
  anchored below it. Waveform-peak and detection work use generations so
  stale futures cannot overwrite current state.
  - **`waveform_panel.py`** / **`waveform_stack_widget.py`** — Stacked
    per-stem waveform lanes in stem colors with a shared playhead, A-B loop
    shading, click/drag-to-seek, and mute/solo shown as lane dimming. The
    stack prefers 280px, yields to a 120px floor in short windows, and in
    tall windows grows into spare height up to about 110px per lane (520px
    at most, never capped below 280px).
  - **`chord_lane.py`** — Fixed-height lane under the stem lanes in the
    same card: an overview row of the whole song's chords on the stem
    lanes' time axis, and a read-ahead row showing the current bar and the
    next seven, scrolling with the playhead. Labels follow transposition.
    The player's `chord_sequence` is the one source of truth for the lane,
    the chord badge, and the saved session, and always covers the whole
    song: a loop re-detection analyses only the loop, so `PlayerControls`
    splices its chords into the song's (`splice_chords`) instead of
    replacing them. The overview layer is cached; frames come from the
    20 Hz position updates.
  - **`section_ruler.py`** — Experimental, fixed-height ruler above the
    stem lanes in the same card, on their time axis: one block per song
    section, the current one in the accent, a section the A-B loop covers
    exactly outlined in the loop colour. Click seeks to a section's start;
    double-click (or Shift+L for the section playing) calls
    `PlayerControls.loop_section`, which sets A and B on the section's bar
    lines, turns looping on, and jumps in only from outside the section.
    Loop re-detections leave the sections alone. The blocks are cached;
    a frame repaints only when the current section or playhead pixel
    changes. View > Song Sections (Experimental) hides it.
  - **`song_info_bar.py`** — Key, live chord, and tempo readout strip.
  - **`practice_rack.py`** — Loop and Trainer, Speed and Pitch, and
    Metronome and Count-in cards, which wrap to two rows in narrow windows.
  - **`stem_mixer.py`** — Stem and recording-take rows (mute, solo, volume,
    take nudge and delete).
  - **`transport_bar.py`** — Play, stop, record, time, and master volume.
- **`library_panel.py`** — Song list with search (titles, artists, and
  tags), a setlist and tag filter, setlist management and drag-reorder,
  practice badges, and metadata edit (double-click / context menu).
  Selecting a row loads the song, so presses that may become a drag, right
  clicks, and Ctrl/Shift clicks never change the current row: marks for
  batch actions are a separate set, dropped when a song leaves the view.
  Removal (single or batch) asks once.
- **`play_queue_widget.py`** — The Play Queue view in the sidebar: Up Next
  and the active setlist's remaining songs.
- **`batch_export_dialog.py`** — Stems, mix or separate files, and format
  for exporting several songs.
- **`import_dialog.py`** — File browser or YouTube URL, metadata fields, and
  2/4/6-stem model selection. Several files share an artist and model,
  with checks and questions asked once and the model downloaded before
  the library changes. Missing models download with progress; failures
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
- Uses bundled QuickJS-ng (`qjs.exe`) and `yt-dlp-ejs` when present (falling back to supported runtimes on PATH), and forwards `yt-dlp` warnings and errors to the `stemma` logger
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
