# Changelog

All notable development sessions are documented here in reverse chronological order.

---

## 2026-03-22 -- v1.0: Library Search, Metadata Editing, and Playback Speed

### Done
- Library search/filter (#54, PR #63)
  - Search bar with clear button in library panel
  - Real-time filter hides non-matching songs as you type
  - Remove button disabled when selection is hidden by filter
  - 9 new tests
- Song metadata editing (#52, PR #64)
  - Double-click a song to edit title/artist
  - Right-click context menu with Edit option
  - Right-click selects the item before opening menu (fixes wrong-item bug)
  - Empty fields fall back to original values
  - 10 new tests
- Pitch-preserving playback speed control (#53, PR #67)
  - librosa.effects.time_stretch pre-stretches stems in background QThread
  - Speed presets: 0.5x, 0.75x, 0.85x, 1.0x, 1.25x, 1.5x, 2.0x
  - Keyboard shortcuts: [ (slower) and ] (faster)
  - Frame index and loop points proportionally adjusted on speed change
  - Peak normalization after stretch fixes volume reduction from phase vocoder
  - Worker detach pattern prevents stale signal callbacks
  - 13 new tests

### Metrics
- 229 fast tests, 5 slow ONNX tests, 1 hardware playback test (235 total)
- v1.0 roadmap: 2 tickets remaining (#61 branding, #56 packaging)

---

## 2026-03-22 -- v1.0: Drag-and-Drop Import + ffmpeg Bundling

### Done
- Drag-and-drop import (#51, PR #60)
  - Drop .mp3/.wav/.flac files onto the main window to trigger import
  - ImportDialog gains `file_path` parameter for pre-fill
  - Multiple dropped files open sequential import dialogs
  - dragMoveEvent override for correct drop cursor on Windows/Qt6
  - Inline import of ImportDialog in main_window.py fixed (rule #13)
  - 17 new tests (3 prefill + 6 drag-enter + 2 drag-move + 6 drop)
- Bundled ffmpeg via imageio-ffmpeg (PR #60)
  - YouTube import now works without ffmpeg on PATH
  - imageio-ffmpeg ships a static ffmpeg binary as package data
  - _get_ffmpeg_exe() prefers bundled binary, falls back to PATH
  - ffmpeg_location passed to yt-dlp options
  - 4 new tests (3 check_ffmpeg fallback + 1 ffmpeg_location in opts)

### Metrics
- 196 fast tests, 5 slow ONNX tests, 1 hardware playback test (202 total)
- Phase 3: 4 of 5 tickets complete; v1.0 roadmap: 5 tickets remaining

---

## 2026-03-22 -- Phase 3: Waveform Visualization

### Done
- Waveform visualization (#43, PR #58)
  - Custom QPainter widget replacing seek slider
  - Mirrored waveform bars, playback cursor, A-B loop region shading and markers
  - Click/drag-to-seek
  - Waveform recomputes on mute/solo/volume changes
  - Vectorized peak computation via numpy (compute_peaks + np.maximum.reduceat)
  - Three review passes; all findings addressed:
    - Zero-width guards, cursor bounds clamping, redundant repaint removal
    - Unused import cleanup, stale waveform cleared on empty stems
  - 19 new tests (9 peak computation + 10 widget)
- Inline import rule added to project conventions

### Metrics
- 173 fast tests, 5 slow ONNX tests, 1 hardware playback test (178 total)
- Phase 3: 3 of 5 tickets complete

---

## 2026-03-21 -- Phase 3: A-B Loop and YouTube Import

### Done
- A-B loop repeat (#44, PR #48)
  - Loop point set/clear API on player with callback wrap-around
  - UI controls: Set A, Set B, Loop toggle, Clear buttons
  - Keyboard shortcuts: A/B/L keys
  - Zero-width loop guard to prevent callback deadlock
  - 9 new tests
- YouTube URL import (#41, PR #49)
  - yt-dlp downloader module: URL validation, metadata extraction, audio download
  - Import dialog reworked: URL input with Fetch button, dual-mode import
  - Background workers for metadata fetch and audio download
  - ffmpeg availability check with user-facing error
  - Worker lifecycle hardening: signal rename to avoid QThread.finished shadowing,
    safe disconnect helper, proper wait/cleanup on dialog close
  - Output file existence verification after download
  - noplaylist option to prevent accidental playlist downloads
  - Error handling in local import path (disk full, missing file, permissions)
  - 33 new tests (26 downloader + 7 import dialog)

### Metrics
- 154 fast tests, 5 slow ONNX tests, 1 hardware playback test (159 total)
- Phase 3: 2 of 5 tickets complete

---

## 2026-03-21 -- Phase 2 Completion and Hardening

### Done
- Audio post-processing: Wiener filtering and soft gating (#19, PR #38)
  - Chunked processing to bound memory usage (~50 MB vs 3.5 GB)
  - 17 tests including multi-chunk boundary verification
- Robustness fixes (PR #40)
  - Corrupted JSON recovery (app no longer crashes on malformed library.json)
  - Thread cleanup on close (export worker, separator worker)
  - Player stream leak fix on audio device errors
  - Encapsulation fix: public API for mute toggle from keyboard shortcuts
- MP3 export via lameenc (#23, PR #36)
  - 320kbps CBR, no ffmpeg dependency
  - Export dialog offers WAV and MP3
  - 5 new tests
- Window state persistence (#27, PR #35)
  - QSettings saves/restores geometry across sessions
- Keyboard shortcuts (#24, PR #34)
  - Space=play/pause, S=stop, Left/Right=seek, 1-6=toggle stem mute
- Per-stem volume sliders (#25, PR #31)
  - 0-200% gain per stem, applied in audio callback
  - Volume reflected in export
- Code quality cleanup (PR #30)
  - Public `has_stems` and `muted_stems` properties on player
  - Stem name constants imported from separator module
- CI pipeline (PR #34)
  - GitHub Actions running fast tests on every push
  - pythonpath fix for module resolution
- Reviewed and merged external PRs:
  - #32: ONNX tensor mapping fix, button padding, remove song button
  - #37: Export worker QThread, peak normalization
  - #39: Integration test fix for soft gating side effect

### Metrics
- 112 fast tests, 5 slow ONNX tests, 1 hardware playback test
- ~2000 lines of source, ~1700 lines of tests
- Phase 1 and Phase 2 fully complete

---

## 2026-03-20 -- Phase 1 MVP

### Done
- Stem separation engine (#2, PR #6)
  - ONNX Runtime + DirectML, STFT/iSTFT via librosa
  - Segmented inference with progress reporting
- Multi-track audio player (#3, PR #14)
  - sounddevice callback API, mute/solo, clipping protection
- Song library (#4, PRs #15, #16)
  - JSON persistence, atomic writes, audio file copying on import
- Stem exporter (#7, PR #20)
  - Single stem or custom mix export as WAV
- UI modules (#8-#12, PR #21)
  - Main window with splitter layout
  - Player controls with transport and stem rows
  - Library panel with song list
  - Import dialog with separation progress
  - Dark theme (Catppuccin Mocha-inspired)
- Integration test suite (#18, PR #17)
  - Library round-trips, player with real WAVs, ONNX inference
  - Hardware playback test using real music
- Overlap-add Hann windowing (#22, PR #29)
  - 50% overlap with Hann window eliminates boundary clicks
- Model manager
  - Download from HuggingFace, DirectML -> CPU fallback

---

## 2026-03-20 -- Session 1: Project Setup

### Done
- Researched stem separation technology landscape (HTDemucs v4, ONNX Runtime, existing projects)
- Chose tech stack: Python 3.14, PySide6, ONNX Runtime + DirectML, sounddevice
- Created `PROJECT.md` with full spec, module descriptions, and phased roadmap
- Created GitHub repo (https://github.com/cyanidesayonara/stemma), pushed initial scaffold
- Set up project folder structure with all source file skeletons
- Created virtual environment and installed dependencies
- Started `separator.py` (SeparatorWorker QThread skeleton) and `model_manager.py` (ModelDownloader)
- Adopted `AGENTS.md` open standard for cross-tool AI handover
- Set up GitHub Projects kanban board with Phase 1 issues
