# AGENTS.md -- stemma

This file provides context and instructions for AI coding agents working on this project.
It follows the AGENTS.md open standard (https://agents.md).

## Project Overview

**stemma** is a Windows desktop music player with AI stem separation.
Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other),
mute/solo any stem, and play along with your instrument.

Local-only. No cloud, no subscriptions, no command line needed by the end user.

For full technical specs, module descriptions, and the phased roadmap, see `PROJECT.md`.

## Tech Stack

- Python 3.14
- PySide6 (Qt 6) for GUI
- ONNX Runtime + DirectML for GPU-accelerated inference (no PyTorch)
- HTDemucs v4 ONNX models (4-stem and 6-stem)
- sounddevice + soundfile for audio playback
- numpy for audio buffer processing
- librosa for STFT/iSTFT (pre/post-processing outside ONNX model)
- yt-dlp + ffmpeg for YouTube audio download
- PyInstaller for packaging

## Project Structure

```
stemma/
  main.py              # Entry point
  requirements.txt
  pyproject.toml       # pytest config (markers, pythonpath)
  PROJECT.md           # Detailed spec and roadmap (reference doc)
  AGENTS.md            # This file (AI context, living document)
  CHANGELOG.md         # Append-only session history
  stemma.spec              # PyInstaller one-file build spec
  requirements-dev.txt     # Dev/build dependencies (pyinstaller)
  .github/workflows/ci.yml      # CI: fast tests on push
  .github/workflows/release.yml # Build .exe + GitHub Release on v* tags
  src/
    app.py             # QApplication setup
    app_settings.py    # Typed QSettings reads (audio, export, import defaults)
    data_paths.py      # Per-user data dir + legacy repo data/ migration
    paths.py           # app_root(): frozen-build-aware root dir (sys._MEIPASS)
    version.py         # __version__ string
    import_messages.py # User-facing import / download / separation error text
    separator.py       # ONNX stem separation engine
    model_manager.py   # Download/cache ONNX models
    player.py          # Multi-track audio player
    library.py         # Song library (JSON-based)
    exporter.py        # Export stems as WAV/MP3
    downloader.py      # YouTube audio download (yt-dlp)
    post_processing.py # Wiener filter + soft gate
    waveform.py        # Waveform peak computation (numpy)
    ui/
      main_window.py
      player_controls.py
      waveform_widget.py # Waveform display (QPainter)
      library_panel.py
      import_dialog.py
      preferences_dialog.py  # Edit > Preferences
      styles.py        # Dark / light themes
  tests/               # pytest test suite (~286 fast + 5 slow + 1 hardware)
    conftest.py        # Shared fixtures
    test_separator.py
    test_model_manager.py
    test_player.py
    test_library.py
    test_downloader.py
    test_exporter.py
    test_post_processing.py
    test_waveform.py
    test_waveform_widget.py
    test_import_dialog.py
    test_import_messages.py
    test_drag_drop.py
    test_library_panel.py
    test_metadata_edit.py
    test_speed_control.py
    test_data_paths.py
    test_app_settings.py
    test_theme.py
    test_integration.py
    test_session_persistence.py
  data/                # Legacy dev-only folder; packaged app uses OS user dir
    models/            # (when using repo data/) Cached ONNX models
    songs/{song-id}/   # Separated stems per song
```

Runtime library and models default to the per-user folder (e.g. `%LOCALAPPDATA%\\stemma` on Windows), with a one-time copy from `./data` when that folder is new. See `src/data_paths.py`.

## Rules

1. No emojis in code, documentation, or commit messages.
2. One CLI command at a time during agentic interactions. No chained commands.
3. Conventional commits: `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, etc.
4. Commit after each major change. Keep git log clean and PRs manageable.
5. PEP 8 for Python code. Clear docstrings for classes and complex functions.
6. Keep dependencies lean. No PyTorch. Use ONNX Runtime for inference.
7. All changes go through PRs. No direct pushes to `main`.
8. TDD when possible: write tests first, then implement to make them pass.
9. No placeholder or mock code in `main` branch. Use `NotImplementedError` with clear TODOs.
10. Work deliberately. Plan each feature, implement carefully, test thoroughly.
11. **Always keep the GitHub Kanban board up to date.** Move issues to In Progress, update subtasks, and close them when PRs merge.
12. **Dual-Agent Workflow:** We use a builder/reviewer model. The "Builder" agent implements the feature and opens a PR. Do not merge your own PRs. The user will pass the PR to a "Reviewer" agent to audit the code, catch bugs, suggest improvements, and approve it.
13. **No inline imports.** All imports belong at the top of the file. Do not place `import` statements inside functions or methods.

## Current Status

Last updated: 2026-03-25

### Phase 1 (MVP) -- Complete
All core functionality implemented and tested:
- ONNX stem separation with overlap-add Hann windowing
- Multi-track player with mute/solo/volume
- Song library with JSON persistence and corruption recovery
- Full Qt UI with dark theme
- Integration test suite (including hardware playback)

### Phase 2 (Polish) -- Complete
- MP3 export (lameenc, 320kbps)
- Keyboard shortcuts (Space, S, arrows, 1-6, A/B/L for loop)
- Per-stem volume sliders
- Window state persistence
- Wiener filter + soft gate post-processing
- Robustness fixes (thread cleanup, stream safety, JSON recovery)
- CI pipeline (GitHub Actions)

### Phase 3 (Advanced) -- Complete
- [x] A-B loop repeat (#44, PR #48)
- [x] YouTube URL import (#41, PR #49)
- [x] Waveform visualization (#43, PR #58)
- [x] Drag-and-drop import (#51, PR #60)
- [x] Bundled ffmpeg via imageio-ffmpeg (PR #60)
- [x] Error handling and model download UX (#73, PR #82)

### v1.0 Release -- Shipped
- [x] Library search/filter (#54, PR #63)
- [x] Song metadata editing (#52, PR #64)
- [x] Playback speed control (#53, PR #67)
- [x] Light theme + theme switch (#70)
- [x] App icon and branding (#61)
- [x] User data directory, preferences, single-instance lock (#72, PR #81)
- [x] PyInstaller packaging + GitHub Release (#56, PR #83)

### Post-1.0 Backlog
Tickets ship as incremental 1.x releases (semver: minor for features, patch for fixes).
- [x] Session persistence (#55, PR #85)
- [x] Metronome with BPM entry (#57, PR #86)
- [x] Count-in before playback/loop start (#78)
- [ ] Record audio track (#79)
- [ ] Tempo/key detection and beat-synced metronome (#42)
- [ ] Animated startup logo (#76)
- [ ] MSIX packaging (#74)
- [ ] Experimental DSP (#28)
- [ ] Real-time streaming (#13)
## Test Suite

```
pytest                                    # ~354 fast tests (~10s)
pytest -m slow                            # 5 ONNX inference tests (~20s, needs model)
pytest -m hardware                        # 1 audible playback test (~30s, needs speakers)
set STEMMA_TEST_SONG=path/to/song.mp3     # Required for slow/hardware tests
```

## Session History

See `CHANGELOG.md` for a detailed log of what was done in each session.
