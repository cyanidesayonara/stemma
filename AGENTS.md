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
- PyInstaller for packaging

## Project Structure

```
stemma/
  main.py              # Entry point
  requirements.txt
  PROJECT.md           # Detailed spec and roadmap (reference doc)
  AGENTS.md            # This file (AI context, living document)
  CHANGELOG.md         # Append-only session history
  src/
    app.py             # QApplication setup
    separator.py       # ONNX stem separation engine
    model_manager.py   # Download/cache ONNX models
    player.py          # Multi-track audio player
    library.py         # Song library (JSON-based)
    exporter.py        # Export stems as WAV/MP3
    ui/
      main_window.py
      player_controls.py
      library_panel.py
      import_dialog.py
      styles.py        # Dark theme
  tests/               # pytest test suite
    conftest.py        # Shared fixtures
    test_separator.py
    test_model_manager.py
    test_player.py
    test_library.py
    test_exporter.py
  data/                # Runtime data (gitignored)
    models/            # Cached ONNX model files
    songs/{song-id}/   # Separated stems per song
```

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

## Current Status

Last updated: 2026-03-20

### Completed
- Repository created and pushed to GitHub (https://github.com/cyanidesayonara/stemma)
- Project scaffolding: all source files created (empty skeletons)
- Virtual environment set up, dependencies installed
- `PROJECT.md` finalized with full spec

### In Progress
- Quality infrastructure: PR workflow, testing setup, code cleanup
- Stem separation engine (`src/separator.py`): skeleton `SeparatorWorker` QThread exists
- Model manager (`src/model_manager.py`): `ModelManager` and `ModelDownloader` classes exist

### Next Steps
- Set up GitHub Projects kanban board with Phase 1 issues
- Complete the ONNX inference pipeline in `separator.py` (STFT via librosa, chunked inference, iSTFT)
- Implement multi-track audio player (`src/player.py`)
- Build the UI components
- See `PROJECT.md` "Implementation Phases" for the full Phase 1 checklist

## Session History

See `CHANGELOG.md` for a detailed log of what was done in each session.
