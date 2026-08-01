# AGENTS.md -- stemma

This file provides context and instructions for AI coding agents working on this project.
It follows the AGENTS.md open standard (https://agents.md).

## Project Overview

**stemma** is a Windows desktop music player with AI stem separation.
Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other),
mute/solo any stem, and play along with your instrument.

Local-only. No cloud, no subscriptions, no command line needed by the end user.

Latest stable release: **v2.5.0**. The current `main` line and this branch
target **v2.6.0**, which is not released.

Canonical documents:

- `PROJECT.md`: durable architecture and technical reference
- `docs/DEVELOPMENT.md`: setup, validation, diagnostics, packaging, and
  contribution workflow
- `docs/ROADMAP.md`: concise roadmap backed by live GitHub issues
- `CHANGELOG.md`: shipped release notes only
- `docs/DEVELOPMENT_LOG.md`: historical session and implementation notes

## Tech Stack

- Python 3.14 (local dev; GitHub Actions CI uses 3.12 per `.github/workflows/ci.yml`)
- PySide6 (Qt 6) for GUI
- ONNX Runtime DirectML for inference; no PyTorch
- MDX-Net two-stem separation selects DirectML when available and reports
  CPU fallback
- HTDemucs v4 four/six-stem separation remains CPU-only; DirectML model
  re-export research is tracked in
  [issue #125](https://github.com/cyanidesayonara/stemma/issues/125)
- sounddevice + soundfile for audio playback
- numpy for audio buffer processing
- librosa for STFT/iSTFT (pre/post-processing outside ONNX model)
- yt-dlp + ffmpeg for YouTube audio download
- PyInstaller for packaging

## Architecture at a Glance

- `main.py` and `src/app.py`: diagnostics or Qt application startup
- `src/separator.py` and `src/mdx_separator.py`: HTDemucs and MDX
  separation engines
- `src/onnx_session.py`: shared DirectML-first session construction with
  CPU fallback
- `src/separation_queue.py`: serialized background separation jobs
- `src/player.py`: real-time multi-track playback, loops, metronome,
  recording, and rendered speed/pitch changes
- `src/library.py` and `src/data_paths.py`: persistent song metadata and
  per-user storage
- `src/beat_detector.py`: tempo, beat/downbeat, key, and chord analysis
- `src/ui/`: Qt windows, dialogs, controls, themes, and visualizations
- `tests/`: fast tests plus explicitly marked slow-model and hardware tests

Runtime data defaults to `%LOCALAPPDATA%\stemma`, with one-time migration
from the legacy repository `data/` directory. See `PROJECT.md` for design
rationale and subsystem boundaries.

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
13. **Imports at module scope by default.** Prefer top-of-file imports. Deferred imports inside a function or method are acceptable when there is a concrete reason (for example: faster cold start, avoiding a heavy or rarely used dependency until needed, or optional/platform-specific modules). Add a short comment at the import site when the reason is not obvious.

## Current Status

Verified against GitHub Releases on 2026-08-01:

- Latest stable: **v2.6.0**
- Current source on `main` targets **v3.0** (unreleased)
- v3.0 focus: practice cockpit visual recomposition ([#131](https://github.com/cyanidesayonara/stemma/issues/131))

Use `docs/ROADMAP.md` for future scope and `CHANGELOG.md` for shipped
releases. Do not infer release status from code already present on a branch.

## Canonical Validation

```powershell
python -m ruff check .
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -m "not slow and not hardware"
python main.py --diagnostics
```

See `docs/DEVELOPMENT.md` for focused, slow-model, hardware, lock-generation,
packaging, and release commands.
