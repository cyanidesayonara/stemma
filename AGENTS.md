# AGENTS.md -- stemma

This file provides context and instructions for AI coding agents working on this project.
It follows the AGENTS.md open standard (https://agents.md).

## Project Overview

**stemma** is a Windows desktop music player with AI stem separation.
Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other),
mute/solo any stem, and play along with your instrument.

Local-only. No cloud, no subscriptions, no command line needed by the end user.

Latest stable release: **v3.2.1**. See "Current Status" below.

Canonical documents:

- `PROJECT.md`: durable architecture and technical reference
- `docs/DEVELOPMENT.md`: setup, validation, diagnostics, packaging, and
  contribution workflow
- `docs/ROADMAP.md`: concise roadmap backed by live GitHub issues
- `CHANGELOG.md`: shipped release notes only
- Pull request descriptions: the record of each change (what, why, how,
  testing). `docs/DEVELOPMENT_LOG.md` is an archive and is no longer
  updated.

## Tech Stack

- Python 3.14 (local dev; GitHub Actions CI uses 3.12 per `.github/workflows/ci.yml`)
- PySide6 (Qt 6) for GUI
- ONNX Runtime DirectML for inference; no PyTorch
- MDX-Net two-stem and HTDemucs v4 four/six-stem separation select DirectML
  when available and report CPU fallback
- `src/htdemucs_dml.py` rewrites HTDemucs DirectML-incompatible graph ops
  into a cached `.dml.onnx` sidecar next to the original model
  ([issue #125](https://github.com/cyanidesayonara/stemma/issues/125))
- sounddevice + soundfile for audio playback
- numpy for audio buffer processing
- librosa for STFT/iSTFT (pre/post-processing outside ONNX model)
- yt-dlp + ffmpeg for YouTube audio download
- PyInstaller for packaging

## Architecture at a Glance

- `main.py` and `src/app.py`: diagnostics or Qt application startup
- `src/separator.py` and `src/mdx_separator.py`: HTDemucs and MDX
  separation engines
- `src/htdemucs_dml.py`: DirectML graph patcher and `.dml.onnx` sidecar
  cache for HTDemucs
- `src/onnx_session.py`: shared DirectML-first session construction with
  CPU fallback
- `src/separation_queue.py`: serialized background separation jobs
- `src/player.py`: real-time multi-track playback, loops, metronome,
  and recording
- `src/stretch.py`: real-time speed/pitch (phase vocoder and `soxr`)
  applied to the mix as it plays
- `src/library.py` and `src/data_paths.py`: persistent song metadata and
  per-user storage
- `src/play_queue.py`: play queue and setlist playback order
- `src/practice_state.py`: loop, speed, and pitch remembered per song, in
  settings
- `src/exporter.py` and `src/batch_export.py`: stem and mix export, for
  one song or several
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
12. **Builder/reviewer workflow:** The builder implements the change and opens a PR, then runs the independent reviewer defined in `.claude/agents/pr-reviewer.md` in a fresh context and its own worktree. The builder posts the reviewer's report on the PR unedited, addresses the findings in follow-up commits, and replies with what changed. Merge (squash) once the review's findings are addressed and CI passes (a docs-only PR runs no CI; see "Going easy on GitHub"); stop and ask the maintainer only when a change needs a human check a render or test cannot give (how something sounds or feels in use), or a product decision.
13. **Imports at module scope by default.** Prefer top-of-file imports. Deferred imports inside a function or method are acceptable when there is a concrete reason (for example: faster cold start, avoiding a heavy or rarely used dependency until needed, or optional/platform-specific modules). Add a short comment at the import site when the reason is not obvious.
14. **Look at UI changes before handing them off.** Any change under `src/ui/` is rendered with `python scripts/render_ui_review.py` and the images inspected at every size and theme before the PR opens. Leave the human acceptance pass for what a render cannot show (see "Verification tiers" in `docs/DEVELOPMENT.md`).

## Current Status

Set with the v3.2.1 release prep on 2026-11-26:

- Latest stable: **v3.2.1** (library, setlists, and play queue, [#132](https://github.com/cyanidesayonara/stemma/issues/132); MSIX virtualized library path fix, [#279](https://github.com/cyanidesayonara/stemma/issues/279))
- Next: DirectML GPU separation for 4/6-stem HTDemucs ([#125](https://github.com/cyanidesayonara/stemma/issues/125))
  implemented for the next release; see `docs/ROADMAP.md`

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

## Going easy on GitHub

Test on this machine before pushing; push a branch once it is believed done,
not after every commit, and open it as a draft until then. Mark it
ready before merging: a draft's skipped run is not a pass. Batch related work:
at most 4 pull requests a day, 2 open at once. No pictures in branches (except
the images the app, site, or Store listing ship, regenerated at most once a
release), no polling of checks, and nothing that runs on every push to main
without Santtu's word.

A pull request that changes only root `*.md` files or `docs/` runs no CI. Run
`python -m pytest tests/test_store_listing.py` before pushing it: it checks
the versions and generated text in those files.
