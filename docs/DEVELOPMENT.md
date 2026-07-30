# Development guide

This is the canonical local-development and validation guide for stemma.
Architecture is documented in `../PROJECT.md`; release operations are
documented in `store-release-pipeline.md`.

## Supported environments

- Windows 10 or 11
- Python 3.14 for normal local development
- Python 3.12 for CI and release-lock generation
- No PyTorch. ONNX inference uses the `onnxruntime-directml` distribution.

Use a virtual environment. Keep runtime and development dependencies
separate so packaging does not accidentally depend on test-only packages.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Run the application from the repository root:

```powershell
python main.py
```

Runtime data defaults to `%LOCALAPPDATA%\stemma`. A repository-local `data/`
directory is legacy development data and is migrated only when the user data
directory is new.

## Release dependency lock

`requirements-release.in` is the human-maintained release input.
`requirements-release.txt` is the fully pinned, hash-checked Windows/Python
3.12 lock used by the release workflow.

Regenerate it on Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv-release
.\.venv-release\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "pip-tools>=7.6.0"
python -m piptools compile --allow-unsafe --generate-hashes --index-url=https://pypi.org/simple --output-file=requirements-release.txt --strip-extras requirements-release.in
python -m pip install --require-hashes -r requirements-release.txt
```

Review dependency and hash changes before committing the regenerated lock.
Do not hand-edit transitive pins.

## Testing

Qt tests should use the offscreen platform for deterministic headless runs:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
```

Run the complete fast suite used by PR CI:

```powershell
python -m pytest -m "not slow and not hardware"
```

Run focused tests while developing:

```powershell
python -m pytest tests/test_library.py
python -m pytest tests/test_diagnostics.py
```

Slow-model tests are scheduled/manual in CI rather than run on every PR.
To run them locally, obtain the integrity-checked model cache, inspect the
available providers, then run the slow non-hardware slice:

```powershell
python -m scripts.cache_slow_test_models
python main.py --diagnostics
python -m pytest -m "slow and not hardware"
```

Hardware playback requires speakers or an audio device and a real song:

```powershell
$env:STEMMA_TEST_SONG = "C:\path\to\song.mp3"
python -m pytest -m hardware
```

Do not convert slow-model or hardware tests into unmarked PR tests.

## Lint

Ruff checks syntax, undefined names, and import correctness:

```powershell
python -m ruff check .
```

The maintained selection is in `pyproject.toml`. Fix real findings rather
than adding broad suppressions.

## Diagnostics

Source diagnostics report the branch build version, ONNX Runtime version,
and available execution providers without opening the GUI:

```powershell
python main.py --diagnostics
```

The release workflow runs the frozen executable with
`--diagnostics-file`. A valid DirectML release must report
`DmlExecutionProvider`; MDX progress separately tells the user whether a
particular session selected DirectML GPU or CPU fallback.

## Packaging

Build the one-folder application:

```powershell
python -m PyInstaller --clean --noconfirm stemma.spec
```

Smoke-test frozen diagnostics:

```powershell
.\dist\stemma\stemma.exe --diagnostics-file dist\frozen-diagnostics.txt
```

Build the MSIX after the one-folder build exists:

```powershell
.\scripts\build_msix.ps1 -Output "dist/stemma.msix"
```

Generated `build/`, `dist/`, model, and package artifacts are not committed.

## Version and release workflow

`src/version.py` and `msix/AppxManifest.xml` describe the current branch
build. A release tag still synchronizes both before packaging:

```powershell
.\scripts\sync_release_version.ps1 -Tag v2.6.0
```

Running that command does not publish a release. The GitHub Release is the
authority for what shipped. Follow `store-release-pipeline.md` for tag,
artifact, checksum, and Store submission details.

## Contribution workflow

1. Start from an issue and add it to
   [GitHub Project 2](https://github.com/users/cyanidesayonara/projects/2).
2. Move the item to In Progress while a branch or pull request is active.
3. Work on a feature branch; never push implementation directly to `main`.
4. Use tests first for behavior and regression changes.
5. Run focused tests, Ruff, and the complete fast suite before handoff.
6. Use a conventional commit message.
7. Open a pull request. The builder does not merge their own pull request;
   a separate reviewer audits and approves it.
8. Close issues and move Project items to Done only after the change is
   merged or the issue's acceptance criteria are otherwise satisfied.

Binding coding-agent rules remain in `../AGENTS.md`.
