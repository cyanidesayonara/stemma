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

Settings (theme, session, window, preferences) live in the registry under
`HKCU\Software\stemma\stemma`. To run from source without touching the
settings of a stemma you also use day to day, point them at a file:

```powershell
$env:STEMMA_SETTINGS_FILE = "$PWD\build\dev-settings.ini"
python main.py
```

The test suite and `scripts/render_ui_review.py` do this automatically.
Packaged and Store builds ignore the variable.

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

## Keeping YouTube import working

YouTube changes how it serves video every few weeks, and `yt-dlp` follows
with a release. A stemma release freezes three pieces into the package:

- `yt-dlp`, pinned in `requirements-release.txt` (`requirements.txt` only
  sets a floor).
- `yt-dlp-ejs`, the challenge-solver scripts. Each `yt-dlp` release names
  the exact version it expects in its `default` extra, and checks the
  scripts' hashes: any other version logs "Hash mismatch on challenge
  solver" and is skipped. `requirements.txt` lists plain `yt-dlp` without
  that extra, so pip-compile does not see the pin, and the version has to
  be chosen by hand (see step 1). `tests/test_dependency_locks.py` keeps
  the `requirements.txt` range on the locked minor version.
- QuickJS-ng, the JavaScript runtime that runs those scripts. Its version
  and SHA-256 are in `scripts/fetch_js_runtime.py`. The release workflow
  checks the frozen diagnostics line
  `YouTube JS runtime: quickjs-ng X (yt-dlp-ejs Y)`.

An installed stemma cannot update these on its own, so an old release
eventually stops importing from YouTube. The lock was created on
2026-07-31 and bumped once, on 2026-11-19, with no routine between.

### Decisions

- **Update by releasing stemma, never by updating `yt-dlp` inside the app.**
  - The app does not download or replace `yt-dlp`.
  - It does not enable `yt-dlp`'s remote solver components
    (`ejs:github`, `ejs:npm`).
  - Both would run code that is not in the signed, certified package, and
    stemma stays local-only. `scripts/smoke_youtube_download.py` already
    fails on a "remote component" warning.
- **The vehicle is a patch release** (`vX.Y.Z`) that changes only the lock
  and anything the new versions need. Store installs update on their own.
  ZIP users get it from the GitHub Release.
- **The user-facing message stays.** A format failure already reads
  "YouTube changed how this video is served. Try again later, or update
  stemma." (`MSG_YT_FORMAT` in `src/import_messages.py`). Under this plan,
  that advice is correct.

### Routine

1. **Every release** refreshes the three pieces in its release prep PR.
   Substeps 1 to 3 are automated: run the "Refresh yt-dlp lock (manual)"
   workflow from `main` (see "Automated checks"), or
   `python scripts/refresh_ytdlp_lock.py` on Windows with Python 3.12 and
   pip-tools. By hand:
   1. Find the newest `yt-dlp` on PyPI. Its `default` extra names the
      `yt-dlp-ejs` version it expects: the `Requires-Dist: yt-dlp-ejs==...`
      line on its PyPI page or in the installed METADATA.
   2. If that is a new `yt-dlp-ejs` minor version, move the
      `requirements.txt` range to it first. Otherwise the compile cannot
      reach it.
   3. Compile the lock with both versions given exactly:

      ```powershell
      python -m piptools compile --allow-unsafe --generate-hashes --index-url=https://pypi.org/simple --output-file=requirements-release.txt --strip-extras --upgrade-package "yt-dlp==YYYY.M.D" --upgrade-package "yt-dlp-ejs==X.Y.Z" requirements-release.in
      ```

   4. If the `yt-dlp-ejs` release notes ask for a newer QuickJS-ng, or
      `yt-dlp` drops support for the current one, update
      `scripts/fetch_js_runtime.py`: the version, URL, and SHA-256 from the
      QuickJS-ng release page.
   5. Run `python main.py --diagnostics`, and run the YouTube smoke test
      with a manual `workflow_dispatch` of CI.
2. **Every week** the scheduled CI run (Mondays) runs `youtube-smoke-test`
   against the lock on `main`.
   - That is what users have installed only while `main`'s lock matches
     the latest release. Once a lock refresh merges, ship it as a patch
     release promptly, so the two do not drift apart.
   - A red smoke test while the fast `test` job is green means the locked
     `yt-dlp` no longer works.
   - GitHub emails scheduled-run failures to whoever last changed the
     workflow's schedule, or re-enabled the workflow. Make sure that is the
     maintainer, or watch the Actions tab.
   - In a public repository, GitHub disables scheduled workflows after 60
     days with no repository activity, without warning. A quiet stretch is
     exactly when `yt-dlp` drift builds up. After a pause in development,
     check that the CI workflow is still enabled.
3. **When the smoke test goes red:**
   1. Re-run it once first. "Sign in to confirm you're not a bot", 403s,
      and timeouts can come from GitHub's runner IP addresses rather than
      from a real change at YouTube.
   2. Check whether the newest `yt-dlp` fixes it. In a throwaway venv,
      install the release lock, then `pip install -U "yt-dlp[default]"`
      (which brings the matching `yt-dlp-ejs`). Run
      `python scripts/fetch_js_runtime.py`, then
      `python scripts/smoke_youtube_download.py`. Without the fetched
      QuickJS-ng, the smoke test fails with "QuickJS challenge solver was
      not invoked".
   3. If it passes, open a `fix:` PR with the step 1 lock refresh.
      - Note the YouTube fix in `CHANGELOG.md` and the Store What's new.
      - Ship a patch release through the normal release and Store
        pipeline.
   4. If the newest `yt-dlp` fails too, YouTube is ahead of `yt-dlp`.
      - Wait for an upstream release, and re-run the smoke test with
        `workflow_dispatch`.
      - Nothing in stemma needs to change until then.

### Automated checks

These checks run after the locked smoke test, even when it fails, and only
warn. They never fail the job (#260).

- **Canary.** `youtube-smoke-test` ends by upgrading `yt-dlp` to the newest
  release, installing the `yt-dlp-ejs` version it names
  (`scripts/check_ytdlp_lock_age.py --print-expected-ejs`), and running the
  smoke test again. The rest of the release lock stays, as in a lock
  refresh. A `yt-dlp canary` notice or warning in the run summary, naming
  the version it tested, answers step 3.2.
- **Lock age.** `scripts/check_ytdlp_lock_age.py --open-issue` compares
  the PyPI upload dates of the locked and the newest `yt-dlp`. When the lock
  is more than 60 days behind, it warns and opens a "yt-dlp release lock is
  behind" issue. If one is open, it retitles it and comments only when a
  newer release has come out since. The issue stays visible even if the
  schedule pauses later. Run it locally without `--open-issue`.
- **Lock-ejs test.** `tests/test_dependency_locks.py` checks that the
  locked `yt-dlp-ejs` is the one the locked `yt-dlp`'s `default` extra
  names. It runs wherever the installed `yt-dlp` is the locked one, as in
  CI, and skips elsewhere.

- **Lock refresh.** The "Refresh yt-dlp lock (manual)" workflow
  (`.github/workflows/refresh-ytdlp-lock.yml`) runs from `main` on
  `windows-latest` with Python 3.12, because Dependabot runs on Linux and
  this lock must be compiled on Windows. Leave the version input empty for
  the newest yt-dlp.
  - `scripts/refresh_ytdlp_lock.py` reads the `yt-dlp-ejs` version from
    the release's `default` extra on PyPI. It moves the `requirements.txt`
    range if the minor version changes, and compiles the lock with both
    versions pinned.
  - The workflow checks that the new lock installs and that
    `tests/test_dependency_locks.py` passes. It then pushes
    `chore/refresh-yt-dlp-<version>`, opens a pull request, and starts CI
    on the branch with `workflow_dispatch`, since a push made with
    `GITHUB_TOKEN` starts no workflow itself.
  - If the repository does not allow Actions to open pull requests, the
    run summary links to the compare page instead.
  - QuickJS-ng is not touched. The pull request lists what is left to
    check before merging.

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

## Verification tiers

Each kind of check happens where it is cheapest, so that human testing is
spent only on what nothing else can judge.

1. **Automated tests, every PR.** Ruff and the fast suite run in PR CI.
   Slow-model tests run weekly and on demand; hardware tests run locally.
2. **Rendered UI review, every PR that changes `src/ui/`.** Layout
   regressions only show in a styled, rendered window, which the fast suite
   never builds. Render one and inspect it before opening the PR:

   ```powershell
   python scripts/render_ui_review.py --out build/ui-review/branch
   ```

   This renders the real `MainWindow` with a generated song in three states
   (empty, loaded, active practice with an A-B loop and a muted stem), at
   900x600, 1366x768, and 1920x1080, in both themes. Output is PNGs plus
   `index.html` under `build/ui-review/`. Use `--stems 2` or `--stems 6`
   for the two- or six-stem layout, and render `main` into a second
   directory to compare. The renderer uses a private data directory and
   never touches your library. Say in the PR what was checked.
3. **Human acceptance pass, once per release.** Some things only a person
   can judge: how speed and pitch renders sound, playback and metronome
   sync, recording through a real interface, separation quality and GPU
   use on real songs, and whether the app feels right in use. These checks
   are collected in the release issue and done in one sitting before
   tagging, not per PR. A PR that needs one lists it under Testing.

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
particular session selected DirectML GPU or CPU fallback. Diagnostics probe
DirectML for every cached model, so ONNX Runtime prints an initialization
error ("The parameter is incorrect") for each HTDemucs model before the
report: that is expected, and those models report `CPUExecutionProvider`.

## Packaging

Fetch the SHA-256 verified QuickJS-ng executable and build the one-folder
application:

```powershell
python scripts/fetch_js_runtime.py
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
.\scripts\sync_release_version.ps1 -Tag v3.1.0
```

Running that command does not publish a release. The GitHub Release is the
authority for what shipped. Follow `store-release-pipeline.md` for tag,
artifact, checksum, and Store submission details. A successful release also
redeploys the website from the new tag's Store listing and screenshots; see
`website.md`.

Before tagging, refresh the YouTube pieces of the release lock (step 1 of
"Keeping YouTube import working").

## Contribution workflow

1. Start from an issue and add it to
   [GitHub Project 2](https://github.com/users/cyanidesayonara/projects/2).
2. Move the item to In Progress while a branch or pull request is active.
3. Work on a feature branch; never push implementation directly to `main`.
4. Use tests first for behavior and regression changes.
5. Run focused tests, Ruff, and the complete fast suite before handoff.
6. Use a conventional commit message.
7. Open a pull request, then have it reviewed independently: the reviewer
   in `.claude/agents/pr-reviewer.md` runs in a fresh context and its own
   worktree, and its report is posted on the PR unedited. Address the
   findings in follow-up commits. Squash-merge once the findings are
   addressed and CI passes. Hand to the maintainer only what needs a human:
   a listening or hands-on check, or a product decision.
8. Close issues and move Project items to Done only after the change is
   merged or the issue's acceptance criteria are otherwise satisfied.

Binding coding-agent rules remain in `../AGENTS.md`.
