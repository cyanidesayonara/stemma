# Store Listing as Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Microsoft Store listing copy machine-checked from `store/listing.yaml`, generate `docs/store-listing.md`, and fail PR/release validation when What's new, features, or screenshots are wrong for the version under test — without calling Partner Center.

**Architecture:** A small `src/store_listing.py` library owns load/validate/render. Two thin CLI scripts (`build_store_listing.py`, `validate_store_release.py`) call it. Pytest covers the library; `release.yml` runs the validator after tag version sync. No Partner Center network calls.

**Tech Stack:** Python 3.12 (CI/release), PyYAML for listing YAML, stdlib `struct` for PNG IHDR dimensions, pytest, existing GitHub Actions workflows.

**Spec:** `docs/superpowers/specs/2026-07-31-store-listing-as-code-design.md`  
**Issue:** [#134](https://github.com/cyanidesayonara/stemma/issues/134)

## Global Constraints

- No Partner Center API calls, draft submission, or publish in this slice.
- Do not invent Store marketing claims; migrate from `docs/store-listing.md` and add a truthful `whats_new["2.6.0"]` for already-merged main capabilities only.
- Do not name HTDemucs/MDX-Net in Store description text; do not claim GPU for 4/6-stem separation.
- Screenshot gate: at least **3** PNGs under `assets/store_listing/screenshots/`, each width >= 1366 and height >= 768.
- Feature gate: `1 <= len(features) <= 20`; each feature length <= `MAX_FEATURE_LENGTH` (200).
- Short description max length: `MAX_SHORT_DESCRIPTION_LENGTH` (1000).
- No new heavy dependencies beyond PyYAML; do not add Pillow solely for PNG size checks.
- Conventional commits; no emojis; TDD (failing test before production code).
- Work on `feat/store-listing-as-code` in the `stemma-store-listing` worktree.

## File map

| Path | Responsibility |
|---|---|
| `src/store_listing.py` | Load YAML, constants, validate, render markdown + skeleton JSON, PNG IHDR reader, version helpers |
| `store/listing.yaml` | Hand-edited source of truth |
| `store/product-update.skeleton.json` | Generated Partner Center-shaped skeleton (not submitted) |
| `scripts/build_store_listing.py` | CLI: write outputs or `--check` drift |
| `scripts/validate_store_release.py` | CLI: validate for a release version |
| `tests/test_store_listing.py` | Fast unit tests |
| `docs/store-listing.md` | Generated human copy |
| `docs/store-release-pipeline.md` | Operator docs update |
| `requirements-dev.txt` / `requirements-release.in` / `requirements-release.txt` | Add PyYAML + regenerate hashed lock |
| `.github/workflows/release.yml` | Run validator after version sync |

---

### Task 1: Store listing library — load, constants, PNG size

**Files:**
- Create: `src/store_listing.py`
- Create: `tests/test_store_listing.py`
- Modify: `requirements-dev.txt` (add `PyYAML>=6.0`)
- Modify: `requirements-release.in` (add `PyYAML>=6.0`)
- Modify: `requirements-release.txt` (regenerate with hashes)

**Interfaces:**
- Produces:
  - `MAX_FEATURES = 20`
  - `MAX_FEATURE_LENGTH = 200`
  - `MAX_SHORT_DESCRIPTION_LENGTH = 1000`
  - `MIN_SCREENSHOTS = 3`
  - `MIN_SCREENSHOT_WIDTH = 1366`
  - `MIN_SCREENSHOT_HEIGHT = 768`
  - `class ListingData` (dataclass or TypedDict-like) with fields: `listing_version: str`, `short_description: str`, `description: str`, `features: list[str]`, `search_terms: list[str]`, `whats_new: dict[str, str]`
  - `load_listing(path: str | Path) -> ListingData`
  - `png_size(path: str | Path) -> tuple[int, int]`  # (width, height) via IHDR
  - `read_app_version(version_py: str | Path) -> str`

- [ ] **Step 1: Add PyYAML to dev requirements and install**

Append to `requirements-dev.txt`:

```text
PyYAML>=6.0
```

Append to `requirements-release.in`:

```text
PyYAML>=6.0
```

Install for local tests:

```powershell
.\.venv\Scripts\python.exe -m pip install "PyYAML>=6.0"
```

(If `.venv` is missing in the worktree, create it with `py -3.12 -m venv .venv` and install `requirements.txt` + `requirements-dev.txt` first.)

- [ ] **Step 2: Write failing tests for load + PNG + version**

Create `tests/test_store_listing.py`:

```python
"""Tests for Store listing-as-code helpers."""

from pathlib import Path

import pytest

from src.store_listing import (
    load_listing,
    png_size,
    read_app_version,
)


def test_load_listing_reads_required_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "listing.yaml"
    yaml_path.write_text(
        """
listing_version: "2.6.0"
short_description: Short text
description: |
  Longer description.
features:
  - Feature one
search_terms:
  - stem separation
whats_new:
  "2.5.0": |
    What's new in version 2.5.0
  "2.6.0": |
    What's new in version 2.6.0
""".lstrip(),
        encoding="utf-8",
    )

    data = load_listing(yaml_path)

    assert data.listing_version == "2.6.0"
    assert data.short_description == "Short text"
    assert "Longer description." in data.description
    assert data.features == ["Feature one"]
    assert data.search_terms == ["stem separation"]
    assert "2.6.0" in data.whats_new


def test_png_size_reads_ihdr(tmp_path: Path) -> None:
    # Minimal 8x4 RGBA PNG
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 8, 4, 8, 2, 0, 0, 0)
    raw = b"".join(
        b"\x00" + bytes([0, 0, 255, 255] * 8) for _ in range(4)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path = tmp_path / "tiny.png"
    path.write_bytes(png)

    assert png_size(path) == (8, 4)


def test_read_app_version(tmp_path: Path) -> None:
    path = tmp_path / "version.py"
    path.write_text('__version__ = "2.6.0"\n', encoding="utf-8")
    assert read_app_version(path) == "2.6.0"
```

- [ ] **Step 3: Run tests to verify they fail**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest tests\test_store_listing.py -q --tb=short
```

Expected: FAIL with `ModuleNotFoundError` or import error for `src.store_listing`.

- [ ] **Step 4: Implement minimal `src/store_listing.py`**

```python
"""Store listing load/validate/render helpers (issue #134 slice 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import yaml

MAX_FEATURES = 20
MAX_FEATURE_LENGTH = 200
MAX_SHORT_DESCRIPTION_LENGTH = 1000
MIN_SCREENSHOTS = 3
MIN_SCREENSHOT_WIDTH = 1366
MIN_SCREENSHOT_HEIGHT = 768

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LISTING_YAML = REPO_ROOT / "store" / "listing.yaml"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "store-listing.md"
DEFAULT_SKELETON = REPO_ROOT / "store" / "product-update.skeleton.json"
DEFAULT_SCREENSHOTS = REPO_ROOT / "assets" / "store_listing" / "screenshots"
DEFAULT_VERSION_PY = REPO_ROOT / "src" / "version.py"


@dataclass(frozen=True)
class ListingData:
    listing_version: str
    short_description: str
    description: str
    features: list[str]
    search_terms: list[str]
    whats_new: dict[str, str]


def load_listing(path: str | Path) -> ListingData:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("listing YAML must be a mapping")
    whats_new = raw.get("whats_new") or {}
    if not isinstance(whats_new, dict):
        raise ValueError("whats_new must be a mapping")
    return ListingData(
        listing_version=str(raw["listing_version"]).strip(),
        short_description=str(raw["short_description"]).strip(),
        description=str(raw["description"]).strip(),
        features=[str(item).strip() for item in raw.get("features") or []],
        search_terms=[
            str(item).strip() for item in raw.get("search_terms") or []
        ],
        whats_new={
            str(key): str(value).strip() for key, value in whats_new.items()
        },
    )


def png_size(path: str | Path) -> tuple[int, int]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    length = struct.unpack(">I", data[8:12])[0]
    tag = data[12:16]
    if tag != b"IHDR" or length < 8:
        raise ValueError(f"PNG missing IHDR: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def read_app_version(version_py: str | Path) -> str:
    text = Path(version_py).read_text(encoding="utf-8")
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"could not parse __version__ from {version_py}")
    return match.group(1)
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_store_listing.py -q --tb=short
```

Expected: PASS (3 tests).

- [ ] **Step 6: Regenerate release lock with hashes**

```powershell
.\.venv\Scripts\python.exe -m pip install pip-tools
.\.venv\Scripts\python.exe -m piptools compile --allow-unsafe --generate-hashes --resolver=backtracking --strip-extras --index-url=https://pypi.org/simple --output-file=requirements-release.txt requirements-release.in
```

Confirm `PyYAML` appears in `requirements-release.txt`.

- [ ] **Step 7: Commit**

```powershell
git add src/store_listing.py tests/test_store_listing.py requirements-dev.txt requirements-release.in requirements-release.txt
git commit -m "feat: add store listing load helpers"
```

---

### Task 2: Validation gates

**Files:**
- Modify: `src/store_listing.py`
- Modify: `tests/test_store_listing.py`
- Create: `scripts/validate_store_release.py`

**Interfaces:**
- Consumes: `ListingData`, `load_listing`, `png_size`, constants from Task 1
- Produces:
  - `class ValidationError(Exception)` with `.errors: list[str]`
  - `validate_listing(data: ListingData, *, version: str, screenshots_dir: Path) -> None` (raises `ValidationError`)
  - CLI `scripts/validate_store_release.py [--version X.Y.Z]` exit 0/1

- [ ] **Step 1: Write failing validation tests**

Append to `tests/test_store_listing.py`:

```python
from src.store_listing import ValidationError, validate_listing, ListingData


def _data(**overrides) -> ListingData:
    base = dict(
        listing_version="2.6.0",
        short_description="Short",
        description="Desc",
        features=["Feature one"],
        search_terms=["stem"],
        whats_new={"2.6.0": "What's new in version 2.6.0\n\nNotes."},
    )
    base.update(overrides)
    return ListingData(**base)


def test_validate_listing_requires_whats_new(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    with pytest.raises(ValidationError) as exc:
        validate_listing(
            _data(whats_new={"2.5.0": "old"}),
            version="2.6.0",
            screenshots_dir=shots,
        )
    assert any("whats_new" in err for err in exc.value.errors)


def test_validate_listing_rejects_too_many_features(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    with pytest.raises(ValidationError):
        validate_listing(
            _data(features=[f"f{i}" for i in range(21)]),
            version="2.6.0",
            screenshots_dir=shots,
        )


def test_validate_listing_rejects_undersized_screenshots(
    tmp_path: Path,
) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    # reuse tiny PNG writer from earlier test or write three small files
    for name in ("a.png", "b.png", "c.png"):
        (shots / name).write_bytes(
            # caller may import a helper; simplest: copy bytes from
            # test_png_size_reads_ihdr construction
            _minimal_png(8, 4)
        )
    with pytest.raises(ValidationError) as exc:
        validate_listing(
            _data(),
            version="2.6.0",
            screenshots_dir=shots,
        )
    assert any("1366" in err or "screenshot" in err.lower()
               for err in exc.value.errors)
```

Add a shared `_minimal_png(width, height)` helper in the test module (extract from Task 1 PNG test).

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_store_listing.py -q --tb=short
```

Expected: FAIL on missing `validate_listing` / `ValidationError`.

- [ ] **Step 3: Implement validation**

Add to `src/store_listing.py`:

```python
class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def validate_listing(
    data: ListingData,
    *,
    version: str,
    screenshots_dir: str | Path,
) -> None:
    errors: list[str] = []
    notes = data.whats_new.get(version, "").strip()
    if not notes:
        errors.append(f"whats_new missing entry for version {version}")
    if not (1 <= len(data.features) <= MAX_FEATURES):
        errors.append(
            f"features count {len(data.features)} not in 1..{MAX_FEATURES}"
        )
    for feature in data.features:
        if not feature:
            errors.append("empty feature entry")
        elif len(feature) > MAX_FEATURE_LENGTH:
            errors.append(
                f"feature exceeds {MAX_FEATURE_LENGTH} chars: {feature!r}"
            )
    if not data.short_description:
        errors.append("short_description is empty")
    elif len(data.short_description) > MAX_SHORT_DESCRIPTION_LENGTH:
        errors.append("short_description exceeds max length")
    if not data.description.strip():
        errors.append("description is empty")

    shot_dir = Path(screenshots_dir)
    shots = sorted(shot_dir.glob("*.png")) if shot_dir.is_dir() else []
    if len(shots) < MIN_SCREENSHOTS:
        errors.append(
            f"need >= {MIN_SCREENSHOTS} screenshots, found {len(shots)}"
        )
    for shot in shots:
        try:
            width, height = png_size(shot)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if width < MIN_SCREENSHOT_WIDTH or height < MIN_SCREENSHOT_HEIGHT:
            errors.append(
                f"{shot.name} is {width}x{height}; "
                f"need >= {MIN_SCREENSHOT_WIDTH}x{MIN_SCREENSHOT_HEIGHT}"
            )
    if errors:
        raise ValidationError(errors)
```

Create `scripts/validate_store_release.py`:

```python
"""Validate Store listing metadata for a release version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import (
    DEFAULT_LISTING_YAML,
    DEFAULT_SCREENSHOTS,
    DEFAULT_VERSION_PY,
    ValidationError,
    load_listing,
    read_app_version,
    validate_listing,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="X.Y.Z release version")
    parser.add_argument("--listing", type=Path, default=DEFAULT_LISTING_YAML)
    parser.add_argument(
        "--screenshots", type=Path, default=DEFAULT_SCREENSHOTS,
    )
    args = parser.parse_args(argv)
    version = args.version or read_app_version(DEFAULT_VERSION_PY)
    try:
        data = load_listing(args.listing)
        validate_listing(
            data, version=version, screenshots_dir=args.screenshots,
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Store listing OK for version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_store_listing.py -q --tb=short
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/store_listing.py tests/test_store_listing.py scripts/validate_store_release.py
git commit -m "feat: validate store listing for release versions"
```

---

### Task 3: Generator + real `store/listing.yaml`

**Files:**
- Modify: `src/store_listing.py`
- Modify: `tests/test_store_listing.py`
- Create: `scripts/build_store_listing.py`
- Create: `store/listing.yaml`
- Create: `store/product-update.skeleton.json` (generated)
- Modify: `docs/store-listing.md` (generated)

**Interfaces:**
- Consumes: `ListingData`
- Produces:
  - `render_markdown(data: ListingData, *, release_version: str) -> str`
  - `render_skeleton(data: ListingData, *, release_version: str) -> dict`
  - `write_outputs(data, *, release_version, markdown_path, skeleton_path) -> None`
  - `outputs_match(data, *, release_version, markdown_path, skeleton_path) -> bool`
  - CLI `build_store_listing.py` and `--check`

- [ ] **Step 1: Write failing generator tests**

```python
from src.store_listing import render_markdown, render_skeleton, outputs_match


def test_render_markdown_marks_generated_and_includes_sections() -> None:
    text = render_markdown(_data(), release_version="2.6.0")
    assert "generated from" in text.lower()
    assert "store/listing.yaml" in text
    assert "## Short description" in text
    assert "Short" in text
    assert "## Product features" in text
    assert "Feature one" in text
    assert "What's new in version 2.6.0" in text


def test_render_skeleton_includes_package_url_placeholder() -> None:
    payload = render_skeleton(_data(), release_version="2.6.0")
    assert "packages" in payload
    assert "v2.6.0/stemma.msix" in payload["packages"][0]["packageUrl"]
    assert payload["listing"]["shortDescription"] == "Short"
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_store_listing.py -q --tb=short
```

Expected: FAIL on missing render helpers.

- [ ] **Step 3: Implement render helpers**

Implement `render_markdown` to produce Partner Center paste-ready markdown with:

- Banner that the file is generated from `store/listing.yaml`
- Field mapping table
- Sections: Short description, Description, What's new (for `release_version`), Product features, Search terms
- Preserve the "Notes for future submissions" bullets from the current hand file (hardcode those three bullets in the renderer; they are process notes, not Partner Center fields)

Implement `render_skeleton` roughly as:

```python
{
  "packages": [
    {
      "packageUrl": (
        "https://github.com/cyanidesayonara/stemma/releases/"
        f"download/v{release_version}/stemma.msix"
      ),
      "languages": ["en-us"],
      "architectures": ["x64"],
      "installerParameters": "",
      "isSilentInstall": True,
    }
  ],
  "listing": {
    "shortDescription": data.short_description,
    "description": data.description,
    "features": data.features,
    "searchTerms": data.search_terms,
    "whatsNew": data.whats_new[release_version],
  },
}
```

(`whatsNew` key only included when present; generator for real files uses validated version.)

- [ ] **Step 4: Create `store/listing.yaml` from current copy**

Migrate text from existing `docs/store-listing.md`:

- `short_description`, `description`, `features` (18 lines), `search_terms` (split the comma list into YAML list items)
- `whats_new["2.5.0"]` = current What's new body
- `whats_new["2.6.0"]` = truthful notes covering:
  - 2-stem GPU-accelerated separation when available, with CPU fallback
  - background import / separation queue
  - stability and integrity hardening (model checksums, async stem loading, release diagnostics)
  - Do **not** claim HTDemucs GPU or v3.0 UI recomposition

Set `listing_version: "2.6.0"`.

- [ ] **Step 5: Add CLI and generate committed outputs**

`scripts/build_store_listing.py`:

```python
"""Generate docs/store-listing.md and store/product-update.skeleton.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import (
    DEFAULT_LISTING_YAML,
    DEFAULT_MARKDOWN,
    DEFAULT_SKELETON,
    DEFAULT_VERSION_PY,
    load_listing,
    outputs_match,
    read_app_version,
    write_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--listing", type=Path, default=DEFAULT_LISTING_YAML)
    args = parser.parse_args(argv)
    version = args.version or read_app_version(DEFAULT_VERSION_PY)
    data = load_listing(args.listing)
    if version not in data.whats_new:
        print(f"whats_new missing {version}", file=sys.stderr)
        return 1
    if args.check:
        if outputs_match(
            data,
            release_version=version,
            markdown_path=DEFAULT_MARKDOWN,
            skeleton_path=DEFAULT_SKELETON,
        ):
            print("Store listing outputs are up to date")
            return 0
        print("Store listing outputs are stale; run build_store_listing.py",
              file=sys.stderr)
        return 1
    write_outputs(
        data,
        release_version=version,
        markdown_path=DEFAULT_MARKDOWN,
        skeleton_path=DEFAULT_SKELETON,
    )
    print(f"Wrote {DEFAULT_MARKDOWN} and {DEFAULT_SKELETON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```powershell
.\.venv\Scripts\python.exe scripts\build_store_listing.py --version 2.6.0
.\.venv\Scripts\python.exe scripts\build_store_listing.py --check --version 2.6.0
```

Expected: writes files; `--check` exits 0.

- [ ] **Step 6: Add drift test and run full store tests**

```python
def test_build_check_detects_stale_markdown(tmp_path: Path) -> None:
    # write yaml, generate to paths, mutate markdown, assert outputs_match False
    ...
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_store_listing.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/store_listing.py tests/test_store_listing.py scripts/build_store_listing.py store/listing.yaml store/product-update.skeleton.json docs/store-listing.md
git commit -m "feat: generate store listing from YAML"
```

---

### Task 4: Wire CI/release + operator docs + issue notes

**Files:**
- Modify: `.github/workflows/release.yml`
- Modify: `docs/store-release-pipeline.md`
- Modify: `docs/ROADMAP.md` (brief #134 note if needed)
- Modify: `AGENTS.md` only if a canonical command belongs there (prefer `docs/store-release-pipeline.md` + `docs/DEVELOPMENT.md` link)

**Interfaces:**
- Consumes: `scripts/validate_store_release.py`, `scripts/build_store_listing.py --check`

- [ ] **Step 1: Add release workflow step after version sync**

In `.github/workflows/release.yml`, immediately after "Sync version from tag" and **after** dependency install (validator needs PyYAML), add:

```yaml
      - name: Validate Store listing
        shell: pwsh
        run: |
          $tag = "${{ github.ref_name }}"
          if ($tag.StartsWith("v")) { $tag = $tag.Substring(1) }
          python scripts/validate_store_release.py --version $tag
          python scripts/build_store_listing.py --check --version $tag
```

Place it after "Install dependencies" and before or after Ruff (after install is required).

- [ ] **Step 2: Run local validation against repo screenshots**

```powershell
.\.venv\Scripts\python.exe scripts\validate_store_release.py --version 2.6.0
.\.venv\Scripts\python.exe scripts\build_store_listing.py --check --version 2.6.0
```

Expected: exit 0 (three 1366x768 screenshots on main).

If screenshots are missing in the worktree, `git checkout origin/main -- assets/store_listing/screenshots` first.

- [ ] **Step 3: Update `docs/store-release-pipeline.md`**

Document:

- `store/listing.yaml` is the source of truth
- `python scripts/build_store_listing.py` regenerates markdown + skeleton
- `python scripts/validate_store_release.py --version X.Y.Z` is required on tags
- Partner Center automation remains manual / later slice of #134
- Editing `docs/store-listing.md` by hand is wrong; edit YAML instead

- [ ] **Step 4: Run full fast suite**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -m "not slow and not hardware" -q --tb=short
.\.venv\Scripts\python.exe -m ruff check src\store_listing.py scripts\build_store_listing.py scripts\validate_store_release.py tests\test_store_listing.py
```

Expected: all green.

- [ ] **Step 5: Commit**

```powershell
git add .github/workflows/release.yml docs/store-release-pipeline.md docs/ROADMAP.md
git commit -m "ci: validate store listing on release tags"
```

- [ ] **Step 6: Update GitHub issue #134**

Using `gh`, comment or edit checklist to mark slice-1 items done in the PR description; leave Partner Center draft automation unchecked. Keep issue open.

---

## Spec coverage self-review

| Spec requirement | Task |
|---|---|
| `store/listing.yaml` source of truth | Task 3 |
| Generate `docs/store-listing.md` | Task 3 |
| Skeleton JSON with package URL placeholder | Task 3 |
| `whats_new` required for release version | Task 2 |
| Feature count/length gates | Task 2 |
| Screenshot >=3 and >=1366x768 | Task 2 |
| Generator `--check` drift | Task 3 |
| `release.yml` validator after sync/install | Task 4 |
| No Partner Center calls | All tasks |
| Truthful 2.6.0 What's new | Task 3 |
| Operator docs | Task 4 |
| PyYAML in release lock | Task 1 |

No intentional placeholders remain. Screenshot minimum stays 3 per approved design.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-31-store-listing-as-code.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
