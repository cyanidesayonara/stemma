# Store listing as code (issue #134, slice 1)

Date: 2026-07-31  
Status: Approved for implementation planning  
Issue: [#134](https://github.com/cyanidesayonara/stemma/issues/134)  
Related: [#137](https://github.com/cyanidesayonara/stemma/issues/137) (v2.6 publish gate)

## Problem

v2.5.0 shipped through the Store with hand-edited Partner Center fields.
The listing drifted (still describing 2.3.0 when 2.5.0 was submitted).
`docs/store-listing.md` is human-edited prose that can disagree with the
tag being released. A `v*` tag already builds GitHub Release artifacts;
it does not yet prove that Store copy matches that tag.

## Goal

Treat Store listing copy as code for the first vertical slice of #134:

- One YAML source of truth for listing fields and version-keyed What's new.
- Generated human-readable markdown that cannot silently drift from YAML.
- Release and PR validation that fails when What's new, features, or
  screenshots are wrong for the version under test.
- No Partner Center API calls, draft submission, or publish in this slice.

Success looks like: tagging `v2.6.0` fails the release job unless
`store/listing.yaml` has a `whats_new` entry for `2.6.0`, features stay
within Partner Center limits, committed screenshots meet the size/count
gate, and regenerating `docs/store-listing.md` would not change the tree.

## Non-goals (later slices)

- Partner Center credentialed `update` / draft submission / publish.
- CI regeneration of brand assets or screenshots from live UI.
- Submission-id tracking and Store status polling.
- Raising the screenshot minimum from 3 to 4 (current committed set is 3).

## Current baseline

- `release.yml` on `v*` tags: version sync, Ruff, fast tests, PyInstaller,
  frozen diagnostics, zip, MSIX, checksums, GitHub Release. Works.
- `partner-center-submit.yml`: manual dispatch; package URL only; no
  draft-only path. Untouched by this slice.
- `docs/store-listing.md`: hand-maintained v2.5.0-oriented copy.
- `assets/store_listing/screenshots/`: three PNGs at 1366x768 (on main).
- No `store/listing.yaml` and no listing validator today.

## Design

### 1. Source of truth: `store/listing.yaml`

Machine-readable listing payload. Suggested shape:

```yaml
# Informational only. Release authority is the git tag / synced version.py.
listing_version: "2.6.0"

short_description: |
  ...

description: |
  ...

features:
  - "AI stem separation: vocals, drums, bass, guitar, piano, other"
  # max 20 items; each within Partner Center length limits

search_terms:
  - stem separation
  - vocal remover
  # ...

whats_new:
  "2.5.0": |
    What's new in version 2.5.0
    ...
  "2.6.0": |
    What's new in version 2.6.0
    ...
```

Rules:

- Migrate current `docs/store-listing.md` content into YAML without
  inventing new marketing claims.
- Keep model-family names (HTDemucs, MDX-Net) out of Store description
  text, matching the existing notes in `docs/store-listing.md`.
- Do not claim GPU for 4/6-stem separation.
- For the upcoming publish, include a truthful `whats_new["2.6.0"]`
  covering MDX DirectML 2-stem, background import queue, stability work,
  and diagnostics/integrity improvements already on `main`. Do not claim
  v3.0 UI work.
- `listing_version` may track the next intended Store submission; the
  validator keys What's new off the **release version under test**, not
  this field alone.

### 2. Generator: `scripts/build_store_listing.py`

Reads `store/listing.yaml` and writes:

1. `docs/store-listing.md` — Partner Center paste-ready markdown, marked
   as **generated from** `store/listing.yaml`.
2. `store/product-update.skeleton.json` — listing fields plus a package
   URL placeholder (`https://github.com/cyanidesayonara/stemma/releases/download/v{version}/stemma.msix`).
   Not submitted in this slice; exists so the later Partner Center work
   has a stable shape to validate against.

CLI:

```text
python scripts/build_store_listing.py
python scripts/build_store_listing.py --check   # exit non-zero on drift
```

`--check` regenerates into memory/temp and compares to the committed
markdown (and skeleton JSON). Used by tests and CI.

### 3. Validator: `scripts/validate_store_release.py`

Inputs:

- `--version X.Y.Z` (required for release; tests may pass explicit values)
- Optional defaults: read `src/version.py` when `--version` omitted in
  local/PR runs

Checks (all hard failures):

| Check | Rule |
|---|---|
| What's new | `whats_new[version]` exists and is non-empty |
| Features count | `1 <= len(features) <= 20` |
| Feature length | each feature string length within Partner Center limit (use 200 chars unless docs prove a tighter limit; encode the constant once) |
| Short description | non-empty; enforce a conservative max length (1000 chars) with a named constant |
| Screenshots | at least **3** PNG files under `assets/store_listing/screenshots/`; each width >= 1366 and height >= 768 |
| Listing drift | running the generator `--check` mode succeeds |

Out of scope for the validator in this slice:

- Regenerating brand/store PNGs and failing on dirty git
- Requiring fonts or a local song library
- Calling Partner Center

### 4. Tests

Prefer pytest over ad-hoc-only scripts where practical:

- YAML load / schema-ish required keys
- Missing `whats_new` for a version fails
- Too many / too-long features fail
- Screenshot count and dimension failures (temp dirs with tiny PNGs)
- Generator round-trip: build markdown from fixture YAML matches golden
  or regenerates stably
- `--check` fails when committed markdown is intentionally stale

Keep tests fast and offline (no network, no Qt UI, no real model loads).
Screenshot dimension checks can use small Pillow/png fixtures or the
stdlib/`struct` PNG IHDR reader if Pillow is already available in
release/dev deps; do not add a heavy new dependency only for this.

### 5. CI and release wiring

**PR / push CI (`ci.yml`):**

- Run the new listing tests with the fast suite (or an explicit path
  already covered by `pytest`).
- Optionally run `validate_store_release.py` against `src/version.py`
  so `main` at 2.6.0 already requires a 2.6.0 What's new entry.

**Tag release (`release.yml`):**

After `sync_release_version.ps1` and before PyInstaller:

```text
python scripts/validate_store_release.py --version <tag-without-v>
```

If validation fails, the release job fails before packaging.

### 6. Documentation updates

- `docs/store-release-pipeline.md`: document YAML as source of truth,
  generator commands, validator in release, and that Partner Center
  automation remains a later slice.
- `docs/ROADMAP.md` / `#134`: note slice 1 landing when the PR merges;
  keep Partner Center draft automation as remaining work.
- Generated `docs/store-listing.md` header must say it is generated and
  point editors at `store/listing.yaml`.

## File map

| Path | Role |
|---|---|
| `store/listing.yaml` | Source of truth |
| `store/product-update.skeleton.json` | Generated payload skeleton (not submitted) |
| `scripts/build_store_listing.py` | YAML → markdown + skeleton |
| `scripts/validate_store_release.py` | Release/PR gates |
| `tests/test_store_listing.py` | Fast unit/integration coverage |
| `docs/store-listing.md` | Generated human copy |
| `docs/store-release-pipeline.md` | Operator docs |
| `.github/workflows/release.yml` | Invoke validator after version sync |
| `.github/workflows/ci.yml` | Covered via pytest (and optional explicit validate) |

## Rollout order

1. Add YAML + generator + tests (TDD).
2. Replace hand-maintained markdown with generated output; add 2.6.0
   What's new.
3. Add validator + screenshot/feature gates.
4. Wire validator into `release.yml`.
5. Update operator docs and issue #134 checklist text.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Partner Center field limits guessed wrong | Use conservative constants in one module; adjust after first live draft validation in a later slice |
| Only 3 screenshots vs earlier “>=4” idea | Explicit gate of >=3 for this slice; raise later when a fourth shot exists |
| Authors edit markdown instead of YAML | Generated banner + `--check` / CI drift failure |
| v2.6 What's new overclaims | Draft from shipped #138/#139 capabilities only; review in PR |

## Acceptance criteria

- [ ] `store/listing.yaml` is the only hand-edited listing source.
- [ ] `docs/store-listing.md` is generated and drift-checked.
- [ ] Missing `whats_new` for the release version fails validation.
- [ ] Feature count/length and screenshot size/count gates fail loudly.
- [ ] `release.yml` runs the validator after version sync.
- [ ] Fast tests cover success and failure paths.
- [ ] No Partner Center network calls are introduced.
- [ ] Issue #134 remains open with slice-1 complete and Partner Center
      draft automation still TODO.

## Open follow-ups (not this PR)

1. Secret-gated Partner Center draft update on tag (stop at draft).
2. Raise screenshot minimum to 4 and/or generate screenshots in CI with
   a fixture song.
3. Brand/store asset freshness gate (`generate_*` then fail if dirty).
4. Submission-id recording and status polling.
