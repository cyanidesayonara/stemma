# Microsoft Store and release pipeline

## What runs automatically today

Pushing a version tag matching `v*` triggers `.github/workflows/release.yml`:

1. **Sync versions** -- `scripts/sync_release_version.ps1` sets `src/version.py` and `msix/AppxManifest.xml` Identity `Version` from the tag (`v2.6.0` becomes app `2.6.0` and MSIX `2.6.0.0`). The repository files describe the current branch build; tag synchronization remains the guard that makes packaged bits match the release tag.
2. **Validate Store listing** -- after dependency install, `scripts/validate_store_release.py` and `scripts/build_store_listing.py --check` run against the synced `src/version.py` (semver from the tag; prerelease suffixes already stripped by sync). Each command's exit code is checked so a failed validation cannot be masked by a later success. The release fails if listing metadata, assets, or generated outputs are invalid or out of date.
3. **Fast tests** -- same pytest slice as CI (`not slow`, `not hardware`).
4. **PyInstaller** -- `dist/stemma/` plus `stemma.zip` and `stemma.msix`.
5. **GitHub Release** -- attaches `stemma.zip` and `stemma.msix`.

CI (`.github/workflows/ci.yml`) also runs on `v*` tag pushes so a tag-only release still gets a test run.

## Store listing as code

`store/listing.yaml` is the single source of truth for Partner Center copy, feature bullets, and per-version **What's new** text. It also records the live Store product identity (`store.product_id` `9P2W12L8F381`, package family name, public URL). An archival capture of the previously published Store copy lives in `store/live-snapshot-2026-07-31.md`.

- **Edit YAML, not markdown.** Do not hand-edit `docs/store-listing.md`; it is generated output.
- **Regenerate outputs** after YAML changes:

  ```powershell
  python scripts/build_store_listing.py --version X.Y.Z
  ```

  This refreshes `docs/store-listing.md` and `store/product-update.skeleton.json`.

- **Validate before tagging** (same checks as release CI):

  ```powershell
  python scripts/validate_store_release.py --version X.Y.Z
  python scripts/build_store_listing.py --check --version X.Y.Z
  ```

  Add a `whats_new` entry for every release version. Screenshots under `assets/store_listing/screenshots/` must meet Store minimum size (1366x768) and count requirements.

Partner Center draft/submit automation uses `.github/workflows/partner-center-submit.yml`
(manual `workflow_dispatch`) with the [Microsoft Store CLI](https://learn.microsoft.com/en-us/windows/apps/publish/msstore-dev-cli/overview) (`msstore`), which supports MSIX products. Modes:

- **`configure`** -- credentials check only (`msstore reconfigure` + `msstore info`)
- **`update_draft`** -- download `stemma.msix` from the release tag, upload it with
  `msstore publish --noCommit` (draft only; does not start certification), then push
  listing metadata and verify via the submission API
- **`update_metadata`** -- ensure a pending draft exists (creating one with
  `--noCommit` if needed), merge listing fields from `store/listing.yaml`, push with
  `submission updateMetadata`, and verify via the submission API
- **`get_draft`** -- print current submission status and package JSON (debug)

Partner Center UI can lag behind the submission API while a draft is in
`PendingCommit`, especially until you inspect the draft in Partner Center.
After `update_draft` or `update_metadata`, the workflow checks the merged
payload and attempts a submission GET (warnings only if GET is stale). Submit
for certification manually in Partner Center when you are ready to ship.

```powershell
python scripts/build_partner_center_payloads.py --tag v2.6.0
```

Writes `store/payloads/product-update.json` and `store/payloads/metadata-update.json`
(gitignored). After `update_draft` or `update_metadata`, confirm listing fields via
the workflow verification step or `get_draft`, then submit for certification
manually in Partner Center.

## Manual Store upload (fallback)

After the GitHub Release exists, download `stemma.msix` (or use the direct URL below) and upload it in [Partner Center](https://partner.microsoft.com/dashboard) under your app submission packages.

Public download URL pattern (public repo):

`https://github.com/<owner>/<repo>/releases/download/<tag>/stemma.msix`

Example: `https://github.com/cyanidesayonara/stemma/releases/download/v2.5.0/stemma.msix`

## Partner Center credentials (GitHub Actions)

Repository secrets for `partner-center-submit.yml`:

- `PARTNER_CENTER_SELLER_ID`
- `PARTNER_CENTER_PRODUCT_ID`
- `PARTNER_CENTER_TENANT_ID`
- `PARTNER_CENTER_CLIENT_ID`
- `PARTNER_CENTER_CLIENT_SECRET`

Use **mode `configure`** first, then **`update_draft`** with a release tag (for example `v2.6.0`). Verify listing metadata via the workflow or **`get_draft`**, then click **Submit for certification** in Partner Center. Manual MSIX upload remains a fallback if automation fails.

**Not automated (manual in Partner Center when they change):** Store listing screenshots (`assets/store_listing/screenshots/`), poster/box/tile art (`assets/store_listing/*.png`, regenerate with `scripts/generate_store_listing_assets.py`), and any category or age-rating fields. Release CI validates screenshot count and size; MSIX package icons come from the uploaded package itself.

**Limitation:** `msstore publish` (MSIX package upload) is [documented as free-products-only](https://learn.microsoft.com/en-us/windows/apps/publish/msstore-dev-cli/overview). If package upload fails with that error, upload `stemma.msix` manually; listing metadata can still be pushed via `update_draft`.

Note: [microsoft/store-submission](https://github.com/microsoft/store-submission) targets EXE/MSI (Win32) packaged apps and does not support MSIX; stemma uses `msstore` instead.

## Local version sync (without tagging)

To align repo files with a would-be tag before committing:

```powershell
.\scripts\sync_release_version.ps1 -Tag v2.6.0
```

This updates local build metadata only; it does not publish v2.6.0. Commit
`src/version.py` and `msix/AppxManifest.xml` when the branch build target
changes.
