# Microsoft Store and release pipeline

## What runs automatically today

Pushing a version tag matching `v*` triggers `.github/workflows/release.yml`:

1. **Sync versions** -- `scripts/sync_release_version.ps1` sets `src/version.py` and `msix/AppxManifest.xml` Identity `Version` from the tag (`v2.6.0` becomes app `2.6.0` and MSIX `2.6.0.0`). The repository files describe the current branch build; tag synchronization remains the guard that makes packaged bits match the release tag.
2. **Validate Store listing** -- after dependency install, `scripts/validate_store_release.py` and `scripts/build_store_listing.py --check` run against the tag version (leading `v` stripped). The release fails if listing metadata, assets, or generated outputs are invalid or out of date.
3. **Fast tests** -- same pytest slice as CI (`not slow`, `not hardware`).
4. **PyInstaller** -- `dist/stemma/` plus `stemma.zip` and `stemma.msix`.
5. **GitHub Release** -- attaches `stemma.zip` and `stemma.msix`.

CI (`.github/workflows/ci.yml`) also runs on `v*` tag pushes so a tag-only release still gets a test run.

## Store listing as code

`store/listing.yaml` is the single source of truth for Partner Center copy, feature bullets, and per-version **What's new** text.

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

Partner Center draft/submit automation remains manual for slice 1 of [#134](https://github.com/cyanidesayonara/stemma/issues/134); use the GitHub Release MSIX upload path below until a later slice wires API submission.

## Manual Store upload (current default)

After the GitHub Release exists, download `stemma.msix` (or use the direct URL below) and upload it in [Partner Center](https://partner.microsoft.com/dashboard) under your app submission packages.

Public download URL pattern (public repo):

`https://github.com/<owner>/<repo>/releases/download/<tag>/stemma.msix`

Example: `https://github.com/cyanidesayonara/stemma/releases/download/v2.5.0/stemma.msix`

## Optional: Partner Center API / GitHub Action

Microsoft publishes [microsoft/store-submission](https://github.com/microsoft/store-submission) for automating submissions. It targets the newer Store submission flow (often used for Win32/MSI-style packages with `packageUrl`). MSIX / Desktop Bridge products may use different API fields than the samples in that README.

Before wiring automation:

1. Complete [Partner Center prerequisites](https://github.com/microsoft/store-submission#prerequisites) (Azure AD app, Manager role, at least one manual submission).
2. Confirm in Partner Center whether your listing uses the **packaged (MSIX)** or **Win32** submission path so you pass the correct `type` and `product-update` JSON to the action.
3. Add repository secrets (names are suggestions; match what you reference in YAML):

   - `PARTNER_CENTER_SELLER_ID`
   - `PARTNER_CENTER_PRODUCT_ID`
   - `PARTNER_CENTER_TENANT_ID`
   - `PARTNER_CENTER_CLIENT_ID`
   - `PARTNER_CENTER_CLIENT_SECRET`

This repo includes `.github/workflows/partner-center-submit.yml`, a **manual** workflow (`workflow_dispatch`). Use **mode `configure`** first to verify Partner Center credentials. **mode `submit_and_publish`** runs update + publish using a **template** `product-update` JSON in the YAML; edit that JSON to match your app type in Partner Center before using it. Until then, keep using manual upload from the GitHub Release asset URL.

## Local version sync (without tagging)

To align repo files with a would-be tag before committing:

```powershell
.\scripts\sync_release_version.ps1 -Tag v2.6.0
```

This updates local build metadata only; it does not publish v2.6.0. Commit
`src/version.py` and `msix/AppxManifest.xml` when the branch build target
changes.
