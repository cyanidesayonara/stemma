# Microsoft Store and release pipeline

## What runs automatically today

Pushing a version tag matching `v*` triggers `.github/workflows/release.yml`:

1. **Sync versions** -- `scripts/sync_release_version.ps1` sets `src/version.py` and `msix/AppxManifest.xml` Identity `Version` from the tag (`v2.0.3` becomes app `2.0.3` and MSIX `2.0.3.0`). The commit on `main` can still say `2.0.2` until the next housekeeping commit; the **tag build** is the source of truth for shipped bits.
2. **Fast tests** -- same pytest slice as CI (`not slow`, `not hardware`).
3. **PyInstaller** -- `dist/stemma/` plus `stemma.zip` and `stemma.msix`.
4. **GitHub Release** -- attaches `stemma.zip` and `stemma.msix`.

CI (`.github/workflows/ci.yml`) also runs on `v*` tag pushes so a tag-only release still gets a test run.

## Manual Store upload (current default)

After the GitHub Release exists, download `stemma.msix` (or use the direct URL below) and upload it in [Partner Center](https://partner.microsoft.com/dashboard) under your app submission packages.

Public download URL pattern (public repo):

`https://github.com/<owner>/<repo>/releases/download/<tag>/stemma.msix`

Example: `https://github.com/cyanidesayonara/stemma/releases/download/v2.0.3/stemma.msix`

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
.\scripts\sync_release_version.ps1 -Tag v2.0.3
```

Then commit `src/version.py` and `msix/AppxManifest.xml` if you want `main` to match the next release before the tag exists.
