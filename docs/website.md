# The website

A free static site on GitHub Pages at
https://cyanidesayonara.github.io/stemma/: what stemma does, a playable
mixer demo, the features, screenshots and What's new, with download links
to the Microsoft Store and the GitHub Release zip.

## What is where

```
site/
  index.html    the page; a template with <!-- gen:NAME --> blocks
  style.css     the stylesheet; colours only through palette.css variables
  palette.css   generated from src/ui/styles.py, never edited by hand
  demo.js       the "Try the mixer" demo
  site.js       the dark/light theme toggle
scripts/
  build_site.py    fills the template and assembles the site
  site_palette.py  writes site/palette.css
.github/workflows/pages.yml   builds and deploys
```

Plain HTML, CSS and JavaScript: no framework, no external requests, no
cookies. Links are relative, so the same files work under `/stemma/` and at
the root of a custom domain.

## Look

The site wears the app's two themes. `site/palette.css` is the dark and light
palettes and the per-stem colours from `src/ui/styles.py`; after changing a
colour there, run

```powershell
python scripts/site_palette.py
```

`tests/test_site.py` fails while it is stale, and fails on any literal colour
in `style.css`, `demo.js` or `site.js`. The page follows the system theme
until the visitor picks one with the sun button, and the logo is the app's
`logo_arpeggio_{dark,light}.svg` wordmark.

## Copy and screenshots

The `<!-- gen:NAME -->` blocks hold no copy in the repository. At deploy time
`scripts/build_site.py` fills them from the **latest stable release's tag**:
the short description, features and What's new from its
`store/listing.yaml`, and its Store screenshots and captions from
`assets/store_listing/screenshots/`. The site therefore always describes the
version people download, never main's unreleased work, and the release
checklist that keeps the Store listing current keeps the site current too.

Preview locally (copy from this checkout):

```powershell
python scripts/build_site.py --out build/site
```

```powershell
python -m http.server 8765 --directory build/site
```

Or preview what the live site shows, from a tag checked out in a worktree:

```powershell
git worktree add ../stemma-rel v2.6.0
```

```powershell
python scripts/build_site.py --from ../stemma-rel --version v2.6.0 --released 2026-07-31 --out build/site
```

## The mixer demo

`site/demo.js` plays an eight-bar tune written for the page, sequenced in the
Web Audio API as six stems, with the app's practice controls: mute, solo and
volume per stem, click-to-seek and drag-to-loop on stacked lanes, the Loop
Trainer, speed, pitch and a metronome. Because the tune is sequenced rather
than recorded, speed changes the tempo without the pitch and pitch changes
the key without the tempo, as the app does for real songs. It separates
nothing; the page says so.

## Deploy

`.github/workflows/pages.yml` builds and deploys the whole site every time:

- after every successful `Release` run, so a new version goes live with no
  change to `site/`
- on a push to main that touches the site or its build
- on manual dispatch

A prerelease is never GitHub's "latest" release, so it leaves the site on the
last stable version. Pages is set to deploy from GitHub Actions
(Settings > Pages > Source).
