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
  demo.js       the "Try the mixer" demo: song picker, controls, lanes
  stretch-worklet.js  its audio engine (mixing, loop, speed, pitch)
  demo/         the demo songs: songs.json and <id>/<stem>.mp3
  site.js       the dark/light theme toggle
scripts/
  build_site.py       fills the template and assembles the site
  site_palette.py     writes site/palette.css
  make_demo_songs.py  writes site/demo/ from stemma's own separations
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

A few blocks come from elsewhere: `themecolor` (the browser theme colour, the
app's mantle per theme from `src/ui/styles.py`), `jsonld` (the schema.org
SoftwareApplication record with the released version) and `credits` (the
"Songs in this demo" list under the cockpit, from `site/demo/songs.json`).

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

The demo plays short clips of free-to-use songs, each separated by stemma
itself, so visitors hear what the app really produces rather than studio
multitracks. A song may have any of the app's stems: `stems` in
`songs.json` lists them (a 4-stem song has vocals, drums, bass and other),
and the lanes and mixer rows follow it, in the app's order and colours. `site/demo.js` is the
cockpit: a song picker, mute, solo and volume per stem, click-to-seek and
drag-to-loop on stacked lanes, the Loop Trainer, speed, pitch and a
metronome. `site/stretch-worklet.js` is the audio engine, an AudioWorklet
that mixes the stems and changes speed and pitch live with the app's phase
vocoder (`src/stretch.py`, ported), so tempo and key stay independent.

The key, chord, tempo and beat readouts are the app's own detector
(`src/beat_detector.py`), run on each clip. The lanes are drawn from a
stored overview, so they show before any audio downloads; a song's stems
(about 2-3 MB) load when it is first played.

Only the playing song is decoded, and its samples move to the worklet
rather than being copied; switching back to a song decodes it again from
the HTTP cache. With no A-B loop, playback stops at the end of the clip and
the next play starts from 0:00. After a pause has faded out the page
suspends the audio context, and a song switch waits for the fade-out
before the new song loads. On the lanes, a tap seeks, a sideways drag sets
a loop, and a vertical swipe scrolls the page; with the lanes focused,
Left and Right (and Down and Up) move 5 seconds as in the app, Page Down
and Page Up move 15, and Home and End go to the start and the end.

### The songs

| Song | Artist | Genre | Licence | Source |
|---|---|---|---|---|
| Makes-Shift Salvation | HoliznaCC0 | Indie rock | CC0 1.0 | [Free Music Archive](https://freemusicarchive.org/music/holiznacc0/makeshift-salvation/makes-shift-salvation) |
| Blue Part II | JCrews (jimmy233) feat. Adisa McKenzie | Soul | CC BY 3.0 | [ccMixter](https://ccmixter.org/files/jimmy233/58818) |
| Free Music and Free Beer | Admiral Bob (admiralbob77) feat. SackJo22 | Blues | CC BY 3.0 | [ccMixter](https://ccmixter.org/files/admiralbob77/38104) |
| Girls Like You | Stefan Kartenberg (JeffSpeed68) feat. Thespinwires | Garage rock | CC BY 3.0 | [ccMixter](https://ccmixter.org/files/JeffSpeed68/44932) |
| Let It In | Josh Woodward | Funk rock | CC BY 4.0 | [joshwoodward.com](https://joshwoodward.com/song/LetItIn) |
| Where You Are Now | AlexBeroza feat. WolfFly | Hard rock | CC BY 3.0 | [ccMixter](https://ccmixter.org/files/AlexBeroza/33065) |

The list mirrors `SONGS` in `scripts/make_demo_songs.py`.

Only licences that allow derivative works without a non-commercial clause
qualify (CC0, CC BY); `tests/test_site.py` enforces that, the credit the
page shows beside the player, and the static credits list under it.

To add or remake a song: download it, separate it in stemma (or with the
6-stem model), put its stem folder beside the others, add it to `SONGS` in
`scripts/make_demo_songs.py`, and run

```powershell
python scripts/make_demo_songs.py <folder of separated songs>
```

The script picks a clip of whole bars where every part plays (or uses the
`start` you give it), runs the detector, and rewrites `site/demo/`.

## Deploy

`.github/workflows/pages.yml` builds and deploys the whole site every time:

- after every successful `Release` run, so a new version goes live with no
  change to `site/`
- on a push to main that touches the site or its build
- on manual dispatch

A prerelease is never GitHub's "latest" release, so it leaves the site on the
last stable version. Pages is set to deploy from GitHub Actions
(Settings > Pages > Source).
