# Microsoft Store listing copy

Generated from `store/listing.yaml`. Edit the YAML, then run
`python scripts/build_store_listing.py` to regenerate this file.
Do not edit this markdown by hand.

This copy reflects listing content for version **3.0.0**.

Public Store listing: https://apps.microsoft.com/detail/9p2w12l8f381 (product id `9P2W12L8F381`).

Fields map to Partner Center as follows:

| Partner Center field | Section below |
|---|---|
| Description | [Description](#description) |
| What's new in this version | [What's new](#whats-new-in-this-version) |
| Product features (max 20, one per line) | [Product features](#product-features) |
| Short description | [Short description](#short-description) |
| Search terms | [Search terms](#search-terms) |

Assets: `assets/store_listing/` (regenerate with
`python scripts/generate_store_listing_assets.py`).
Screenshots: `assets/store_listing/screenshots/` (regenerate with
`python scripts/generate_screenshots.py`).

---

## Short description

Practice any song with the band. stemma splits a track into vocals, drums, bass, guitar, piano, and other, so you can mute your part and play along. Slow it down without changing pitch, loop the hard bars, transpose to your key, and record your take. A stem splitter and vocal remover that runs entirely on your PC: no account, no subscription, no uploads.

---

## Description

stemma turns any song into a practice tool.

Import a track and stemma separates it into individual stems -- vocals, drums, bass, guitar, piano, and everything else -- so you can mute the part you play and perform it yourself. Silence the guitar and it is your guitar in the mix. Solo the drums to lock in with them. Pull the vocal down and sing the line yourself. Choose two stems (vocals and the backing track, fast, and GPU-accelerated when your PC supports it), four stems, or all six.

Everything else is built around learning a part properly. Set an A-B loop over the two bars that keep tripping you up and drill them. Slow the passage down without the pitch dropping, or turn on the Loop Trainer and let stemma step the speed up a notch on every repeat until you are at full tempo. Transpose the whole song into a key that suits your voice or instrument, up to seven semitones either way, and the key and chord readouts follow. Count yourself in, play along to the metronome, and record your take against the backing to hear how it really sat.

stemma reads the song as you work: tempo, musical key, and the chord under the playhead, updated as it plays. Every stem gets its own colored lane in the waveform, so you can see where each part plays and click anywhere to jump there. There are per-stem volume faders, and a library that remembers exactly where you left off -- song, position, mix, loop, speed, and pitch -- so practice picks up where it stopped. Browse with previous and next, loop a single song, shuffle the collection, or autoplay through it.

Separation runs on your own machine. Nothing is uploaded, there is no account, no subscription, and no internet connection needed once the models are downloaded. Import an MP3, WAV, or FLAC file or paste a YouTube link, and export your own mix, a single stem, or just the loop as WAV or MP3 when you want to take it elsewhere.

Built for Windows, with keyboard shortcuts for the whole practice loop that work on any keyboard layout. Dark and light themes.

---

## What's new in this version

What's new in version 3.0.0

A new practice cockpit: every stem gets its own colored lane in the waveform, a muted stem dims in place, and the A-B loop shows its times. The practice controls are grouped into cards. Key, chord, and tempo share one readout under the waveform, and the transport stays at the bottom.

A new look: a new app icon, stronger contrast in both the light and dark themes, and tidier dialogs.

Your library is safer: a damaged library file no longer removes songs, and an interrupted import is cleaned up on the next launch.

Export Mix now exports the stems you hear, including solo and stem volumes, at the song's original speed and pitch.

Keyboard and accessibility: the whole app can be used from the keyboard, controls have names for screen readers, and View > Switch Theme changes the theme.

Safer imports: stemma checks a file before importing it, asks before downloading a separation model, and downloads a fresh copy if one is damaged. Quitting during a separation asks first.

Clearer errors: a problem shows a readable message, and the details go to a log file. If stemma cannot start, it says why.

Recording: with no microphone connected, stemma turns recording off and keeps playing, and a take that cannot be saved stays in memory so you can try again.

Fixes: changing loop points quickly no longer crashes the app, and the chord readout follows the pitch shift.

---

## Product features

Split any song into vocals, drums, bass, guitar, piano, and other
Mute your part and play along with the rest of the band
Per-stem mute, solo, and volume
Fast two-stem vocal remover, GPU-accelerated when available
Slow down or speed up from 0.5x to 2x without changing pitch
A-B loop to drill a difficult passage
Loop Trainer: speed steps up on every repeat until you reach full tempo
Transpose up or down seven semitones, tempo unchanged
Automatic key and tempo detection, with a live chord readout
Beat-synced metronome with tap tempo and nudge
Count-in before playback and before each loop repeat
Record your own take over the backing track
Line up recorded takes with a latency offset and per-take nudge
Stacked waveform with a colored lane per stem, click-to-seek, and loop shading
Import an MP3, WAV, or FLAC file, or a YouTube link
Export your mix, a single stem, or just the loop as WAV or MP3
Picks up where you left off: song, position, mix, loop, speed, and pitch
Library with search, repeat, shuffle, and autoplay
Keyboard shortcuts for the whole practice loop
Runs entirely on your PC: no account, no subscription, no uploads

---

## Search terms

stem splitter, vocal remover, backing track, slow down music, guitar practice, play along, stem separation

---

## Notes for future submissions

- Update `What's new` for every Store submission; keep the version
  number in the first line (Partner Center shows it verbatim).
- The Description avoids naming specific model versions (HTDemucs,
  MDX-Net): those change, and the Store copy should not need a rewrite
  when they do. Attribution for the models lives in the README.
- Do not claim GPU acceleration for 4/6-stem separation: only the
  2-stem path runs on the GPU today (see issue #125).
