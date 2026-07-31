# Microsoft Store listing copy

Generated from `store/listing.yaml`. Edit the YAML, then run
`python scripts/build_store_listing.py` to regenerate this file.
Do not edit this markdown by hand.

This copy reflects listing content for version **2.6.0**.

Fields map to Partner Center as follows:

| Partner Center field | Section below |
|---|---|
| Description | [Description](#description) |
| What's new in this version | [What's new](#whats-new-in-this-version) |
| Product features (max 20, one per line) | [Product features](#product-features) |
| Short description | [Short description](#short-description) |
| Search terms | [Search terms](#search-terms) |

Assets: `assets/store_listing/` (regenerate with
`python scripts/generate_brand.py` then
`python scripts/generate_store_listing_assets.py`).
Screenshots: `assets/store_listing/screenshots/` (regenerate with
`python scripts/generate_screenshots.py`).

---

## Short description

Practice any song: split it into stems, mute the part you play, slow it
down, and loop the hard bars.

---

## Description

stemma turns any song into a practice tool.

Import a track and stemma separates it into individual stems -- vocals,
drums, bass, guitar, piano, and everything else -- so you can mute the
part you play and perform it yourself. Silence the guitar and it is your
guitar in the mix. Solo the drums to lock in with them. Pull the vocal
down and sing the line yourself.

Everything else is built around learning a part properly. Set an A-B
loop over the two bars that keep tripping you up and drill them. Slow
the passage down without the pitch dropping, or turn on the Loop
Trainer and let stemma step the speed up a notch on every repeat until
you are at full tempo. Transpose the whole song into a key that suits
your voice or instrument, up to seven semitones either way, and the
displayed key follows. Count yourself in, play along to the metronome,
and record your take against the backing to hear how it really sat.

stemma reads the song as you work: tempo, musical key, and the chord
under the playhead, updated as it plays. There is a
waveform to scrub, per-stem volume faders, and a library that remembers
exactly where you left off -- song, position, mix, loop, speed, and
pitch -- so practice picks up where it stopped.

Separation runs on your own machine. Nothing is uploaded, there is no
account, no subscription, and no internet connection needed once the
models are downloaded. Import from a file or paste a YouTube link, and
export any stem or your own custom mix as WAV or MP3 when you want to
take it elsewhere.

Built for Windows, keyboard-first, dark and light themes.

---

## What's new in this version

What's new in version 2.6.0

Faster 2-stem separation: when a compatible GPU is available, 2-stem
separation runs with GPU acceleration and falls back to CPU
automatically when it is not.

Background imports: importing and separating songs no longer blocks
the rest of the app -- jobs run in a background queue so you can keep
browsing your library while work finishes.

Stability and integrity: model downloads are checksum-verified,
stems load asynchronously so the UI stays responsive, and release
diagnostics make packaged builds easier to verify before Store
submission.

---

## Product features

AI stem separation: vocals, drums, bass, guitar, piano, other
Per-stem mute, solo, and volume faders
A-B loop for drilling a difficult passage
Loop Trainer: speed steps up automatically on every loop repeat
Pitch-preserving speed control from 50% to 200%
Transpose up or down seven semitones, tempo unchanged
Automatic tempo and key detection
Live chord readout that follows the playhead
Beat-synced metronome with tap tempo and nudge
Count-in before playback and before each loop repeat
Record your own take against the backing track
Manual timing offset for recorded takes
Waveform with click-to-seek, playback cursor, and loop markers
Export stems or a custom mix as WAV or MP3
Import from an audio file or a YouTube link
Session memory: song, position, mix, loop, speed, and pitch
Keyboard shortcuts for the whole practice loop
Runs entirely on your PC: no account, no subscription, no uploads

---

## Search terms

stem separation, vocal remover, play along, backing track, practice, karaoke, transcribe, slow down music, loop, metronome, tempo, key detection, chord detection, guitar practice, bass practice, drum practice, singer, music learning, isolate instrument, minus one

---

## Notes for future submissions

- Update `What's new` for every Store submission; keep the version
  number in the first line (Partner Center shows it verbatim).
- The Description avoids naming specific model versions (HTDemucs,
  MDX-Net): those change, and the Store copy should not need a rewrite
  when they do. Attribution for the models lives in the README.
- Do not claim GPU acceleration for 4/6-stem separation: only the
  2-stem path runs on the GPU today (see issue #125).
