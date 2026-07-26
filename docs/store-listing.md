# Microsoft Store listing copy

Paste-ready text for Partner Center. Keep this file in sync when
features ship -- it is the source of truth for the listing, so the
listing never drifts the way it did between v2.3.0 and v2.5.0.

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

stemma reads the song as you work: tempo, musical key, time signature,
and the chord under the playhead, updated as it plays. There is a
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

What's new in version 2.5.0

Loop Trainer: with an A-B loop running, stemma can raise the playback
speed one step on every repeat -- start at 75% and work up to full
tempo without touching the controls.

Pitch transposition: shift a whole song up or down by up to seven
semitones with the tempo unchanged. The key readout shows both the
original and the transposed key, and recordings can either follow the
shift or stay at their own pitch.

Faster, smoother imports: a new 2-stem mode splits vocals from the
backing track in seconds using your GPU, and separation now runs in the
background -- the import window closes immediately and the song shows
its progress in the library while you keep playing something else.

Reliability: this release fixes a number of issues found in a full
audit, including recorded takes that could play back silent, audio
dropouts when deleting a take or clearing a loop mid-playback, the
chord readout being wrong at reduced speed, export problems when
exporting a loop region at a slower speed, and several cases where
closing the app during a long task could interrupt it uncleanly. Model
downloads now verify themselves and resume cleanly if interrupted.

---

## Product features

AI stem separation: vocals, drums, bass, guitar, piano, other
2-stem fast mode: vocals and backing track in seconds, GPU accelerated
Background separation: keep using the app while songs process
Per-stem mute, solo, and volume faders
A-B loop for drilling a difficult passage
Loop Trainer: speed steps up automatically on every loop repeat
Pitch-preserving speed control from 50% to 200%
Transpose up or down seven semitones, tempo unchanged
Automatic tempo, key, and time-signature detection
Live chord readout that follows the playhead
Beat-synced metronome with tap tempo and nudge
Count-in before playback and before each loop repeat
Record your own take against the backing track
Latency compensation for recorded takes
Waveform with click-to-seek, playback cursor, and loop markers
Export stems or a custom mix as WAV or MP3
Import from an audio file or a YouTube link
Session memory: song, position, mix, loop, speed, and pitch
Keyboard shortcuts for the whole practice loop
Runs entirely on your PC: no account, no subscription, no uploads

---

## Search terms

stem separation, vocal remover, play along, backing track, practice,
karaoke, transcribe, slow down music, loop, metronome, tempo, key
detection, chord detection, guitar practice, bass practice, drum
practice, singer, music learning, isolate instrument, minus one

---

## Notes for future submissions

- Update `What's new` for every Store submission; keep the version
  number in the first line (Partner Center shows it verbatim).
- The Description avoids naming specific model versions (HTDemucs,
  MDX-Net): those change, and the Store copy should not need a rewrite
  when they do. Attribution for the models lives in the README.
- Do not claim GPU acceleration for 4/6-stem separation: only the
  2-stem path runs on the GPU today (see issue #125).
