# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, adjust volumes, and play along with your instrument.

Latest stable release: **v2.6.0**. The current `main` line targets
**v3.0**, which is not released yet.

## Download

**Microsoft Store (recommended):** [stemma on the Microsoft Store](https://apps.microsoft.com/detail/9p2w12l8f381)

**Portable zip:** Download **stemma.zip** from the [latest GitHub Release](https://github.com/cyanidesayonara/stemma/releases/latest), extract anywhere, and run `stemma.exe`. No installation required. ONNX models download automatically on first import.

> Windows SmartScreen may show a warning for unsigned executables. Click **More info** then **Run anyway**.

### Requirements

- Windows 10/11

## Features

### Separation

- HTDemucs v4 four-stem (vocals, drums, bass, other) and six-stem (adds
  guitar and piano) separation. CPU-only for now; GPU support is research in
  [issue #125](https://github.com/cyanidesayonara/stemma/issues/125).
- MDX-Net two-stem separation (vocals + backing) on the GPU via
  DirectML, with automatic CPU fallback and a clear report of which ran.
- Imports run in the background, one after another, with progress in the
  library. Import from a file, by drag and drop, or from a YouTube URL.
- Post-processing (Wiener filtering and soft gating) to reduce bleed
  between stems.
- ONNX Runtime inference, no PyTorch. Models download on first use and are
  checksum-verified.

### Practice

- Per-stem mute, solo, and volume.
- A-B loop, pitch-preserving speed presets, and pitch transposition
  (plus or minus 7 semitones). The Key readout follows the transposition.
- Loop Trainer: with a loop set, speed steps up one preset on each repeat,
  from a chosen start speed up to 1.0x.
- Automatic tempo, key, beat, and live chord detection.
- Metronome with tap tempo, beat sync to the track, and a nudge offset;
  optional count-in before playback and before each loop repeat.

### Playing along

- Record takes over the stems through your audio interface. Takes appear
  as mixer rows and can be nudged to line up.
- Export individual stems, a custom mix, or just the loop region, as WAV or
  MP3, optionally with the count-in prepended.

### Everything else

- Library with search, metadata editing, repeat, shuffle, and autoplay.
- Session restore: last song, position, mixer, loop, speed, metronome,
  count-in, and takes.
- Keyboard shortcuts for nearly everything (below).
- Dark and light themes; choose the data folder and audio devices under
  **Edit > Preferences**.
- 100% local processing. No cloud, no account, no subscription.

## v3.0 Target (Unreleased)

The current source tree includes the v3.0 practice cockpit. It is not part
of the stable v2.6.0 Store or portable downloads yet:

- A tall stacked waveform with one colored lane per stem, sharing the
  playhead and A-B loop region; muted stems dim in place.
- Practice controls grouped into three cards: Loop and Trainer, Speed and
  Pitch, Metronome and Count-in.
- Key, chord, and tempo in one readout strip under the waveform.
- Play, stop, record, and master volume anchored at the bottom while the
  practice content scrolls.

## Development

Setup, validation, packaging, and the contribution workflow are in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). The short version, in
PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python main.py
```

Run the fast test suite:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -m "not slow and not hardware"
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| S | Stop |
| Left / Right | Seek -/+ 5 seconds |
| Home / End | Jump to start / end |
| 0-9 | Jump to 0%–90% position |
| Up / Down | Master volume |
| Shift+Up / Down | Speed up / down |
| Shift+Left / Right | Transpose -/+ 1 semitone |
| Ctrl+1-6 | Toggle mute on stem |
| A / B | Set loop point A / B |
| L | Toggle A-B loop |
| M | Toggle metronome |
| C | Toggle count-in |
| R | Arm / disarm recording |
| N / P | Next / previous song |
| F1 | Keyboard shortcuts dialog |

Use **Help > Keyboard Shortcuts** in the app for the authoritative list (same bindings as above).

## Project Documentation

- [PROJECT.md](PROJECT.md): architecture and technical reference
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): setup, testing, diagnostics,
  packaging, and contribution workflow
- [docs/ROADMAP.md](docs/ROADMAP.md): short roadmap backed by GitHub issues
- [CHANGELOG.md](CHANGELOG.md): shipped release notes
- [AGENTS.md](AGENTS.md): guidance for coding agents
- [docs/store-release-pipeline.md](docs/store-release-pipeline.md): release
  and Microsoft Store operations
- [docs/privacy-policy.md](docs/privacy-policy.md): Store privacy policy

## License

MIT

## Credits

- **HTDemucs v4** — Meta AI Research (MIT)
- **MDX-Net models** — trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project and its developers (MIT); thank you to UVR for making them available
- **beat_this** — beat/downbeat tracking model (ISMIR 2024, MIT)
