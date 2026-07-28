# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, adjust volumes, and play along with your instrument.

Latest stable release: **v2.5.0**. The current `main` line targets
**v2.6.0**, which is not released yet.

## Download

**Microsoft Store (recommended):** [stemma on the Microsoft Store](https://apps.microsoft.com/detail/9p2w12l8f381)

**Portable zip:** Download **stemma.zip** from the [latest GitHub Release](https://github.com/cyanidesayonara/stemma/releases/latest), extract anywhere, and run `stemma.exe`. No installation required. ONNX models download automatically on first import.

> Windows SmartScreen may show a warning for unsigned executables. Click **More info** then **Run anyway**.

### Requirements

- Windows 10/11

## Features

- AI-powered HTDemucs v4 stem separation (4-stem and 6-stem, CPU-only)
- ONNX Runtime inference without PyTorch; DirectML support for HTDemucs
  four/six-stem remains research in
  [issue #125](https://github.com/cyanidesayonara/stemma/issues/125)
- Multi-track player with per-stem mute/solo/volume controls
- Audio post-processing pipeline with Wiener filtering and soft gating
- Real-time chord detection (major/minor) with Viterbi smoothing, updated 4×/s during playback
- Automatic tempo and key detection; beat-synced metronome mode
- Beat/downbeat tracking via the auto-downloaded beat_this ONNX model
- Import from YouTube URL (bundled ffmpeg when available; otherwise ffmpeg on PATH)
- Clear errors and progress when ONNX models download on first use; large-file warning before heavy imports
- Export individual stems or custom mixes as WAV or MP3
- Waveform visualization with click-to-seek, playback cursor, and loop markers
- A-B loop for practice sections (Stop returns to loop A while looping; seek stays inside the loop); pitch-preserving playback speed presets
- Pitch transposition (±7 semitones), rendered in a single pass with the speed change; the Key badge shows the transposed key
- Loop Trainer: with an A-B loop active, playback speed steps up one preset each repeat, from a chosen start speed up to 1.0x — learn a passage slow and work it up to tempo hands-free
- Metronome with BPM entry, tap tempo, and beat-sync nudge (±500ms)
- Optional count-in beats before playback (and optionally before each loop repeat)
- Session persistence: restore last song, position, mixer, loop, speed, metronome, count-in, and recording take state after restart
- Library panel shows artist and title on separate lines with teal selection highlight
- Keyboard shortcuts for transport, stems, loop, speed, pitch, metronome, count-in, and recording; full list under **Help > Keyboard Shortcuts**
- Dark / light Qt themes; window geometry/state persistence; configurable data folder and audio device (Edit > Preferences)
- 100% local processing -- no cloud, no subscriptions

## v2.6.0 Target (Unreleased)

The current source tree includes these v2.6.0 target features. They are not
part of the stable v2.5.0 Store or portable downloads yet:

- MDX-Net two-stem separation (vocals + backing) requests DirectML when
  available and explicitly reports whether it selected DirectML GPU or CPU
  fallback. HTDemucs four/six-stem remains CPU-only.
- Imports run in the background: the dialog closes after separation is
  queued, the library shows progress, and multiple imports run serially.

## Development Setup

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the complete setup,
dependency-lock, validation, diagnostics, packaging, and contribution
workflow.

```bash
git clone https://github.com/cyanidesayonara/stemma.git
cd stemma
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Running Tests

```bash
python -m pytest -m "not slow and not hardware"
```

Slow-model and hardware commands are documented in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#testing).

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

- [PROJECT.md](PROJECT.md) -- architecture and technical reference
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) -- setup, testing, lint,
  diagnostics, packaging, and contribution workflow
- [docs/ROADMAP.md](docs/ROADMAP.md) -- short release roadmap backed by
  live GitHub issues
- [CHANGELOG.md](CHANGELOG.md) -- shipped release notes
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) -- historical
  development-session record
- [AGENTS.md](AGENTS.md) -- binding guidance for coding agents
- [docs/store-listing.md](docs/store-listing.md) and
  [docs/store-release-pipeline.md](docs/store-release-pipeline.md) --
  Store copy and release operations
- [docs/privacy-policy.md](docs/privacy-policy.md) -- Store privacy policy

## License

MIT

## Credits

- **HTDemucs v4** — Meta AI Research (MIT)
- **MDX-Net models** — trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project and its developers (MIT); thank you to UVR for making them available
- **beat_this** — beat/downbeat tracking model (ISMIR 2024, MIT)
