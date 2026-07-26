# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, adjust volumes, and play along with your instrument.

## Download

**Microsoft Store (recommended):** [stemma on the Microsoft Store](https://apps.microsoft.com/detail/9p2w12l8f381)

**Portable zip:** Download **stemma.zip** from the [latest GitHub Release](https://github.com/cyanidesayonara/stemma/releases/latest), extract anywhere, and run `stemma.exe`. No installation required. ONNX models download automatically on first import.

> Windows SmartScreen may show a warning for unsigned executables. Click **More info** then **Run anyway**.

### Requirements

- Windows 10/11
- NVIDIA GPU recommended (DirectML, falls back to CPU)

## Features

- AI-powered stem separation: HTDemucs v4 (4-stem and 6-stem, CPU) and MDX-Net 2-stem (vocals + backing) that runs on the GPU via DirectML — seconds instead of minutes
- ONNX Runtime inference with DirectML GPU acceleration where the model supports it, automatic CPU fallback otherwise (full multi-stem GPU is tracked in issue #125)
- Multi-track player with per-stem mute/solo/volume controls
- Audio post-processing: Wiener filter and soft gating for cleaner stems
- Real-time chord detection (major/minor) with Viterbi smoothing, updated 4×/s during playback
- Automatic tempo and key detection; beat-synced metronome mode
- High-accuracy beat/downbeat tracking via beat_this ONNX model (auto-downloaded)
- Import from YouTube URL (bundled ffmpeg when available; otherwise ffmpeg on PATH)
- Imports run in the background: the dialog closes as soon as separation starts, the library row shows live progress, and the app stays fully usable (multiple imports queue up)
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

## Development Setup

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
# Fast tests (~25 seconds, ~845 tests)
pytest

# Include ONNX inference tests (~20 seconds, needs model file)
set STEMMA_TEST_SONG=path\to\song.mp3
pytest -m slow

# Include hardware playback test (~30 seconds, needs speakers)
pytest -m hardware
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

- **PROJECT.md** -- Full technical spec, module descriptions, and roadmap
- **AGENTS.md** -- AI coding agent context (cross-tool standard)
- **CHANGELOG.md** -- Development history
- **docs/privacy-policy.md** -- Store privacy policy (Markdown)
- **docs/privacy-policy-plain.txt** -- Same policy as plain text (Partner Center paste)
- **assets/store_listing/** -- Store listing PNGs (poster/box: main + arpeggio SVGs; tiny icon: `icon_256.png`); see `scripts/generate_store_listing_assets.py`

## License

MIT

## Credits

- **HTDemucs v4** — Meta AI Research (MIT)
- **MDX-Net models** — trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project and its developers (MIT); thank you to UVR for making them available
- **beat_this** — beat/downbeat tracking model (ISMIR 2024, MIT)
