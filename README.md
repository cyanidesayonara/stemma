# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, adjust volumes, and play along with your instrument.

## Features

- AI-powered stem separation using HTDemucs v4 (4-stem and 6-stem models)
- GPU-accelerated inference via ONNX Runtime + DirectML
- Multi-track player with per-stem mute/solo/volume controls
- Audio post-processing: Wiener filter and soft gating for cleaner stems
- Import from YouTube URL (bundled ffmpeg when available; otherwise ffmpeg on PATH)
- Clear errors and progress when ONNX models download on first use; large-file warning before heavy imports
- Export individual stems or custom mixes as WAV or MP3
- Waveform visualization with click-to-seek, playback cursor, and loop markers
- A-B loop for practice sections (Stop returns to loop A while looping; seek stays inside the loop); pitch-preserving playback speed presets
- Metronome with BPM entry, tap tempo, mute/solo-friendly click track; automatic tempo/key detection with beat-synced mode
- Beat-sync nudge: shift metronome clicks ±500ms relative to detected beat positions to align the click with any song
- Optional count-in beats before playback (and optionally before each loop repeat)
- Session persistence: restore last song, position, mixer, loop, speed, metronome, count-in, and recording take state after restart
- Library panel shows artist and title on separate lines with teal selection highlight
- Keyboard shortcuts for transport, stems, loop, speed, metronome, and count-in; full list under **Help > Keyboard Shortcuts**
- Dark / light Qt themes; window geometry/state persistence; configurable data folder and audio device (Edit > Preferences)
- 100% local processing -- no cloud, no subscriptions

## Download

Grab **stemma.exe** from the [latest GitHub Release](https://github.com/cyanidesayonara/stemma/releases/latest). No installation required -- just run the .exe. ONNX models download automatically on first import.

Windows SmartScreen may show a warning for unsigned executables. Click **More info** then **Run anyway**.

### Requirements

- Windows 10/11
- NVIDIA GPU recommended (DirectML, falls back to CPU)

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
# Fast tests (~10 seconds, ~360 tests)
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
| 1-6 | Toggle mute on stem |
| A | Set loop start point |
| B | Set loop end point |
| L | Toggle A-B loop |
| [ / ] | Slower / faster playback speed |
| M | Toggle metronome |
| C | Toggle count-in |

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
