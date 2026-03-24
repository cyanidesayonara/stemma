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
- A-B loop for practice sections; pitch-preserving playback speed presets
- Keyboard shortcuts for transport, stems, loop, and speed
- Dark / light Qt themes; window state persistence; configurable data folder and audio device (Edit > Preferences)
- 100% local processing -- no cloud, no subscriptions

## Requirements

- Windows 10/11
- Python 3.14
- NVIDIA GPU recommended (DirectML, falls back to CPU)
- ffmpeg on PATH only if the bundled binary (imageio-ffmpeg) is unavailable (YouTube import)

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
# Fast tests (~10 seconds, ~286 tests)
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

## Project Documentation

- **PROJECT.md** -- Full technical spec, module descriptions, and roadmap
- **AGENTS.md** -- AI coding agent context (cross-tool standard)
- **CHANGELOG.md** -- Development history

## License

MIT
