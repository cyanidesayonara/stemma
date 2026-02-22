# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, adjust volumes, and play along with your instrument.

## Features

- AI-powered stem separation using HTDemucs v4 (4-stem and 6-stem models)
- GPU-accelerated inference via ONNX Runtime + DirectML
- Multi-track player with per-stem mute/solo/volume controls
- Audio post-processing: Wiener filter and soft gating for cleaner stems
- Export individual stems or custom mixes as WAV or MP3
- Keyboard shortcuts for transport and stem control
- Dark-themed Qt desktop interface with window state persistence
- 100% local processing -- no cloud, no subscriptions

## Requirements

- Windows 10/11
- Python 3.14
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
# Fast tests (~8 seconds)
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

## Project Documentation

- **PROJECT.md** -- Full technical spec, module descriptions, and roadmap
- **AGENTS.md** -- AI coding agent context (cross-tool standard)
- **CHANGELOG.md** -- Development history

## License

MIT
