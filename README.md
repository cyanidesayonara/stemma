# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, and play along with your instrument.

## Features

- AI-powered stem separation using HTDemucs v4 (4-stem and 6-stem models)
- GPU-accelerated inference via ONNX Runtime + DirectML
- Multi-track player with per-stem mute/solo controls
- Export individual stems or custom mixes as WAV/MP3
- Dark-themed Qt desktop interface
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

## Running Tests

```bash
pytest
```

## Project Documentation

- **PROJECT.md** -- Full technical spec, module descriptions, and roadmap
- **AGENTS.md** -- AI coding agent context (cross-tool standard)
- **CHANGELOG.md** -- Session-by-session development history

## License

MIT
