# stemma

A Windows desktop music player with AI stem separation. Import a song, separate it into stems, mute/solo individual instruments, play along.

**Personal-use tool. No cloud, no subscriptions, no command line needed.**

---

## Concept

1. Import a song (MP3/WAV/FLAC)
2. AI separates it into stems (vocals, drums, bass, guitar, piano, other)
3. Mute/solo any stem — play along with your instrument
4. Export stems or custom mixes

---

## Decisions

| Decision | Choice | Notes |
|---|---|---|
| **Name** | stemma (always lowercase) | |
| **Platform** | Windows desktop only | No mobile/web for now |
| **GPU** | NVIDIA RTX 4070 Ti | DirectML acceleration via ONNX Runtime |
| **Stems** | 4-stem and 6-stem models | Both available per song |
| **Distribution** | Single `.exe` via GitHub Releases | Models download on first run |
| **Python** | 3.14 | All deps confirmed compatible (March 2026) |

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| **Language** | Python 3.14 | Rich audio ecosystem |
| **GUI** | PySide6 (Qt 6) | Modern look, powerful widgets, LGPL |
| **Inference** | ONNX Runtime + DirectML | Lean (~50MB), GPU-accelerated, no PyTorch bloat |
| **Stem Models** | HTDemucs v4 ONNX | 4-stem + 6-stem, downloaded on first run (~80-300MB) |
| **Audio Playback** | `sounddevice` + `soundfile` | Low-latency multi-track mixing via NumPy |
| **Audio Processing** | `numpy` | Efficient buffer manipulation |
| **Export** | `soundfile` (WAV), `pydub` (MP3) | Individual stems or custom mix |
| **Packaging** | PyInstaller | Single `.exe` (~150-250MB without models) |
| **Future** | `yt-dlp`, `librosa`/`audiotsm` | YouTube import, tempo/pitch changes |

### Why HTDemucs v4?

- **Developer**: Meta AI Research — MIT license (fully open, free)
- **Quality**: State-of-the-art SDR (Signal-to-Distortion Ratio)
- **4-stem**: Vocals, Drums, Bass, Other
- **6-stem variant**: Adds Guitar + Piano (exactly what we need)
- **Speed**: ~30-60 seconds for a 4-min song on GPU; ~5-15 min on CPU
- **Model size**: ~80MB (quantized) to ~300MB (full)

### Why ONNX Runtime over PyTorch?

- PyTorch adds ~2GB+ to the installer — ONNX Runtime is ~50MB
- DirectML provider gives native GPU acceleration on Windows (no CUDA toolkit needed)
- Proven approach: Intel's OpenVINO Audacity plugin and others use the same pattern
- Models are converted once from PyTorch → ONNX format and hosted on HuggingFace

---

## Project Structure

```
stemma/
├── main.py                    # App entry point
├── requirements.txt
├── README.md
├── LICENSE                    # MIT
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── app.py                 # QApplication setup
│   ├── separator.py           # ONNX Runtime stem separation
│   ├── model_manager.py       # Download/cache ONNX models on first run
│   ├── player.py              # Multi-track audio player (sounddevice)
│   ├── library.py             # Song library (JSON-based)
│   ├── exporter.py            # Export stems as WAV/MP3
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py     # Main window layout
│       ├── player_controls.py # Transport + stem mute/solo
│       ├── library_panel.py   # Song list / playlist
│       ├── import_dialog.py   # Import songs dialog
│       └── styles.py          # Dark theme stylesheet
├── assets/
│   └── icons/
└── data/                      # Created at runtime
    ├── library.json
    ├── models/                # Downloaded ONNX models cached here
    └── songs/{song-id}/
        ├── original.mp3
        ├── vocals.wav
        ├── drums.wav
        ├── bass.wav
        ├── guitar.wav         # (6-stem only)
        ├── piano.wav          # (6-stem only)
        └── other.wav
```

---

## Module Specifications

### `separator.py` — Stem Separation Engine
- Loads HTDemucs ONNX model via `onnxruntime.InferenceSession`
- Uses DirectML execution provider for GPU acceleration
- Handles STFT/iSTFT pre/post-processing in NumPy (stripped from ONNX model)
- Runs in background `QThread`, emits progress signals
- Supports both `htdemucs` (4-stem) and `htdemucs_6s` (6-stem)

### `model_manager.py` — Model Download & Cache
- Checks if ONNX model files exist in `data/models/`
- Downloads from HuggingFace on first run (~80-300MB per model)
- Shows download progress in UI
- Manages both 4-stem and 6-stem model files

### `player.py` — Multi-Track Audio Player
- Loads stem WAVs as NumPy arrays
- `sounddevice.OutputStream` callback: reads buffers per stem, applies gain, sums to output
- API: `play()`, `pause()`, `stop()`, `seek()`, `set_stem_gain(stem, value)`
- Tracks playback position for UI sync

### `library.py` — Song Library
- JSON song index: `{id, title, artist, stems_path, model_used, date_added}`
- CRUD operations on the song list

### `exporter.py` — Stem Export
- Export individual stems or custom mix (with current mute/solo state) as WAV or MP3

### UI Modules
- **`main_window.py`** — Left panel: song library list. Center: player controls + stem mixer. Menu: File > Import / Export
- **`player_controls.py`** — Transport (Play/Pause/Stop + seek slider + time display). Per-stem row: label + Mute + Solo + volume slider. Color-coded stems (vocals=purple, drums=orange, bass=blue, guitar=red, piano=green, other=gray)
- **`styles.py`** — Dark theme, modern aesthetic, good contrast

---

## Implementation Phases

### Phase 1 — MVP
- [ ] Project setup (GitHub repo, deps, structure)
- [ ] Model manager (download ONNX models on first run)
- [ ] Stem separation engine (ONNX Runtime + DirectML)
- [ ] Multi-track player with mute/solo
- [ ] Song library (import, list, remove)
- [ ] Export stems as WAV
- [ ] UI: main window, player controls, library panel, import dialog
- [ ] Dark theme styling
- [ ] Manual testing & bug fixes

### Phase 2 — Polish
- [ ] MP3 export support
- [ ] Separation progress bar
- [ ] Keyboard shortcuts (Space=play/pause, M=mute, S=solo)
- [ ] Per-stem volume sliders
- [ ] Window state persistence
- [ ] Error handling & edge cases

### Phase 3 — Advanced
- [ ] YouTube URL import (yt-dlp)
- [ ] Tempo change (time-stretch)
- [ ] Key transposition (pitch-shift)
- [ ] Waveform visualization
- [ ] A-B loop repeat
- [ ] General music player features (shuffle, repeat, EQ)

---

## Reference Projects

| Project | Relevance |
|---|---|
| **OpenVINO Audacity Plugin** | Proves HTDemucs ONNX works; source code reference |
| **MISST** | Desktop stem separation + player GUI — closest to our vision |
| **Ultimate Vocal Remover (UVR5)** | Full-featured separator — complex but good reference |
| **Demucs Web** | Runs Demucs in browser via ONNX Runtime Web + WebGPU |
| **audio-separator** | Python CLI/library for stem separation |
| **deanturpin/stems** | C++ HTDemucs via ONNX Runtime — high-performance reference |

---

## Performance Expectations

| Scenario | Expected Speed (4-min song) |
|---|---|
| GPU (DirectML, RTX 4070 Ti) | ~30-60 seconds |
| CPU fallback | ~5-15 minutes |

## Audio Quality Notes

- Bass and drums separate cleanly (primary use case ✅)
- Guitar separation (6-stem) is good but not perfect — some bleed is normal
- For practice/busking purposes, quality is more than sufficient
