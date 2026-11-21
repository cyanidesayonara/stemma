# stemma

A Windows desktop music player with AI stem separation.

Import a song, separate it into stems (vocals, drums, bass, guitar, piano, other), mute/solo any stem, adjust volumes, and play along with your instrument.

Latest stable release: **v3.0.0**.

## Download

Website, with a playable mixer demo: https://cyanidesayonara.github.io/stemma/

**Microsoft Store (recommended):** [stemma on the Microsoft Store](https://apps.microsoft.com/detail/9p2w12l8f381)

**Portable zip:** Download **stemma.zip** from the [latest GitHub Release](https://github.com/cyanidesayonara/stemma/releases/latest), extract anywhere, and run `stemma.exe`. No installation required. ONNX models download automatically on first import.

> Windows SmartScreen may show a warning for unsigned executables. Click **More info** then **Run anyway**.

### Requirements

- Windows 10/11

## Features

### The practice cockpit

- A tall stacked waveform with one colored lane per stem, sharing the
  playhead and A-B loop region; muted stems dim in place.
- Practice controls grouped into three cards: Loop and Trainer, Speed and
  Pitch, Metronome and Count-in.
- Key, chord, and tempo in one readout strip under the waveform.
- Play, stop, record, and master volume anchored at the bottom while the
  practice content scrolls.

### Separation

- HTDemucs v4 four-stem (vocals, drums, bass, other) and six-stem (adds
  guitar and piano) separation. CPU-only for now; GPU support is research in
  [issue #125](https://github.com/cyanidesayonara/stemma/issues/125).
- MDX-Net two-stem separation (vocals + other) on the GPU via
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
  (plus or minus 7 semitones), applied live while the song plays. The key
  and chord readouts follow the transposition.
- Loop Trainer: with a loop set, speed steps up one preset on each repeat,
  from a chosen start speed up to 1.0x.
- Automatic tempo, key, beat, and live chord detection.
- A chord lane under the stem waveforms: the whole song's chords on the
  waveform's timeline, plus the next few bars large enough to read ahead
  while playing. Click a chord to jump to it; View > Chord Lane hides it.
- Song sections (experimental): a ruler above the stem waveforms splits
  the song into sections on its bar lines, with the same letter for
  sections that sound alike. Click a section to jump to it; double-click
  it, or press Shift+L, to loop it bar-accurately. View > Song Sections
  hides it.
- Metronome with tap tempo, beat sync to the track, and a nudge offset;
  optional count-in before playback and before each loop repeat.

### Playing along

- Record takes over the stems through your audio interface, at 1.0x speed
  and the original pitch. Takes appear as mixer rows and can be nudged to
  line up.
- Export the mix as you hear it (mute, solo, and volumes), a single stem,
  or just the loop region, as WAV or MP3, optionally with the count-in
  prepended.

### Everything else

- Library with search, metadata editing, repeat, shuffle, and autoplay.
- Session restore: last song, position, mixer, loop, speed, metronome,
  count-in, and takes.
- Keyboard shortcuts for nearly everything (below).
- Dark and light themes; choose the data folder and audio devices under
  **Edit > Preferences**.
- 100% local processing. No cloud, no account, no subscription.

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
| Space | Play / Pause (presses the focused button, if one has focus) |
| Enter | Press the focused button |
| Tab | Move keyboard focus between controls |
| S | Stop |
| Left / Right | Seek -/+ 5 seconds |
| Home / End | Jump to start / end |
| 0-9 | Jump to 0%–90% position |
| Up / Down | Master volume |
| Shift+Up / Down | Speed up / down |
| Shift+Left / Right | Transpose -/+ 1 semitone |
| Ctrl+1-6 | Mute/unmute vocals, drums, bass, other, guitar, piano |
| A / B | Set loop point A / B |
| L | Toggle A-B loop |
| Shift+L | Loop the song section playing (experimental) |
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
- [docs/website.md](docs/website.md): the GitHub Pages site and its deploy

## License

MIT

## Credits

- **HTDemucs v4** — Meta AI Research (MIT); ONNX export from
  [rysertio/Demucs-onnx](https://huggingface.co/rysertio/Demucs-onnx)
- **MDX-Net models** — trained by the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) project and its developers (MIT); thank you to UVR for making them available
- **beat_this** — beat/downbeat tracking model by the Institute of
  Computational Perception, JKU Linz (ISMIR 2024, MIT); ONNX export from
  [mosynthkey/beat_this_cpp](https://github.com/mosynthkey/beat_this_cpp)
