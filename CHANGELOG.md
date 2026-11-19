# Changelog

Shipped releases are listed in reverse chronological order. The detail
behind each change lives in its pull request; implementation notes up to
August 2026 are archived in
[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

Release publication on GitHub is authoritative. Source code present on
`main` or a feature branch is not considered shipped until a release is
published.

## [v3.0.0](https://github.com/cyanidesayonara/stemma/releases/tag/v3.0.0) - 2026-11-19

- A new practice cockpit: a stacked waveform with one colored lane per stem
  (muted stems dim in place), the practice controls grouped into Loop and
  Trainer, Speed and Pitch, and Metronome and Count-in cards, and key, chord,
  and tempo in one readout strip.
- Speed and pitch apply live while the song plays, at the song's own level,
  instead of rendering first.
- While a song separates, the empty player shows its progress and time left.
- A new app icon, Store art, and stronger contrast in both themes; the whole
  app works from the keyboard, and controls have names for screen readers.
- Safer imports and downloads: files are checked before import, MP3s decode
  in full (header-less VBR files used to import truncated), every model
  download asks first (the beat model included), and a damaged model is
  replaced. An offline beat-model failure no longer stalls each song load.
- Export Mix exports the stems you hear, at the original speed and pitch.
- A damaged library file no longer removes songs, an interrupted import is
  cleaned up on the next launch, quitting during a separation asks first,
  and errors are readable, with details in a log file.
- Recording without a microphone turns recording off and keeps playing.
- Fixes: rapid loop-point changes no longer crash, the chord readout follows
  the pitch shift, the synced metronome keeps the real tempo, and a YouTube
  download that fails once is retried.
- A website with a playable mixer demo:
  https://cyanidesayonara.github.io/stemma/

## [v2.6.0](https://github.com/cyanidesayonara/stemma/releases/tag/v2.6.0) - 2026-07-31

- Added DirectML-capable MDX two-stem separation with automatic CPU fallback.
- Moved imports and separation onto a background queue so the UI stays usable.
- Hardened model download checksums, async stem loading, lifecycle safety, and
  packaged release diagnostics.

## [v2.5.0](https://github.com/cyanidesayonara/stemma/releases/tag/v2.5.0) - 2026-07-26

- Added Loop Trainer speed ramps for A-B loop practice.
- Added pitch shift and key transposition.
- Shipped the v2.4.1 recording, callback, export, download, library, and
  lifecycle stability pass.

## [v2.3.0](https://github.com/cyanidesayonara/stemma/releases/tag/v2.3.0) - 2026-04-17

- Added library playback controls, master volume, revised keyboard
  shortcuts, and UI/session polish.

## [v2.2.0](https://github.com/cyanidesayonara/stemma/releases/tag/v2.2.0) - 2026-04-14

- Added real-time major/minor chord display and beat_this beat/downbeat
  analysis.
- Improved worker lifecycle and light-theme detection badges.

## [v2.1.0](https://github.com/cyanidesayonara/stemma/releases/tag/v2.1.0) - 2026-04-03

- Added metronome beat-sync nudge, recording-session state, and transport,
  library, and volume-control polish.

## [v2.0.5](https://github.com/cyanidesayonara/stemma/releases/tag/v2.0.5) - 2026-04-02

- Added automatic tempo/key detection and beat-synced metronome support.

## [v2.0.4](https://github.com/cyanidesayonara/stemma/releases/tag/v2.0.4) - 2026-03-31

- Shipped UI, theme, mixer-state, branding, and Store-asset polish.

## [v2.0.3](https://github.com/cyanidesayonara/stemma/releases/tag/v2.0.3) - 2026-03-30

- Reduced separation peak memory use for Store certification.
- Added preflight memory checks and clearer ONNX out-of-memory errors.

## [v2.0.2](https://github.com/cyanidesayonara/stemma/releases/tag/v2.0.2) - 2026-03-27

- Added clean ONNX session fallback when DirectML initialization fails.
- Improved ONNX and memory-related import errors.

## [v2.0.1](https://github.com/cyanidesayonara/stemma/releases/tag/v2.0.1) - 2026-03-27

- Fixed splash/logo timing and Microsoft Store audio behavior.
- Added Store listing assets.

## [v2.0.0](https://github.com/cyanidesayonara/stemma/releases/tag/v2.0.0) - 2026-03-27

- Added the redesigned interface, per-stem waveforms, recording alignment,
  export options, animated startup, and MSIX distribution.

## [v1.2.0](https://github.com/cyanidesayonara/stemma/releases/tag/v1.2.0) - 2026-03-26

- Added full-duplex play-along recording, multiple takes, input-device
  selection, and manual timing offset.

## [v1.1.0](https://github.com/cyanidesayonara/stemma/releases/tag/v1.1.0) - 2026-03-26

- Added session persistence, metronome, count-in, loop-aware transport, and
  the keyboard-shortcuts dialog.

## [v1.0.0](https://github.com/cyanidesayonara/stemma/releases/tag/v1.0.0) - 2026-03-24

- Shipped the initial Windows desktop release with local ONNX stem
  separation, multi-track playback, library, export, YouTube import,
  waveform, practice controls, themes, preferences, and portable packaging.
