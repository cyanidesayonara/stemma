# Development log

Historical development-session and implementation notes are preserved here
in reverse chronological order. These notes describe work performed; they do
not establish release status. See `../CHANGELOG.md` and GitHub Releases for
shipped versions.

---

## 2026-07-17 -- Unreleased v2.6.0 MDX-Net 2-stem GPU separation and background imports

### Done
- **Background separation queue:** importing no longer blocks the app. After picking the file/model (and any first-time model download), the import dialog closes immediately and separation runs in a background queue. The library row shows live "Queued... / Separating... N%" progress, stays unselectable until its stems exist, and offers a right-click "Cancel separation". Failures roll the row back with a message; songs interrupted by an app crash are pruned on the next launch. Multiple imports queue serially.
- **Second separation engine:** UVR's MDX-Net (Inst HQ 3) splits a song into vocals + backing. Unlike the HTDemucs export, MDX ONNX compiles on DirectML, so this path genuinely uses the GPU -- verified on an RTX 4070 Ti at 73 ms/chunk (DML) vs 3.3 s (CPU), ~45x. A 72 s song separates in ~26 s end-to-end including one-time session setup; HTDemucs took minutes.
- **New import option:** "2-stem fast (vocals + backing, GPU)" in the import dialog's model picker. Output maps to `vocals.wav` + `other.wav`, so the player, mixer, export, and recording features work unchanged.
- **Engine details:** numpy/librosa STFT packing matching UVR's torch.stft conventions, classic context-trimmed windowing, DML to CPU fallback, secondary stem derived as mix minus primary. No PyTorch. Cross-validated against HTDemucs stems of a real song: vocal-stem correlation 0.90.
- **Model download integrity:** the MDX registry carries UVR's published tail hash; ModelDownloader verifies it after the transfer and deletes/reports a mismatched file instead of caching it.
- Model weights by the Ultimate Vocal Remover project (MIT) -- credit to UVR and its developers (see README Credits).

### Metrics recorded at the time
- 847 fast tests passed. Ten engine tests were added, including an identity-model reconstruction proof of the STFT/window math and a gated real-model integration test.

---

## 2026-07-09 -- v2.5.0 Loop Trainer

### Done
- **Loop Trainer:** New "Loop Trainer" toggle in the transport area. With an A-B loop active, it steps playback speed up one preset each time the loop repeats -- from a chosen start speed (0.5x / 0.75x / 0.85x) up to 1.0x, then holds -- so a passage can be learned slow and worked up to tempo hands-free. Start speed and enabled state persist across restarts (reset per song).
- **Player loop-wrap signal:** `MultiTrackPlayer` now exposes a `loop_wrapped` signal. A realtime-safe wrap counter is incremented in the audio callback and surfaced from the GUI-thread position timer, so loop-driven UI logic (the trainer) never runs on the audio thread. Reuses the existing render serialization, so rapid ramps do not pile up librosa workers.

### Metrics recorded at the time
- 846 fast tests passed, one skipped. Thirteen tests covered wrap counting/signals, ramp progression and cap, reset-on-load, and session round-trip.

---

## 2026-07-09 -- v2.4.1 Stability Pass

A returning-from-dormancy audit (deep review of the whole project plus PR #124) turned up a batch of correctness bugs; this session fixed them and restored green CI.

### Player and audio callback
- **Silent recording takes (critical):** `add_recording_stem` never invalidated the active-stems cache, so a freshly recorded take mixed at gain 0 -- visible in the mixer but inaudible -- until a mute/solo toggle. The cache is now invalidated on add/remove.
- **Callback vs. GUI-thread races:** recording-stem add/remove now replaces the stems dict copy-on-write (deleting a take mid-playback used to abort the stream with `dictionary changed size`); the callback snapshots the stems dict and loop frames once per block so `clear_loop` cannot null a frame mid-read.
- **`chord_at` speed mapping:** inverted the stretch factor -- at 0.5x the chord badge read from the wrong playhead time. Fixed and tested.
- **Render desync (from #124 follow-up):** a render discarded by `cancel_stretch` (spinbox scrubbing) could leave the knobs claiming a speed/pitch the audio did not have; the player now tracks applied render state and re-renders when stale.

### Recording lifecycle
- Empty takes (stop during count-in) no longer save a full-length silent WAV that consumes a take slot.
- Speed/pitch changes are refused while recording because they rescale the playhead under the live input stream.
- The take limit and speed/pitch guards can no longer be bypassed via the `R` shortcut; opening a song already at the take limit disables recording.
- Close Song and removing the loaded song now fully unload the player (`unload()`) instead of leaving it playing invisibly.

### Export
- Refuse a second export while one runs (double-start dropped the last reference to a live QThread and crashed Qt).
- Loop-region export maps stretched-timeline loop points back to original time.
- The count-in export option appears only when count-in is enabled.

### Robustness
- **Model downloads** stream to a `.partial` file, verify the byte count, and rename into place atomically, with a socket timeout.
- **Library** recovery preserves a corrupt index as `library.json.bak` and rebuilds from `songs/` instead of destroying it; `_load` handles `OSError` and non-UTF-8; writes fsync before rename.
- Detection/peak workers are drained on close (`PlayerControls.shutdown`).
- Re-detection badges clear on failure instead of sticking on "detecting...".

### CI and packaging
- Tests run under Qt's offscreen platform. Player fixtures join render threads at teardown and conftest collects garbage at module boundaries.
- Updated checkout/setup-python actions.
- Dropped unused `pydub`; split pytest dependencies into `requirements-dev.txt`.

### Investigations
- **DirectML GPU separation (#125):** verified on an RTX 4070 Ti that both HTDemucs ONNX exports fail DML kernel compilation and fall back to CPU. Kept plain `onnxruntime` at that time and filed #125 with diagnostics for a future model re-export.

### Metrics recorded at the time
- 825 fast tests passed (slow/hardware deselected) on local Python 3.14 and a CI-parity Python 3.12 environment.

---

## 2026-04-18 -- v2.4.0 Pitch Shift / Transposition

### Done
- **Pitch transposition:** New Pitch spinbox in the transport bar (plus or minus seven semitones). Uses `librosa.effects.pitch_shift` for pitch-preserving transposition. `Shift+Left`/`Shift+Right` shortcuts nudge by one semitone.
- **Unified stretch worker:** `SpeedWorker` replaced by `StretchWorker`, which applies pitch shift and time stretch in a single pass per channel from the original buffers, avoiding artifact compounding. Identity state is a fast-path no-op.
- **Effective key display:** When pitch is non-zero, the Key badge shows detected to transposed key. `transpose_key()` handles sharp/flat alias parsing and wrap-around.
- **Recording guard:** Recording cannot be armed while pitch is non-zero.
- **Sync-recording-pitch preference:** New preference to pitch-shift recording takes with stems.
- **Per-song session persistence:** Pitch saves/restores alongside speed.
- **Auto-reset on song switch:** Loading a new song resets pitch before applying saved song state.

### Metrics recorded at the time
- 708 tests passed, one skipped. Thirty-four tests covered pitch clamping/signals, render ordering, recording takes, and key transposition.

---

## 2026-04-17 -- v2.3.0 Library, Polish and Shortcuts

### Done
- **Library playback controls:** Repeat (off/all/one), shuffle, previous/next, autoplay, and now-playing indicator.
- **Keyboard shortcuts overhaul (#119):** layout-independent YouTube-style controls, position jumps, master volume, seek, speed, stem mute, next/previous, and shortcuts dialog.
- **Master volume:** Added gain multiplier persisted in session.
- **UI polish pass:** Centralized QSS object-name styling and reduced hardcoded styles.
- **Teal contrast fix:** Added a near-black foreground token for controls on teal accent.
- **Shortcuts dialog refresh:** Grouped categories with section headers and dividers.
- **Session persistence:** Repeat mode, shuffle, and master volume save/restore.

### Metrics recorded at the time
- 670 tests passed, five skipped.

---

## 2026-04-14 -- v2.2.0 Chord Detection and beat_this ONNX Model

### Done
- **Chord detection:** Real-time major/minor chord display using chromagram templates, Viterbi smoothing, silence gating, and playback-position lookup.
- **beat_this ONNX model:** Auto-download, chunked inference, and overlap handling for beat/downbeat tracking with librosa fallback.
- **Detection badge UI:** Key, tempo, and chord badges with confidence colors and theme regeneration.
- **QThread crash fix:** Worker orphaning with identity-checked finish callbacks.
- **Light mode readability:** Explicit badge foreground colors.
- **MSIX logo sound fix:** Windows in-memory sound fallback for Store sandbox behavior.
- **Session backward compatibility:** Schema migration forces re-detection for older chord data.
- **Downbeat sensitivity:** Adjusted downbeat peak-picking threshold.

### Metrics recorded at the time
- 625 tests passed, five skipped.

---

## 2026-04-03 -- v2.1.0 Metronome Nudge, UI Polish, Library Improvements and Session Persistence

### Done
- **Metronome beat-sync nudge:** Added plus/minus 500 ms offset for click sources.
- **Count-in controls in transport bar:** Moved count-in controls to the transport.
- **Live volume combos:** Editable percentage combos synchronized with sliders.
- **Icon button fix:** Corrected collapsed metronome/count-in buttons.
- **Speed combo alignment:** Aligned speed control to the transport edge.
- **Library two-row display:** Artist/title rows, separators, and teal selection.
- **Recording session persistence:** Nudge, mute, solo, and volume restore per take.

### Metrics recorded at the time
- 600 fast tests passed, five skipped.

---

## 2026-04-02 -- v2.0.5 Automatic BPM and Key Detection

### Done
- **Beat-synced metronome:** Sync toggle locks clicks to detected beat positions and carries click tails across callback blocks.
- **BPM/key detection:** Background beat_this/librosa and Krumhansl-Schmuckler analysis.
- **Suggestion-only display:** Read-only detected values with confidence colors.
- **Per-song caching:** Detection result caching in session settings.
- **A-B region detection:** Loop-region re-analysis.
- **Double-click to re-detect:** Per-value refresh.
- **Logo sound fix (Microsoft Store):** Windows playback path.
- **Menu padding:** Improved menu spacing.

### Metrics recorded at the time
- 31 new detector tests; 580 total passed, five skipped.

---

## 2026-03-31 -- v2.0.4 UI Polish and Store Alignment

### Done
- Standardized mute/solo button styling.
- Replaced ambiguous metronome/count-in icons.
- Added synchronized metronome volume presets.
- Preserved stem mute/solo state across song changes.
- Refined logo baselines and tooltips.
- Regenerated Store listing assets.

### Metrics recorded at the time
- 549 fast tests passed, five skipped.

---

## 2026-03-30 -- Low-RAM Stem Separation

### Done
- **In-place post-processing:** Wiener filter and soft gate write into input arrays.
- **Early resource release:** Releases ORT session and source audio before post-processing.
- **Pre-flight memory check:** Warns when estimated separation memory exceeds available memory.
- **ORT error messages:** Distinguishes allocation errors from generic initialization failures.

### Metrics recorded at the time
- Peak memory for a four-minute six-stem song was measured as reduced by about 1.3-1.6 GB in that environment.

---

## 2026-03-28 -- ONNX External Data Download for HuggingFace Models

### Done
- **Model cache:** Downloads ONNX external weights and requires all artifacts.
- **Library metadata:** Stores the actual selected HTDemucs model.

---

## 2026-03-28 -- Release Pipeline Housekeeping

### Done
- **Tag-driven versions:** Release workflow synchronizes source and manifest versions from the tag.
- **Tests before release build:** Release workflow runs the fast CI slice.
- **CI on tags:** CI runs for `v*` tag pushes.
- **Qt helper:** Added shared `safe_disconnect`.
- **Docs:** Added Store release pipeline documentation and optional Partner Center workflow.
- **Roadmap:** Synced then-current post-2.0 work.

---

## 2026-03-27 -- v2.0.2 ONNX DirectML Fallback for Store Certification

### Done
- **ONNX CPU fallback:** Session creation retries on CPU when DirectML fails.
- **ONNX error messages:** Added user-readable ONNX and memory errors.
- **MSIX manifest:** Updated package version for the release.

---

## 2026-03-27 -- v2.0.1 Splash and Store Listing

### Done
- **Splash screen:** Preserved animation visibility and sound synchronization when startup blocks the event loop.
- **Store listing PNGs:** Regenerated poster and box assets with refined spacing.

---

## 2026-03-26 -- Animated Logos and Splash Synchronization (#76)

### Done
- Deferred splash sound to a rendered frame and restarted timing after blocked startup.
- Added animated main chord logo and footer arpeggio logo.
- Added synthesized chord and arpeggio WAV assets.
- Added click replay behavior respecting the startup-sound preference.
- Reused animated branding in the About dialog.
- Triggered the main animation from `showEvent`.
- Removed obsolete SVG-rendering imports.

### Earlier work in the same pull request
- Added the splash widget and letter animation.
- Restructured startup so the splash appears before heavy imports.
- Added the generated startup sound and preference.
- Bundled audio assets in PyInstaller.

### Metrics recorded at the time
- 488 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-26 -- v1.2.0 Release

Shipped as GitHub Release **v1.2.0**:

- Full-duplex play-along recording with multiple takes (#79, PR #91).
- Input-device selection and manual timing offset.
- Recordings in mix export.
- Count-in at any play position.

## 2026-03-26 -- Record Audio Track (#79, PR #91)

### Done
- Added synchronized full-duplex recording.
- Saved recordings as mixer stems with mute/solo/volume.
- Added record controls, input-device settings, and multiple takes.
- Supported loop recording and mix export.
- Added recording deletion and discovery on song load.
- Hardened mono input, device selection, finalization, sample-rate checks, take numbering, and count-in behavior.

### Metrics recorded at the time
- 395 fast tests, five slow ONNX tests, and one hardware test existed.
- Thirty-three recording tests were added.

---

## 2026-03-26 -- v1.1.0 Release

Shipped as GitHub Release **v1.1.0**:

- Metronome (#57, PR #86).
- Count-in (#78, PR #87).
- Session persistence (#55, PR #85).
- Loop-aware Stop and seek.
- Keyboard shortcuts dialog.

## 2026-03-26 -- Metronome and Count-In (#57, #78)

### Done
- Added BPM, tap tempo, click mixing, toggle, and volume.
- Added optional pre-roll and loop-repeat count-in.
- Added loop-aware Stop/seek behavior and persisted settings.

### Metrics recorded at the time
- 362 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-25 -- Session Persistence (#55, PR #85)

### Done
- Added save/restore for current song, position, stem state, loop, and speed.
- Added deferred restore and safe missing-song handling.
- Added programmatic library selection and mixer/loop state helpers.

### Metrics recorded at the time
- 286 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-24 -- v1.0.0 Release and Packaging (#56, PR #83)

### Done
- Added frozen-build root resolution.
- Added PyInstaller one-folder packaging and dependency collection.
- Added tag-triggered GitHub Release build.
- Added development packaging dependencies.
- Added frozen asset paths.
- Set the first stable version.

### Metrics recorded at the time
- The local executable measured 202 MB.
- 282 fast tests, five slow ONNX tests, and one hardware test existed.
- All v1.0 tickets were closed.

---

## 2026-03-24 -- Error Handling and Model Download UX (#73, PR #82)

### Done
- Added model-download progress and user-facing import errors.
- Added retry and rollback for failed imports.
- Added large-file confirmation.
- Hardened library partial-copy cleanup.
- Added playback/no-device warnings.
- Renamed the downloader success signal to avoid shadowing `QThread.finished`.

### Metrics recorded at the time
- 286 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-22 -- v1.0 Library Search, Metadata Editing and Playback Speed

### Done
- Added library search/filter (#54, PR #63).
- Added song metadata editing (#52, PR #64).
- Added pitch-preserving playback speed control (#53, PR #67).
- Added background stretch work, state updates, and peak normalization.

### Metrics recorded at the time
- 229 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-22 -- v1.0 Drag-and-Drop Import and ffmpeg Bundling

### Done
- Added drag-and-drop import (#51, PR #60).
- Added bundled ffmpeg through imageio-ffmpeg.
- Preserved PATH fallback.

### Metrics recorded at the time
- 196 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-22 -- Phase 3 Waveform Visualization

### Done
- Added waveform visualization (#43, PR #58).
- Added click/drag seek, loop markers, active-mix peak computation, and review fixes.
- Added the module-scope import convention.

### Metrics recorded at the time
- 173 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-21 -- Phase 3 A-B Loop and YouTube Import

### Done
- Added A-B repeat and loop controls (#44, PR #48).
- Added YouTube URL import with metadata, download workers, ffmpeg checks, and lifecycle cleanup (#41, PR #49).

### Metrics recorded at the time
- 154 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-21 -- Phase 2 Completion and Hardening

### Done
- Added Wiener filtering and soft gating (#19, PR #38).
- Hardened corrupted JSON recovery, thread cleanup, stream safety, and public APIs (PR #40).
- Added MP3 export (#23, PR #36).
- Added window persistence (#27, PR #35).
- Added keyboard shortcuts (#24, PR #34).
- Added per-stem volume (#25, PR #31).
- Added CI and integrated reviewed external fixes.

### Metrics recorded at the time
- 112 fast tests, five slow ONNX tests, and one hardware test existed.

---

## 2026-03-20 -- Phase 1 MVP

### Done
- Implemented HTDemucs separation (#2, PR #6).
- Implemented multi-track playback (#3, PR #14).
- Implemented persistent song library (#4, PRs #15 and #16).
- Implemented WAV export (#7, PR #20).
- Implemented Qt UI modules (#8-#12, PR #21).
- Added integration tests (#18, PR #17).
- Added overlap-add Hann windowing (#22, PR #29).
- Added first-run model management.

---

## 2026-03-20 -- Session 1 Project Setup

### Done
- Researched HTDemucs, ONNX Runtime, and related projects.
- Chose the Windows/Python/PySide6/ONNX/sounddevice stack.
- Created the initial technical plan and repository.
- Set up source, test, data, and model structure.
- Created the development environment and installed dependencies.
- Started separator and model-manager skeletons.
- Adopted `AGENTS.md`.
- Set up the GitHub Projects Kanban board with Phase 1 issues.
