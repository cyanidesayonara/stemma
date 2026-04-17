# Changelog

All notable development sessions are documented here in reverse chronological order.

---

## 2026-04-17 -- v2.3.0 Library, Polish & Shortcuts

### Done
- **Library playback controls:** Repeat (off / all / one), shuffle with randomized queue, previous/next buttons. Autoplay on track end respects the repeat mode. Now-playing indicator (accent bar) on the currently loaded song in the list.
- **Keyboard shortcuts overhaul (#119):** YouTube-style controls, layout-independent. Zero symbol keys — works on Nordic/Finnish and all layouts. `0-9` jumps to 0%-90% position, `Left/Right` seek ±5s, `Home/End` start/end, `Up/Down` master volume ±5%, `Shift+Up/Down` speed cycle (replaces `[`/`]`), `Ctrl+1-6` stem mute toggle (replaces bare `1-6`), `N/P` next/previous song, `F1` shortcuts dialog. Focus guard prevents shortcuts firing while typing in the search box or spinboxes.
- **Master volume:** New gain multiplier on `MultiTrackPlayer`, applied in the audio callback alongside per-stem volumes. Persisted in session. Changes show a toast overlay.
- **UI polish pass:** Centralised styling via QSS objectName selectors (`card-frame`, `icon-btn`, `subtle-label`, `footer`, `copyright`, `QSpinBox`). Removed ~20 hardcoded `setStyleSheet` calls. Transport buttons (play/stop/record) share `icon-btn` styling with stem buttons. Confidence colours and delegate defaults moved to `styles.py`.
- **Teal contrast fix:** Introduced `styles.ON_ACCENT = "#11111b"` (near-black) as the foreground token for anything on the teal accent fill. Fixes poor contrast in light mode (previously used theme.base, which is near-white in light mode). Applied to checked toggle buttons, selected library rows, repeat/shuffle icons on teal, and stem mute/solo icons.
- **Shortcuts dialog refresh:** Grouped by category with accent-coloured section headers and hairline dividers.
- **Session persistence:** `repeat_mode`, `shuffle_enabled`, and `master_volume` save and restore across app restarts.

### Metrics
- 670 tests pass, 5 skipped (slow ONNX/hardware).

---

## 2026-04-14 -- v2.2.0 Chord Detection & beat_this ONNX Model

### Done
- **Chord detection:** Real-time chord display using chromagram template matching with cosine similarity across 24 chord templates (major, minor). Viterbi HMM smoothing (self_prob=0.99) for temporal stability. Silence gating prevents stale chords. Binary search lookup at playback position (O(log n)). Updates at 4 Hz during playback. Shows "Chord: --" placeholder when stopped or during silence.
- **beat_this ONNX model:** Auto-downloads the beat_this model (MIT license, ISMIR 2024) for high-accuracy beat/downbeat tracking. Chunked inference with 1500-frame segments and 6-frame border overlap to satisfy rotary position embedding constraints. Falls back to librosa when unavailable.
- **Detection badge UI:** Three detection labels (key, tempo, chord) rendered as styled badges with rich text HTML — label text with confidence-coloured values on rounded surface0 backgrounds. Theme switching regenerates badge HTML with new theme colours.
- **QThread crash fix:** Worker orphaning pattern with identity-checked `finished` callback prevents "QThread: Destroyed while thread is still running" crashes when switching songs during detection.
- **Light mode readability:** Badge labels include explicit `color:` property; dim text uses theme text color for proper contrast in both dark and light modes.
- **MSIX logo sound fix:** Logo click and splash screen sounds now use `winsound.SND_MEMORY` with daemon thread playback, fixing silent sounds in MSIX sandbox (path virtualization breaks `SND_FILENAME`) and Python's `SND_ASYNC | SND_MEMORY` restriction.
- **Session backward compatibility:** Schema version 4 forces re-detection when cached sessions have older chord data.
- **Downbeat sensitivity:** Lowered downbeat peak-picking threshold (0.15 vs 0.3 for beats) to capture weaker downbeat activations from the ONNX model.

### Metrics
- 625 tests pass, 5 skipped (slow ONNX/hardware).

---

## 2026-04-03 -- v2.1.0 Metronome Nudge, UI Polish, Library Improvements, Session Persistence

### Done
- **Metronome beat-sync nudge:** New ±500ms spinbox on the metronome row shifts all click sources (continuous metronome, count-in, beat-synced mode) relative to detected beat positions. Useful for aligning the click to songs with a consistent offset. Offset resets on Stop.
- **Count-in controls in transport bar:** Beat counter and count-in controls (label, spinbox, repeat checkbox) moved from the metronome row into the transport bar, reducing crowding and keeping the metronome row focused.
- **Live volume combos:** Stem and metronome volume combo boxes are now editable and update in real time as the slider moves (e.g. "137%"). The combo opens on a click anywhere in the widget, not just the dropdown arrow.
- **Icon button fix:** On/off and repeat buttons in the metronome/count-in area now render at their correct 36x36 size instead of collapsing to slivers.
- **Speed combo alignment:** Playback speed combo box is now flush with the right edge of the transport bar.
- **Library two-row display:** Songs in the library panel now show artist name (bold) on the top line and song title (subdued, smaller) below, with a separator line between items. Selection uses the teal accent colour to match the app theme.
- **Recording session persistence:** Per-song QSettings now save and restore each recording take's nudge offset, mute, solo, and volume. Values are re-applied after the takes finish loading so they survive song switches and restarts.

### Metrics
- 600 fast tests pass, 5 skipped (slow ONNX/hardware).

---

## 2026-04-02 -- v2.0.5 Automatic BPM & Key Detection

### Done
- **Beat-synced metronome:** Added a "Sync" toggle to lock metronome clicks to the detected beat positions instead of a fixed BPM grid. Handles time-stretching correctly via scaled beat frames. Fixed an auditory glitch by properly carrying over click tails that fall across audio chunk playback boundaries.
- **BPM/key detection:** New background analysis engine detects tempo and musical key for every song automatically after it loads. Uses the beat_this ONNX model (MIT, ISMIR 2024, 89–97% F1) for beat tracking with a librosa fallback, and the Krumhansl-Schmuckler algorithm on chroma features for key detection.
- **Suggestion-only display:** Detected values are shown as read-only labels — "Detected key: A minor" on the loop row and "~98 BPM" on the metronome row — colour-coded by confidence (green/yellow/red). The metronome BPM spinbox is never modified.
- **Per-song caching:** Results (including confidence level) are stored per-song in settings. Switching back to a previously analysed song restores the cached values instantly without re-running analysis.
- **A-B region detection:** Setting loop points A and B re-triggers detection within that region only, letting you identify the key/tempo of a specific section. Clearing the loop reverts to full-song analysis.
- **Double-click to re-detect:** Double-clicking either the key or BPM label forces a fresh analysis for that value individually — useful after the first run or when A-B points change.
- **Logo sound fix (MS Store):** Logo click sound now uses `winsound` on Windows, matching the proven splash-screen path and fixing silent clicks in MSIX builds.
- **Menu padding:** QMenu dropdown items now have proper vertical padding for a less cramped look.

### Metrics
- 31 new tests (test_beat_detector.py); 580 total pass, 5 skipped.

---

## 2026-03-31 -- v2.0.4 UI Polish and Store Alignment

### Done
- **UI button polish:** Standardized stem mute/solo buttons with explicit object name (`#icon-btn`) and QSS matching toolbar buttons.
- **Icons:** Replaced complex metronome and count-in icons with universal power (on/off) icons for clarity at small sizes. Fixed visibility of active-state icons on the accent background.
- **Metronome volume:** Replaced the plain percentage label with a drop-down preset combo box (0% to 200% in 20% steps), synced bidirectionally with the slider.
- **State persistence:** The player now preserves the solo and mute state of stem tracks when switching to a different song.
- **Logo alignment:** Fine-tuned the vertical baselines of letters in both the animated splash screen and the static SVG logos so text sits perfectly centered within the staff lines.
- **Tooltips:** Capitalized stem names in the mixer tooltips.
- **Store Assets:** Regenerated and preserved Store listing assets.

### Metrics
- 549 fast tests pass, 5 skipped (slow ONNX/hardware).

---

## 2026-03-30 -- Low-RAM stem separation (Store certification fix)

### Done
- **In-place post-processing:** `wiener_filter` and `soft_gate` now write results back into the input array instead of allocating full-song copies, saving ~1 GB of peak RAM on a 4-minute 6-stem track.
- **Early resource release:** ORT inference session and input audio are freed before post-processing begins, saving ~300-500 MB.
- **Pre-flight memory check:** Before starting separation, the estimated peak RAM is compared against available physical memory via Windows `GlobalMemoryStatusEx`. If insufficient, a warning dialog lets the user close other apps or abort.
- **Smarter ORT error messages:** `format_import_error` now distinguishes ORT errors containing OOM phrases (e.g. `failed to allocate`, `bad_alloc`) from generic init failures, giving a more accurate user message.

### Metrics
- Peak memory for a 4-minute 6-stem song reduced by ~1.3-1.6 GB.
- 549 fast tests pass (7 new tests for in-place behavior, memory estimation, and ORT error split).

---

## 2026-03-28 -- ONNX external data download for HuggingFace models

### Done
- **Model cache:** HuggingFace `htdemucs*.onnx` files use external weights in `*.onnx.data`. `ModelDownloader` now fetches every artifact for each variant; `is_model_downloaded` requires both files. Fixes ONNX Runtime init errors when only the small `.onnx` stub was present.
- **Library metadata:** Successful 6-stem import stores `model_used=htdemucs_6s` instead of always `htdemucs`.

---

## 2026-03-28 -- Release pipeline housekeeping

### Done
- **Tag-driven versions:** `scripts/sync_release_version.ps1` sets `src/version.py` and `msix/AppxManifest.xml` from the pushed tag before PyInstaller and MSIX packaging in `release.yml`, so GitHub Releases stay aligned with app and package identity without a pre-tag version commit.
- **Tests before release build:** `release.yml` runs the same fast pytest slice as CI before `pyinstaller`.
- **CI on tags:** `ci.yml` also triggers on `v*` tag pushes.
- **Qt helper:** `src/qt_signal_utils.safe_disconnect` replaces duplicated logic in `player.py` and `import_dialog.py`.
- **Docs:** `docs/store-release-pipeline.md` describes the release and Store upload flow; optional `.github/workflows/partner-center-submit.yml` (`workflow_dispatch`, default **configure** only) for future Partner Center API automation.
- **Roadmap:** `PROJECT.md` post-2.0 backlog synced with shipped work and open GitHub issues.

---

## 2026-03-27 -- v2.0.2 ONNX DirectML fallback for Store certification

### Done
- **ONNX CPU fallback:** `_create_session()` now tries DirectML first, but if session creation raises (e.g. `RUNTIME_EXCEPTION` on Surface Go 4 integrated GPU), retries with `CPUExecutionProvider` only. Fixes Store certification failure 10.1.2.10 ("Import & Separate unusable").
- **ONNX error messages:** `import_messages.py` maps raw `ONNXRuntimeError` and out-of-memory style messages to user-friendly text. OOM matching uses specific phrases (`bad_alloc`, `failed to allocate`, etc.), not a bare ``alloc`` substring.
- **MSIX manifest:** Version updated to 2.0.2.0.

### Metrics
- New unit tests for DML-fail->CPU fallback, DML-not-available paths, and `format_import_error` ONNX/OOM branches.

---

## 2026-03-27 -- v2.0.1 splash and Store listing

### Done
- **Splash screen:** Letter animation stays visible when the event loop is blocked during heavy imports: minimum paint-frame check resets the animation timeline in `finish()` so letters fade in over the full minimum display window. Sound defers when the second paint arrives late (blocked loop) so the arpeggio is not started twice; one playback aligns with the resynced animation. Bitmap `drawText` for letters (reliable on Windows); `finish()` measures minimum display time from animation start via `_sound_start_ms`.
- **Store listing PNGs:** `generate_store_listing_assets.py` -- wider vertical gap between chord and arpeggio (poster/box), optional extra margin constants, arpeggio width capped to chord width. Regenerated `poster_720x1080.png` and `box_1080x1080.png`.

### Metrics
- Splash tests expanded; full `pytest` suite unchanged in scope.

---

## 2026-03-26 -- Animated logos and splash sync fix (#76)

### Done
- **Splash screen sync fix:** Sound playback deferred to the second rendered frame. If the event loop was blocked by heavy imports, the animation clock is restarted so letters and sound stay synchronized.
- **Animated main logo:** New `AnimatedLogoWidget` (src/ui/animated_logo.py) replaces the static QLabel in the player empty state. Notes light up near-simultaneously as a chord (40ms stagger), with bounce and wave growth effects. Waves undulate gently then settle. Plays on app startup via `showEvent` (visual only, no sound since splash just played it).
- **Chord sound:** Synthesized Cmaj7 chord WAV (assets/audio/chord.wav) with all four notes (C3-E3-G3-B3) playing simultaneously. Used by the main logo on click (distinct from the arpeggio used by the footer and splash).
- **Animated footer arpeggio logo:** New `AnimatedArpeggioWidget` (src/ui/animated_arpeggio.py) replaces the static QLabel in the footer. Letters glow sequentially at arpeggio timing with a bloom effect.
- **Clickable Easter egg:** Both logos replay their animation with sound when clicked (pointing hand cursor for discoverability). Main logo plays chord.wav; footer plays arpeggio.wav. Respects the "Play startup sound" preference.
- **About dialog:** Static SVG logo replaced with interactive `AnimatedLogoWidget` (plays intro animation on open, clickable to replay with chord sound).
- **Startup animation timing fix:** Main logo animation now triggers via `MainWindow.showEvent` instead of being called before the window is visible. Ensures the chord animation plays reliably after the splash fades.
- **Cleanup:** Removed unused `_render_svg`, `_logo_variant`, `_ROOT_DIR`, `QImage`, `QSvgRenderer` imports from `main_window.py` and `player_controls.py`.

### Previous (same PR)
- **Splash screen:** `SplashScreen` widget (src/ui/splash_screen.py) shows the arpeggio logo animation while the app loads. Staff lines and bass clef appear immediately; the six letters of "stemma" fade in one by one at arpeggio timing (Cmaj7: C-E-G-B-G-E). A pulsing "Loading..." indicator shows after the animation completes.
- **Deferred imports:** `main.py` restructured to show the splash before importing heavy modules (onnxruntime, librosa, sounddevice). `app.py` gains `build_and_show()` for the heavy construction phase, called via `QTimer.singleShot` after the splash is visible. `processEvents()` calls between construction steps keep the animation responsive.
- **Startup sound:** Synthesized Cmaj7 arpeggio WAV (assets/audio/arpeggio.wav) played via `winsound.PlaySound` (async, Windows-only). Generated by scripts/generate_startup_audio.py.
- **Preference:** "Play startup sound" checkbox in Edit > Preferences (Startup group), backed by `startup/play_sound` QSettings key (default True).
- **Smooth transition:** Minimum display time (1.8s) ensures the animation completes; splash fades out over 250ms before the main window appears.
- **PyInstaller:** `stemma.spec` updated to bundle `assets/audio/`.

### Metrics
- 488 fast tests, 5 slow ONNX tests, 1 hardware playback test.

---

## 2026-03-26 -- v1.2.0 release

Shipped as GitHub Release **v1.2.0** (tag `v1.2.0`). User-facing highlights:

- Record audio track: full-duplex play-along recording with multiple takes (#79, PR #91).
- Input device selection and latency compensation in Preferences.
- Recordings included in mix export.
- Count-in now fires at any play position (not just position 0 or loop A).

## 2026-03-26 -- Record audio track (#79, PR #91)

### Done
- **Recording engine:** Full-duplex `sd.Stream` callback captures input audio at the exact playback frame position, guaranteeing frame-synchronised recording with stem playback. Recording buffer is position-indexed so A-B loop recording naturally overwrites the same region on each pass.
- **Record button:** Red circle icon in the transport bar, checkable (arm/disarm). `R` keyboard shortcut. Disabled when speed != 1.0x.
- **Multiple takes:** Recordings saved as `recording_take1.wav`, `recording_take2.wav`, etc. in the song's stems directory. Take numbering auto-increments. Each take gets its own stem row in the mixer.
- **Recording stem rows:** `RecordingStemRow` subclass with mute/solo/volume controls and a delete button (X). Colored in Material Red. Shown in a separate "Recordings" section below stems.
- **Input device selection:** Preferences > Audio now has an input device dropdown (mirrors output device). Stored as `audio/input_device` in QSettings.
- **Latency compensation:** Manual ms offset in Preferences (-200 to +200 ms). Applied via `np.roll` when saving the recording. Stored as `audio/latency_offset_ms`.
- **Export integration:** Recordings are included in mix export alongside stems.
- **Take management:** Delete individual takes with confirmation dialog. Removes file from disk, stem from player, and row from mixer.
- **Song loading:** Existing recordings in a song's stems directory are auto-discovered and loaded on song selection.
- **Review fixes:** Mono mic support (channels tuple), safe default device resolution, stop-only finalization (pause keeps buffer for resume, stop saves take), `_total_frames` recalculation on recording deletion, encapsulated recording stem management via public methods, sample rate validation, count-in at any play position with device-switch suppression, record button sync on pause-then-stop.

### Metrics
- 395 fast tests, 5 slow ONNX tests, 1 hardware playback test.
- 33 new recording-specific tests.

---

## 2026-03-26 -- v1.1.0 release

Shipped as GitHub Release **v1.1.0** (tag `v1.1.0`). User-facing highlights:

- Metronome with BPM (20--300), tap tempo, toggle and volume (#57, PR #86).
- Count-in pre-roll (1--8 beats), optional on loop repeats, `C` shortcut (#78, PR #87).
- Session persistence for song, position, stems, loop, speed, metronome, and count-in (#55, PR #85).
- A-B loop: while **L** is on, Stop jumps to loop A; seek clamps into the loop region (PR #87).
- **Help > Keyboard Shortcuts** dialog (PR #87).

`src/version.py` set to **1.1.0**.

---

## 2026-03-26 -- Metronome (#57, PR #86) and count-in (#78, PR #87)

### Done
- **Metronome:** BPM spinbox (20--300), tap tempo, toggle (M), volume slider, click mixed in the audio callback with phase tied to full PortAudio block size; `set_metronome_bpm` rejects non-finite values; `tap_tempo` in `src/metronome.py`.
- **Count-in:** Optional pre-roll (1--8 beats, default 4) before stems start; uses metronome BPM/volume; optional count-in before each A-B repeat; arms only from position 0 or loop A (not on resume mid-song); `C` shortcut; Help > Keyboard Shortcuts dialog; session keys for count-in state; QCheckBox styling for loop-repeat option.
- **A-B loop UX:** With looping active and a valid region, Stop jumps to loop A; seek clamps into `[loop_a, loop_b)` (outside snaps to A). Count-in boundary uses the same valid-region check as the callback.

### Metrics
- 362 fast tests, 5 slow ONNX tests, 1 hardware playback test.

---

## 2026-03-25 -- Session persistence (#55, PR #85)

### Done
- `MainWindow._save_session()`: writes 9 QSettings keys on close -- song ID, position, muted/soloed stems (JSON lists), per-stem volumes (JSON dict), loop A/B (float, -1 for unset), looping flag, and playback speed.
- `MainWindow._restore_session()`: deferred via `QTimer.singleShot(0)`, validates song still in library, selects it to trigger stem load, then restores stem state, loop points, and speed. Speed != 1.0 uses a one-shot `speed_changed` connection to seek after time-stretch completes.
- `LibraryPanel.select_song(song_id)`: programmatic selection by ID, returns bool.
- `StemRow.set_soloed()` / `set_volume_slider()`: programmatic setters for restore path.
- `PlayerControls.restore_stem_state()` / `restore_loop_state()`: batch-apply saved state through UI widgets so player and UI stay in sync.

### Metrics
- 286 fast tests, 5 slow ONNX tests, 1 hardware playback test.

---

## 2026-03-24 -- v1.0.0 Release: PyInstaller packaging + GitHub Release (#56, PR #83)

### Done
- `src/paths.py`: `app_root()` returns `sys._MEIPASS` in frozen builds, repo root otherwise.
- `stemma.spec`: one-file windowed PyInstaller build. Collects ONNX Runtime DirectML DLLs, PortAudio, imageio-ffmpeg binary, and PySide6 QtSvg. UPX disabled to avoid corrupting native DLLs.
- `.github/workflows/release.yml`: builds .exe on `v*` tag push, creates GitHub Release with the artifact via `softprops/action-gh-release@v2`.
- `requirements-dev.txt` for pyinstaller as dev dependency.
- `.gitignore` updated to track `stemma.spec` while still ignoring other `.spec` files.
- Updated `src/app.py` and `src/ui/player_controls.py` to use `app_root()` for asset path resolution.
- Version bumped from `1.0.0-dev` to `1.0.0`.

### Metrics
- Local build: 202 MB exe, launches correctly with icon.
- 282 fast tests, 5 slow ONNX tests, 1 hardware playback test (288 total).
- All v1.0 tickets closed. First GitHub Release published.

---

## 2026-03-24 -- Error handling, model download UX, playback warnings (#73, PR #82)

### Done
- Import when ONNX is missing: `ModelDownloader` runs inside the import dialog with progress and status; separation starts when the file exists.
- `format_import_error()` in `import_messages.py` for disk full, permission, timeout, network, SSL, HTTP/404, cancellation, and long message truncation.
- Failed import (including model download, separation, and `add_song` errors) removes the new library row; **Retry import** re-runs the flow.
- Large source files (100 MiB and up): confirmation before import (local path and YouTube temp file after download).
- `SongLibrary.add_song`: on `OSError` during copy or JSON save, removes partial song directory and re-raises.
- `MultiTrackPlayer.playback_failed` when `PortAudioError` on play; main window shows a **Playback** dialog. Startup warning if PortAudio reports no output devices.
- `ModelDownloader` success signal renamed to **`download_complete(str)`** so `QThread.finished` remains available for thread cleanup (`deleteLater` on cancel).
- Tests: `test_import_messages.py`, extended import dialog and library tests, `ModelDownloader` signal naming, `playback_failed` on PortAudio error.

### Metrics
- 286 fast tests, 5 slow ONNX tests, 1 hardware playback test (292 total)

---

## 2026-03-22 -- v1.0: Library Search, Metadata Editing, and Playback Speed

### Done
- Library search/filter (#54, PR #63)
  - Search bar with clear button in library panel
  - Real-time filter hides non-matching songs as you type
  - Remove button disabled when selection is hidden by filter
  - 9 new tests
- Song metadata editing (#52, PR #64)
  - Double-click a song to edit title/artist
  - Right-click context menu with Edit option
  - Right-click selects the item before opening menu (fixes wrong-item bug)
  - Empty fields fall back to original values
  - 10 new tests
- Pitch-preserving playback speed control (#53, PR #67)
  - librosa.effects.time_stretch pre-stretches stems in background QThread
  - Speed presets: 0.5x, 0.75x, 0.85x, 1.0x, 1.25x, 1.5x, 2.0x
  - Keyboard shortcuts: [ (slower) and ] (faster)
  - Frame index and loop points proportionally adjusted on speed change
  - Peak normalization after stretch fixes volume reduction from phase vocoder
  - Worker detach pattern prevents stale signal callbacks
  - 13 new tests

### Metrics
- 229 fast tests, 5 slow ONNX tests, 1 hardware playback test (235 total)
- v1.0 roadmap: 2 tickets remaining (#61 branding, #56 packaging)

---

## 2026-03-22 -- v1.0: Drag-and-Drop Import + ffmpeg Bundling

### Done
- Drag-and-drop import (#51, PR #60)
  - Drop .mp3/.wav/.flac files onto the main window to trigger import
  - ImportDialog gains `file_path` parameter for pre-fill
  - Multiple dropped files open sequential import dialogs
  - dragMoveEvent override for correct drop cursor on Windows/Qt6
  - Inline import of ImportDialog in main_window.py fixed (rule #13)
  - 17 new tests (3 prefill + 6 drag-enter + 2 drag-move + 6 drop)
- Bundled ffmpeg via imageio-ffmpeg (PR #60)
  - YouTube import now works without ffmpeg on PATH
  - imageio-ffmpeg ships a static ffmpeg binary as package data
  - _get_ffmpeg_exe() prefers bundled binary, falls back to PATH
  - ffmpeg_location passed to yt-dlp options
  - 4 new tests (3 check_ffmpeg fallback + 1 ffmpeg_location in opts)

### Metrics
- 196 fast tests, 5 slow ONNX tests, 1 hardware playback test (202 total)
- Phase 3: 4 of 5 tickets complete; v1.0 roadmap: 5 tickets remaining

---

## 2026-03-22 -- Phase 3: Waveform Visualization

### Done
- Waveform visualization (#43, PR #58)
  - Custom QPainter widget replacing seek slider
  - Mirrored waveform bars, playback cursor, A-B loop region shading and markers
  - Click/drag-to-seek
  - Waveform recomputes on mute/solo/volume changes
  - Vectorized peak computation via numpy (compute_peaks + np.maximum.reduceat)
  - Three review passes; all findings addressed:
    - Zero-width guards, cursor bounds clamping, redundant repaint removal
    - Unused import cleanup, stale waveform cleared on empty stems
  - 19 new tests (9 peak computation + 10 widget)
- Inline import rule added to project conventions

### Metrics
- 173 fast tests, 5 slow ONNX tests, 1 hardware playback test (178 total)
- Phase 3: 3 of 5 tickets complete

---

## 2026-03-21 -- Phase 3: A-B Loop and YouTube Import

### Done
- A-B loop repeat (#44, PR #48)
  - Loop point set/clear API on player with callback wrap-around
  - UI controls: Set A, Set B, Loop toggle, Clear buttons
  - Keyboard shortcuts: A/B/L keys
  - Zero-width loop guard to prevent callback deadlock
  - 9 new tests
- YouTube URL import (#41, PR #49)
  - yt-dlp downloader module: URL validation, metadata extraction, audio download
  - Import dialog reworked: URL input with Fetch button, dual-mode import
  - Background workers for metadata fetch and audio download
  - ffmpeg availability check with user-facing error
  - Worker lifecycle hardening: signal rename to avoid QThread.finished shadowing,
    safe disconnect helper, proper wait/cleanup on dialog close
  - Output file existence verification after download
  - noplaylist option to prevent accidental playlist downloads
  - Error handling in local import path (disk full, missing file, permissions)
  - 33 new tests (26 downloader + 7 import dialog)

### Metrics
- 154 fast tests, 5 slow ONNX tests, 1 hardware playback test (159 total)
- Phase 3: 2 of 5 tickets complete

---

## 2026-03-21 -- Phase 2 Completion and Hardening

### Done
- Audio post-processing: Wiener filtering and soft gating (#19, PR #38)
  - Chunked processing to bound memory usage (~50 MB vs 3.5 GB)
  - 17 tests including multi-chunk boundary verification
- Robustness fixes (PR #40)
  - Corrupted JSON recovery (app no longer crashes on malformed library.json)
  - Thread cleanup on close (export worker, separator worker)
  - Player stream leak fix on audio device errors
  - Encapsulation fix: public API for mute toggle from keyboard shortcuts
- MP3 export via lameenc (#23, PR #36)
  - 320kbps CBR, no ffmpeg dependency
  - Export dialog offers WAV and MP3
  - 5 new tests
- Window state persistence (#27, PR #35)
  - QSettings saves/restores geometry across sessions
- Keyboard shortcuts (#24, PR #34)
  - Space=play/pause, S=stop, Left/Right=seek, 1-6=toggle stem mute
- Per-stem volume sliders (#25, PR #31)
  - 0-200% gain per stem, applied in audio callback
  - Volume reflected in export
- Code quality cleanup (PR #30)
  - Public `has_stems` and `muted_stems` properties on player
  - Stem name constants imported from separator module
- CI pipeline (PR #34)
  - GitHub Actions running fast tests on every push
  - pythonpath fix for module resolution
- Reviewed and merged external PRs:
  - #32: ONNX tensor mapping fix, button padding, remove song button
  - #37: Export worker QThread, peak normalization
  - #39: Integration test fix for soft gating side effect

### Metrics
- 112 fast tests, 5 slow ONNX tests, 1 hardware playback test
- ~2000 lines of source, ~1700 lines of tests
- Phase 1 and Phase 2 fully complete

---

## 2026-03-20 -- Phase 1 MVP

### Done
- Stem separation engine (#2, PR #6)
  - ONNX Runtime + DirectML, STFT/iSTFT via librosa
  - Segmented inference with progress reporting
- Multi-track audio player (#3, PR #14)
  - sounddevice callback API, mute/solo, clipping protection
- Song library (#4, PRs #15, #16)
  - JSON persistence, atomic writes, audio file copying on import
- Stem exporter (#7, PR #20)
  - Single stem or custom mix export as WAV
- UI modules (#8-#12, PR #21)
  - Main window with splitter layout
  - Player controls with transport and stem rows
  - Library panel with song list
  - Import dialog with separation progress
  - Dark theme (Catppuccin Mocha-inspired)
- Integration test suite (#18, PR #17)
  - Library round-trips, player with real WAVs, ONNX inference
  - Hardware playback test using real music
- Overlap-add Hann windowing (#22, PR #29)
  - 50% overlap with Hann window eliminates boundary clicks
- Model manager
  - Download from HuggingFace, DirectML -> CPU fallback

---

## 2026-03-20 -- Session 1: Project Setup

### Done
- Researched stem separation technology landscape (HTDemucs v4, ONNX Runtime, existing projects)
- Chose tech stack: Python 3.14, PySide6, ONNX Runtime + DirectML, sounddevice
- Created `PROJECT.md` with full spec, module descriptions, and phased roadmap
- Created GitHub repo (https://github.com/cyanidesayonara/stemma), pushed initial scaffold
- Set up project folder structure with all source file skeletons
- Created virtual environment and installed dependencies
- Started `separator.py` (SeparatorWorker QThread skeleton) and `model_manager.py` (ModelDownloader)
- Adopted `AGENTS.md` open standard for cross-tool AI handover
- Set up GitHub Projects kanban board with Phase 1 issues
