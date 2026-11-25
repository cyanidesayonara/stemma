# Roadmap

Latest stable release: **v3.2.0**.

This document is intentionally short. GitHub issues hold acceptance criteria,
discussion, and live status; `../CHANGELOG.md` records only shipped releases.
Version groupings below describe the intended sequence, not a promise of
scope or date. Each upcoming version has a GitHub milestone.

## v2.6 -- stability and release hardening (shipped)

- [#137: v2.6 stability and release hardening](https://github.com/cyanidesayonara/stemma/issues/137)
  — DirectML/runtime truth, correctness and lifecycle work, generation-safe
  background loading, reproducible release validation, and Store listing
  automation ([#134](https://github.com/cyanidesayonara/stemma/issues/134)).

## v3.0 -- practice cockpit (shipped)

- [#131: practice cockpit interface](https://github.com/cyanidesayonara/stemma/issues/131)
  — the recomposition (stacked stem lanes, practice cards, song readout
  strip, anchored transport), released in v3.0.0, with follow-up fixes
  found by rendering the window in PRs
  [#153](https://github.com/cyanidesayonara/stemma/pull/153) (layout) and
  [#156](https://github.com/cyanidesayonara/stemma/pull/156) (mute/solo
  button feedback).
- [#159: acceptance pass and release checklist](https://github.com/cyanidesayonara/stemma/issues/159)
  — one hands-on pass on the merged build, then screenshots
  ([#146](https://github.com/cyanidesayonara/stemma/issues/146)),
  changelog, tag, and Store submission.

## v3.1 -- song structure and chord lane (shipped)

- [#133: song sections and scrolling chord lane](https://github.com/cyanidesayonara/stemma/issues/133)
  — released in v3.1.0: a chord lane with an overview row and an
  eight-bar read-ahead row under the stem lanes, and experimental song
  sections with snap-to-section loops, both built on the existing
  beat/chord analysis.
- [#188: post-3.0 robustness and accessibility](https://github.com/cyanidesayonara/stemma/issues/188)
  and [#237: post-3.0 audit](https://github.com/cyanidesayonara/stemma/issues/237)
  — output-device loss, the bundled YouTube JavaScript runtime, silent-take
  warning, atomic exports, data folder on a disconnected drive, menu
  accelerators, and an accessible waveform shipped in v3.1.0. The remaining
  items in #188 (Tab order, skippable splash and reduced motion, Windows
  text scaling, and the yt-dlp update plan and automation) shipped in
  v3.2.0 and #188 is closed; #237 stays open for the Store-build
  microphone check.

## v3.2 -- library and setlists (shipped)

- [#132: library, setlists, and play queue](https://github.com/cyanidesayonara/stemma/issues/132)
  — released in v3.2.0: tags and setlists, a play queue, practice state per song,
  batch actions, multi-file import, and the library schema migration.
- [#272: acceptance pass and release checklist](https://github.com/cyanidesayonara/stemma/issues/272)
  — the hands-on pass on the merged build, and the release steps.

## Research and cross-release tracks

- [#125: DirectML-compatible HTDemucs exports](https://github.com/cyanidesayonara/stemma/issues/125)
  investigates GPU inference for four/six-stem separation. Those paths are
  CPU-only today.
- [#28: experimental DSP extensions](https://github.com/cyanidesayonara/stemma/issues/28)
  evaluates separation and post-processing approaches without promising a
  quality tier.
- [#13: real-time streaming separation](https://github.com/cyanidesayonara/stemma/issues/13)
  investigates progressive playback/separation constraints.
