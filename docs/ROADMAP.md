# Roadmap

Latest stable release: **v2.6.0**.

This document is intentionally short. GitHub issues hold acceptance criteria,
discussion, and live status; `../CHANGELOG.md` records only shipped releases.
Version groupings below describe the intended sequence, not a promise of
scope or date. Each upcoming version has a GitHub milestone.

## v2.6 -- stability and release hardening (shipped)

- [#137: v2.6 stability and release hardening](https://github.com/cyanidesayonara/stemma/issues/137)
  — DirectML/runtime truth, correctness and lifecycle work, generation-safe
  background loading, reproducible release validation, and Store listing
  automation ([#134](https://github.com/cyanidesayonara/stemma/issues/134)).

## v3.0 -- practice cockpit

- [#131: practice cockpit interface](https://github.com/cyanidesayonara/stemma/issues/131)
  — the recomposition (stacked stem lanes, practice cards, song readout
  strip, anchored transport) is on `main`, with follow-up fixes found by
  rendering the window in PRs
  [#153](https://github.com/cyanidesayonara/stemma/pull/153) (layout) and
  [#156](https://github.com/cyanidesayonara/stemma/pull/156) (mute/solo
  button feedback).
- [#159: acceptance pass and release checklist](https://github.com/cyanidesayonara/stemma/issues/159)
  — one hands-on pass on the merged build, then screenshots
  ([#146](https://github.com/cyanidesayonara/stemma/issues/146)),
  changelog, tag, and Store submission.

## v3.1 -- library and setlists

- [#132: library, setlists, and play queue](https://github.com/cyanidesayonara/stemma/issues/132)
  adds collection structure, practice state, queueing, and the required
  storage migration.

## v3.2 -- song structure and chord lane

- [#133: song sections and scrolling chord lane](https://github.com/cyanidesayonara/stemma/issues/133)
  builds section-aware loops and read-ahead chord presentation on the
  existing beat/chord analysis.

## Research and cross-release tracks

- [#125: DirectML-compatible HTDemucs exports](https://github.com/cyanidesayonara/stemma/issues/125)
  investigates GPU inference for four/six-stem separation. Those paths are
  CPU-only today.
- [#154: librosa 1.0](https://github.com/cyanidesayonara/stemma/issues/154)
  checks the reworked phase vocoder by ear before lifting the `<1.0` cap.
- [#28: experimental DSP extensions](https://github.com/cyanidesayonara/stemma/issues/28)
  evaluates separation and post-processing approaches without promising a
  quality tier.
- [#13: real-time streaming separation](https://github.com/cyanidesayonara/stemma/issues/13)
  investigates progressive playback/separation constraints.
