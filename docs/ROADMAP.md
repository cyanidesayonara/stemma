# Roadmap

Latest stable release: **v2.5.0**. The current source tree targets
**v2.6.0**, which is not released.

This document is intentionally short. GitHub issues hold acceptance criteria,
discussion, and live status; `../CHANGELOG.md` records only shipped releases.
Version groupings below describe the intended sequence, not a promise of
scope or date.

## v2.6 -- stability and release hardening

- [#137: v2.6 stability and release hardening](https://github.com/cyanidesayonara/stemma/issues/137)
  tracks the DirectML/runtime truth, correctness and lifecycle work,
  generation-safe background loading, reproducible release validation, and
  documentation/Project repair.

Do not close the issue or describe v2.6 as shipped until the builder pull
request is reviewed and merged, required validation passes, and the release
is published.

## v3.0 -- practice cockpit

- [#131: interface re-composition](https://github.com/cyanidesayonara/stemma/issues/131)
  splits the oversized controls module behind the current behavior before
  recomposing the practice-focused interface.

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
- [#28: experimental DSP extensions](https://github.com/cyanidesayonara/stemma/issues/28)
  evaluates separation and post-processing approaches without promising a
  quality tier.
- [#13: real-time streaming separation](https://github.com/cyanidesayonara/stemma/issues/13)
  investigates progressive playback/separation constraints.
- [#134: Store submission and listing automation](https://github.com/cyanidesayonara/stemma/issues/134)
  keeps release metadata and Store assets reproducible across versions.
