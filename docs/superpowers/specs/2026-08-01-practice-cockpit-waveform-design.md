# Practice cockpit waveform stack (issue #131, slice 1)

Date: 2026-08-01  
Status: Approved  
Issue: [#131](https://github.com/cyanidesayonara/stemma/issues/131)

## Goal

Replace the single 140px composite waveform with a DAW-style stacked stem lane
view (~280px), and remove redundant per-stem mini waveforms from mixer rows.

This is the first visual slice of the v3.0 practice cockpit. Later slices
(practice cards, song readout strip, transport repositioning) stay out of scope
here.

## Current state

- `TransportBar` hosts one `WaveformWidget` (140px, accent-colored composite
  peaks).
- `PlayerControls._on_peaks_computed` already computes `main_peaks` and
  `stem_peaks` on a background thread; composite peaks go to the main waveform,
  per-stem peaks go to `StemMixer.set_mini_peaks`.
- Each `StemRow` embeds a `MiniWaveformWidget` (24px) beside mute/solo/volume.

## Design

### New `WaveformStackWidget`

Add `src/ui/waveform_stack_widget.py` (or extend `waveform_widget.py` if the
file stays readable):

| Property | Value |
|---|---|
| Height | 280px fixed (roughly 2x current main waveform) |
| Lanes | One row per stem in stable stem order (same order as mixer) |
| Peaks | Per-stem arrays from existing `compute_stem_peaks` |
| Colors | `STEM_COLORS_DARK` / `STEM_COLORS_LIGHT` per stem name |
| Interaction | Click/drag seek on any lane; shared playhead and A–B loop region |
| Mute/solo | Muted lanes drawn dimmed; hidden or near-zero when not in effective mix (match composite peak semantics) |
| Loading | Reuse shimmer/loading pattern from `WaveformWidget` across the stack |

Each lane shows a small stem label (abbreviated or capitalized name) in the lane
color, left-aligned, similar to a DAW track header.

### Transport integration

- `TransportBar` replaces `WaveformWidget` with `WaveformStackWidget`.
- Public surface on `TransportBar`: `waveform` property returns the stack widget
  (or a shared protocol) so `PlayerControls` keeps working with minimal renames.
- Loop markers, position, theme colors, and seek wiring stay on the stack.

### Mixer simplification

- Remove `MiniWaveformWidget` from `StemRow`.
- Remove `StemRow.set_mini_peaks`, `StemMixer.set_mini_peaks`, and the
  `set_mini_peaks` call in `PlayerControls._on_peaks_computed`.
- Stem rows become: label + mute + solo + volume (optionally slightly wider
  label column now that waveform space is freed).

### Data flow

```
_on_peaks_computed(main_peaks, stem_peaks)
  -> waveform_stack.set_stem_peaks(stem_peaks, stem_order, mute/solo state)
  -> (optional) keep main_peaks for a faint composite overlay or drop if redundant
```

Prefer driving the stack purely from `stem_peaks` plus mix state. Composite
`main_peaks` may still be computed for tests or future overlay but is not
required for rendering if lanes sum visually.

### Keep `WaveformWidget` and `MiniWaveformWidget`

Do not delete `WaveformWidget` in this slice. `MiniWaveformWidget` can remain in
the module but unused, or be removed if nothing references it after mixer
changes; prefer removal if tests are updated.

## Non-goals

- Practice rack card grouping (#131 slice 2).
- Song readout strip (#131 slice 3).
- Scrolling chord lane (#133).
- Store screenshot regeneration (#146).

## Acceptance criteria

- [ ] Main waveform area shows stacked, stem-colored lanes at ~280px height.
- [ ] Seek, playhead, loop markers, and theme switching work as before.
- [ ] Mute/solo changes update lane dimming without reloading audio.
- [ ] Mixer rows no longer show mini waveforms.
- [ ] Fast tests cover stack widget seek/position/loop and updated stem row layout.
- [ ] No regression in peak computation or background-thread generation safety.

## Risks

| Risk | Mitigation |
|---|---|
| Many stems (6+) make lanes too thin | Minimum lane height with vertical scroll, or cap visible lanes with "+N more" — defer unless testing shows problem |
| Peak bin mismatch between composite and stems | Use same `mini_bins` count for stack lanes as current mini waveforms |
| Paint cost with 6 lanes | Reuse cached `QPainterPath` per lane keyed on size + peaks |

## References

- `src/ui/waveform_widget.py` — existing paint/seek/loop logic
- `src/ui/transport_bar.py` — integration point
- `src/ui/stem_mixer.py` — row layout after mini removal
- `src/ui/player_controls.py` — peak routing
