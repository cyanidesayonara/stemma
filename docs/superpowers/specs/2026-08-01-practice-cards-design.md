# Practice cards and song readout strip (issue #131, slice 2)

Date: 2026-08-01
Status: Approved
Issue: [#131](https://github.com/cyanidesayonara/stemma/issues/131)

## Goal

Replace the three flat, equal-weight practice rows with three labeled cards,
and gather the tempo/key/chord readouts into one strip.

This covers intended moves 2 and 3 from #131. Move 1 (stacked stem lanes)
landed in #149. Move 4 (persistent transport bar, shorter mixer eye travel)
stays out of scope.

Moves 2 and 3 are taken together because they are physically entangled:
`SongInfoBar` is currently a child of the loop row that move 2 dismantles, and
`SongInfoBar.detected_bpm_label` is re-parented into the metronome row. Doing
move 2 alone would mean parking those readouts somewhere arbitrary and moving
them again in the next slice.

## Current state

`PracticeRack` owns one `QVBoxLayout` with three `QHBoxLayout` rows:

1. Set A / Set B / Loop / Clear / loop label / `SongInfoBar` / stretch /
   Speed / speed status / Pitch
2. Loop Trainer / from / start combo / arrow / trainer status / stretch
3. Metronome / toggle / BPM / Tap / Sync / nudge / volume slider / volume
   combo / `detected_bpm_label` / stretch

`count_in_controls` is built by `PracticeRack` but passed to `TransportBar`,
which places it in the top-right transport corner.

Rendering the window shows the problems #131 describes: unrelated groups share
a row, Speed and Pitch sit far to the right of the loop buttons they are
unrelated to, count-in is separated from the metronome it belongs with, and
"detecting..." appears twice because the key and BPM readouts live in
different rows.

## Design

### Cards

A card is a `title-label` `QLabel` above a `QFrame` with object name
`card-frame`, which is the idiom `StemMixer` already uses for Stems and
Recordings. No new stylesheet rules are required.

Three cards sit side by side in a `QHBoxLayout`:

| Card | Contents |
|---|---|
| Loop and Trainer | Set A, Set B, Loop, Clear, loop range label; Loop Trainer check, start speed, target, trainer status |
| Speed and Pitch | Speed combo and status; Pitch spin |
| Metronome and Count-in | Metronome toggle, BPM, Tap, Sync, nudge, volume slider and combo; count-in toggle, beats, repeat-each-loop |

Side by side rather than stacked: it groups related controls into scannable
blocks, and it reclaims vertical space, which is the same complaint as "a
large area below the mixer is empty".

Stretch factors weight the row so the metronome card, which holds the most
controls, is not squeezed by the narrow speed card.

### Song readout strip

`SongInfoBar` adds `detected_bpm_label` to its own layout instead of handing it
to `PracticeRack`, so key, chord, and tempo read as one line. The strip sits
directly under the waveform, above the cards, where it reads against the
playhead.

### Count-in

`TransportBar` no longer takes a `count_in_controls` argument. The controls
move into the metronome card next to the tempo they count.

## Constraints

- `PracticeRack`'s signal surface stays exactly as it is. `PlayerControls`
  wiring, session persistence, and keyboard shortcuts must not need changes.
- Accessible names and tooltips are preserved on every moved widget.
- The `count_in_controls` property stays on `PracticeRack` so existing lookups
  keep resolving, even though `TransportBar` no longer receives it.

## Non-goals

- Persistent transport bar and mixer eye travel (#131 move 4).
- Scrolling chord lane (#133), library/setlists/queue (#132).
- Any change to playback, detection, or persistence behavior.

## Acceptance criteria

- [ ] Practice controls render as three labeled cards.
- [ ] Count-in sits with the metronome, not in the transport corner.
- [ ] Key, chord, and tempo render once, together, in one strip.
- [ ] Every practice signal still fires from the same widget as before.
- [ ] Fast tests cover card composition and the readout strip.
- [ ] Rendered dark and light screenshots inspected before merge.

## Risks

| Risk | Mitigation |
|---|---|
| Cards squeeze controls at narrow window widths | Check the rendered layout at the minimum supported width, not just 1366px |
| Re-parenting widgets loses theme icons | `apply_theme` already rebuilds toggle icons; assert after a theme switch |
| Silent signal breakage from re-parenting | Keep the signal surface untouched and assert emissions in tests |
