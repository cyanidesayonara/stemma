# Practice Cockpit Waveform Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 140px composite waveform with a ~280px DAW-style stacked stem lane view and remove per-stem mini waveforms from mixer rows.

**Architecture:** New `WaveformStackWidget` owns multi-lane paint/seek/loop logic, reused from `WaveformWidget` patterns. `TransportBar` hosts the stack instead of `WaveformWidget`. `PlayerControls` routes existing `stem_peaks` to the stack and drops `StemMixer.set_mini_peaks`. `StemRow` layout simplifies to label + controls.

**Tech Stack:** Python 3.12+, PySide6 (QPainter, signals), numpy peaks, pytest + offscreen Qt.

**Spec:** `docs/superpowers/specs/2026-08-01-practice-cockpit-waveform-design.md`  
**Issue:** [#131](https://github.com/cyanidesayonara/stemma/issues/131)

## Global Constraints

- Issue #131 slice 1 only: waveform stack + mixer mini removal. No practice cards, song readout strip, or transport repositioning.
- Preserve `PlayerControls` as the facade; `MainWindow` integration points unchanged where possible.
- Keep generation-safe peak computation in `PlayerControls` (background thread + `_peak_generation` guard).
- Mute/solo dimming must match effective mix semantics used for composite peaks today.
- Stem order in the stack matches mixer row order.
- No emojis; conventional commits; TDD for new widget behavior.
- Canonical validation after each task: `python -m ruff check .` and `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest -m "not slow and not hardware"`.

## File map

| Path | Responsibility |
|---|---|
| `src/ui/waveform_stack_widget.py` | New stacked lane widget |
| `src/ui/transport_bar.py` | Host stack instead of single waveform |
| `src/ui/player_controls.py` | Route `stem_peaks` to stack; drop mini peaks |
| `src/ui/stem_mixer.py` | Remove mini waveforms and `set_mini_peaks` |
| `tests/test_waveform_stack_widget.py` | Unit tests for stack widget |
| `tests/test_waveform_widget.py` | Update/remove mini-waveform stem row assertions |
| `tests/widget_visual.py` | Offscreen widget PNG snapshot helpers |
| `tests/fixtures/widget_snapshots/` | Golden PNGs for visual regression |

---

### Task 0: Widget visual snapshot helpers

**Files:**
- Create: `tests/widget_visual.py`
- Create: `tests/fixtures/widget_snapshots/.gitkeep`
- Create: `tests/test_widget_visual.py`

**Interfaces:**
- Produces:
  - `render_widget_png(widget, *, width: int, height: int) -> bytes`
  - `assert_widget_snapshot(widget, name: str, *, width: int, height: int) -> None`
  - Set env `UPDATE_WIDGET_SNAPSHOTS=1` to regenerate golden PNGs locally/CI.

Use `QApplication.processEvents()` before grab; offscreen platform only. Snapshots
live under `tests/fixtures/widget_snapshots/{name}.png`. Compare SHA-256 of PNG
bytes for stable CI. Include one smoke test that grabs a simple QLabel to prove
the harness works.

- [ ] Implement helpers + smoke test
- [ ] Commit: `test: add offscreen widget snapshot helpers`

---

### Task 1: `WaveformStackWidget` core API and lane rendering

**Files:**
- Create: `src/ui/waveform_stack_widget.py`
- Create: `tests/test_waveform_stack_widget.py`

**Interfaces:**
- Produces: `WaveformStackWidget` with:
  - `set_stem_lanes(stems: list[tuple[str, np.ndarray, str]], *, muted: set[str], soloed: set[str]) -> None`
  - `set_position(ratio: float) -> None`
  - `set_loop_markers(a: float | None, b: float | None) -> None`
  - `set_total_seconds(seconds: float) -> None`
  - `set_theme_colors(colors: dict[str, str]) -> None`
  - `set_loading(loading: bool) -> None`
  - Signal `seek_requested(float)` (seconds)

- [ ] **Step 1: Write failing tests** (include one `assert_widget_snapshot` for two-lane layout)

```python
# tests/test_waveform_stack_widget.py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from src.ui.waveform_stack_widget import WaveformStackWidget, STACK_HEIGHT
from tests.widget_visual import assert_widget_snapshot


@pytest.fixture(scope="module")
def app():
    inst = QApplication.instance() or QApplication([])
    return inst


def test_stack_has_fixed_height(app):
    w = WaveformStackWidget()
    assert w.height() == STACK_HEIGHT


def test_set_stem_lanes_stores_order(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    w.set_stem_lanes([("vocals", peaks, "#ff0000")], muted=set(), soloed=set())
    assert w.lane_count() == 1


def test_seek_emits_seconds(app):
    w = WaveformStackWidget()
    w.set_total_seconds(100.0)
    w.resize(400, STACK_HEIGHT)
    got = []
    w.seek_requested.connect(got.append)
    w._emit_seek_at_ratio(0.25)
    assert got == [25.0]


def test_two_lane_stack_snapshot(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 0.8, 0.2, 0.9], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#e78284"), ("drums", peaks, "#a6e3a1")],
        muted=set(),
        soloed=set(),
    )
    w.set_total_seconds(60.0)
    assert_widget_snapshot(w, "waveform_stack_two_lanes", width=640, height=280)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/test_waveform_stack_widget.py -v`  
Expected: `ModuleNotFoundError` or missing methods.

- [ ] **Step 3: Implement minimal widget**

Create `WaveformStackWidget` with `STACK_HEIGHT = 280`, lane list storage, basic
`paintEvent` drawing one bar path per lane using stem color, shared cursor line,
and loop region fill copied from `WaveformWidget` patterns.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add WaveformStackWidget with stacked stem lanes"
```

---

### Task 2: Mute/solo dimming and loading shimmer

**Files:**
- Modify: `src/ui/waveform_stack_widget.py`
- Modify: `tests/test_waveform_stack_widget.py`

**Interfaces:**
- Consumes: lane tuples from Task 1.
- Produces: `lane_opacity(stem_name) -> float` used in paint (1.0 active, ~0.15 muted when not soloed, solo-only lanes at 1.0).

- [ ] **Step 1: Write failing tests**

```python
def test_muted_lane_low_opacity(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes([("drums", peaks, "#00ff00")], muted={"drums"}, soloed=set())
    assert w.lane_opacity("drums") < 0.5


def test_solo_hides_non_solo_lanes(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#f00"), ("drums", peaks, "#0f0")],
        muted=set(),
        soloed={"vocals"},
    )
    assert w.lane_opacity("drums") < 0.5
    assert w.lane_opacity("vocals") == 1.0
```

- [ ] **Step 2–4:** Implement opacity rules matching mixer composite behavior; wire loading shimmer from `WaveformWidget`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: dim waveform lanes for mute and solo state"
```

---

### Task 3: Integrate stack into `TransportBar`

**Files:**
- Modify: `src/ui/transport_bar.py`
- Modify: `tests/test_waveform_widget.py` (if transport tests reference type)

**Interfaces:**
- Consumes: `WaveformStackWidget` from Tasks 1–2.
- Produces: `TransportBar.waveform` returns the stack widget (type widened or renamed property `waveform_stack` with `waveform` alias).

- [ ] **Step 1: Replace widget construction**

Swap `WaveformWidget()` for `WaveformStackWidget()` inside the card frame. Keep
`seek_requested` forwarding and `set_theme_colors`.

- [ ] **Step 2: Run fast tests**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest -m "not slow and not hardware" -q`  
Expected: PASS (stack is drop-in for properties used by PlayerControls).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: use waveform stack in transport bar"
```

---

### Task 4: Route peaks in `PlayerControls`

**Files:**
- Modify: `src/ui/player_controls.py`

**Interfaces:**
- Consumes: `stem_peaks: dict[str, np.ndarray]` from existing peak job.
- Produces: `_waveform.set_stem_lanes(...)` call with stem order from `_stem_mixer` row order and current mute/solo sets.

- [ ] **Step 1: Write/adjust test** in `tests/test_waveform_widget.py` or player test if one exists for `_on_peaks_computed` routing (optional smoke via widget test).

- [ ] **Step 2: Update `_on_peaks_computed`**

Build lane list from `self._stem_mixer.stem_names()` (add accessor if missing),
colors from theme palette, pass mute/solo from player/mixer state.

- [ ] **Step 3: Update mute/solo handlers** to refresh lane opacities without recomputing peaks when only mix changed.

- [ ] **Step 4: Run fast tests — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: feed stem peaks into waveform stack from player controls"
```

---

### Task 5: Remove mini waveforms from mixer rows

**Files:**
- Modify: `src/ui/stem_mixer.py`
- Modify: `tests/test_waveform_widget.py`

- [ ] **Step 1: Update failing stem row test**

Change `test_stem_row_has_mini_waveform` to assert mini waveform is **gone**
(label + buttons remain).

- [ ] **Step 2: Remove `MiniWaveformWidget` usage**

Delete mini widget from `StemRow`, remove `set_mini_peaks` from row and mixer,
remove import if unused.

- [ ] **Step 3: Remove `MiniWaveformWidget` class** from `waveform_widget.py` if no references remain; otherwise leave class but delete dead tests for it.

- [ ] **Step 4: Run fast tests — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: remove per-stem mini waveforms from mixer rows"
```

---

### Task 6: Manual smoke and docs touch

**Files:**
- Modify: `src/ui/player_controls.py` module docstring (note waveform stack shipped)
- Modify: `docs/DEVELOPMENT_LOG.md` (short entry)

- [ ] Run `python main.py --diagnostics` locally.
- [ ] Offscreen smoke: load a song in a test if available, or document manual GUI check for seek/loop/mute.
- [ ] Commit docs-only follow-up if needed.

---

## Spec coverage self-review

| Spec requirement | Task |
|---|---|
| ~280px stacked lanes | Task 1 |
| Stem colors + order | Tasks 1, 4 |
| Seek/playhead/loop | Tasks 1, 3 |
| Mute/solo dimming | Task 2, 4 |
| Remove mixer minis | Task 5 |
| Tests | Tasks 1–5 |
| Out-of-scope slices untouched | Global constraints |

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-08-01-practice-cockpit-waveform.md`.

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
**2. Inline Execution** — implement in this session with checkpoints

Which approach do you want?
