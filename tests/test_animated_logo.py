"""Tests for the animated main logo widget."""

import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPointingDevice
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from src.ui.animated_logo import (
    AnimatedLogoWidget,
    _ANIM_END_MS,
    _BOUNCE_MS,
    _DAMPEN_START_MS,
    _FADE_MS,
    _NOTE_SPACING_MS,
    _UNDULATE_DEPTH,
    _UNDULATE_START_MS,
    _W,
    _WAVE_GROW_MS,
    _bounce_y,
    _load_logo,
    _logo_path,
    _note_alpha,
    _parse_polyline,
    _parse_transform,
    _wave_path,
    _wave_reveal,
    _wave_swell,
)
from src.ui import animated_logo
from src.ui.svg_source import local_name, read_svg

THEMES = ("dark", "light")


def _left_mouse_press(local_x: float, local_y: float) -> QMouseEvent:
    """Build a non-deprecated QMouseEvent (Qt 6 single-point API)."""
    pos = QPointF(local_x, local_y)
    dev = QPointingDevice.primaryPointingDevice()
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        dev,
    )


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _is_animated_part(element) -> bool:
    """A notehead or a wave, by what it is rather than how it is written."""
    name = local_name(element)
    if name == "ellipse":
        return True
    return (
        name == "path"
        and element.get("fill") == "none"
        and element.get("stroke") not in (None, "none")
    )


def _pixels(image: QImage) -> np.ndarray:
    """An (h, w, 4) uint8 array of *image* in premultiplied ARGB32."""
    image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    width, height = image.width(), image.height()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(
        height, image.bytesPerLine(),
    )
    return rows[:, : width * 4].reshape(height, width, 4).copy()


def _frame(widget: AnimatedLogoWidget, t: int) -> np.ndarray:
    """The widget painted at animation time *t* on a transparent image."""
    widget._static_t = t
    image = QImage(
        widget.width(), widget.height(),
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(Qt.GlobalColor.transparent)
    widget.render(image)
    return _pixels(image)


def _svg_reference(theme: str, width: int, height: int) -> np.ndarray:
    """The brand SVG itself, rendered by Qt at *width* x *height*."""
    renderer = QSvgRenderer(_logo_path(theme))
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    return _pixels(image)


def _read_svg_text(text: str) -> ET.Element:
    """Parse SVG markup (a static layer) into an element tree root."""
    return ET.fromstring(text)


def _svg_static_reference(logo, width: int, height: int) -> np.ndarray:
    """Only the static layers of *logo*, composited in order."""
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    for layer in logo.static_layers:
        renderer = QSvgRenderer(layer.encode("utf-8"))
        renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    return _pixels(image)


class TestParseLogo:
    """The notes and waves are read from the brand SVG, not hard-coded."""

    @pytest.mark.parametrize("theme", THEMES)
    def test_four_notes_and_four_waves(self, theme):
        logo = _load_logo(theme)
        assert len(logo.notes) == 4
        assert len(logo.waves) == 4

    @pytest.mark.parametrize("theme", THEMES)
    def test_notes_run_bottom_to_top(self, theme):
        heights = [note.cy for note in _load_logo(theme).notes]
        assert heights == sorted(heights, reverse=True)

    @pytest.mark.parametrize("theme", THEMES)
    def test_each_wave_leaves_its_note(self, theme):
        logo = _load_logo(theme)
        for note, wave in zip(logo.notes, logo.waves):
            start_x, start_y = wave.points[0]
            assert start_y == pytest.approx(note.cy)
            assert start_x > note.cx
            assert wave.points[-1][0] > start_x

    @pytest.mark.parametrize("theme", THEMES)
    def test_each_wave_has_its_notes_colour(self, theme):
        logo = _load_logo(theme)
        for note, wave in zip(logo.notes, logo.waves):
            assert wave.color.lower() == note.color.lower()

    def test_themes_share_geometry_but_not_colours(self):
        dark, light = _load_logo("dark"), _load_logo("light")
        assert dark.view_box == light.view_box
        assert [n.cy for n in dark.notes] == [n.cy for n in light.notes]
        assert [w.points for w in dark.waves] == [
            w.points for w in light.waves
        ]
        assert [n.color for n in dark.notes] != [n.color for n in light.notes]

    def test_known_brand_colours(self):
        assert _load_logo("dark").notes[0].color.lower() == "#4fb8b8"
        assert _load_logo("light").notes[0].color.lower() == "#3da8a8"

    @pytest.mark.parametrize("theme", THEMES)
    def test_static_layers_hold_no_notes_or_waves(self, theme):
        for layer in _load_logo(theme).static_layers:
            root = _read_svg_text(layer)
            assert not [e for e in root.iter() if _is_animated_part(e)]

    @pytest.mark.parametrize("theme", THEMES)
    def test_no_element_is_lost(self, theme):
        source = list(read_svg(_logo_path(theme)))
        logo = _load_logo(theme)
        static = sum(
            len(list(_read_svg_text(layer))) for layer in logo.static_layers
        )
        assert static + len(logo.notes) + len(logo.waves) == len(source)

    @pytest.mark.parametrize("theme", THEMES)
    def test_static_layers_keep_staff_and_wordmark(self, theme):
        text = "".join(_load_logo(theme).static_layers)
        assert 'y1="48"' in text
        assert 'y1="108"' in text
        assert "translate(76.7,23)" in text

    def test_paint_order_follows_the_document(self):
        # The stem is drawn over the noteheads and under the waves, so a
        # static layer must sit between the last note and the first wave.
        kinds = [kind for kind, _ in _load_logo("dark").paint_order]
        last_note = max(i for i, k in enumerate(kinds) if k == "note")
        first_wave = kinds.index("wave")
        assert "static" in kinds[last_note:first_wave]


_MINI_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="140" '
    'viewBox="10 4 240 140">'
    '<line x1="18" y1="48" x2="228" y2="48" stroke="#ccc"/>'
    "{notes}"
    '<path d="M105,78 L225,78" fill="none" stroke="#bfa3dc" '
    'stroke-width="1.8" stroke-linecap="{cap}" stroke-opacity="0.5"/>'
    "</svg>"
)
_MINI_NOTE = (
    '<ellipse cx="94" cy="78" rx="10.5" ry="6.5" fill="#bfa3dc" '
    'opacity="0.8"/>'
)


def _load_mini(tmp_path, monkeypatch, *, notes=_MINI_NOTE, cap="round"):
    """Parse a small hand-written logo SVG in place of the brand one."""
    icons = tmp_path / "assets" / "icons"
    icons.mkdir(parents=True)
    (icons / "logo_main_dark.svg").write_text(
        _MINI_LOGO.format(notes=notes, cap=cap), encoding="utf-8",
    )
    monkeypatch.setattr(animated_logo, "_ROOT", str(tmp_path))
    # Bypass the cache so the brand logo's cached parse is untouched.
    return animated_logo._load_logo.__wrapped__("dark")


class TestParseRobustness:
    def test_hand_written_logo_parses(self, tmp_path, monkeypatch):
        logo = _load_mini(tmp_path, monkeypatch)
        assert len(logo.notes) == len(logo.waves) == 1
        assert logo.waves[0].cap == Qt.PenCapStyle.RoundCap

    def test_nested_notes_fail_loudly(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="noteheads"):
            _load_mini(tmp_path, monkeypatch, notes=f"<g>{_MINI_NOTE}</g>")

    def test_unknown_cap_falls_back_to_the_svg_default(self, tmp_path,
                                                        monkeypatch):
        logo = _load_mini(tmp_path, monkeypatch, cap="inherit")
        assert logo.waves[0].cap == Qt.PenCapStyle.FlatCap

    def test_opacity_is_read(self, tmp_path, monkeypatch):
        logo = _load_mini(tmp_path, monkeypatch)
        assert logo.notes[0].opacity == pytest.approx(0.8)
        assert logo.waves[0].opacity == pytest.approx(0.5)


class TestViewBox:
    @pytest.mark.parametrize("theme", THEMES)
    def test_no_ink_is_cut_off(self, app, theme):
        # Render with a 20-unit band added on every side: none of it may
        # hold ink, or the widget (which shows only the viewBox) clips it.
        with open(_logo_path(theme), encoding="utf-8") as fh:
            text = fh.read()
        vx, vy, vw, vh = _load_logo(theme).view_box
        pad, scale = 20, 4
        text = text.replace(
            f'viewBox="{vx:g} {vy:g} {vw:g} {vh:g}"',
            f'viewBox="{vx - pad:g} {vy - pad:g} {vw + 2 * pad:g} '
            f'{vh + 2 * pad:g}"',
            1,
        )
        width, height = int((vw + 2 * pad) * scale), int((vh + 2 * pad) * scale)
        renderer = QSvgRenderer(text.encode("utf-8"))
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, width, height))
        painter.end()
        alpha = _pixels(image)[:, :, 3]
        inner = alpha.copy()
        inner[pad * scale: height - pad * scale,
              pad * scale: width - pad * scale] = 0
        assert np.count_nonzero(inner) == 0
        assert np.count_nonzero(alpha) > 0


class TestParsePolyline:
    def test_absolute_move_and_lines(self):
        assert _parse_polyline("M1,2 L3,4 L5,6") == [
            (1.0, 2.0), (3.0, 4.0), (5.0, 6.0),
        ]

    def test_implicit_lineto_after_move(self):
        assert _parse_polyline("M1 2 3 4") == [(1.0, 2.0), (3.0, 4.0)]

    def test_relative_commands(self):
        assert _parse_polyline("m1,1 l2,0 l0,-1") == [
            (1.0, 1.0), (3.0, 1.0), (3.0, 0.0),
        ]

    def test_brand_style_decimals(self):
        points = _parse_polyline("M105.00,78.00 L105.40,77.99")
        assert points == [(105.0, 78.0), (105.4, 77.99)]

    def test_curves_are_rejected(self):
        with pytest.raises(ValueError, match="polyline"):
            _parse_polyline("M105,78 Q111,69 117,78")


class TestParseTransform:
    def test_rotate_about_a_point_keeps_the_point(self):
        transform = _parse_transform("rotate(-20,94,78)")
        mapped = transform.map(QPointF(94, 78))
        assert mapped.x() == pytest.approx(94)
        assert mapped.y() == pytest.approx(78)

    def test_rotate_turns_other_points(self):
        mapped = _parse_transform("rotate(90)").map(QPointF(1, 0))
        assert mapped.x() == pytest.approx(0, abs=1e-9)
        assert mapped.y() == pytest.approx(1)

    def test_translate(self):
        mapped = _parse_transform("translate(3 4)").map(QPointF(1, 1))
        assert (mapped.x(), mapped.y()) == pytest.approx((4, 5))

    def test_empty_is_identity(self):
        assert _parse_transform("").isIdentity()

    def test_unsupported_is_rejected(self):
        with pytest.raises(ValueError):
            _parse_transform("skewX(10)")


class TestNoteAlpha:
    def test_before_onset(self):
        assert _note_alpha(0, 100) == 0.0

    def test_at_onset(self):
        assert _note_alpha(100, 100) == pytest.approx(0.0)

    def test_after_full_fade(self):
        assert _note_alpha(100 + _FADE_MS + 1, 100) == 1.0

    def test_mid_fade(self):
        alpha = _note_alpha(100 + _FADE_MS // 2, 100)
        assert 0.0 < alpha < 1.0


class TestBounceY:
    def test_before_onset(self):
        assert _bounce_y(0, 100) == 0.0

    def test_at_onset(self):
        assert _bounce_y(100, 100) == pytest.approx(-4.0)

    def test_after_bounce(self):
        assert _bounce_y(100 + _BOUNCE_MS + 1, 100) == 0.0

    def test_mid_bounce_negative(self):
        val = _bounce_y(100 + _BOUNCE_MS // 2, 100)
        assert -4.0 < val < 0.0


class TestWaveReveal:
    def test_hidden_before_onset(self):
        assert _wave_reveal(50, 100) == 0.0

    def test_fully_drawn_after_growing(self):
        assert _wave_reveal(100 + _WAVE_GROW_MS, 100) == 1.0

    def test_partly_drawn_while_growing(self):
        assert 0.0 < _wave_reveal(100 + _WAVE_GROW_MS // 2, 100) < 1.0


class TestWaveSwell:
    @pytest.mark.parametrize("index", range(4))
    def test_rests_before_undulating(self, index):
        assert _wave_swell(_UNDULATE_START_MS, index) == 1.0

    @pytest.mark.parametrize("index", range(4))
    def test_settles_exactly_at_the_end(self, index):
        assert _wave_swell(_ANIM_END_MS, index) == 1.0
        assert _wave_swell(_ANIM_END_MS + 500, index) == 1.0

    @pytest.mark.parametrize("index", range(4))
    def test_starts_smoothly(self, index):
        # No jump when the undulation begins.
        swell = _wave_swell(_UNDULATE_START_MS + 5, index)
        assert swell == pytest.approx(1.0, abs=0.01)

    def test_moves_while_undulating(self):
        samples = [
            _wave_swell(t, index)
            for t in range(_UNDULATE_START_MS, _DAMPEN_START_MS, 20)
            for index in range(4)
        ]
        assert max(abs(s - 1.0) for s in samples) > _UNDULATE_DEPTH / 2

    def test_stays_within_its_depth(self):
        for t in range(0, _ANIM_END_MS + 1, 10):
            for index in range(4):
                swell = _wave_swell(t, index)
                assert abs(swell - 1.0) <= _UNDULATE_DEPTH + 1e-9


class TestWavePath:
    def _wave(self):
        return _load_logo("dark").waves[0]

    def test_full_wave_is_the_svg_polyline(self):
        wave = self._wave()
        path = _wave_path(wave, 1.0, 1.0)
        assert path.elementCount() == len(wave.points)
        for i, (x, y) in enumerate(wave.points):
            element = path.elementAt(i)
            assert (element.x, element.y) == pytest.approx((x, y))

    def test_half_reveal_stops_midway(self):
        wave = self._wave()
        path = _wave_path(wave, 0.5, 1.0)
        start_x, end_x = wave.points[0][0], wave.points[-1][0]
        tip = path.currentPosition().x()
        assert start_x < tip < end_x
        assert tip == pytest.approx((start_x + end_x) / 2, abs=10)

    def test_swell_scales_height_about_the_staff_line(self):
        wave = self._wave()
        base_y = wave.points[0][1]
        path = _wave_path(wave, 1.0, 2.0)
        for i, (_, y) in enumerate(wave.points):
            assert path.elementAt(i).y == pytest.approx(
                base_y + 2.0 * (y - base_y)
            )


class TestAnimatedLogoConstruction:
    def test_size_follows_the_svg_aspect_ratio(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        _, _, view_w, view_h = _load_logo("dark").view_box
        assert w.width() == _W
        assert w.height() == round(_W * view_h / view_w)

    def test_light_theme(self, app):
        w = AnimatedLogoWidget(theme="light", play_sound=False)
        assert w._is_dark is False

    def test_cursor_is_pointing_hand(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_initial_static_t_shows_final_frame(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w._static_t == _ANIM_END_MS


class TestRenderedFrames:
    """The resting frame is the brand logo; earlier frames are not."""

    @pytest.mark.parametrize("theme", THEMES)
    def test_final_frame_matches_the_svg(self, app, theme):
        w = AnimatedLogoWidget(theme=theme, play_sound=False)
        frame = _frame(w, _ANIM_END_MS).astype(int)
        reference = _svg_reference(theme, w.width(), w.height()).astype(int)
        # Identical today. The small tolerance only absorbs a rounding
        # difference another Qt build might make where separately drawn
        # parts meet; a moved, doubled, or stretched part changes
        # thousands of pixels by far more.
        assert np.abs(frame - reference).max() <= 2

    @pytest.mark.parametrize("theme", THEMES)
    def test_mid_animation_differs_from_rest(self, app, theme):
        w = AnimatedLogoWidget(theme=theme, play_sound=False)
        rest = _frame(w, _ANIM_END_MS).astype(int)
        for t in (_NOTE_SPACING_MS, _UNDULATE_START_MS + 150):
            frame = _frame(w, t).astype(int)
            changed = np.abs(frame - rest).max(axis=2) > 32
            assert np.count_nonzero(changed) > 200

    def test_nothing_coloured_before_the_first_note(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        blank = _frame(w, 0)
        logo = _load_logo("dark")
        without_parts = _svg_static_reference(logo, w.width(), w.height())
        diff = np.abs(blank.astype(int) - without_parts.astype(int))
        assert diff.max() <= 2


class TestRestFrameCache:
    def test_repaints_at_rest_reuse_one_pixmap(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        _frame(w, _ANIM_END_MS)
        cached = w._rest_pixmap
        assert cached is not None
        _frame(w, _ANIM_END_MS)
        _frame(w, _ANIM_END_MS + 5000)
        assert w._rest_pixmap is cached

    def test_intro_frames_do_not_use_the_cache(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        rest = _frame(w, _ANIM_END_MS)
        assert not np.array_equal(_frame(w, _UNDULATE_START_MS + 150), rest)

    def test_theme_switch_rebuilds_the_cache(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        _frame(w, _ANIM_END_MS)
        dark = w._rest_pixmap
        w.set_theme("light")
        assert w._rest_pixmap is None
        _frame(w, _ANIM_END_MS)
        assert w._rest_pixmap is not dark


class TestPlayIntro:
    def test_starts_timer(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        assert w._timer.isActive()
        assert w._animating is True
        assert w._clock.isValid()
        w._timer.stop()

    def test_sets_static_t(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w._static_t == _ANIM_END_MS
        w.play_intro(with_sound=False)
        assert w._static_t == _ANIM_END_MS
        w._timer.stop()

    def test_restart_resets_clock(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        first_start = w._clock.elapsed()
        time.sleep(0.05)
        w.play_intro(with_sound=False)
        assert w._clock.elapsed() < first_start + 100
        w._timer.stop()


class TestThemeSwitching:
    def test_set_theme_switches_the_logo(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w._is_dark is True
        w.set_theme("light")
        assert w._is_dark is False
        assert w._logo is _load_logo("light")

    def test_set_theme_repaints_in_the_new_colours(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.set_theme("light")
        frame = _frame(w, _ANIM_END_MS).astype(int)
        reference = _svg_reference("light", w.width(), w.height()).astype(int)
        assert np.abs(frame - reference).max() <= 2


class TestPaintEvent:
    def test_paint_does_not_crash_static(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.repaint()

    def test_paint_does_not_crash_animating(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        w.repaint()
        w._timer.stop()

    def test_paint_light_theme(self, app):
        w = AnimatedLogoWidget(theme="light", play_sound=False)
        w.repaint()


class TestAnimationConstants:
    def test_chord_style_fast_spacing(self):
        assert _NOTE_SPACING_MS <= 50

    def test_anim_end_after_dampen(self):
        assert _ANIM_END_MS > _DAMPEN_START_MS

    def test_undulate_before_dampen(self):
        assert _UNDULATE_START_MS < _DAMPEN_START_MS

    def test_every_wave_is_drawn_before_the_end(self):
        last_onset = 3 * _NOTE_SPACING_MS
        assert last_onset + _WAVE_GROW_MS < _ANIM_END_MS


class TestClickReplay:
    def test_click_triggers_play_intro(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.mousePressEvent(_left_mouse_press(10, 10))
        assert w._animating is True
        w._timer.stop()


def test_logo_path_points_at_the_brand_svgs():
    assert _logo_path("dark").endswith(os.path.join("icons",
                                                    "logo_main_dark.svg"))
    assert _logo_path("light").endswith(os.path.join("icons",
                                                     "logo_main_light.svg"))
