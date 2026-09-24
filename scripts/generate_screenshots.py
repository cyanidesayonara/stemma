"""Generate the Microsoft Store screenshot set from the real application UI.

    python scripts/generate_screenshots.py [--song-dir DIR]
        [--title TITLE] [--artist ARTIST] [--out DIR]

Renders the real MainWindow offscreen, scripted into each state, so the Store
art always matches the build instead of going stale. The set follows the
plan in issue #146: two composed shots (headline and trust line beside the
window) lead, then bare full-window shots, each showing one feature. Every
shot is 1920x1080. Captions for Partner Center are written to
``captions.txt`` next to the images.

``--song-dir`` points at a separated song folder (``vocals.wav``,
``drums.wav``, ...). Use a song you have the rights to show: the generated
fixture used without it is for layout checks only, since its waveforms look
too regular for Store art. Like ``render_ui_review.py``, it runs against a
private data directory and throwaway settings files, never your library.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import shutil
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from scripts.qt_capture import pump  # noqa: E402
from scripts.render_ui_review import (  # noqa: E402
    MODEL_KEYS,
    _stage_practice,
    _wait_for_detection,
    close_window,
    load_song,
    open_window,
    prepare_library,
    start_app,
)

DEFAULT_OUT = os.path.join(_ROOT, "assets", "store_listing", "screenshots")
CANVAS = (1920, 1080)
# Composed shots frame a window at the Partner Center minimum size, so its
# UI text stays legible after scaling into the art.
COMPOSED_WINDOW = (1366, 768)
STEM_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")
_DATA_DIR = os.path.join(_ROOT, "build", "store-screenshots", ".data")


@dataclasses.dataclass(frozen=True)
class Shot:
    """One Store screenshot: what to stage, and how to present it."""

    name: str
    state: str
    theme: str
    caption: str
    headline: str = ""
    subline: str = ""
    chips: tuple[str, ...] = ()

    @property
    def composed(self) -> bool:
        return bool(self.headline)


SHOTS = (
    Shot(
        "01_practice", "practice", "dark",
        caption="Mute your part and play along with the band",
        headline="Mute your part.\nPlay along.",
        subline=(
            "stemma splits any song into stems, so you can drop the part "
            "you play and loop the bars you are learning."
        ),
        chips=("Offline", "No account", "No subscription"),
    ),
    Shot(
        "02_loop_trainer", "loop_trainer", "dark",
        caption="Slow down, transpose, and loop the hard bars",
        headline="Slow it down.\nLoop the hard bars.",
        subline=(
            "Loop Trainer speeds a passage up one step each time it "
            "repeats, until you play it at full tempo."
        ),
        chips=("Speed without pitch change", "Transpose", "Metronome"),
    ),
    Shot(
        "03_detected", "loaded", "light",
        caption="Key, chords, and tempo detected for you",
    ),
    Shot(
        "04_import", "import", "dark",
        caption="Split into 2, 4, or 6 stems; two-stem mode runs on the GPU",
    ),
    Shot(
        "05_takes", "takes", "dark",
        caption="Record takes over the stems and line them up",
    ),
)


def import_song_dir(stemma_dir, song_dir, title, artist):
    """Copy a separated song folder into the private library."""
    from src.library import SongLibrary
    from src.separation_state import write_completion_marker

    stems = [s for s in STEM_NAMES
             if os.path.isfile(os.path.join(song_dir, f"{s}.wav"))]
    by_count = {len(v): k for k, v in _expected_stems().items()}
    model_key = by_count.get(len(stems))
    if model_key is None:
        raise SystemExit(
            f"{song_dir}: found stems {stems}; need a 2-, 4-, or 6-stem set"
        )
    for stale in ("songs", "library.json", "library.json.bak"):
        path = os.path.join(stemma_dir, stale)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    os.makedirs(stemma_dir, exist_ok=True)
    library = SongLibrary(stemma_dir)
    first = os.path.join(song_dir, f"{stems[0]}.wav")
    song = library.add_song(title, artist, first, model_key)
    for name in stems:
        shutil.copy2(os.path.join(song_dir, f"{name}.wav"), song.stems_path)
    write_completion_marker(song.stems_path, model_key)
    return library, song.id


def _expected_stems():
    from src.separation_state import EXPECTED_STEMS

    return {k: v for k, v in EXPECTED_STEMS.items() if k in MODEL_KEYS.values()}


def write_takes(song_dir: str) -> None:
    """Add a recording take derived from the song's own stems.

    It stands in for a player's take: a quieter, offset copy of an
    instrument stem, so the take lane reads as music rather than silence.
    """
    source = next(
        (os.path.join(song_dir, f"{s}.wav") for s in ("guitar", "other", "bass")
         if os.path.isfile(os.path.join(song_dir, f"{s}.wav"))),
        None,
    )
    if source is None:
        return
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    # One take: at the two-take limit Record disables itself, which would
    # read as broken in Store art.
    shifted = np.roll(data, int(sr * 0.25), axis=0) * 0.7
    sf.write(os.path.join(song_dir, "recording_take1.wav"), shifted, sr)


def show_current_chord(window) -> None:
    """Show the chord at the playhead, as the badge does during playback.

    Paused, the badge deliberately reads "--"; the shots are stills of a
    playing session.
    """
    controls = window._player_controls
    player = window._player
    chord = player.chord_at(int(player.current_seconds * player.sample_rate))
    if chord:
        controls._chord_label.setText(controls._badge_html("Chord:", chord))


def stage(app, window, shot, song_id) -> None:
    """Drive the window into the state *shot* shows."""
    load_song(app, window, song_id, 90.0)
    if shot.state in ("practice", "loop_trainer", "takes"):
        _stage_practice(app, window)
        pump(app, 0.5)
        _wait_for_detection(app, window._player_controls, 20.0)
    if shot.state == "loop_trainer":
        renders = {"finished": 0, "last": time.monotonic()}

        def note(finished: bool) -> None:
            renders["last"] = time.monotonic()
            if finished:
                renders["finished"] += 1

        player = window._player
        player.stretch_started.connect(lambda: note(False))
        player.stretch_finished.connect(lambda: note(True))
        rack = window._player_controls.practice_rack
        rack._trainer_check.setChecked(True)
        index = rack.speed_combo.findData(0.75)
        if index >= 0:
            rack.speed_combo.setCurrentIndex(index)
        rack.pitch_spin.setValue(-2)
        _wait_for_renders(app, renders)
    show_current_chord(window)
    pump(app, 0.2)


def _wait_for_renders(app, renders, timeout_s: float = 120.0) -> None:
    """Wait until speed/pitch rendering has finished and gone quiet.

    Otherwise the capture shows progress text such as "-2 semi (0/6)".
    Counting starts against finishes does not work: a render superseded by
    the next change starts but never finishes. So wait for at least one
    finish followed by two quiet seconds.
    """
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        pump(app, 0.25)
        quiet = time.monotonic() - renders["last"]
        if renders["finished"] and quiet >= 2.0:
            return
    print("  note: speed/pitch render still running at capture")


def grab_import_over(app, window, library, stemma_dir):
    """The window with the import dialog open over it, as the user sees it."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QPainter

    from src.model_manager import ModelManager
    from src.ui.import_dialog import ImportDialog

    dialog = ImportDialog(library, ModelManager(stemma_dir), window)
    # Show the GPU two-stem option, one of stemma's differentiators.
    index = dialog._model_combo.findData(MODEL_KEYS[2])
    if index >= 0:
        dialog._model_combo.setCurrentIndex(index)
    dialog.show()
    pump(app, 0.5)
    base = window.grab().toImage()
    top = dialog.grab().toImage()
    painter = QPainter(base)
    painter.fillRect(base.rect(), _rgba(0, 0, 0, 90))
    origin = QPoint((base.width() - top.width()) // 2,
                    (base.height() - top.height()) // 2)
    _shadow(painter, origin.x(), origin.y(), top.width(), top.height())
    painter.drawImage(origin, top)
    painter.end()
    dialog.close()
    pump(app, 0.2)
    return base


def _rgba(r, g, b, a):
    from PySide6.QtGui import QColor

    return QColor(r, g, b, a)


def _shadow(painter, x, y, w, h, radius=10, spread=18) -> None:
    """A soft drop shadow: stacked translucent rounded rectangles."""
    from PySide6.QtCore import QRectF, Qt

    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    for i in range(spread, 0, -3):
        painter.setBrush(_rgba(0, 0, 0, max(4, 40 - i * 2)))
        painter.drawRoundedRect(
            QRectF(x - i, y - i + 8, w + 2 * i, h + 2 * i),
            radius + i, radius + i,
        )
    painter.restore()


HEADLINE_PX = 56


def _text_block_height(painter, shot, font_family, text_width) -> float:
    """Height of the brand, headline, subline, and chips as compose() lays them."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QFont

    height = 34.0
    headline = QFont(font_family, 1)
    headline.setPixelSize(HEADLINE_PX)
    headline.setWeight(QFont.Weight.Bold)
    painter.setFont(headline)
    height += painter.boundingRect(
        QRectF(0, 0, text_width, 400), Qt.TextFlag.TextWordWrap, shot.headline,
    ).height() + 28
    sub = QFont(font_family, 1)
    sub.setPixelSize(25)
    painter.setFont(sub)
    height += painter.boundingRect(
        QRectF(0, 0, text_width - 20, 400), Qt.TextFlag.TextWordWrap,
        shot.subline,
    ).height() + 36
    chip = QFont(font_family, 1)
    chip.setPixelSize(20)
    painter.setFont(chip)
    rows, x = 1, 0.0
    for label in shot.chips:
        width = painter.fontMetrics().horizontalAdvance(label) + 36
        if x and x + width > text_width:
            rows, x = rows + 1, 0.0
        x += width + 12
    return height + rows * 52 - 12


def compose(window_image, shot, colors, font_family):
    """Lay the window out beside the shot's headline, subline, and chips."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (
        QColor,
        QFont,
        QImage,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QRadialGradient,
    )

    canvas = QImage(*CANVAS, QImage.Format.Format_ARGB32)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    ground = QLinearGradient(0, 0, 0, CANVAS[1])
    ground.setColorAt(0.0, QColor("#11111b"))
    ground.setColorAt(1.0, QColor(colors["base"]))
    painter.fillRect(canvas.rect(), ground)
    glow = QRadialGradient(QPointF(CANVAS[0] * 0.78, CANVAS[1] * 0.9), 900)
    accent = QColor(colors["accent"])
    glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 60))
    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.fillRect(canvas.rect(), glow)

    left, text_width = 110, 560
    # Lay the text out once off-canvas to measure it, then center the
    # block on the window beside it.
    y = (CANVAS[1] - _text_block_height(painter, shot, font_family,
                                         text_width)) / 2
    brand = QFont(font_family, 1)
    brand.setPixelSize(26)
    brand.setItalic(True)
    painter.setFont(brand)
    painter.setPen(accent)
    painter.drawText(QPointF(left, y), "stemma")
    y += 34

    headline = QFont(font_family, 1)
    headline.setPixelSize(HEADLINE_PX)
    headline.setWeight(QFont.Weight.Bold)
    painter.setFont(headline)
    painter.setPen(QColor("#f4f5fb"))
    rect = QRectF(left, y, text_width, 260)
    bounds = painter.boundingRect(rect, Qt.TextFlag.TextWordWrap, shot.headline)
    painter.drawText(rect, Qt.TextFlag.TextWordWrap, shot.headline)
    y += bounds.height() + 28

    sub = QFont(font_family, 1)
    sub.setPixelSize(25)
    painter.setFont(sub)
    painter.setPen(QColor(colors["text"]))
    rect = QRectF(left, y, text_width - 20, 200)
    bounds = painter.boundingRect(rect, Qt.TextFlag.TextWordWrap, shot.subline)
    painter.drawText(rect, Qt.TextFlag.TextWordWrap, shot.subline)
    y += bounds.height() + 36

    chip = QFont(font_family, 1)
    chip.setPixelSize(20)
    painter.setFont(chip)
    x = float(left)
    for label in shot.chips:
        width = painter.fontMetrics().horizontalAdvance(label) + 36
        if x + width > left + text_width:
            x = float(left)
            y += 52
        box = QRectF(x, y, width, 40)
        painter.setPen(accent)
        painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 28))
        painter.drawRoundedRect(box, 20, 20)
        painter.setPen(QColor("#e6e9f5"))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)
        x += width + 12

    scaled = window_image.scaledToWidth(
        1130, Qt.TransformationMode.SmoothTransformation,
    )
    wx = CANVAS[0] - scaled.width() - 70
    wy = (CANVAS[1] - scaled.height()) // 2
    _shadow(painter, wx, wy, scaled.width(), scaled.height())
    clip = QPainterPath()
    clip.addRoundedRect(QRectF(wx, wy, scaled.width(), scaled.height()), 10, 10)
    painter.setClipPath(clip)
    painter.drawImage(wx, wy, scaled)
    painter.setClipping(False)
    painter.setPen(QColor(255, 255, 255, 28))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(
        QRectF(wx + 0.5, wy + 0.5, scaled.width() - 1, scaled.height() - 1),
        10, 10,
    )
    painter.end()
    return canvas


def generate(out_dir, song_dir=None, title="", artist=""):
    """Render every shot in SHOTS into *out_dir*; return the written paths."""
    app = start_app(_DATA_DIR)

    from scripts.qt_capture import load_ui_font
    from src.data_paths import platform_user_data_dir
    from src.ui.styles import get_colors

    font_family = load_ui_font(app) or "Segoe UI"
    stemma_dir = platform_user_data_dir()
    if song_dir:
        library, song_id = import_song_dir(
            stemma_dir, song_dir, title or "Song", artist or "Artist",
        )
    else:
        print("  note: no --song-dir; using the generated fixture "
              "(layout checks only, not Store art)")
        library, song_id = prepare_library(stemma_dir, 6)

    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(stale)

    written = []
    for shot in SHOTS:
        if shot.state == "takes":
            write_takes(library.get_song(song_id).stems_path)
        size = COMPOSED_WINDOW if shot.composed else CANVAS
        settings = os.path.join(_DATA_DIR, "settings", f"{shot.name}.ini")
        window = open_window(
            app, library, stemma_dir, settings, shot.theme, size,
        )
        if shot.state == "import":
            load_song(app, window, song_id, 90.0)
            image = grab_import_over(app, window, library, stemma_dir)
        else:
            stage(app, window, shot, song_id)
            image = window.grab().toImage()
        if shot.composed:
            image = compose(image, shot, get_colors(shot.theme), font_family)
        path = os.path.join(out_dir, f"{shot.name}.png")
        image.save(path)
        written.append(path)
        print("wrote", os.path.relpath(path, _ROOT))
        close_window(app, window)

    with open(os.path.join(out_dir, "captions.txt"), "w",
              encoding="utf-8") as handle:
        for shot in SHOTS:
            handle.write(f"{shot.name}.png: {shot.caption}\n")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--song-dir", help="separated song folder to show")
    parser.add_argument("--title", default="")
    parser.add_argument("--artist", default="")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    generate(os.path.abspath(args.out), args.song_dir, args.title,
             args.artist)


if __name__ == "__main__":
    main()
