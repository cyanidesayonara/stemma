"""Render the real MainWindow in review states, sizes, and themes.

Layout regressions (clipped cards, stray backgrounds, dead space) only show
up in a rendered, styled window, which the fast suite never builds. This
script renders one so a change can be checked by eye -- by a reviewer or a
coding agent -- without launching the app and importing a song by hand.

    python scripts/render_ui_review.py [--out DIR] [--stems 2|4|6]
        [--sizes 900x600,1366x768,1920x1080] [--themes dark,light]

It writes one PNG per state, theme, and size plus an ``index.html`` contact
sheet to ``build/ui-review/`` (ignored by git). Render once on ``main`` and
once on a branch with different ``--out`` directories to compare.

The window runs against a private data directory with a generated song and
a throwaway settings file per window, so the real library, session, and
preferences are never touched. That directory persists between runs so
the beat-detection model downloads only once.
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from scripts.qt_capture import load_ui_font, pump  # noqa: E402

DEFAULT_OUT = os.path.join(_ROOT, "build", "ui-review")
DEFAULT_SIZES = ((900, 600), (1366, 768), (1920, 1080))
DEFAULT_THEMES = ("dark", "light")
FIXTURE_TITLE = "Review Fixture"
FIXTURE_ARTIST = "stemma"
SAMPLE_RATE = 44100
DURATION_S = 48.0
_BPM = 100.0
# Settle time for detection (tempo/key) before each capture. The first run
# may also download the beat model, which the longer first wait covers.
_DETECTION_TIMEOUT_S = 90.0
_BUSY_MARKERS = ("detecting", "downloading")
# The model each stem count comes from, so the fixture looks exactly like a
# finished import of that kind: MDX two-stem, HTDemucs four- or six-stem.
MODEL_KEYS = {2: "mdx_inst_hq3", 4: "htdemucs", 6: "htdemucs_6s"}


def synth_stems(stem_count: int = 4, seconds: float = DURATION_S):
    """Return deterministic stereo float32 stems that read clearly as lanes.

    Each stem gets a distinct envelope -- sparse vocal phrases, percussive
    drums, a steady bass, sustained chords -- so lane dimming and loop
    shading are easy to judge in a screenshot. Six-stem adds guitar and
    piano for the taller stack.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    beat = 60.0 / _BPM
    phase = np.mod(t, beat)
    bar = (t // (beat * 4)).astype(int) % 4
    roots = np.array([110.0, 87.31, 130.81, 98.0])[bar]  # A F C G

    def tone(freq):
        return np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)

    drums = 0.8 * np.exp(-30 * phase) * np.sin(2 * np.pi * 60 * t)
    off = np.mod(t + beat / 2, beat)
    drums += 0.3 * np.exp(-60 * off) * rng.standard_normal(t.size)
    bass = 0.5 * tone(roots) * (0.6 + 0.4 * np.exp(-4 * phase))
    other = sum(0.15 * tone(roots * m) for m in (2.0, 2.5198, 2.9966))
    melody = 440 * 2 ** (np.round(4 * np.sin(2 * np.pi * t / 7)) / 12)
    phrase = np.sin(2 * np.pi * t / 5) > -0.3
    vocals = 0.35 * tone(melody) * phrase

    stems = {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
    if stem_count == 2:
        # MDX two-stem output: vocals plus everything else as the backing.
        stems = {"vocals": vocals, "other": drums + bass + other}
    elif stem_count == 6:
        strum = np.exp(-6 * np.mod(t, beat * 2))
        stems["guitar"] = 0.3 * tone(roots * 4) * strum
        stems["piano"] = 0.25 * tone(roots * 3) * np.exp(-3 * phase)
    # Peak-normalize each stem: the noise hits on the drums can otherwise
    # exceed full scale, and relative stem loudness is irrelevant here.
    return {
        name: np.stack([data, data], axis=1).astype(np.float32)
        * np.float32(0.8 / max(float(np.abs(data).max()), 1e-9))
        for name, data in stems.items()
    }


def prepare_library(data_dir: str, stem_count: int):
    """Reset the private library to exactly one generated song.

    Models under ``data_dir/models`` are kept so detection does not
    re-download on every run.
    """
    from src.library import SongLibrary
    from src.separation_state import write_completion_marker

    for stale in ("songs", "library.json", "library.json.bak"):
        path = os.path.join(data_dir, stale)
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
    os.makedirs(data_dir, exist_ok=True)

    stems = synth_stems(stem_count)
    mix_path = os.path.join(data_dir, "fixture-mix.wav")
    sf.write(mix_path, sum(stems.values()) / len(stems), SAMPLE_RATE)
    # A real model key plus the completion marker, exactly as a finished
    # separation leaves them; otherwise MainWindow prunes the song at startup
    # as an interrupted import.
    model_key = MODEL_KEYS[stem_count]
    library = SongLibrary(data_dir)
    song = library.add_song(FIXTURE_TITLE, FIXTURE_ARTIST, mix_path, model_key)
    for name, data in stems.items():
        sf.write(
            os.path.join(song.stems_path, f"{name}.wav"), data, SAMPLE_RATE,
        )
    write_completion_marker(song.stems_path, model_key)
    return library, song.id


def _wait_for_detection(app, controls, timeout_s: float) -> bool:
    """Pump events until key and tempo both show a detected result.

    Both badges start empty and pass through "detecting" (and, on the first
    run, "downloading model") states, so "not busy" alone would also match
    the empty moment before detection starts.
    """
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        texts = (
            controls._key_label.text().lower(),
            controls._detected_bpm_label.text().lower(),
        )
        if all(texts) and not any(
            marker in text for marker in _BUSY_MARKERS for text in texts
        ):
            return True
        pump(app, 0.2)
    return False


def _stage_practice(app, window) -> None:
    """Drive the UI into an active practice state through its own buttons.

    An A-B loop around the playhead and a muted drum stem exercise the loop
    shading and lane dimming, and go through the same signal wiring a click
    would rather than poking the player directly.
    """
    controls = window._player_controls
    player = window._player
    total = player.total_seconds
    player.seek(total * 0.25)
    pump(app, 0.1)
    controls._loop_a_btn.click()
    player.seek(total * 0.45)
    pump(app, 0.1)
    controls._loop_b_btn.click()
    if not controls._loop_toggle_btn.isChecked():
        controls._loop_toggle_btn.click()
    drums = controls._stem_rows.get("drums")
    if drums is not None:
        drums._mute_btn.click()
    player.seek(total * 0.34)
    pump(app, 0.4)


STATES = ("empty", "loaded", "practice")


def start_app(data_dir: str):
    """Create the offscreen QApplication with a private LOCALAPPDATA."""
    os.environ["LOCALAPPDATA"] = data_dir
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Deferred so importing this module stays light: the tests import its
    # fixture helpers without creating a QApplication or loading src.ui.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    load_ui_font(app)
    return app


def open_window(app, library, stemma_dir, settings_path, theme, size):
    """Open a MainWindow in *theme* at *size* on a fresh settings file.

    A fresh settings file per window, never the user's real store: each
    window saves its session on close, and the next would otherwise restore
    it (auto-loading the song into the "empty" state, and re-applying the
    previous loop and mute so the practice staging toggled them back off).
    """
    from src.app_settings import SETTINGS_FILE_ENV, open_settings
    from src.model_manager import ModelManager
    from src.player import MultiTrackPlayer
    from src.ui.main_window import MainWindow
    from src.ui.styles import (
        apply_tooltip_palette,
        get_colors,
        get_stylesheet,
    )

    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    if os.path.exists(settings_path):
        os.remove(settings_path)
    os.environ[SETTINGS_FILE_ENV] = settings_path
    # Seed the theme and start the way main.py and src/app.py do, so the
    # window picks it up through its real startup path: theme toggle icon,
    # tooltip palette, and the apply_theme() call that repaints icons drawn
    # before the theme was known.
    seed = open_settings()
    seed.setValue("theme", theme)
    seed.sync()
    app.setStyleSheet(get_stylesheet(theme))
    apply_tooltip_palette(theme)
    window = MainWindow(library, MultiTrackPlayer(), ModelManager(stemma_dir))
    window.apply_theme(theme, get_colors(theme))
    window.resize(*size)
    window.show()
    pump(app, 0.4)
    return window


def load_song(app, window, song_id, timeout_s: float) -> None:
    """Select *song_id* and wait for its detection to settle."""
    window._library_panel.select_song(song_id)
    pump(app, 1.5)
    if not _wait_for_detection(app, window._player_controls, timeout_s):
        print("  note: detection still busy at capture")
    window._player.seek(window._player.total_seconds * 0.34)
    pump(app, 0.4)


def close_window(app, window) -> None:
    window._player.shutdown()
    window._player_controls.shutdown()
    window.close()
    pump(app, 0.2)


def render(out_dir, sizes, themes, stem_count) -> list[dict]:
    """Render every state/theme/size and return one record per image."""
    data_dir = os.path.join(DEFAULT_OUT, ".data")
    app = start_app(data_dir)

    from src.data_paths import platform_user_data_dir

    stemma_dir = platform_user_data_dir()
    library, song_id = prepare_library(stemma_dir, stem_count)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    first_load = True
    for theme in themes:
        for width, height in sizes:
            settings_path = os.path.join(
                data_dir, "settings", f"{theme}-{width}x{height}.ini",
            )
            window = open_window(
                app, library, stemma_dir, settings_path, theme,
                (width, height),
            )

            for state in STATES:
                if state == "loaded":
                    load_song(
                        app, window, song_id,
                        _DETECTION_TIMEOUT_S if first_load else 20.0,
                    )
                    first_load = False
                elif state == "practice":
                    _stage_practice(app, window)
                    # Setting loop points re-runs detection on the loop.
                    pump(app, 0.5)
                    if not _wait_for_detection(
                        app, window._player_controls, 20.0,
                    ):
                        print("  note: detection still busy at capture")

                name = f"{state}_{theme}_{width}x{height}.png"
                window.grab().save(os.path.join(out_dir, name))
                records.append({
                    "file": name, "state": state, "theme": theme,
                    "size": f"{width}x{height}",
                })
                print("wrote", name)

            close_window(app, window)
    return records


def write_index(out_dir: str, records: list[dict], title: str) -> str:
    """Write a contact sheet grouping the images by state, then theme."""
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font:14px system-ui;margin:16px;background:#111;"
        "color:#ddd}figure{display:inline-block;margin:0 12px 16px 0;"
        "vertical-align:top}img{max-width:640px;border:1px solid #444}"
        "figcaption{font-size:12px;color:#aaa}</style>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for state in STATES:
        rows = [r for r in records if r["state"] == state]
        if not rows:
            continue
        parts.append(f"<h2>{html.escape(state)}</h2>")
        for record in rows:
            src = html.escape(record["file"])
            caption = html.escape(f"{record['theme']} {record['size']}")
            parts.append(
                f"<figure><a href='{src}'><img src='{src}'></a>"
                f"<figcaption>{caption}</figcaption></figure>"
            )
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return path


def _parse_sizes(text: str) -> tuple[tuple[int, int], ...]:
    sizes = []
    for token in text.split(","):
        width, height = token.lower().split("x")
        sizes.append((int(width), int(height)))
    return tuple(sizes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(DEFAULT_OUT, "latest"))
    parser.add_argument(
        "--stems", type=int, choices=sorted(MODEL_KEYS), default=4,
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="comma-separated WxH list",
    )
    parser.add_argument(
        "--themes",
        default=",".join(DEFAULT_THEMES),
        help="comma-separated: dark,light",
    )
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    themes = tuple(t for t in args.themes.split(",") if t)
    records = render(out_dir, args.sizes, themes, args.stems)
    index = write_index(
        out_dir, records, f"stemma UI review ({os.path.basename(out_dir)})",
    )
    print("index", index)


if __name__ == "__main__":
    main()
