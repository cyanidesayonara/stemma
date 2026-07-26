"""Generate Microsoft Store screenshots from the real application UI.

Renders the actual MainWindow offscreen at Store dimensions, so the
screenshots always match the shipped build instead of being hand-captured
and going stale.

    python scripts/generate_screenshots.py [song_dir]

Outputs 1366x768 PNGs (Partner Center desktop minimum) to
assets/store_listing/screenshots/.

By default it uses the newest fully separated song in the per-user data
directory. Pass a song directory to pick one deliberately -- for store
art, choose a track whose waveform looks interesting.

Note: the offscreen QPA platform registers no system fonts, so text
would render as empty boxes. Segoe UI is loaded from disk explicitly
before any widget is built.
"""

from __future__ import annotations

import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_OUT_DIR = os.path.join(_ROOT, "assets", "store_listing", "screenshots")
_SIZE = (1366, 768)
_UI_FONTS = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)
_STEM_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")


def _load_ui_font(app) -> None:
    """Register a real UI font: offscreen Qt otherwise draws tofu boxes."""
    from PySide6.QtGui import QFont, QFontDatabase

    for path in _UI_FONTS:
        if not os.path.isfile(path):
            continue
        fid = QFontDatabase.addApplicationFont(path)
        families = QFontDatabase.applicationFontFamilies(fid)
        if families:
            app.setFont(QFont(families[0], 9))
            print(f"  ui font: {families[0]}")
            return
    print("  WARNING: no UI font found; text may render as boxes")


def _pump(app, seconds: float) -> None:
    """Process events for *seconds* so async work (peaks) can land."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _pick_song(data_dir: str) -> str | None:
    """Newest song directory that has at least two stems on disk."""
    songs_dir = os.path.join(data_dir, "songs")
    if not os.path.isdir(songs_dir):
        return None
    candidates = []
    for name in os.listdir(songs_dir):
        d = os.path.join(songs_dir, name)
        stems = [
            s for s in _STEM_NAMES
            if os.path.isfile(os.path.join(d, f"{s}.wav"))
        ]
        if len(stems) >= 2:
            candidates.append((os.path.getmtime(d), d))
    if not candidates:
        return None
    return max(candidates)[1]


def main() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    _load_ui_font(app)

    from src.data_paths import platform_user_data_dir
    from src.library import SongLibrary
    from src.model_manager import ModelManager
    from src.player import MultiTrackPlayer
    from src.ui.main_window import MainWindow
    from src.ui.styles import get_colors, get_stylesheet

    data_dir = platform_user_data_dir()
    library = SongLibrary(data_dir)

    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    if wanted is None:
        wanted = _pick_song(data_dir)
    song_id = None
    if wanted:
        target = os.path.normcase(os.path.abspath(wanted))
        for song in library.songs:
            if os.path.normcase(os.path.abspath(song.stems_path)) == target:
                song_id = song.id
                break
    if song_id is None and library.songs:
        song_id = library.songs[0].id

    os.makedirs(_OUT_DIR, exist_ok=True)
    written: list[str] = []

    for theme in ("dark", "light"):
        app.setStyleSheet(get_stylesheet(theme))
        window = MainWindow(library, MultiTrackPlayer(), ModelManager(data_dir))
        window._theme = theme
        window.apply_theme(theme, get_colors(theme))
        window.resize(*_SIZE)
        window.show()
        _pump(app, 0.6)

        if theme == "dark":
            out = os.path.join(_OUT_DIR, "01_empty_dark.png")
            window.grab().save(out)
            written.append(out)

        if song_id is not None:
            window._library_panel.select_song(song_id)
            # Peaks are computed on a worker thread; give it time to land
            # so the waveform is populated rather than a flat line.
            _pump(app, 3.5)
            window._player.seek(window._player.total_seconds * 0.34)
            _pump(app, 0.5)
            out = os.path.join(_OUT_DIR, f"02_player_{theme}.png")
            window.grab().save(out)
            written.append(out)

        window._player.shutdown()
        window._player_controls.shutdown()
        window.close()
        _pump(app, 0.2)

    for path in written:
        print("wrote", os.path.relpath(path, _ROOT))


if __name__ == "__main__":
    main()
