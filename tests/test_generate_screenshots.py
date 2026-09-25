"""Tests for the Store screenshot generator's definitions and fixtures."""

import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from scripts.generate_screenshots import (
    CANVAS,
    LEGACY_SHOTS,
    SHOTS,
    clear_previous_set,
    compose,
    import_song_dir,
    write_takes,
)
from scripts.render_ui_review import part_to_mute
from src.separation_state import separation_is_complete
from src.ui.styles import DARK_COLORS


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _write_stems(folder, names, seconds=1.0, sr=22050):
    os.makedirs(folder, exist_ok=True)
    tone = np.sin(np.linspace(0, 440 * 2 * np.pi, int(sr * seconds)))
    data = np.stack([tone, tone], axis=1).astype(np.float32) * 0.5
    for name in names:
        sf.write(os.path.join(folder, f"{name}.wav"), data, sr)


def test_shot_definitions_are_complete():
    names = [shot.name for shot in SHOTS]
    assert names == sorted(names), "file names must sort into Store order"
    assert len(set(names)) == len(names)
    assert all(shot.caption for shot in SHOTS)
    assert all(shot.theme in ("dark", "light") for shot in SHOTS)
    # Composed shots lead, as the #146 plan says.
    composed = [shot.composed for shot in SHOTS]
    assert composed == sorted(composed, reverse=True)
    for shot in SHOTS:
        if shot.composed:
            assert shot.subline and shot.chips


@pytest.mark.parametrize("stems, model", [
    (("vocals", "other"), "mdx_inst_hq3"),
    (("vocals", "drums", "bass", "other"), "htdemucs"),
    (("vocals", "drums", "bass", "other", "guitar", "piano"), "htdemucs_6s"),
])
def test_import_song_dir_looks_like_a_finished_import(tmp_path, stems, model):
    source = tmp_path / "song"
    _write_stems(source, stems)

    library, song_id = import_song_dir(
        str(tmp_path / "data"), str(source), "Title", "Artist",
    )
    song = library.get_song(song_id)

    assert song.model_used == model
    assert separation_is_complete(song.stems_path, song.model_used)
    assert [s.title for s in library.songs] == ["Title"]


@pytest.mark.parametrize("stems", [
    ("vocals", "drums", "bass"),
    # Two stems, but not the vocals + other pair MDX writes.
    ("vocals", "drums"),
])
def test_import_song_dir_rejects_an_unsupported_stem_set(tmp_path, stems):
    source = tmp_path / "song"
    _write_stems(source, stems)

    with pytest.raises(SystemExit, match="need one of"):
        import_song_dir(str(tmp_path / "data"), str(source), "T", "A")


def test_import_song_dir_rejects_a_missing_folder(tmp_path):
    with pytest.raises(SystemExit, match="not a folder"):
        import_song_dir(
            str(tmp_path / "data"), str(tmp_path / "missing"), "T", "A",
        )


def test_clear_previous_set_removes_only_screenshot_files(tmp_path):
    ours = [f"{shot.name}.png" for shot in SHOTS] + list(LEGACY_SHOTS)
    for name in ours + ["holiday.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")

    clear_previous_set(str(tmp_path))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "holiday.png", "notes.txt",
    ]


@pytest.mark.parametrize("stems, muted", [
    (("vocals", "drums", "bass", "other"), "drums"),
    (("vocals", "drums", "bass", "other", "guitar", "piano"), "drums"),
    # Two-stem songs have no drums: mute the vocal and sing along.
    (("vocals", "other"), "vocals"),
])
def test_practice_staging_mutes_a_part_every_stem_set_has(stems, muted):
    assert part_to_mute(stems) == muted


def test_write_takes_adds_one_take(tmp_path):
    """One take only: at the two-take limit Record disables itself."""
    _write_stems(tmp_path, ("vocals", "other"))

    write_takes(str(tmp_path))

    takes = sorted(p.name for p in tmp_path.glob("recording_take*.wav"))
    assert takes == ["recording_take1.wav"]
    take, _ = sf.read(tmp_path / "recording_take1.wav")
    source, _ = sf.read(tmp_path / "other.wav")
    assert take.shape == source.shape


def test_compose_fills_the_store_canvas(app):
    window = QImage(1366, 768, QImage.Format.Format_ARGB32)
    window.fill(0xFF303040)

    image = compose(window, SHOTS[0], DARK_COLORS, app.font().family())

    assert (image.width(), image.height()) == CANVAS
    # The window lands in the right half, the text in the left.
    assert image.pixelColor(1500, 540).name() == "#303040"
    assert image.pixelColor(1500, 540) != image.pixelColor(300, 900)
