"""Tests for the UI review renderer's fixture and contact sheet."""

import numpy as np
import pytest

from scripts.render_ui_review import (
    FIXTURE_TITLE,
    SAMPLE_RATE,
    _parse_sizes,
    prepare_library,
    synth_stems,
    write_index,
)
from src.separation_state import separation_is_complete


@pytest.mark.parametrize("count, names", [
    (4, {"vocals", "drums", "bass", "other"}),
    (6, {"vocals", "drums", "bass", "other", "guitar", "piano"}),
])
def test_synth_stems_are_deterministic_stereo(count, names):
    first = synth_stems(count, seconds=1.0)
    second = synth_stems(count, seconds=1.0)

    assert set(first) == names
    for name, data in first.items():
        assert data.shape == (SAMPLE_RATE, 2)
        assert data.dtype == np.float32
        assert np.abs(data).max() <= 1.0
        np.testing.assert_array_equal(data, second[name])


def test_prepared_song_survives_the_startup_prune(tmp_path):
    """MainWindow prunes songs that do not look fully separated; the fixture
    must look exactly like a finished import or it silently vanishes."""
    library, song_id = prepare_library(str(tmp_path), 4)
    song = library.get_song(song_id)

    assert [s.title for s in library.songs] == [FIXTURE_TITLE]
    assert separation_is_complete(song.stems_path, song.model_used)


def test_prepare_library_resets_to_one_song(tmp_path):
    prepare_library(str(tmp_path), 4)
    library, _ = prepare_library(str(tmp_path), 6)

    assert len(library.songs) == 1
    assert library.songs[0].model_used == "htdemucs_6s"


def test_contact_sheet_groups_by_state(tmp_path):
    records = [
        {"file": "loaded_dark_900x600.png", "state": "loaded",
         "theme": "dark", "size": "900x600"},
        {"file": "empty_dark_900x600.png", "state": "empty",
         "theme": "dark", "size": "900x600"},
    ]
    html = open(write_index(str(tmp_path), records, "t"), encoding="utf-8").read()

    assert html.index("<h2>empty</h2>") < html.index("<h2>loaded</h2>")
    assert "loaded_dark_900x600.png" in html


def test_parse_sizes():
    assert _parse_sizes("900x600,1920X1080") == ((900, 600), (1920, 1080))
