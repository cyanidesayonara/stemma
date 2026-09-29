"""The empty player shows a separating import instead of sitting blank (#159)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.separation_view import SeparationEta, SeparationView, format_eta


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class TestEta:
    def test_unsure_before_the_separation_itself_runs(self):
        eta = SeparationEta()
        assert eta.update(5, 0.0) is None
        assert eta.update(10, 3.0) is None

    def test_steady_progress_gives_the_time_left(self):
        eta = SeparationEta()
        # 1 % every 2 s from 20 %.
        for i in range(21):
            left = eta.update(20 + i, 2.0 * i)
        assert left == pytest.approx((100 - 40) * 2.0, rel=0.15)

    def test_needs_a_few_seconds_of_progress_first(self):
        eta = SeparationEta()
        eta.update(20, 0.0)
        assert eta.update(21, 1.0) is None

    @pytest.mark.parametrize("seconds, text", [
        (None, "Estimating time left…"),
        (4, "Almost done"),
        (42, "About 50 s left"),
        (61, "About 2 min left"),
        (179, "About 3 min left"),
    ])
    def test_format(self, seconds, text):
        assert format_eta(seconds) == text


class TestView:
    def test_start_and_progress(self, app):
        view = SeparationView()
        view.start("Estranged", "Guns N' Roses")
        assert view.title == "Estranged"
        assert view.stage == "Starting…"
        view.update_progress(15, "Separating stems...", now=0.0)
        view.update_progress(40, "Separating stems...", now=50.0)
        assert view.percent == 40
        assert view.stage == "Separating stems..."
        assert view.time_left == "40% · About 2 min left"


def _stub(has_stems=False, loading=None, pending=1):
    stub = MagicMock()
    stub._player.has_stems = has_stems
    stub._loading_song_id = loading
    stub._separation_queue.pending_count = pending
    stub._library.get_song.return_value = SimpleNamespace(
        title="Estranged", artist="Guns N' Roses",
    )
    stub._watched_separation = None
    stub._watch_separation = lambda sid: MainWindow._watch_separation(
        stub, sid,
    )
    return stub


class TestMainWindowFlow:
    def test_empty_player_shows_the_separation(self, app):
        stub = _stub(pending=3)
        MainWindow._on_separation_started(stub, "s1")
        stub._player_controls.show_separation.assert_called_once_with(
            "Estranged", "Guns N' Roses",
        )
        stub._player_controls.separation_view.set_queued.assert_called_with(2)
        assert stub._watched_separation == "s1"

    @pytest.mark.parametrize("has_stems, loading", [(True, None), (False, "x")])
    def test_a_song_in_the_player_is_left_alone(self, app, has_stems, loading):
        stub = _stub(has_stems=has_stems, loading=loading)
        MainWindow._on_separation_started(stub, "s1")
        stub._player_controls.show_separation.assert_not_called()
        assert stub._watched_separation is None

    def test_progress_reaches_the_view(self, app):
        stub = _stub()
        MainWindow._on_separation_started(stub, "s1")
        MainWindow._on_separation_progress(stub, "s1", 42, "Separating...")
        stub._player_controls.update_separation.assert_called_once_with(
            42, "Separating...",
        )

    def test_finished_song_opens(self, app):
        stub = _stub()
        MainWindow._on_separation_started(stub, "s1")
        MainWindow._on_separation_finished(stub, "s1", "htdemucs_6s")
        stub._player_controls.hide_separation.assert_called()
        stub._library_panel.select_song.assert_called_once_with("s1")
        assert stub._watched_separation is None

    def test_finished_song_waits_if_another_was_opened(self, app):
        stub = _stub()
        MainWindow._on_separation_started(stub, "s1")
        stub._player.has_stems = True
        MainWindow._on_separation_finished(stub, "s1", "htdemucs")
        stub._library_panel.select_song.assert_not_called()

    def test_failure_hides_the_view(self, app):
        stub = _stub()
        stub._damaged_prompt_open = False
        MainWindow._on_separation_started(stub, "s1")
        stub._library.get_song.return_value = None
        MainWindow._on_separation_failed(stub, "s1", "Separation cancelled.")
        stub._player_controls.hide_separation.assert_called()
        assert stub._watched_separation is None
