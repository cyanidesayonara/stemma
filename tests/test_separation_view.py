"""The empty player shows a separating import instead of sitting blank (#159)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.separation_view import (
    SeparationEta,
    SeparationView,
    format_eta,
    stage_for,
)


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
        # 1 % every 2 s from 20 %. The separation ends at 90 %; saving
        # after it is a short fixed tail, not 10 % more at this pace.
        for i in range(21):
            left = eta.update(20 + i, 2.0 * i)
        assert left == pytest.approx((90 - 40) * 2.0 + 5.0, rel=0.05)

    def test_near_the_end_counts_only_the_separation_left(self):
        eta = SeparationEta()
        # About 3.2 s per % (a 4-minute separation): at 85 % some 20 s
        # remain, not the 50 s that counting up to 100 % would say.
        for p in range(15, 86):
            left = eta.update(p, 3.2 * (p - 15))
        assert format_eta(left) == "About 30 s left"

    @pytest.mark.parametrize("percent", [90, 95, 100])
    def test_almost_done_once_the_separation_ends(self, percent):
        eta = SeparationEta()
        assert format_eta(eta.update(percent, 0.0)) == "Almost done"

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
        view.update_progress(15, now=0.0)
        view.update_progress(40, now=50.0)
        assert view.percent == 40
        assert view.stage == "Separating stems…"
        assert view.time_left == "40% · About 2 min left"
        view.update_progress(92, now=160.0)
        assert view.stage == "Saving stems…"
        assert view.time_left == "92% · Almost done"

    @pytest.mark.parametrize("percent, stage", [
        (0, "Loading audio…"),
        (5, "Preparing the model…"),
        (12, "Preparing the model…"),
        (57, "Separating stems…"),
        (95, "Saving stems…"),
    ])
    def test_plain_stages(self, percent, stage):
        assert stage_for(percent) == stage

    def test_long_title_is_elided_and_leaves_the_minimum_width(self, app):
        view = SeparationView()
        view.start("Short", "Artist")
        short_min = view.minimumSizeHint().width()
        long_title = "Symphony No. 9 in D minor, Op. 125 " * 6
        long_artist = "Berliner Philharmoniker, Herbert von Karajan, " * 4
        view.start(long_title, long_artist)
        assert view.minimumSizeHint().width() == short_min
        view.resize(short_min, 300)
        view.show()
        app.processEvents()
        label = view._title
        assert label.width() <= short_min
        assert label.text() != long_title
        assert label.text().endswith("…")
        assert label.toolTip() == long_title
        assert view.title == long_title
        assert view._artist.toolTip() == long_artist
        view.close()


def _stub(has_stems=False, loading=None, pending=1):
    stub = MagicMock()
    stub._player.has_stems = has_stems
    stub._loading_song_id = loading
    stub._separation_queue.pending_count = pending
    stub._library.get_song.return_value = SimpleNamespace(
        title="Estranged", artist="Guns N' Roses",
    )
    stub._watched_separation = None
    stub._separation_percent = (None, 0)
    stub._current_song_id = None
    stub._refresh_separation_queued = (
        lambda: MainWindow._refresh_separation_queued(stub)
    )
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
        MainWindow._on_separation_progress(stub, "s1", 42, "Segment 3/9")
        stub._player_controls.update_separation.assert_called_once_with(42)

    def test_rewatching_a_running_job_resumes_its_progress(self, app):
        stub = _stub()
        MainWindow._on_separation_started(stub, "s1")
        stub._player.has_stems = True  # A song was opened...
        MainWindow._watch_separation(stub, "s1")
        MainWindow._on_separation_progress(stub, "s1", 57, "Segment")
        stub._player_controls.update_separation.reset_mock()
        stub._player.has_stems = False  # ...and closed mid-job.
        MainWindow._watch_separation(stub, "s1")
        stub._player_controls.update_separation.assert_called_once_with(57)

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
        stub._current_song_id = "other"
        MainWindow._on_separation_finished(stub, "s1", "htdemucs")
        stub._library_panel.select_song.assert_not_called()

    def test_queue_count_follows_new_imports_and_cancels(self, app):
        stub = _stub(pending=1)
        MainWindow._on_separation_started(stub, "s1")
        view = stub._player_controls.separation_view
        view.set_queued.assert_called_with(0)
        stub._separation_queue.pending_count = 3
        MainWindow._on_separation_queued(stub, "s2")
        view.set_queued.assert_called_with(2)
        stub._damaged_prompt_open = False
        stub._separation_queue.pending_count = 2
        MainWindow._on_separation_failed(stub, "s2", "Separation cancelled.")
        view.set_queued.assert_called_with(1)
        assert stub._watched_separation == "s1"

    def test_failure_hides_the_view(self, app):
        stub = _stub()
        stub._damaged_prompt_open = False
        MainWindow._on_separation_started(stub, "s1")
        stub._library.get_song.return_value = None
        MainWindow._on_separation_failed(stub, "s1", "Separation cancelled.")
        stub._player_controls.hide_separation.assert_called()
        assert stub._watched_separation is None
