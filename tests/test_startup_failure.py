"""A failed startup must say so, log it, and exit (#182).

Any exception while building the main window used to leave the splash up
forever with nothing written anywhere (the frozen build has no console), and
the still-held single-instance lock made every relaunch say "already
running".
"""

import json
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import QApplication

import main as stemma_main
from src import crash_log
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def fresh_log(monkeypatch, tmp_path):
    """Install the crash log into a private folder; undo it afterwards."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(crash_log, "_installed_path", None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    handlers = list(crash_log.logger.handlers)
    yield tmp_path
    for handler in crash_log.logger.handlers[len(handlers):]:
        handler.close()
        crash_log.logger.removeHandler(handler)


def _flush():
    for handler in crash_log.logger.handlers:
        handler.flush()


class TestCrashLog:
    def test_install_logs_under_the_data_folder(self, fresh_log):
        path = crash_log.install()

        assert path == str(
            fresh_log / "appdata" / "stemma" / "logs" / "stemma.log"
        )
        assert crash_log.install() == path  # idempotent

    def test_falls_back_to_temp_when_the_data_folder_is_unusable(
        self, fresh_log, monkeypatch,
    ):
        blocker = fresh_log / "appdata"
        blocker.write_text("a file where the folder should be")
        monkeypatch.setattr(
            crash_log.tempfile, "gettempdir", lambda: str(fresh_log / "tmp"),
        )

        path = crash_log.install()

        assert path == str(fresh_log / "tmp" / "stemma" / "stemma.log")

    def test_uncaught_exceptions_reach_the_log(self, fresh_log, monkeypatch):
        path = crash_log.install()
        monkeypatch.setattr(sys, "stderr", None)  # as in the frozen build

        try:
            raise RuntimeError("boom in a slot")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
        worker = threading.Thread(
            target=lambda: (_ for _ in ()).throw(ValueError("boom in thread")),
            name="worker-x",
        )
        worker.start()
        worker.join()
        _flush()

        text = open(path, encoding="utf-8").read()
        assert "boom in a slot" in text
        assert "boom in thread" in text and "worker-x" in text

    def test_real_log_path_resolves_to_an_existing_file(self, fresh_log):
        crash_log.install()
        crash_log.logger.info("hello")
        _flush()

        real = crash_log.real_log_path()

        assert real is not None
        assert open(real, encoding="utf-8").read().endswith("hello\n")


class TestStartupFailure:
    def _run(self, qapp, exc):
        splash = MagicMock()
        fake_app = MagicMock()
        with patch("src.app.build_and_show", side_effect=exc), patch.object(
            stemma_main, "QMessageBox",
        ) as box:
            stemma_main._finish_startup(fake_app, MagicMock(), "dark", splash)
        return splash, fake_app, box

    def test_failure_closes_the_splash_explains_and_exits(
        self, qapp, fresh_log,
    ):
        log = crash_log.install()
        folder = r"C:\Users\someone\AppData\Local\stemma"

        splash, fake_app, box = self._run(
            qapp, PermissionError(13, "Access is denied", folder),
        )

        splash.abort.assert_called_once_with()
        fake_app.exit.assert_called_once_with(1)
        title, text = box.critical.call_args.args[1:3]
        assert title == "stemma could not start"
        assert folder in text and "write to it" in text
        assert "Details were saved to" in text
        _flush()
        assert "Startup failed" in open(log, encoding="utf-8").read()

    def test_other_failures_get_a_generic_message(self, qapp, fresh_log):
        crash_log.install()

        _, _, box = self._run(qapp, KeyError("window/geometry"))

        text = box.critical.call_args.args[2]
        assert text.startswith("stemma ran into a problem while starting")
        assert "KeyError" not in text


class TestDamagedSettings:
    @pytest.mark.parametrize("value", ["garbage", 42, ["a"], {"x": 1}])
    def test_wrong_typed_geometry_is_ignored(self, qapp, value):
        stub = MagicMock()
        stub._settings.value.return_value = value

        MainWindow._restore_state(stub)

        stub.restoreGeometry.assert_not_called()
        stub.restoreState.assert_not_called()

    def test_saved_geometry_is_still_restored(self, qapp):
        stub = MagicMock()
        stub._settings.value.return_value = QByteArray(b"\x01\xd9\xd0\xcb")

        MainWindow._restore_state(stub)

        stub.restoreGeometry.assert_called_once()
        stub.restoreState.assert_called_once()

    @pytest.mark.parametrize("stored, names, numbers", [
        (json.dumps(["drums", 3, ["x"]]), {"drums"}, {}),
        (json.dumps({"drums": 0.5, "bass": "loud", "x": True}),
         set(), {"drums": 0.5}),
        ('{"drums": NaN, "bass": Infinity, "other": 1}', set(),
         {"other": 1.0}),
        ("not json", set(), {}),
        (None, set(), {}),
    ])
    def test_session_values_of_the_wrong_shape_fall_back(
        self, qapp, stored, names, numbers,
    ):
        stub = MagicMock()
        stub._settings.value.return_value = stored
        stub._session_json = lambda *a: MainWindow._session_json(stub, *a)

        assert MainWindow._session_names(stub, "session/muted_stems") == names
        assert MainWindow._session_numbers(stub, "session/volumes") == numbers


class TestEarlyStartupFailure:
    def test_failure_before_the_splash_still_explains_itself(
        self, qapp, fresh_log, monkeypatch,
    ):
        """get_stylesheet, the splash, or settings failing inside main()
        used to escape with no message at all."""
        monkeypatch.setattr(
            stemma_main, "QApplication", MagicMock(return_value=qapp),
        )
        monkeypatch.setattr(stemma_main, "QSharedMemory", MagicMock())
        monkeypatch.setattr(
            stemma_main, "get_stylesheet",
            MagicMock(side_effect=RuntimeError("bad theme")),
        )
        with patch.object(stemma_main, "QMessageBox") as box:
            assert stemma_main.main() == 1

        assert box.critical.call_args.args[1] == "stemma could not start"
