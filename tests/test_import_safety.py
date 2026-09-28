"""Import safety: file validation, download consent, damaged models.

Covers the "Import and model download" part of issue #185.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

from src import downloader
from src.import_messages import (
    MSG_MODEL_DAMAGED,
    MSG_UNREADABLE_AUDIO,
    MSG_YT_ONLY,
    ModelDamagedError,
)
from src.library import SongLibrary
from src.mdx_separator import MdxSeparatorWorker
from src.model_manager import ModelManager
from src.onnx_session import create_onnx_session
from src.separator import SeparatorWorker, to_stereo
from src.ui import import_dialog as import_dialog_module
from src.ui.import_dialog import ImportDialog

_YES = QMessageBox.StandardButton.Yes
_NO = QMessageBox.StandardButton.No


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wav(path, channels=2, seconds=0.1, sample_rate=44100):
    frames = int(sample_rate * seconds)
    data = np.full((frames, channels), 0.1, dtype=np.float32)
    sf.write(str(path), data, sample_rate)
    return str(path)


def _dialog(tmp_path, *, queue=None):
    library = SongLibrary(str(tmp_path / "data"))
    manager = ModelManager(data_dir=str(tmp_path / "data"))
    dlg = ImportDialog(library, manager, separation_queue=queue or MagicMock())
    return dlg, library, manager


def _ok(dlg):
    return dlg._button_box.button(QDialogButtonBox.StandardButton.Ok)


def _cancel(dlg):
    return dlg._button_box.button(QDialogButtonBox.StandardButton.Cancel)


class TestToStereo:
    def test_mono_is_duplicated(self):
        out = to_stereo(np.ones((1, 10), dtype=np.float32))
        assert out.shape == (2, 10)

    def test_stereo_is_unchanged(self):
        audio = np.random.default_rng(0).random((2, 10)).astype(np.float32)
        assert to_stereo(audio) is audio

    def test_surround_is_downmixed_not_truncated(self):
        audio = np.zeros((6, 100), dtype=np.float32)
        audio[2] = 0.5  # center channel only
        out = to_stereo(audio)
        assert out.shape == (2, 100)
        # The center channel is heard on both sides, not dropped.
        assert np.all(out[0] > 0)
        assert np.allclose(out[0], out[1])

    def test_downmix_never_clips(self):
        out = to_stereo(np.ones((6, 50), dtype=np.float32))
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6

    @pytest.mark.parametrize("worker_cls", [SeparatorWorker, MdxSeparatorWorker])
    def test_workers_load_six_channels_as_stereo(self, tmp_path, worker_cls):
        # Signal only in the centre channel: keeping the first two channels
        # would load silence.
        data = np.zeros((4410, 6), dtype=np.float32)
        data[:, 2] = 0.5
        path = str(tmp_path / "surround.wav")
        sf.write(path, data, 44100)
        worker = worker_cls(
            input_path=path, output_dir=str(tmp_path / "out"),
            model_path="unused.onnx",
        )
        audio, _sr = worker._load_audio()
        assert audio.shape[0] == 2
        assert np.abs(audio[0]).max() > 0.05
        assert np.abs(audio[1]).max() > 0.05


class TestStatusLabelFits:
    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_long_status_is_not_clipped(self, qapp, tmp_path, theme):
        from src.ui.styles import get_stylesheet
        from tests.widget_visual import deterministic_render_state

        with deterministic_render_state():
            qapp.setStyleSheet(get_stylesheet(theme))
            dlg, _library, _manager = _dialog(tmp_path)
            dlg.show()
            qapp.processEvents()
            dlg._on_error(
                "The separation model on this PC was damaged, so stemma "
                "removed it. Import the song again to download a fresh "
                "copy. " + MSG_UNREADABLE_AUDIO
            )
            qapp.processEvents()
            qapp.processEvents()
            label = dlg._status_label
            assert label.height() >= label.heightForWidth(label.width())
            dlg.close()
            dlg.deleteLater()


class TestValidateBeforeImport:
    def test_unreadable_file_adds_nothing_and_downloads_nothing(
        self, qapp, tmp_path,
    ):
        bad = tmp_path / "broken.mp3"
        bad.write_bytes(b"ID3" + bytes(500))
        dlg, library, manager = _dialog(tmp_path)
        with patch.object(manager, "download_model") as download, patch.object(
            import_dialog_module.QMessageBox, "question",
        ) as ask:
            dlg._start_local_import(str(bad))
        assert library.songs == []
        download.assert_not_called()
        ask.assert_not_called()
        assert dlg._status_label.text() == MSG_UNREADABLE_AUDIO
        assert _ok(dlg).isEnabled()
        dlg.close()


class TestDownloadConsent:
    def test_asks_with_model_name_and_size_before_adding_song(
        self, qapp, tmp_path,
    ):
        source = _wav(tmp_path / "song.wav")
        dlg, library, manager = _dialog(tmp_path)
        with patch.object(
            import_dialog_module.QMessageBox, "question", return_value=_NO,
        ) as ask, patch.object(manager, "download_model") as download:
            dlg._start_local_import(source)
        text = ask.call_args.args[2]
        assert "4-stem model" in text
        assert "172 MB" in text
        assert ".onnx" not in text
        assert library.songs == []
        download.assert_not_called()
        assert _ok(dlg).isEnabled()
        dlg.close()

    def test_yes_adds_song_and_starts_download(self, qapp, tmp_path):
        source = _wav(tmp_path / "song.wav")
        dlg, library, manager = _dialog(tmp_path)
        fake_dl = MagicMock()
        with patch.object(
            import_dialog_module.QMessageBox, "question", return_value=_YES,
        ), patch.object(manager, "download_model", return_value=fake_dl):
            dlg._start_local_import(source)
        assert len(library.songs) == 1
        fake_dl.start.assert_called_once()
        # Cancel stays available so the user can stop the download.
        assert _cancel(dlg).isEnabled()
        assert not _ok(dlg).isEnabled()
        assert "4-stem model" in dlg._status_label.text()
        dlg.close()

    def test_no_prompt_when_model_is_cached(self, qapp, tmp_path):
        source = _wav(tmp_path / "song.wav")
        queue = MagicMock()
        dlg, library, manager = _dialog(tmp_path, queue=queue)
        with patch.object(
            manager, "is_model_downloaded", return_value=True,
        ), patch.object(
            import_dialog_module.QMessageBox, "question",
        ) as ask:
            dlg._start_local_import(source)
        ask.assert_not_called()
        assert queue.enqueue.call_count == 1

    def test_youtube_asks_before_downloading_audio(self, qapp, tmp_path):
        dlg, _library, _manager = _dialog(tmp_path)
        dlg._url_edit.setText("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        with patch.object(
            import_dialog_module, "check_ffmpeg", return_value=True,
        ), patch.object(
            import_dialog_module.QMessageBox, "question", return_value=_NO,
        ) as ask, patch.object(
            import_dialog_module, "_DownloadWorker",
        ) as worker_cls:
            dlg._on_import()
        ask.assert_called_once()
        worker_cls.assert_not_called()
        assert _ok(dlg).isEnabled()
        dlg.close()


class TestYouTubeField:
    def test_non_youtube_url_says_so(self, qapp, tmp_path):
        dlg, library, _manager = _dialog(tmp_path)
        dlg._url_edit.setText("https://vimeo.com/12345")
        dlg._on_import()
        assert dlg._status_label.text() == MSG_YT_ONLY
        assert library.songs == []
        assert _ok(dlg).isEnabled()
        dlg.close()

    def test_ffmpeg_message_for_installed_build_never_mentions_path(
        self, monkeypatch,
    ):
        monkeypatch.setattr(downloader.sys, "frozen", True, raising=False)
        text = downloader.ffmpeg_missing_message()
        assert "PATH" not in text
        assert "reinstall" in text.lower()

    def test_ffmpeg_message_for_source_run(self, monkeypatch):
        monkeypatch.delattr(downloader.sys, "frozen", raising=False)
        assert "PATH" in downloader.ffmpeg_missing_message()

    def test_dialog_uses_the_install_aware_message(self, qapp, tmp_path):
        dlg, _library, _manager = _dialog(tmp_path)
        dlg._url_edit.setText("https://youtu.be/abc")
        with patch.object(
            import_dialog_module, "check_ffmpeg", return_value=False,
        ), patch.object(
            import_dialog_module, "ffmpeg_missing_message",
            return_value="reinstall",
        ), patch.object(
            import_dialog_module.QMessageBox, "critical",
        ) as critical:
            dlg._on_import()
        assert critical.call_args.args[2] == "reinstall"
        dlg.close()


class TestDamagedModel:
    def _garbage_model(self, tmp_path):
        path = tmp_path / "models" / "htdemucs.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(range(200)))
        (tmp_path / "models" / "htdemucs.onnx.data").write_bytes(b"w" * 10)
        return str(path)

    def test_session_reports_damaged_model(self, tmp_path):
        with pytest.raises(ModelDamagedError):
            create_onnx_session(self._garbage_model(tmp_path))

    def test_worker_deletes_it_and_says_so(self, tmp_path):
        model = self._garbage_model(tmp_path)
        worker = SeparatorWorker(
            input_path=_wav(tmp_path / "in.wav"),
            output_dir=str(tmp_path / "out"),
            model_path=model,
        )
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        assert errors == [MSG_MODEL_DAMAGED]
        assert not os.path.exists(model)
        assert not os.path.exists(model + ".data")

    def test_failure_dialog_offers_a_new_import(self, qapp):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.get_song.return_value = None
        stub._damaged_prompt_open = False
        with patch("src.ui.main_window.QMessageBox") as mb:
            mb.StandardButton = QMessageBox.StandardButton
            mb.question.return_value = QMessageBox.StandardButton.Yes
            MainWindow._on_separation_failed(stub, "s1", MSG_MODEL_DAMAGED)
        mb.question.assert_called_once()
        mb.warning.assert_not_called()
        stub._on_import.assert_called_once()
        assert stub._damaged_prompt_open is False

    def test_queued_failures_prompt_once(self, qapp):
        """The model is deleted by the first failure, so every queued job
        fails the same way; only the first one asks."""
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.get_song.return_value = None
        stub._damaged_prompt_open = False

        def import_now():
            # A second queued job fails while the first prompt's import
            # dialog is still open.
            MainWindow._on_separation_failed(stub, "s2", MSG_MODEL_DAMAGED)

        stub._on_import.side_effect = import_now
        with patch("src.ui.main_window.QMessageBox") as mb:
            mb.StandardButton = QMessageBox.StandardButton
            mb.question.return_value = QMessageBox.StandardButton.Yes
            MainWindow._on_separation_failed(stub, "s1", MSG_MODEL_DAMAGED)
        mb.question.assert_called_once()
        mb.warning.assert_not_called()

    def test_damaged_beat_model_is_deleted_for_redownload(self, tmp_path):
        from src import beat_detector

        model = tmp_path / "beat_this.onnx"
        model.write_bytes(b"not a model")
        sr = 22050
        audio = np.random.default_rng(0).standard_normal((sr * 12, 2)) * 0.1
        with patch.object(
            beat_detector, "create_onnx_session",
            side_effect=ModelDamagedError(str(model)),
        ):
            beat_detector.detect_bpm_and_key(
                {"mix": audio.astype(np.float32)}, sr, model_path=str(model),
            )
        assert not model.exists()
