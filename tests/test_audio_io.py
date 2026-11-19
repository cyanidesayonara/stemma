"""Source audio loading gets the full decoded length (issue #206).

A VBR MP3 without a Xing/VBRI header passes ``sf.info``, but libsndfile
estimates its frame count from the first frame's bitrate and then reads
only that many frames. Import would then separate a truncated song with
no error.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from src import audio_io
from src.mdx_separator import MdxSeparatorWorker
from src.separator import SeparatorWorker
from src.ui.import_dialog import ImportDialog

_SR = 44100
_SECONDS = 3
_FRAMES = _SR * _SECONDS
# MP3 decoders differ in how much encoder delay and padding they trim
# once the LAME tag is gone; allow a couple of 1152-sample MP3 frames.
_SLACK = 3 * 1152


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _headerless_vbr_mp3(path: str) -> str:
    """Write a 3 s stereo VBR MP3 and blank its Xing tag.

    Quiet first second, then a loud noisy tone, so the first frame's
    bitrate is far below the average and libsndfile under-estimates the
    length.
    """
    t = np.arange(_FRAMES) / _SR
    rng = np.random.default_rng(0)
    loud = 0.5 * np.sin(2 * np.pi * 440 * t) + rng.normal(0, 0.2, t.size)
    mono = np.where(t < 1.0, 0.0, loud).astype(np.float32)
    sf.write(
        path, np.stack([mono, mono], axis=1), _SR,
        format="MP3", bitrate_mode="VARIABLE", compression_level=0.5,
    )
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    # The Xing (or Info) tag sits inside the first MPEG frame.
    for tag in (b"Xing", b"Info"):
        pos = data.find(tag, 0, 200)
        if pos >= 0:
            data[pos:pos + 4] = b"\0\0\0\0"
            break
    else:
        raise AssertionError("encoder wrote no Xing/Info tag to strip")
    with open(path, "wb") as fh:
        fh.write(bytes(data))
    return path


@pytest.fixture
def headerless_mp3(tmp_path):
    path = _headerless_vbr_mp3(str(tmp_path / "headerless_vbr.mp3"))
    # Guard: the fixture must actually trigger the libsndfile estimate.
    assert sf.info(path).frames < _FRAMES * 0.9
    return path


def _assert_full_length(frames: int) -> None:
    assert _FRAMES <= frames <= _FRAMES + _SLACK


class TestReadAudio:
    def test_headerless_vbr_mp3_reads_full_length(self, headerless_mp3):
        audio, sr = audio_io.read_audio(headerless_mp3)
        assert sr == _SR
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        assert audio.dtype == np.float32
        _assert_full_length(audio.shape[0])
        # The loud part at the end is really there, not zero padding.
        assert np.abs(audio[-_SR // 2:]).max() > 0.1

    def test_wav_reads_through_soundfile(self, tmp_path):
        path = str(tmp_path / "tone.wav")
        data = np.full((4410, 2), 0.25, dtype=np.float32)
        sf.write(path, data, _SR)
        audio, sr = audio_io.read_audio(path)
        assert sr == _SR
        assert audio.shape == (4410, 2)
        assert audio.dtype == np.float32

    def test_mono_mp3_stays_2d(self, tmp_path):
        path = str(tmp_path / "mono.mp3")
        sf.write(path, np.zeros(_SR, dtype=np.float32), _SR, format="MP3")
        audio, _sr = audio_io.read_audio(path)
        assert audio.ndim == 2
        assert audio.shape[1] == 1

    def test_falls_back_to_soundfile_without_ffmpeg(self, headerless_mp3):
        with patch.object(audio_io, "_ffmpeg_exe", return_value=None):
            audio, sr = audio_io.read_audio(headerless_mp3)
        assert sr == _SR
        assert audio.shape[0] == sf.info(headerless_mp3).frames

    def test_falls_back_when_ffmpeg_fails(self, headerless_mp3, tmp_path):
        missing = str(tmp_path / "no-ffmpeg.exe")
        with patch.object(audio_io, "_ffmpeg_exe", return_value=missing):
            audio, _sr = audio_io.read_audio(headerless_mp3)
        assert audio.shape[0] == sf.info(headerless_mp3).frames

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(sf.LibsndfileError):
            audio_io.read_audio(str(tmp_path / "nope.mp3"))


class TestAudioDuration:
    def test_headerless_vbr_mp3_duration(self, headerless_mp3):
        duration = audio_io.audio_duration(headerless_mp3)
        _assert_full_length(round(duration * _SR))

    def test_wav_duration(self, tmp_path):
        path = str(tmp_path / "tone.wav")
        sf.write(path, np.zeros((_SR // 2, 2), dtype=np.float32), _SR)
        assert audio_io.audio_duration(path) == pytest.approx(0.5)


@pytest.mark.parametrize("worker_cls", [SeparatorWorker, MdxSeparatorWorker])
def test_separators_load_full_headerless_mp3(
    tmp_path, headerless_mp3, worker_cls
):
    worker = worker_cls(
        input_path=headerless_mp3,
        output_dir=str(tmp_path / "out"),
        model_path="unused.onnx",
    )
    audio, sr = worker._load_audio()
    assert sr == _SR
    assert audio.shape[0] == 2
    _assert_full_length(audio.shape[1])


def test_memory_check_uses_decoded_duration(qapp, headerless_mp3):
    # Called unbound: the check needs nothing from a built dialog when
    # memory is sufficient.
    with patch(
        "src.ui.import_dialog.estimate_separation_memory", return_value=0,
    ) as estimate:
        assert ImportDialog._check_memory_ok(
            MagicMock(), headerless_mp3, False
        )
    duration = estimate.call_args.args[0]
    _assert_full_length(round(duration * _SR))


def test_fixture_is_tiny(headerless_mp3):
    assert os.path.getsize(headerless_mp3) < 64 * 1024
