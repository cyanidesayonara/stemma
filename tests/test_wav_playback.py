"""Tests for deferred WAV playback helpers."""

import os

import pytest
from PySide6.QtWidgets import QApplication

from src.paths import app_root
from src.ui.wav_playback import play_wav_async


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def test_play_missing_file_is_noop() -> None:
    play_wav_async("/nonexistent/stemma/arpeggio.wav")


def test_play_impl_module_loads() -> None:
    from src.ui import _wav_playback_impl

    assert hasattr(_wav_playback_impl, "play_impl")


def test_play_real_wav_no_crash(app) -> None:
    path = os.path.join(app_root(), "assets", "audio", "arpeggio.wav")
    if not os.path.isfile(path):
        pytest.skip("arpeggio.wav not present")
    play_wav_async(path)


def test_two_different_wavs_no_crash(app) -> None:
    root = app_root()
    a = os.path.join(root, "assets", "audio", "arpeggio.wav")
    b = os.path.join(root, "assets", "audio", "chord.wav")
    if not (os.path.isfile(a) and os.path.isfile(b)):
        pytest.skip("audio assets not present")
    play_wav_async(a)
    play_wav_async(b)
