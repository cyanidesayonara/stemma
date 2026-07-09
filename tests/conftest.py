"""Shared test fixtures for the stemma test suite."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_audio_path(tmp_dir):
    """Create a minimal valid WAV file and return its path.

    The file contains 1 second of silence in stereo at 44100Hz,
    which is sufficient for testing I/O without real audio content.
    """
    import numpy as np
    import soundfile as sf

    path = os.path.join(tmp_dir, "test_audio.wav")
    silence = np.zeros((44100, 2), dtype=np.float32)
    sf.write(path, silence, 44100)
    return path


@pytest.fixture(autouse=True, scope="module")
def _collect_qt_garbage_between_modules():
    """Run a full garbage collection at every test-module boundary.

    CPython's cyclic GC otherwise destroys unreferenced QObject trees
    (old MainWindows, players, dialogs from earlier tests) at arbitrary
    later moments -- sometimes while the interpreter is inside a Qt
    call such as QWidget.show(). Deleting top-level windows reentrantly
    from within Qt's own stack intermittently corrupts native state and
    crashes with an access violation in whichever test touches Qt next.
    Collecting at module boundaries destroys that garbage at a quiescent
    point instead. (Per-test collection also works but adds ~70s to the
    suite; module scope catches the cross-module garbage that actually
    crashed.)"""
    yield
    import gc
    gc.collect()
