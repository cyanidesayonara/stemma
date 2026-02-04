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
