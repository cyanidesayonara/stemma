"""Detection must never run several workers at once.

Each A-B loop change starts tempo/key detection on the new region. A
superseded worker was detached but kept running, so rapid loop clicks
stacked six or more DetectionWorkers running ONNX, librosa, and scipy
concurrently, and the process crashed with an access violation (#168).
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from src.ui import player_controls as player_controls_module
from src.ui.player_controls import PlayerControls


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _GatedWorker(QThread):
    """Stands in for DetectionWorker; runs until its gate opens."""

    completed = Signal(object)
    error = Signal(str)
    instances: list["_GatedWorker"] = []

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.gate = threading.Event()
        _GatedWorker.instances.append(self)

    def run(self) -> None:
        self.gate.wait(10)


@pytest.fixture
def controls(qapp, monkeypatch):
    _GatedWorker.instances = []
    monkeypatch.setattr(player_controls_module, "DetectionWorker", _GatedWorker)
    player = MagicMock()
    player.stems = {"vocals": np.zeros((100, 2), dtype=np.float32)}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    player.beat_times = []
    player.chord_sequence = []
    player.total_seconds = 10.0
    player.current_seconds = 0.0
    player.sample_rate = 44100
    player.loop_a = None
    player.loop_b = None
    player.is_playing = False
    player.speed = 1.0
    player.pitch_semitones = 0
    result = PlayerControls(player)
    yield result
    for worker in _GatedWorker.instances:
        worker.gate.set()
        if isValid(worker):
            worker.wait(2000)
    result.shutdown()
    result.setParent(None)
    result.deleteLater()
    qapp.processEvents()


def _running():
    """Workers still running. Finished ones are deleted once reaped."""
    return [w for w in _GatedWorker.instances if isValid(w) and w.isRunning()]


def _settle(worker) -> None:
    """Let *worker* finish and its queued finished signal be handled."""
    worker.gate.set()
    if isValid(worker):
        worker.wait(2000)
    end = time.monotonic() + 2.0
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def test_rapid_requests_run_one_worker_at_a_time(controls):
    for start in range(5):
        controls.start_detection(float(start), float(start) + 1.0)

    assert len(_GatedWorker.instances) == 1
    assert len(_running()) == 1


def test_the_latest_queued_request_runs_when_the_worker_finishes(controls):
    for start in range(5):
        controls.start_detection(float(start), float(start) + 1.0)
    first = _GatedWorker.instances[0]

    _settle(first)

    # Only the newest request runs next; the three in between are dropped.
    assert len(_GatedWorker.instances) == 2
    assert _GatedWorker.instances[1].kwargs["start_sec"] == 4.0
    assert len(_running()) == 1

    _settle(_GatedWorker.instances[1])
    assert len(_GatedWorker.instances) == 2
    assert _running() == []


def test_a_request_while_idle_starts_at_once(controls):
    controls.start_detection(1.0, 2.0)
    _settle(_GatedWorker.instances[0])

    controls.start_detection(3.0, 4.0)

    assert len(_GatedWorker.instances) == 2
    assert _GatedWorker.instances[1].kwargs["start_sec"] == 3.0
