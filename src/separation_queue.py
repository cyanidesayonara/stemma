"""Background separation queue.

Runs stem-separation jobs one at a time on background threads so the
import dialog can close immediately and the app stays usable while a
song separates. The library panel subscribes to the queue's signals to
show per-song progress; the main window finalizes or rolls back the
library row when a job ends.

Jobs run strictly serially: separation saturates either the CPU
(HTDemucs) or the GPU (MDX), so overlapping jobs would only slow each
other down and multiply peak memory.
"""

from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from src.mdx_separator import MdxSeparatorWorker
from src.qt_signal_utils import safe_disconnect as _safe_disconnect
from src.separator import SeparatorWorker


@dataclass
class SeparationJob:
    """One queued separation request."""

    song_id: str
    input_path: str
    output_dir: str
    model_path: str
    model_key: str  # "htdemucs" | "htdemucs_6s" | "mdx_*"


class SeparationQueue(QObject):
    """Serial queue of separation jobs with per-song progress signals.

    Signals:
        job_queued(str): song_id -- accepted into the queue (fires for
            every job, before job_started).
        job_started(str): song_id -- a job's worker began running.
        job_progress(str, int, str): song_id, percent, status message.
        job_finished(str, str): song_id, model_key -- stems are on disk.
        job_failed(str, str): song_id, error message. Emitted for real
            failures and for cancellations (message contains
            "cancelled"), so the owner can roll the row back either way.
    """

    job_queued = Signal(str)
    job_started = Signal(str)
    job_progress = Signal(str, int, str)
    job_finished = Signal(str, str)
    job_failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pending: deque[SeparationJob] = deque()
        self._active_job: SeparationJob | None = None
        self._active_worker = None  # SeparatorWorker | MdxSeparatorWorker

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active_song_id(self) -> str | None:
        """The song currently separating, or None when idle."""
        return self._active_job.song_id if self._active_job else None

    def is_song_pending(self, song_id: str) -> bool:
        """True when *song_id* is separating now or waiting in the queue."""
        if self._active_job and self._active_job.song_id == song_id:
            return True
        return any(j.song_id == song_id for j in self._pending)

    @property
    def pending_count(self) -> int:
        """Number of jobs not yet finished (active + queued)."""
        return len(self._pending) + (1 if self._active_job else 0)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def enqueue(self, job: SeparationJob) -> None:
        """Add a job; it starts immediately when the queue is idle."""
        self._pending.append(job)
        self.job_queued.emit(job.song_id)
        if self._active_job is None:
            self._start_next()

    def cancel_song(self, song_id: str) -> None:
        """Cancel *song_id*'s job, whether active or still queued.

        A queued job is dropped synchronously (job_failed fires with a
        cancellation message). Cancelling the active job asks the worker
        to stop; its error path then reports through job_failed.
        """
        for job in list(self._pending):
            if job.song_id == song_id:
                self._pending.remove(job)
                self.job_failed.emit(song_id, "Separation cancelled.")
                return
        if self._active_job and self._active_job.song_id == song_id:
            if self._active_worker is not None:
                self._active_worker.cancel()

    def shutdown(self, wait_ms: int = 5000) -> None:
        """Cancel the active job and drop queued ones (app close).

        Interrupted songs are left without stems on disk; the startup
        prune removes their library rows on the next launch.
        """
        self._pending.clear()
        worker = self._active_worker
        if worker is not None:
            worker.cancel()
            self._detach_active()
            if worker.isRunning():
                worker.wait(wait_ms)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_worker(self, job: SeparationJob):
        if job.model_key.startswith("mdx_"):
            return MdxSeparatorWorker(
                input_path=job.input_path,
                output_dir=job.output_dir,
                model_path=job.model_path,
                model_key=job.model_key,
            )
        return SeparatorWorker(
            input_path=job.input_path,
            output_dir=job.output_dir,
            model_path=job.model_path,
            is_6_stem=job.model_key == "htdemucs_6s",
        )

    def _start_next(self) -> None:
        if not self._pending:
            return
        job = self._pending.popleft()
        worker = self._make_worker(job)
        self._active_job = job
        self._active_worker = worker

        sid = job.song_id
        worker.progress.connect(
            lambda pct, msg, s=sid: self.job_progress.emit(s, pct, msg)
        )
        # Both engines emit their custom `finished(dict)` on success and
        # `error(str)` on failure/cancellation, exactly once.
        worker.finished.connect(lambda _files: self._on_worker_done(None))
        worker.error.connect(lambda msg: self._on_worker_done(msg))

        self.job_started.emit(sid)
        worker.start()

    def _detach_active(self) -> None:
        """Disconnect and release the active worker/job references."""
        worker = self._active_worker
        if worker is not None:
            _safe_disconnect(worker.progress)
            _safe_disconnect(worker.finished)
            _safe_disconnect(worker.error)
        self._active_job = None
        self._active_worker = None

    def _on_worker_done(self, error_message: str | None) -> None:
        """Worker completed (successfully or not); report and advance."""
        job = self._active_job
        worker = self._active_worker
        if job is None:
            return
        self._detach_active()

        if worker is not None:
            # run() has returned (the completion signal is its last act);
            # wait for the OS thread to fully stop before deleteLater so
            # Qt never reaps a live QThread.
            if worker.isRunning():
                worker.wait(5000)
            worker.deleteLater()

        if error_message is None:
            self.job_finished.emit(job.song_id, job.model_key)
        else:
            self.job_failed.emit(job.song_id, error_message)

        self._start_next()
