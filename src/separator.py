"""Stem separation engine using ONNX Runtime with DirectML acceleration.

Loads an HTDemucs v4 ONNX model and separates an audio file into individual
stems (vocals, drums, bass, other, and optionally guitar + piano for the
6-stem variant). Pre/post-processing (STFT/iSTFT) is handled via librosa
outside the ONNX graph.

Runs inference on a background QThread to avoid freezing the GUI.
"""

import os

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal


# Stem names for each model variant.
STEMS_4 = ("vocals", "drums", "bass", "other")
STEMS_6 = ("vocals", "drums", "bass", "other", "guitar", "piano")

# HTDemucs expects audio at this sample rate.
SAMPLE_RATE = 44100


class SeparatorWorker(QThread):
    """Background thread that runs ONNX stem separation.

    Signals:
        progress(int, str): Percentage (0-100) and a human-readable status.
        finished(dict): Mapping of stem name to output file path on success.
        error(str): Error description if separation fails.
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, input_path: str, output_dir: str, model_path: str,
                 is_6_stem: bool = False) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.is_6_stem = is_6_stem
        self._is_cancelled = False

    @property
    def stems(self) -> tuple[str, ...]:
        """Return the stem names for the active model variant."""
        return STEMS_6 if self.is_6_stem else STEMS_4

    def cancel(self) -> None:
        """Request cancellation of the running separation."""
        self._is_cancelled = True

    def run(self) -> None:
        """Execute the full separation pipeline.

        Steps:
            1. Load and validate the input audio file.
            2. Create an ONNX Runtime session (DirectML -> CPU fallback).
            3. Apply STFT pre-processing.
            4. Run chunked model inference.
            5. Apply iSTFT post-processing.
            6. Write each stem to a WAV file.

        TODO: Steps 3-5 are not yet implemented. This skeleton validates the
        surrounding I/O and session setup so that the inference pipeline can
        be added incrementally with tests.
        """
        try:
            self._separate()
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _separate(self) -> None:
        """Core separation logic, called inside run().

        Raises NotImplementedError for the inference steps that are still
        pending implementation.
        """
        self.progress.emit(0, "Loading audio file...")
        audio, sr = self._load_audio()

        self.progress.emit(10, "Initializing ONNX model...")
        session = self._create_session()

        # TODO: Implement the actual inference pipeline.
        # The following steps need to be added:
        #   1. Resample audio to SAMPLE_RATE if sr != SAMPLE_RATE.
        #   2. Compute STFT via librosa.
        #   3. Feed spectrogram chunks to the ONNX model.
        #   4. Reassemble chunks and apply iSTFT.
        #   5. Write separated stems to output_dir.
        raise NotImplementedError(
            "ONNX inference pipeline not yet implemented. "
            "See separator.py TODO comments for the required steps."
        )

    def _load_audio(self) -> tuple[np.ndarray, int]:
        """Load the input audio file and return (audio, sample_rate).

        Returns:
            A tuple of (audio_array, sample_rate) where audio_array has
            shape (channels, samples).
        """
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(
                f"Input audio file not found: {self.input_path}"
            )

        audio, sr = sf.read(self.input_path, always_2d=True)
        # Transpose to (channels, samples) for consistency with Demucs.
        return audio.T, sr

    def _create_session(self):
        """Create an ONNX Runtime inference session.

        Attempts DirectML first for GPU acceleration, falling back to CPU.

        Returns:
            An onnxruntime.InferenceSession instance.
        """
        import onnxruntime as ort

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"ONNX model file not found: {self.model_path}"
            )

        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        return ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers,
        )
