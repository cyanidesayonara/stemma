"""Stem separation engine using ONNX Runtime.

Loads an HTDemucs v4 ONNX model (with a cached DirectML sidecar when GPU
acceleration is available) and separates an audio file into individual
stems (vocals, drums, bass, other, and optionally guitar + piano for the
6-stem variant). Pre/post-processing (STFT/iSTFT via ``src/stft_utils.py``
and ``soxr`` resampling) is handled outside the ONNX graph.

The ONNX model expects two inputs:
    - "input": raw waveform [1, 2, segment_samples]
    - "x": complex-as-channels spectrogram [1, 4, nfft//2, time_frames]

And returns two outputs:
    - "output": spectral stem masks [1, n_stems, 4, nfft//2, time_frames]
    - temporal output: per-stem waveforms [1, n_stems, 2, segment_samples]

Runs inference on a background QThread to avoid freezing the GUI.
"""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
import sys

import numpy as np
import soundfile as sf
import soxr
from PySide6.QtCore import QThread, Signal

from src.audio_io import read_audio
from src.htdemucs_dml import is_dml_sidecar, prepare_htdemucs_dml_path
from src.import_messages import ModelDamagedError, describe_error
from src.model_manager import discard_model_files
from src.onnx_session import (
    create_onnx_session,
    selected_session_provider,
    session_provider_label,
)
from src.separation_state import (
    EXPECTED_STEMS,
    clear_completion_marker,
    write_completion_marker,
)
from src.stft_utils import istft_frames_first, stft_batch


# Stem names for each model variant.
# HTDemucs natively outputs tensors in this exact order: 0=drums, 1=bass, etc.
STEMS_4 = EXPECTED_STEMS["htdemucs"]
STEMS_6 = EXPECTED_STEMS["htdemucs_6s"]

# HTDemucs expects audio at this sample rate.
SAMPLE_RATE = 44100

# STFT parameters matching the ONNX model export.
NFFT = 4096
HOP_LENGTH = 1024

# The fixed segment length the model was exported with.
# Derived from inspecting the model: input shape [1, 2, 343980].
SEGMENT_SAMPLES = 343980
SEGMENT_SECONDS = SEGMENT_SAMPLES / SAMPLE_RATE  # ~7.8 seconds


_ORT_OVERHEAD_BYTES = 400 * 1024 * 1024  # ~400 MB for ORT session + runtime

# Gain for folding the extra channels of a multichannel file into both
# sides, as a center channel is folded into stereo (-3 dB).
_FOLD_GAIN = 0.7071


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Return *audio* (channels, samples) as two channels.

    Mono is duplicated. For more than two channels (such as a 5.1 WAV),
    the first two are kept as left and right and the average of the rest
    is folded into both at -3 dB, then the result is scaled down only if
    that would clip. The models are stereo-only.
    """
    channels = audio.shape[0]
    if channels == 2:
        return audio
    if channels == 1:
        return np.repeat(audio, 2, axis=0)
    rest = audio[2:].mean(axis=0) * _FOLD_GAIN
    stereo = np.stack([audio[0] + rest, audio[1] + rest])
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1.0:
        stereo /= peak
    return stereo.astype(np.float32, copy=False)


def estimate_separation_memory(
    duration_seconds: float,
    is_6_stem: bool,
) -> int:
    """Return estimated peak RAM in bytes for separating a track.

    The dominant allocations are:
    * Input audio (stereo float32)
    * The overlap-add accumulator (n_stems * 2 * total_samples * float32)
    * Per-segment ONNX temporaries (~100 MB, bounded by segment size)
    * ORT session / model weights

    ``wiener_filter`` sums the mixture per chunk in the frequency domain,
    so it no longer holds a full-track mixture copy.
    """
    n_stems = 6 if is_6_stem else 4
    total_samples = int(SAMPLE_RATE * duration_seconds)
    bps = 4  # float32

    audio = 2 * total_samples * bps
    accumulator = n_stems * 2 * total_samples * bps
    segment_temps = 100 * 1024 * 1024

    return audio + accumulator + segment_temps + _ORT_OVERHEAD_BYTES


def available_memory_bytes() -> int | None:
    """Return available physical RAM in bytes, or None if unknown."""
    if sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys
        except Exception:
            return None
    return None


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

    @property
    def model_key(self) -> str:
        """Return the persisted model key for this worker."""
        return "htdemucs_6s" if self.is_6_stem else "htdemucs"

    def cancel(self) -> None:
        """Request cancellation of the running separation."""
        self._is_cancelled = True

    def run(self) -> None:
        """Execute the full separation pipeline.

        Steps:
            1. Load and validate the input audio file.
            2. Resample to 44100Hz if necessary.
            3. Create a DirectML or CPU ONNX Runtime session.
            4. Process audio in overlapping segments.
            5. For each segment: compute STFT, run inference, apply iSTFT.
            6. Write each stem to a WAV file.
        """
        try:
            self._separate()
        except Exception as exc:
            self.error.emit(describe_error(exc, "Separation failed"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _separate(self) -> None:
        """Core separation logic, called inside run()."""
        self.progress.emit(0, "Loading audio file...")
        audio, sr = self._load_audio()

        self.progress.emit(5, "Resampling audio...")
        audio = self._resample(audio, sr)

        self.progress.emit(10, "Initializing ONNX model...")
        session = self._create_session()
        self.progress.emit(
            12,
            f"Using {session_provider_label(session)} for stem separation.",
        )

        self.progress.emit(15, "Separating stems...")
        separated = self._run_segmented_inference(audio, session)

        del session, audio

        if self._is_cancelled:
            self.error.emit("Separation cancelled by user.")
            return

        self.progress.emit(90, "Post-processing stems...")
        separated = self._post_process(separated)

        if self._is_cancelled:
            self.error.emit("Separation cancelled by user.")
            return

        self.progress.emit(95, "Saving separated files...")
        result_files = self._save_stems(separated)

        self.progress.emit(100, "Done")
        self.finished.emit(result_files)

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

        audio, sr = read_audio(self.input_path)
        # Transpose to (channels, samples) for consistency with Demucs.
        return to_stereo(audio.T.astype(np.float32)), sr

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample audio to SAMPLE_RATE if necessary.

        Args:
            audio: Audio array with shape (channels, samples).
            sr: Current sample rate.

        Returns:
            Resampled audio array with shape (channels, new_samples).
        """
        if sr == SAMPLE_RATE or audio.shape[1] == 0:
            return audio

        resampled = soxr.resample(audio.T, sr, SAMPLE_RATE, quality="HQ")
        return np.ascontiguousarray(resampled.T, dtype=np.float32)

    def _create_session(self):
        """Create the ONNX session, using a DirectML-patched sidecar if possible.

        When DirectML is available, HTDemucs graphs are patched on first load
        to a cached ``.dml.onnx`` sidecar next to the original model. If that
        sidecar fails to load as damaged, it is removed and the original
        model is retried on CPU before deleting the downloaded model files.
        """
        session_path = prepare_htdemucs_dml_path(self.model_path)
        try:
            return create_onnx_session(session_path)
        except ModelDamagedError:
            if is_dml_sidecar(session_path) and session_path != self.model_path:
                try:
                    os.remove(session_path)
                except OSError:
                    pass
                try:
                    return create_onnx_session(self.model_path, use_gpu=False)
                except ModelDamagedError:
                    discard_model_files(self.model_path)
                    raise
            discard_model_files(self.model_path)
            raise

    def _run_segmented_inference(
        self, audio: np.ndarray, session
    ) -> np.ndarray:
        """Process audio in overlapping segments through the ONNX model.

        Uses 50% overlap with a Hann window to eliminate boundary clicks.
        Each segment is windowed before accumulation, and a weight buffer
        tracks the total window contribution per sample for normalization.

        Args:
            audio: Full audio array, shape (channels, total_samples).
            session: ONNX Runtime inference session.

        Returns:
            Separated stems array, shape (n_stems, channels, total_samples).
        """
        n_channels, total_samples = audio.shape
        n_stems = len(self.stems)

        # Ensure stereo input (model expects 2 channels).
        if n_channels == 1:
            audio = np.repeat(audio, 2, axis=0)

        # 50% overlap: step by half a segment.
        step = SEGMENT_SAMPLES // 2

        # Add half a segment of context at both ends. The Hann window is
        # exactly zero at its endpoints, so placing real audio directly at
        # those endpoints would erase the first and last samples. Context
        # padding keeps every original sample under a positive window weight;
        # it is trimmed after overlap-add normalization.
        context = step
        audio = np.pad(audio, ((0, 0), (context, context)))
        working_samples = audio.shape[1]

        # Pad the right side further so the last segment fits fully.
        if working_samples > SEGMENT_SAMPLES:
            n_steps = (
                (working_samples - SEGMENT_SAMPLES + step - 1) // step + 1
            )
            required_length = (n_steps - 1) * step + SEGMENT_SAMPLES
        else:
            required_length = SEGMENT_SAMPLES
        right_pad = required_length - working_samples

        if right_pad > 0:
            audio = np.pad(audio, ((0, 0), (0, right_pad)))

        padded_length = audio.shape[1]

        # Build segment start positions.
        starts = list(range(0, padded_length - SEGMENT_SAMPLES + 1, step))
        n_segments = len(starts)

        # Hann window for crossfading.
        window = np.hanning(SEGMENT_SAMPLES).astype(np.float32)

        # Accumulators.
        result = np.zeros(
            (n_stems, 2, padded_length), dtype=np.float32
        )
        weight = np.zeros(padded_length, dtype=np.float32)

        use_dml_pipeline = (
            selected_session_provider(session) == "DmlExecutionProvider"
            and n_segments > 1
        )
        use_custom_istft = (
            type(self)._compute_istft_cac is not SeparatorWorker._compute_istft_cac
            or "_compute_istft_cac" in self.__dict__
        )
        stem_workers = 1 if use_custom_istft else min(n_stems, 6)
        stem_pool = (
            ThreadPoolExecutor(max_workers=stem_workers)
            if stem_workers > 1
            else None
        )
        pipeline_pool = (
            ThreadPoolExecutor(max_workers=1) if use_dml_pipeline else None
        )

        def _accumulate(start_pos: int, outputs: list[np.ndarray]) -> None:
            stems_out = self._postprocess_outputs(
                outputs,
                stem_pool=stem_pool,
                use_custom_istft=use_custom_istft,
            )
            end_pos = start_pos + SEGMENT_SAMPLES
            result[:, :, start_pos:end_pos] += stems_out * window
            weight[start_pos:end_pos] += window

        pending = None
        try:
            for seg_idx, start in enumerate(starts):
                if self._is_cancelled:
                    return result

                pct = 15 + int(75 * seg_idx / max(n_segments, 1))
                self.progress.emit(
                    pct, f"Processing segment {seg_idx + 1}/{n_segments}..."
                )

                segment = audio[:, start : start + SEGMENT_SAMPLES]
                outputs = self._run_model(segment, session)

                if pending is not None:
                    pending.result()
                    pending = None

                if self._is_cancelled:
                    return result

                if pipeline_pool is not None:
                    pending = pipeline_pool.submit(_accumulate, start, outputs)
                else:
                    _accumulate(start, outputs)

            if pending is not None:
                pending.result()
                pending = None
        finally:
            if pending is not None:
                try:
                    pending.result()
                except Exception:
                    pass
            if pipeline_pool is not None:
                pipeline_pool.shutdown(wait=True)
            if stem_pool is not None:
                stem_pool.shutdown(wait=True)

        # Normalize by the accumulated window weight.
        # Avoid division by zero in silent padding regions.
        np.maximum(weight, 1e-8, out=weight)
        result /= weight

        # Remove left context and all right-side padding.
        return result[:, :, context:context + total_samples]

    def _run_model(
        self, segment: np.ndarray, session
    ) -> list[np.ndarray]:
        """Run STFT and ONNX Runtime forward pass for a single segment."""
        spec = self._compute_stft_cac(segment)
        return session.run(
            None,
            {
                "input": segment[np.newaxis, :, :],
                "x": spec[np.newaxis, :, :, :],
            },
        )

    def _postprocess_outputs(
        self,
        outputs: list[np.ndarray],
        stem_pool: ThreadPoolExecutor | None = None,
        use_custom_istft: bool = False,
    ) -> np.ndarray:
        """Invert spectral outputs via iSTFT and add temporal stem waveforms."""
        spec_out = outputs[0]
        temporal_out = outputs[1]
        n_stems = spec_out.shape[1]
        if stem_pool is not None and not use_custom_istft:
            spectral = list(
                stem_pool.map(
                    lambda s: self._compute_istft_cac(
                        spec_out[0, s], length=SEGMENT_SAMPLES
                    ),
                    range(n_stems),
                )
            )
            spectral_stems = np.stack(spectral, axis=0)
        else:
            spectral_stems = np.empty(
                (n_stems, 2, SEGMENT_SAMPLES), dtype=np.float32
            )
            for s in range(n_stems):
                spectral_stems[s] = self._compute_istft_cac(
                    spec_out[0, s], length=SEGMENT_SAMPLES
                )
        return spectral_stems + temporal_out[0]

    def _infer_segment(
        self, segment: np.ndarray, session
    ) -> np.ndarray:
        """Run inference on a single segment.

        Args:
            segment: Audio segment, shape (2, SEGMENT_SAMPLES).
            session: ONNX Runtime inference session.

        Returns:
            Separated stems for this segment,
            shape (n_stems, 2, SEGMENT_SAMPLES).
        """
        return self._postprocess_outputs(self._run_model(segment, session))

    def _compute_stft_cac(self, audio: np.ndarray) -> np.ndarray:
        """Compute the complex-as-channels STFT spectrogram.

        The model expects the spectrogram in "complex-as-channels" format:
        for stereo (2 channels), the complex STFT is split into real and
        imaginary parts, yielding 4 channels: [left_real, left_imag,
        right_real, right_imag].

        Args:
            audio: Stereo audio, shape (2, samples).

        Returns:
            CaC spectrogram, shape (4, NFFT//2, time_frames).
        """
        spec = stft_batch(
            audio, n_fft=NFFT, hop_length=HOP_LENGTH, pad_mode="constant"
        )[:, : NFFT // 2, :]
        cac = np.empty(
            (spec.shape[0] * 2, spec.shape[1], spec.shape[2]),
            dtype=np.float32,
        )
        cac[0::2] = spec.real
        cac[1::2] = spec.imag
        return cac

    def _compute_istft_cac(
        self, cac_spec: np.ndarray, length: int
    ) -> np.ndarray:
        """Convert a complex-as-channels spectrogram back to audio.

        Args:
            cac_spec: CaC spectrogram, shape (4, NFFT//2, time_frames).
            length: Target number of output samples.

        Returns:
            Stereo audio, shape (2, length).
        """
        n_frames = cac_spec.shape[2]
        spec_t = np.zeros((2, n_frames, NFFT // 2 + 1), dtype=np.complex64)
        real_t = spec_t.real[:, :, : NFFT // 2]
        imag_t = spec_t.imag[:, :, : NFFT // 2]
        real_t[0] = cac_spec[0].T
        imag_t[0] = cac_spec[1].T
        real_t[1] = cac_spec[2].T
        imag_t[1] = cac_spec[3].T
        return istft_frames_first(
            spec_t,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            length=length,
            overwrite_x=True,
        )

    def _post_process(self, stems: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering and soft gating to improve stem quality.

        Args:
            stems: Separated stems, shape (n_stems, 2, total_samples).

        Returns:
            Post-processed stems with same shape.
        """
        from src.post_processing import wiener_filter, soft_gate

        def cancelled() -> bool:
            return self._is_cancelled

        stems = wiener_filter(stems, should_cancel=cancelled)
        stems = soft_gate(stems, should_cancel=cancelled)
        return stems

    def _raise_if_cancelled(self) -> None:
        """Stop between steps that take a while (filters, stem writes)."""
        if self._is_cancelled:
            raise InterruptedError("Separation cancelled by user.")

    def _save_stems(self, separated: np.ndarray) -> dict[str, str]:
        """Write separated stems to WAV files.

        Args:
            separated: Array of shape (n_stems, 2, total_samples).

        Returns:
            Mapping of stem name to file path.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        clear_completion_marker(self.output_dir)
        result_files = {}

        for i, stem_name in enumerate(self.stems):
            # No completion marker is written after a cancel, so the
            # interrupted import is rolled back rather than kept half-done.
            self._raise_if_cancelled()
            out_path = os.path.join(self.output_dir, f"{stem_name}.wav")
            # Transpose to (samples, channels) for soundfile.
            stem_audio = separated[i].T
            sf.write(out_path, stem_audio, SAMPLE_RATE, subtype="PCM_16")
            result_files[stem_name] = out_path

        write_completion_marker(self.output_dir, self.model_key)
        return result_files
