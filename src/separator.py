"""Stem separation engine using ONNX Runtime with DirectML acceleration.

Loads an HTDemucs v4 ONNX model and separates an audio file into individual
stems (vocals, drums, bass, other, and optionally guitar + piano for the
6-stem variant). Pre/post-processing (STFT/iSTFT) is handled via numpy
outside the ONNX graph.

The ONNX model expects two inputs:
    - "input": raw waveform [1, 2, segment_samples]
    - "x": complex-as-channels spectrogram [1, 4, nfft//2, time_frames]

And returns two outputs:
    - "output": spectral stem masks [1, n_stems, 4, nfft//2, time_frames]
    - temporal output: per-stem waveforms [1, n_stems, 2, segment_samples]

Runs inference on a background QThread to avoid freezing the GUI.
"""

import os

import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal


# Stem names for each model variant.
# HTDemucs natively outputs tensors in this exact order: 0=drums, 1=bass, etc.
STEMS_4 = ("drums", "bass", "other", "vocals")
STEMS_6 = ("drums", "bass", "other", "vocals", "guitar", "piano")

# HTDemucs expects audio at this sample rate.
SAMPLE_RATE = 44100

# STFT parameters matching the ONNX model export.
NFFT = 4096
HOP_LENGTH = 1024

# The fixed segment length the model was exported with.
# Derived from inspecting the model: input shape [1, 2, 343980].
SEGMENT_SAMPLES = 343980
SEGMENT_SECONDS = SEGMENT_SAMPLES / SAMPLE_RATE  # ~7.8 seconds


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
            2. Resample to 44100Hz if necessary.
            3. Create an ONNX Runtime session (DirectML -> CPU fallback).
            4. Process audio in overlapping segments.
            5. For each segment: compute STFT, run inference, apply iSTFT.
            6. Write each stem to a WAV file.
        """
        try:
            self._separate()
        except Exception as exc:
            self.error.emit(str(exc))

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

        self.progress.emit(15, "Separating stems...")
        separated = self._run_segmented_inference(audio, session)

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

        audio, sr = sf.read(self.input_path, always_2d=True)
        # Transpose to (channels, samples) for consistency with Demucs.
        return audio.T.astype(np.float32), sr

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample audio to SAMPLE_RATE if necessary.

        Args:
            audio: Audio array with shape (channels, samples).
            sr: Current sample rate.

        Returns:
            Resampled audio array with shape (channels, new_samples).
        """
        if sr == SAMPLE_RATE:
            return audio

        resampled_channels = []
        for ch in range(audio.shape[0]):
            resampled = librosa.resample(
                audio[ch], orig_sr=sr, target_sr=SAMPLE_RATE
            )
            resampled_channels.append(resampled)
        return np.stack(resampled_channels)

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

        # Pad so the last segment fits fully.
        pad_needed = 0
        if total_samples > SEGMENT_SAMPLES:
            # Number of steps to cover all samples.
            n_steps = (total_samples - SEGMENT_SAMPLES + step - 1) // step + 1
            required_length = (n_steps - 1) * step + SEGMENT_SAMPLES
            pad_needed = required_length - total_samples
        else:
            # Single segment: pad to SEGMENT_SAMPLES.
            pad_needed = SEGMENT_SAMPLES - total_samples
            required_length = SEGMENT_SAMPLES

        if pad_needed > 0:
            audio = np.pad(audio, ((0, 0), (0, pad_needed)))

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

        for seg_idx, start in enumerate(starts):
            if self._is_cancelled:
                return result

            pct = 15 + int(75 * seg_idx / max(n_segments, 1))
            self.progress.emit(
                pct, f"Processing segment {seg_idx + 1}/{n_segments}..."
            )

            end = start + SEGMENT_SAMPLES
            segment = audio[:, start:end]

            stems_out = self._infer_segment(segment, session)

            # Apply Hann window and accumulate.
            for s in range(n_stems):
                for ch in range(2):
                    result[s, ch, start:end] += stems_out[s, ch] * window

            weight[start:end] += window

        # Normalize by the accumulated window weight.
        # Avoid division by zero in silent padding regions.
        weight = np.maximum(weight, 1e-8)
        for s in range(n_stems):
            for ch in range(2):
                result[s, ch] /= weight

        # Remove padding.
        if pad_needed > 0:
            result = result[:, :, :total_samples]

        return result

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
        # Compute the complex-as-channels (CaC) spectrogram.
        spec = self._compute_stft_cac(segment)

        # Prepare model inputs.
        # "input": raw waveform [1, 2, SEGMENT_SAMPLES]
        mix_input = segment[np.newaxis, :, :]  # Add batch dim.
        # "x": CaC spectrogram [1, 4, NFFT//2, time_frames]
        spec_input = spec[np.newaxis, :, :, :]  # Add batch dim.

        # Run inference.
        outputs = session.run(
            None,
            {
                "input": mix_input,
                "x": spec_input,
            },
        )

        # outputs[0]: spectral path [1, n_stems, 4, NFFT//2, time_frames]
        # outputs[1]: temporal path [1, n_stems, 2, SEGMENT_SAMPLES]
        spec_out = outputs[0]  # [1, n_stems, 4, freq, time]
        temporal_out = outputs[1]  # [1, n_stems, 2, samples]

        # Convert spectral output back to time domain via iSTFT.
        n_stems = spec_out.shape[1]
        spectral_stems = np.zeros(
            (n_stems, 2, SEGMENT_SAMPLES), dtype=np.float32
        )
        for s in range(n_stems):
            spectral_stems[s] = self._compute_istft_cac(
                spec_out[0, s], length=SEGMENT_SAMPLES
            )

        # Combine spectral and temporal paths (the model sums them).
        combined = spectral_stems + temporal_out[0]
        return combined

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
        cac_channels = []
        for ch in range(audio.shape[0]):
            stft = librosa.stft(
                audio[ch],
                n_fft=NFFT,
                hop_length=HOP_LENGTH,
                center=True,
            )
            # stft shape: (NFFT//2 + 1, time_frames)
            # Trim to NFFT//2 to match model expectation.
            stft = stft[:NFFT // 2, :]
            cac_channels.append(stft.real)
            cac_channels.append(stft.imag)

        return np.stack(cac_channels).astype(np.float32)

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
        result = np.zeros((2, length), dtype=np.float32)

        for ch in range(2):
            real = cac_spec[ch * 2]      # shape: (NFFT//2, time)
            imag = cac_spec[ch * 2 + 1]  # shape: (NFFT//2, time)

            # Reconstruct the full-size complex spectrogram.
            # Add back the Nyquist bin (zeroed) to get NFFT//2 + 1 bins.
            nyquist = np.zeros((1, real.shape[1]), dtype=np.float32)
            real_full = np.concatenate([real, nyquist], axis=0)
            imag_full = np.concatenate([imag, nyquist], axis=0)

            complex_spec = real_full + 1j * imag_full

            audio = librosa.istft(
                complex_spec,
                hop_length=HOP_LENGTH,
                n_fft=NFFT,
                length=length,
                center=True,
            )
            result[ch] = audio

        return result

    def _post_process(self, stems: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering and soft gating to improve stem quality.

        Args:
            stems: Separated stems, shape (n_stems, 2, total_samples).

        Returns:
            Post-processed stems with same shape.
        """
        from src.post_processing import wiener_filter, soft_gate

        stems = wiener_filter(stems)
        stems = soft_gate(stems)
        return stems

    def _save_stems(self, separated: np.ndarray) -> dict[str, str]:
        """Write separated stems to WAV files.

        Args:
            separated: Array of shape (n_stems, 2, total_samples).

        Returns:
            Mapping of stem name to file path.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        result_files = {}

        for i, stem_name in enumerate(self.stems):
            out_path = os.path.join(self.output_dir, f"{stem_name}.wav")
            # Transpose to (samples, channels) for soundfile.
            stem_audio = separated[i].T
            sf.write(out_path, stem_audio, SAMPLE_RATE, subtype="PCM_16")
            result_files[stem_name] = out_path

        return result_files
