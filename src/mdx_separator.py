"""MDX-Net 2-stem separation engine (ONNX Runtime, DirectML-accelerated).

Runs a UVR MDX-Net model to split a song into a primary stem and its
complement (e.g. Instrumental + Vocals). Unlike the HTDemucs export,
MDX-Net ONNX graphs initialize and run on the DirectML execution
provider, so this path is GPU-accelerated on any DX12 device (~45x
faster than CPU on an RTX 4070 Ti; a 4-minute song separates in
seconds instead of minutes).

The model operates on spectrogram chunks:
    input  [1, 4, dim_f, dim_t]  -- (L.re, L.im, R.re, R.im) x freq x time
    output [1, 4, dim_f, dim_t]  -- the primary stem's spectrogram

STFT/iSTFT run in numpy/librosa outside the graph, matching UVR's
torch.stft settings (periodic Hann, center=True, reflect padding).
The secondary stem is derived as (mix - primary).

Inference windows use the classic UVR scheme: each window carries
``trim = n_fft // 2`` samples of context on both sides and only the
central ``gen_size`` region is kept, so no cross-fade is needed.

Model weights are trained by the Ultimate Vocal Remover project
(https://github.com/Anjok07/ultimatevocalremovergui, MIT) -- credit to
UVR and its developers. The chunking/packing scheme follows UVR /
python-audio-separator (MIT).
"""

import os

import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from src.onnx_session import create_onnx_session, session_provider_label
from src.separation_state import (
    clear_completion_marker,
    write_completion_marker,
)

SAMPLE_RATE = 44100

# Registry of supported MDX models. Parameters come from UVR's model-data
# database keyed by the md5 of the file's last 10,000 KiB (UVR's hashing
# convention). Downloads use the immutable GitHub release asset ID and a
# reviewed full-file SHA-256 before publishing the model locally.
MDX_MODELS: dict[str, dict] = {
    "mdx_inst_hq3": {
        "display_name": "MDX-Net Inst HQ 3",
        "file": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "url": (
            "https://api.github.com/repos/TRvlvr/model_repo/"
            "releases/assets/112310332"
        ),
        "sha256": (
            "317554b07fe1ea5279a77f2b1520a41e"
            "a4b93432560c4ffd08792c30fddf9adc"
        ),
        "md5_tail": "55657dd70583b0fedfba5f67df11d711",
        "n_fft": 6144,
        "hop": 1024,
        "dim_f": 3072,
        "dim_t": 256,
        "compensate": 1.022,
        # The model predicts this stem; the other is (mix - primary).
        "primary_stem": "Instrumental",
    },
}

# Stem-file mapping for the 2-stem result. The player's mixer knows the
# canonical stem names, so the instrumental complement is stored as
# "other" (giving a Vocals + Other mixer, same as a 4-stem song with
# drums/bass absent).
PRIMARY_TO_FILES = {
    "Instrumental": ("other", "vocals"),  # (primary file, secondary file)
    "Vocals": ("vocals", "other"),
}


def hash_model_file(path: str) -> str:
    """Return the md5 of the file's last 10,000 KiB (UVR's convention)."""
    import hashlib

    with open(path, "rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()


class MdxSeparatorWorker(QThread):
    """Background thread that runs MDX-Net 2-stem separation.

    Signals match SeparatorWorker so the import dialog can drive either
    engine identically:
        progress(int, str): Percentage (0-100) and a status message.
        finished(dict): Mapping of stem name to output file path.
        error(str): Error description if separation fails.
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        input_path: str,
        output_dir: str,
        model_path: str,
        model_key: str = "mdx_inst_hq3",
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.model_key = model_key
        self.params = MDX_MODELS[model_key]
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the running separation."""
        self._is_cancelled = True

    def run(self) -> None:
        try:
            self._separate()
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _separate(self) -> None:
        self.progress.emit(0, "Loading audio file...")
        audio, sr = self._load_audio()

        self.progress.emit(5, "Resampling audio...")
        audio = self._resample(audio, sr)

        self.progress.emit(10, "Initializing MDX model...")
        session = self._create_session()
        self.progress.emit(
            12,
            f"Using {session_provider_label(session)} for MDX separation.",
        )

        self.progress.emit(15, "Separating (2-stem)...")
        primary = self._demix(audio, session)
        del session

        if self._is_cancelled:
            self.error.emit("Separation cancelled by user.")
            return

        secondary = audio - primary

        self.progress.emit(95, "Saving separated files...")
        result_files = self._save(primary, secondary)

        self.progress.emit(100, "Done")
        self.finished.emit(result_files)

    def _load_audio(self) -> tuple[np.ndarray, int]:
        """Return (audio, sample_rate) with audio shaped (2, samples)."""
        if not os.path.isfile(self.input_path):
            raise FileNotFoundError(
                f"Input audio file not found: {self.input_path}"
            )
        audio, sr = sf.read(self.input_path, always_2d=True)
        audio = audio.T.astype(np.float32)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        return audio[:2], sr

    def _resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == SAMPLE_RATE:
            return audio
        return np.stack([
            librosa.resample(ch, orig_sr=sr, target_sr=SAMPLE_RATE)
            for ch in audio
        ]).astype(np.float32)

    def _create_session(self):
        return create_onnx_session(self.model_path)

    # ------------------------------------------------------------------
    # STFT packing (matches UVR's torch.stft usage)
    # ------------------------------------------------------------------

    def _stft(self, chunk: np.ndarray) -> np.ndarray:
        """(2, chunk_size) waveform -> (1, 4, dim_f, dim_t) model input."""
        n_fft = self.params["n_fft"]
        hop = self.params["hop"]
        dim_f = self.params["dim_f"]
        specs = []
        for ch in range(2):
            spec = librosa.stft(
                chunk[ch],
                n_fft=n_fft,
                hop_length=hop,
                window="hann",
                center=True,
                pad_mode="reflect",
            )
            # Crop the top bin(s): the model sees dim_f of n_fft//2+1 bins.
            spec = spec[:dim_f]
            specs.append(spec.real.astype(np.float32))
            specs.append(spec.imag.astype(np.float32))
        # Channel order (L.re, L.im, R.re, R.im), matching UVR's
        # reshape([B, ch, 2, f, t]) -> [B, 4, f, t] flattening.
        return np.stack(specs)[np.newaxis]

    def _istft(self, spec4: np.ndarray, length: int) -> np.ndarray:
        """(4, dim_f, dim_t) model output -> (2, length) waveform."""
        n_fft = self.params["n_fft"]
        hop = self.params["hop"]
        bins = n_fft // 2 + 1
        out = []
        for ch in range(2):
            re = spec4[ch * 2]
            im = spec4[ch * 2 + 1]
            full = np.zeros((bins, re.shape[1]), dtype=np.complex64)
            full[: re.shape[0]] = re + 1j * im
            wave = librosa.istft(
                full,
                hop_length=hop,
                n_fft=n_fft,
                window="hann",
                center=True,
                length=length,
            )
            out.append(wave.astype(np.float32))
        return np.stack(out)

    # ------------------------------------------------------------------
    # Windowed inference
    # ------------------------------------------------------------------

    def _demix(self, mix: np.ndarray, session) -> np.ndarray:
        """Run the model across the track; return the primary stem.

        Classic UVR windowing: each window is ``chunk_size`` samples with
        ``trim`` of context at both ends; only the central ``gen_size``
        samples of each window's result are kept.
        """
        n_fft = self.params["n_fft"]
        hop = self.params["hop"]
        dim_t = self.params["dim_t"]

        trim = n_fft // 2
        chunk_size = hop * (dim_t - 1)
        gen_size = chunk_size - 2 * trim

        n = mix.shape[1]
        remainder = n % gen_size
        pad = (gen_size - remainder) if remainder else 0
        padded = np.concatenate(
            [
                np.zeros((2, trim), dtype=np.float32),
                mix,
                np.zeros((2, pad + trim), dtype=np.float32),
            ],
            axis=1,
        )

        input_name = session.get_inputs()[0].name
        out = np.zeros((2, n + pad), dtype=np.float32)
        num_windows = (n + pad) // gen_size

        for k in range(num_windows):
            if self._is_cancelled:
                return out[:, :n]
            start = k * gen_size
            window = padded[:, start:start + chunk_size]
            spec = self._stft(window)
            pred = session.run(None, {input_name: spec})[0][0]
            wave = self._istft(pred, chunk_size)
            out[:, start:start + gen_size] = wave[:, trim:trim + gen_size]
            # 15..90% across the windows.
            pct = 15 + int(75 * (k + 1) / num_windows)
            self.progress.emit(
                pct, f"Separating (2-stem)... {k + 1}/{num_windows}"
            )

        return out[:, :n] * self.params["compensate"]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save(
        self, primary: np.ndarray, secondary: np.ndarray,
    ) -> dict[str, str]:
        os.makedirs(self.output_dir, exist_ok=True)
        clear_completion_marker(self.output_dir)
        primary_file, secondary_file = PRIMARY_TO_FILES[
            self.params["primary_stem"]
        ]
        result = {}
        for name, data in ((primary_file, primary), (secondary_file, secondary)):
            path = os.path.join(self.output_dir, f"{name}.wav")
            sf.write(path, data.T, SAMPLE_RATE, subtype="PCM_16")
            result[name] = path
        write_completion_marker(self.output_dir, self.model_key)
        return result
