"""Read imported source audio at its real decoded length.

libsndfile cannot trust the frame count of every MP3. A VBR MP3 without
a Xing/VBRI header gets a length estimated from the first frame's
bitrate, and libsndfile then reads only that many frames, so the song
silently stops early (issue #206). MP3 sources are therefore decoded
with the bundled ffmpeg, which reads until the end of the stream.
Everything else, and MP3s when ffmpeg is unavailable or fails, goes
through soundfile.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import soundfile as sf

from src.crash_log import logger

# Keep a console window from flashing up in the windowed build.
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _ffmpeg_exe() -> str | None:
    """Return the bundled ffmpeg, or one on PATH, or None."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any failure means not bundled
        return shutil.which("ffmpeg")


def _decode_with_ffmpeg(
    path: str, sample_rate: int, channels: int, sample_format: str,
) -> bytes | None:
    """Decode the first audio stream to raw interleaved PCM.

    Returns None when ffmpeg is missing, fails, or produces no audio, so
    the caller can fall back to soundfile.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        return None
    cmd = [
        exe, "-nostdin", "-v", "error", "-i", path,
        "-map", "0:a:0", "-vn",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-f", sample_format, "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, check=False,
            creationflags=_NO_WINDOW,
        )
    except OSError as exc:
        logger.warning("ffmpeg could not start for %s: %s", path, exc)
        return None
    if result.returncode != 0 or not result.stdout:
        logger.warning(
            "ffmpeg failed to decode %s (exit %s): %s",
            path, result.returncode,
            result.stderr.decode("utf-8", "replace").strip()[:500],
        )
        return None
    return result.stdout


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Return ``(audio, sample_rate)`` with audio float32, (samples, ch).

    Raises soundfile's error when the file cannot be opened at all.
    """
    info = sf.info(path)
    if info.format == "MP3":
        raw = _decode_with_ffmpeg(
            path, info.samplerate, info.channels, "f32le",
        )
        if raw is not None:
            audio = np.frombuffer(raw, dtype="<f4")
            usable = audio.size - audio.size % info.channels
            return (
                audio[:usable].reshape(-1, info.channels).copy(),
                info.samplerate,
            )
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    return audio, sr


def audio_duration(path: str) -> float:
    """Return the decoded length of *path* in seconds.

    For MP3s this decodes the whole file (mono 16-bit, to keep it light),
    because the header-derived length can be a bitrate estimate.
    """
    info = sf.info(path)
    if info.format == "MP3":
        raw = _decode_with_ffmpeg(path, info.samplerate, 1, "s16le")
        if raw is not None:
            return (len(raw) // 2) / info.samplerate
    return info.duration
