"""Generate the startup arpeggio WAV for stemma's splash screen.

Synthesizes an Am7 arpeggio (A2-C3-E3-G3-E3-C3) matching the six letters
of "stemma" placed on staff lines.  Each note is a sine wave with soft
attack/release envelope.  The output is a single WAV file designed to be
played once via winsound while the splash animation runs.

Run:  python scripts/generate_startup_audio.py
"""

import os

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100

NOTE_FREQS = {
    "A2": 110.00,
    "C3": 130.81,
    "E3": 164.81,
    "G3": 196.00,
    "B3": 246.94,
}

ARPEGGIO_NOTES = ["A2", "C3", "E3", "G3", "E3", "C3"]

NOTE_ONSET_S = 0.0
NOTE_SPACING_S = 0.28
NOTE_DURATION_S = 0.60
ATTACK_S = 0.02
RELEASE_S = 0.25
AMPLITUDE = 0.35


def _envelope(length: int, attack: int, release: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float64)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    if release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release)
    return env


def generate_arpeggio(out_path: str) -> None:
    total_duration = NOTE_ONSET_S + NOTE_SPACING_S * 5 + NOTE_DURATION_S + 0.3
    total_samples = int(total_duration * SAMPLE_RATE)
    audio = np.zeros(total_samples, dtype=np.float64)

    attack_samples = int(ATTACK_S * SAMPLE_RATE)
    release_samples = int(RELEASE_S * SAMPLE_RATE)
    note_samples = int(NOTE_DURATION_S * SAMPLE_RATE)

    for i, note_name in enumerate(ARPEGGIO_NOTES):
        freq = NOTE_FREQS[note_name]
        onset = int((NOTE_ONSET_S + i * NOTE_SPACING_S) * SAMPLE_RATE)
        t = np.arange(note_samples, dtype=np.float64) / SAMPLE_RATE

        tone = np.sin(2.0 * np.pi * freq * t)
        tone += 0.3 * np.sin(2.0 * np.pi * 2 * freq * t)
        tone += 0.1 * np.sin(2.0 * np.pi * 3 * freq * t)

        env = _envelope(note_samples, attack_samples, release_samples)
        tone *= env * AMPLITUDE

        end = min(onset + note_samples, total_samples)
        audio[onset : end] += tone[: end - onset]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
        audio *= AMPLITUDE

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {out_path}  ({len(audio) / SAMPLE_RATE:.2f}s, {SAMPLE_RATE} Hz)")


CHORD_NOTES = ["C3", "E3", "G3", "B3"]
CHORD_STRUM_S = 0.04
CHORD_DURATION_S = 1.0
CHORD_RELEASE_S = 0.50


def generate_chord(out_path: str) -> None:
    """Synthesize a Cmaj7 strummed chord (notes staggered ~40ms apart)."""
    last_onset = CHORD_STRUM_S * (len(CHORD_NOTES) - 1)
    total_duration = last_onset + CHORD_DURATION_S + 0.3
    total_samples = int(total_duration * SAMPLE_RATE)
    audio = np.zeros(total_samples, dtype=np.float64)

    attack_samples = int(ATTACK_S * SAMPLE_RATE)
    release_samples = int(CHORD_RELEASE_S * SAMPLE_RATE)
    note_samples = int(CHORD_DURATION_S * SAMPLE_RATE)

    for i, note_name in enumerate(CHORD_NOTES):
        freq = NOTE_FREQS[note_name]
        onset = int(i * CHORD_STRUM_S * SAMPLE_RATE)
        t = np.arange(note_samples, dtype=np.float64) / SAMPLE_RATE

        tone = np.sin(2.0 * np.pi * freq * t)
        tone += 0.3 * np.sin(2.0 * np.pi * 2 * freq * t)
        tone += 0.1 * np.sin(2.0 * np.pi * 3 * freq * t)

        env = _envelope(note_samples, attack_samples, release_samples)
        tone *= env * AMPLITUDE

        end = min(onset + note_samples, total_samples)
        audio[onset:end] += tone[: end - onset]

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
        audio *= AMPLITUDE

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {out_path}  ({len(audio) / SAMPLE_RATE:.2f}s, {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "assets", "audio", "arpeggio.wav")
    generate_arpeggio(out)
    chord_out = os.path.join(root, "assets", "audio", "chord.wav")
    generate_chord(chord_out)
