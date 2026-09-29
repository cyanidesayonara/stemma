"""StreamingStretcher: the real-time speed and pitch engine (src/stretch.py).

The engine replaced a whole-song librosa render that took a minute and
several gigabytes on a ten-minute song (#159). These tests pin that it
sounds the same as librosa, fed in audio-callback-sized blocks, and that
its position, loop-wrap, and parameter changes behave.
"""

import numpy as np
import pytest
import librosa

from src.stretch import HOP, StreamingStretcher, pitch_rate

SR = 44100


def _signal(seconds=3.0, seed=0):
    """Stereo test signal: two tones, a click train, and a little noise."""
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    left = (0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.2 * np.sin(2 * np.pi * 1330 * t)
            + 0.02 * rng.standard_normal(len(t)))
    left[::SR // 4] += 0.8
    right = np.roll(left, 301) * 0.9
    return np.stack([left, right], axis=1).astype(np.float32)


def _source(audio, start=0):
    """A SourceRead over *audio*, from song frame *start*."""
    head = [start]

    def read(count):
        begin = head[0]
        chunk = audio[begin:begin + count]
        head[0] += len(chunk)
        return chunk, begin

    return read


def _stream(audio, speed=1.0, semitones=0, block=1024):
    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(speed, semitones)
    read = _source(audio)
    want = int(round(len(audio) / speed))
    parts = []
    while sum(len(p) for p in parts) < want:
        parts.append(stretcher.read(block, read))
    return np.concatenate(parts)[:want], stretcher


def _relative_error(a, b, trim=8192):
    n = min(len(a), len(b))
    a, b = a[trim:n - trim], b[trim:n - trim]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _librosa(audio, speed=1.0, semitones=0):
    out = []
    for ch in range(audio.shape[1]):
        mono = audio[:, ch]
        if semitones:
            mono = librosa.effects.pitch_shift(
                mono, sr=SR, n_steps=semitones, res_type="soxr_hq",
            )
        if speed != 1.0:
            mono = librosa.effects.time_stretch(mono, rate=speed)
        out.append(mono)
    return np.stack(out, axis=1)


@pytest.mark.parametrize("speed", [0.5, 0.75, 1.25, 2.0])
def test_speed_matches_librosa_fed_in_callback_blocks(speed):
    audio = _signal()
    ours, _ = _stream(audio, speed=speed)
    assert _relative_error(ours, _librosa(audio, speed=speed)) < 1e-3


@pytest.mark.parametrize("semitones", [-7, -2, 3, 7])
def test_pitch_matches_librosa_fed_in_callback_blocks(semitones):
    audio = _signal()
    ours, _ = _stream(audio, semitones=semitones)
    assert _relative_error(
        ours, _librosa(audio, semitones=semitones),
    ) < 1e-3


def test_combined_keeps_the_level_two_passes_lose():
    """One pass for speed and pitch; librosa's two passes lose ~3 dB."""
    audio = _signal()
    ours, _ = _stream(audio, speed=0.75, semitones=-2)
    rms = float(np.sqrt(np.mean(ours[8192:-8192] ** 2)))
    original = float(np.sqrt(np.mean(audio ** 2)))
    assert rms == pytest.approx(original, rel=0.15)


@pytest.mark.parametrize("block", [64, 441, 1024, 4096])
def test_block_size_does_not_change_the_output(block):
    audio = _signal(seconds=1.5)
    reference, _ = _stream(audio, speed=0.8, semitones=2, block=1024)
    other, _ = _stream(audio, speed=0.8, semitones=2, block=block)
    np.testing.assert_allclose(other, reference, atol=1e-5)


def test_position_follows_the_song_at_the_speed():
    audio = _signal(seconds=4.0)
    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(0.5, 0)
    read = _source(audio)
    for _ in range(100):
        stretcher.read(1024, read)
    heard = 100 * 1024 * 0.5
    assert stretcher.position == pytest.approx(heard, abs=2 * HOP)


def test_reset_starts_at_the_given_song_frame():
    audio = _signal(seconds=4.0)
    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(0.75, 0)
    stretcher.reset(SR)
    read = _source(audio, start=SR)
    for _ in range(20):
        stretcher.read(1024, read)
    assert stretcher.position == pytest.approx(
        SR + 20 * 1024 * 0.75, abs=2 * HOP,
    )


def test_speed_change_takes_effect_without_a_reset():
    audio = _signal(seconds=6.0)
    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(1.0, 0)
    read = _source(audio)
    for _ in range(20):
        stretcher.read(1024, read)
    before = stretcher.position
    stretcher.set_params(0.5, 0)
    for _ in range(40):
        stretcher.read(1024, read)
    assert stretcher.position - before == pytest.approx(
        40 * 1024 * 0.5, abs=4 * HOP,
    )


def test_pitch_change_mid_stream_has_no_jump():
    """The old resampler is flushed into the queue before the new one."""
    audio = _signal(seconds=4.0)
    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(1.0, 2)
    read = _source(audio)
    parts = [stretcher.read(1024, read) for _ in range(40)]
    stretcher.set_params(1.0, -3)
    parts += [stretcher.read(1024, read) for _ in range(40)]
    out = np.concatenate(parts)[:, 0]
    steps = np.abs(np.diff(out))
    switch = 40 * 1024
    near = steps[switch - 4096:switch + 4096]
    # No step at the change beyond the largest the signal itself makes
    # (the start is skipped: the test signal opens on a click).
    assert near.max() <= steps[8192:].max()
    before = np.sqrt(np.mean(out[switch - 8192:switch] ** 2))
    after = np.sqrt(np.mean(out[switch:switch + 8192] ** 2))
    assert after == pytest.approx(before, rel=0.1)


def test_loop_wrap_is_found_where_it_is_heard():
    audio = _signal(seconds=4.0)
    loop_a, loop_b = SR // 2, SR * 2
    head = [0]

    def looping(count):
        if head[0] >= loop_b:
            head[0] = loop_a
        begin = head[0]
        count = min(count, loop_b - begin)
        head[0] += count
        return audio[begin:begin + count], begin

    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(0.75, 0)
    heard = 0
    wrapped_at = None
    while heard < SR * 4:
        stretcher.fill(1024, looping)
        jump = stretcher.jump_offset()
        if jump == 0:
            wrapped_at = heard
            stretcher.clear_jump()
            break
        n = 1024 if jump is None else min(1024, jump)
        stretcher.read(n, looping)
        heard += n
    assert wrapped_at is not None
    # 2 s of song at 0.75x is heard after 2.67 s.
    assert wrapped_at == pytest.approx(loop_b / 0.75, abs=2 * HOP)


def test_end_of_song_finishes():
    audio = _signal(seconds=1.0)
    stretcher = StreamingStretcher(SR, 2)
    stretcher.set_params(1.25, 0)
    read = _source(audio)
    blocks = 0
    while not stretcher.finished and blocks < 200:
        stretcher.read(1024, read)
        blocks += 1
    assert stretcher.finished
    assert blocks * 1024 == pytest.approx(len(audio) / 1.25, abs=4 * 1024)


def test_pitch_rate():
    assert pitch_rate(12) == pytest.approx(0.5)
    assert pitch_rate(-12) == pytest.approx(2.0)
    assert pitch_rate(0) == 1.0
