"""StreamingStretcher: the real-time speed and pitch engine (src/stretch.py).

The engine replaced a whole-song librosa render that took a minute and
several gigabytes on a ten-minute song (#159). These tests pin that it
keeps the song's pitch, interval, tonal balance, and level (#227) when fed
in audio-callback-sized blocks, and that its position, loop-wrap, and
parameter changes behave.
"""

import numpy as np
import pytest

from src.stretch import HOP, StreamingStretcher, _peak_owners, pitch_rate

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


def _band(seconds=3.0, seed=0):
    """Stereo polyphonic signal: six notes with harmonics, and noise.

    A pure sine hardly loses level in a phase vocoder; real music does,
    because the partials around each peak drift out of phase with one
    another and partly cancel. This is enough partials for that to show.
    """
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    left = 0.03 * rng.standard_normal(len(t))
    for note in [110.0, 220.0, 261.63, 329.63, 392.0, 493.88]:
        for harmonic in range(1, 11):
            if note * harmonic < 12000:
                left += 0.05 / harmonic * np.sin(
                    2 * np.pi * note * harmonic * t
                    + rng.uniform(0, 2 * np.pi),
                )
    right = np.roll(left, 301) * 0.9
    return np.stack([left, right], axis=1).astype(np.float32)


def _db(x, reference, trim=8192):
    level = np.sqrt(np.mean(x[trim:-trim] ** 2))
    return float(20 * np.log10(level / np.sqrt(np.mean(reference ** 2))))


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


def _peak_hz(mono, trim=8192):
    """Frequency of the loudest spectral peak, to a fraction of a bin."""
    x = mono[trim:-trim]
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(np.argmax(spectrum))
    # Parabolic interpolation on the log magnitude around the peak.
    a, b, c = np.log(spectrum[k - 1:k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * SR / len(x)


def _band_db(stereo, edges=(60, 250, 1000, 4000, 12000), trim=8192):
    """Energy (dB) in each band between *edges*, both channels summed."""
    x = stereo[trim:-trim]
    power = np.sum(np.abs(np.fft.rfft(x, axis=0)) ** 2, axis=1)
    power /= len(x) ** 2  # mean power, whatever the length
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    return np.array([
        10 * np.log10(np.sum(power[(freqs >= lo) & (freqs < hi)]))
        for lo, hi in zip(edges[:-1], edges[1:])
    ])


def test_unity_speed_and_pitch_reproduce_the_input():
    """The phase vocoder is transparent at 1.0x (phase locking included)."""
    audio = _signal()
    ours, _ = _stream(audio)
    assert _relative_error(ours, audio) < 1e-4


@pytest.mark.parametrize("speed", [0.5, 0.75, 1.25, 2.0])
def test_speed_keeps_the_pitch(speed):
    t = np.arange(SR * 3) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ours, _ = _stream(np.stack([tone, tone], axis=1), speed=speed)
    assert len(ours) == int(round(len(tone) / speed))
    assert _peak_hz(ours[:, 0]) == pytest.approx(440, rel=1e-3)


@pytest.mark.parametrize("semitones", [-7, -2, 3, 7])
def test_pitch_moves_by_the_interval_and_keeps_the_length(semitones):
    t = np.arange(SR * 3) / SR
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ours, _ = _stream(np.stack([tone, tone], axis=1), semitones=semitones)
    assert len(ours) == len(tone)
    assert _peak_hz(ours[:, 0]) == pytest.approx(
        440 * 2 ** (semitones / 12), rel=1e-3,
    )


@pytest.mark.parametrize("speed", [0.75, 0.95, 1.25])
def test_tonal_balance_holds(speed):
    """No band of the spectrum gains or loses more than about 1.5 dB.

    Without phase locking the highs lost up to 3.5 dB (#227).
    """
    audio = _band()
    ours, _ = _stream(audio, speed=speed)
    shift = _band_db(ours) - _band_db(audio)
    assert np.all(np.abs(shift) < 1.5), shift


def test_combined_keeps_the_level_two_passes_lose():
    """One pass for speed and pitch; two passes (as librosa) lose ~3 dB."""
    audio = _signal()
    ours, _ = _stream(audio, speed=0.75, semitones=-2)
    rms = float(np.sqrt(np.mean(ours[8192:-8192] ** 2)))
    original = float(np.sqrt(np.mean(audio ** 2)))
    assert rms == pytest.approx(original, rel=0.15)


@pytest.mark.parametrize("speed, semitones", [
    (0.5, 0), (0.75, 0), (0.95, 0), (1.25, 0),
    (1.0, 2), (1.0, -2), (1.0, 7), (1.0, -7), (0.75, -2),
])
def test_polyphonic_level_holds_when_speed_or_pitch_moves(speed, semitones):
    """Touching Speed or Pitch must not drop the song by ~3 dB (#227)."""
    audio = _band()
    ours, _ = _stream(audio, speed=speed, semitones=semitones)
    assert abs(_db(ours, audio)) < 1.0


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


def test_peak_owners_take_the_nearest_peak_the_lower_on_a_tie():
    """The rule site/stretch-worklet.js copies, edges and ties included."""
    magnitude = np.array([[1, 3, 2, 0, 5, 4, 4],
                          [2, 1, 1, 1, 3, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(_peak_owners(magnitude), [
        [1, 1, 1, 4, 4, 4, 4],  # bin 6 ties bin 5: not a peak
        [0, 0, 0, 4, 4, 4, 4],  # bin 0 counts; bin 2 ties at distance 2
    ])
