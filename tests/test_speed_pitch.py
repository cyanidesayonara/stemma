"""Speed and pitch on MultiTrackPlayer, applied live in the audio callback.

They used to be rendered for the whole song before they could be heard,
which took a minute and several gigabytes on a ten-minute song and made
playback stutter (#159). Now ``StreamingStretcher`` paths in the callback
stretch the mix as it plays; positions, loops, beats, and chords stay in
song time. These tests drive the real callback the way PortAudio does.
"""

import numpy as np
import pytest
import sounddevice as sd
from PySide6.QtWidgets import QApplication

from src.player import (
    PITCH_MAX_SEMITONES,
    PITCH_MIN_SEMITONES,
    RECORDING_STEM_PREFIX,
    MultiTrackPlayer,
)

SR = 44100
BLOCK = 1024


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _tone(seconds, freq=220.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


@pytest.fixture
def player(app):
    p = MultiTrackPlayer()
    p.apply_loaded_stems(
        {"vocals": _tone(6.0, 440.0), "drums": _tone(6.0, 110.0)}, SR,
    )
    p._is_playing = True  # drive the callback without a real stream
    yield p
    p._is_playing = False
    p.shutdown()


def _run(player, blocks):
    """Call the audio callback *blocks* times; return the audio it made."""
    out = []
    for _ in range(blocks):
        buf = np.zeros((BLOCK, 2), dtype=np.float32)
        try:
            player._audio_callback(buf, BLOCK, {}, None)
        except sd.CallbackStop:
            out.append(buf)
            break
        out.append(buf)
    return np.concatenate(out)


class TestKnobs:
    def test_defaults(self, player):
        assert player.speed == 1.0
        assert player.pitch_semitones == 0
        assert not player.stretching

    def test_speed_is_clamped(self, player):
        player.set_speed(0.1)
        assert player.speed == 0.5
        player.set_speed(9.0)
        assert player.speed == 2.0

    def test_pitch_is_clamped_and_coerced(self, player):
        player.set_pitch(99)
        assert player.pitch_semitones == PITCH_MAX_SEMITONES
        player.set_pitch(-99)
        assert player.pitch_semitones == PITCH_MIN_SEMITONES
        player.set_pitch(2.9)
        assert player.pitch_semitones == 2
        player.set_pitch("x")
        assert player.pitch_semitones == 2

    def test_changes_signal_at_once(self, player):
        speeds, pitches = [], []
        player.speed_changed.connect(speeds.append)
        player.pitch_changed.connect(pitches.append)
        player.set_speed(0.75)
        player.set_pitch(-2)
        player.set_pitch(-2)  # no-op
        assert speeds == [0.75]
        assert pitches == [-2]

    def test_identity_needs_no_stretcher(self, player):
        player.set_speed(0.75)
        assert player.stretching
        player.set_speed(1.0)
        assert not player.stretching

    def test_refused_while_recording(self, player):
        player._recording = True
        player.set_speed(0.75)
        player.set_pitch(3)
        assert player.speed == 1.0
        assert player.pitch_semitones == 0

    def test_recording_needs_original_speed_and_pitch(self, player):
        player.set_pitch(2)
        player.arm_recording(True)
        assert not player.recording_armed
        player.set_pitch(0)
        player.arm_recording(True)
        assert player.recording_armed

    def test_loading_a_song_resets_both(self, player):
        player.set_speed(0.5)
        player.set_pitch(3)
        player.apply_loaded_stems({"vocals": _tone(1.0)}, SR)
        assert player.speed == 1.0
        assert player.pitch_semitones == 0
        assert not player.stretching


class TestLivePlayback:
    def test_speed_plays_at_once_in_song_time(self, player):
        player.set_speed(0.5)
        out = _run(player, 40)
        assert np.abs(out).max() > 0.1  # sound from the first blocks
        assert player._current_frame == pytest.approx(
            40 * BLOCK * 0.5, abs=2048,
        )
        assert player.total_seconds == pytest.approx(6.0)

    def test_pitch_moves_the_tone(self, player):
        player.set_mute("drums", True)
        player.set_pitch(7)
        out = _run(player, 60)[8192:, 0]
        spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        peak_hz = np.argmax(spectrum) * SR / len(out)
        assert peak_hz == pytest.approx(440.0 * 2 ** (7 / 12), rel=0.01)

    def test_speed_change_mid_play_keeps_going(self, player):
        player.set_speed(0.75)
        _run(player, 20)
        before = player._current_frame
        player.set_speed(1.25)
        _run(player, 20)
        assert player._current_frame - before == pytest.approx(
            20 * BLOCK * 1.25, abs=3000,
        )

    def test_back_to_original_continues_from_the_heard_frame(self, player):
        player.set_speed(0.5)
        _run(player, 30)
        heard = player._current_frame
        player.set_speed(1.0)
        assert not player.stretching
        assert player._current_frame == pytest.approx(heard, abs=4)
        _run(player, 10)
        assert player._current_frame == pytest.approx(
            heard + 10 * BLOCK, abs=4,
        )

    def test_seek_while_stretched(self, player):
        player.set_speed(0.75)
        _run(player, 10)
        player.seek(4.0)
        assert player._current_frame == 4 * SR
        _run(player, 10)
        assert player._current_frame == pytest.approx(
            4 * SR + 10 * BLOCK * 0.75, abs=2048,
        )

    def test_mute_reaches_the_stretched_mix(self, player):
        player.set_speed(0.75)
        _run(player, 10)
        player.set_mute("vocals", True)
        player.set_mute("drums", True)
        out = _run(player, 20)
        assert np.abs(out[8192:]).max() < 1e-3

    def test_end_of_song_stops(self, player):
        player.set_speed(2.0)
        player.seek(5.0)
        out = _run(player, 200)
        assert not player._is_playing
        assert player._current_frame == player._total_frames
        assert len(out) < 200 * BLOCK


class TestLoopsAndBeats:
    def test_loop_wraps_where_it_is_heard(self, player):
        player.set_loop_a(1.0)
        player.set_loop_b(2.0)
        player.set_looping(True)
        player.seek(1.0)
        player.set_speed(0.5)
        blocks = int(2.5 * SR / BLOCK)  # 1 s of song at 0.5x is 2 s
        _run(player, blocks)
        assert player._loop_wrap_count == 1
        assert SR <= player._current_frame < 2 * SR

    def test_count_in_on_repeats_while_stretched(self, player):
        player.set_loop_a(1.0)
        player.set_loop_b(1.5)
        player.set_looping(True)
        player.set_count_in_enabled(True)
        player.set_count_in_on_repeats(True)
        player.set_count_in_beats(2)
        player.set_metronome_bpm(240)  # 0.25 s beats
        player.seek(1.0)
        player.set_speed(0.5)
        _run(player, int(1.2 * SR / BLOCK))  # past the first wrap
        assert player._loop_wrap_count == 1
        assert player.counting_in or player._current_frame < 1.2 * SR

    def test_beats_stay_in_song_time_at_any_speed(self, player):
        player.set_beat_times([0.5, 1.0, 1.5, 2.0], [])
        frames = player._beat_frames.copy()
        player.set_speed(0.5)
        player._recompute_beat_frames()
        np.testing.assert_array_equal(player._beat_frames, frames)

    def test_heard_bpm_follows_the_speed(self, player):
        player.set_beat_times([0.5, 1.0, 1.5, 2.0], [])
        assert player.instantaneous_bpm_at(int(1.2 * SR)) == pytest.approx(120)
        player.set_speed(0.5)
        assert player.instantaneous_bpm_at(int(1.2 * SR)) == pytest.approx(60)

    def test_synced_clicks_land_on_heard_beats(self, player):
        player.set_mute("vocals", True)
        player.set_mute("drums", True)
        player.set_beat_times([1.0, 2.0], [])
        player.set_beat_sync_enabled(True)
        player.set_metronome_enabled(True)
        player.set_speed(0.5)
        out = _run(player, int(4.5 * SR / BLOCK))[:, 0]
        loud = np.flatnonzero(np.abs(out) > 0.05)
        onsets = loud[np.insert(np.diff(loud) > SR // 10, 0, True)]
        # Song beats at 1 s and 2 s are heard at 2 s and 4 s at 0.5x.
        assert onsets / SR == pytest.approx([2.0, 4.0], abs=0.05)

    def test_chords_are_looked_up_in_song_time(self, player):
        player.set_chord_sequence([(0.0, "C"), (2.0, "G")])
        player.set_speed(0.5)
        assert player.chord_at(int(2.5 * SR)) == "G"


class TestRecordingTakes:
    def test_takes_keep_their_pitch_unless_synced(self, player):
        take = f"{RECORDING_STEM_PREFIX}1"
        player.add_recording_stem(take, _tone(6.0, 330.0))
        player.set_pitch(2)
        names = [reader.names for _, reader in player._stretch_paths]
        assert len(names) == 2
        assert take in names[1] and take not in names[0]
        semitones = [s.semitones for s, _ in player._stretch_paths]
        assert semitones == [2, 0]
        player.set_sync_recording_pitch(True)
        assert [r.names for _, r in player._stretch_paths] == [None]

    def test_speed_alone_stretches_takes_with_the_song(self, player):
        player.add_recording_stem(
            f"{RECORDING_STEM_PREFIX}1", _tone(6.0, 330.0),
        )
        player.set_speed(0.75)
        assert [r.names for _, r in player._stretch_paths] == [None]


def _onsets(out, sr=SR, threshold=0.05, gap=0.1):
    loud = np.flatnonzero(np.abs(out) > threshold)
    if not len(loud):
        return np.array([])
    return loud[np.insert(np.diff(loud) > int(sr * gap), 0, True)] / sr


class TestReviewFindings:
    @pytest.mark.parametrize("speed, semitones", [
        (1.0, 3), (0.75, 3), (0.5, -7), (1.25, 2),
    ])
    def test_every_synced_click_sounds_with_pitch_shifted(
        self, player, speed, semitones,
    ):
        """Hop-mark positions jump with a resampler; clicks went missing."""
        player.set_mute("vocals", True)
        player.set_mute("drums", True)
        beats = list(np.arange(0.2, 5.8, 0.4371))
        player.set_beat_times(beats, [])
        player.set_beat_sync_enabled(True)
        player.set_metronome_enabled(True)
        player.set_speed(speed)
        player.set_pitch(semitones)
        out = _run(player, int(5.7 / speed * SR / BLOCK))[:, 0]
        heard = _onsets(out)
        expected = [b / speed for b in beats if b / speed < len(out) / SR]
        assert len(heard) == len(expected)
        assert heard == pytest.approx(expected, abs=0.03)

    def test_returning_to_original_speed_does_not_click(self, player):
        player.set_mute("drums", True)
        player.set_speed(0.85)
        before = _run(player, 30)
        player.set_speed(1.0)
        after = _run(player, 10)
        out = np.concatenate([before, after])[:, 0]
        steady = np.abs(np.diff(before[4096:, 0])).max()
        assert np.abs(np.diff(out[len(before) - 64:])).max() < 2 * steady

    def test_pitch_change_crossfades(self, player):
        player.set_mute("drums", True)
        player.set_pitch(2)
        before = _run(player, 30)
        player.set_pitch(5)
        after = _run(player, 30)
        out = np.concatenate([before, after])[:, 0]
        steady = np.abs(np.diff(before[4096:, 0])).max()
        switch = len(before)
        assert np.abs(np.diff(out[switch - 64:switch + 4096])).max() < 2 * steady

    def test_seek_is_not_lost_to_a_running_callback(self, player):
        """The frame is set under the callback's lock."""
        player.set_speed(0.75)
        _run(player, 5)
        with player._stretch_lock:
            pass  # the callback holds this while it runs
        player.seek(3.0)
        _run(player, 1)
        assert player._current_frame >= 3 * SR

    def test_the_last_moments_of_the_song_are_heard(self, player):
        player.set_speed(2.0)
        player.set_pitch(-7)
        player.seek(5.5)
        out = _run(player, 200)
        # 0.5 s of song at 2x is 0.25 s; allow a hop of analysis edge.
        tail = np.flatnonzero(np.abs(out[:, 0]) > 0.01)
        assert tail[-1] / SR == pytest.approx(0.25, abs=0.02)
