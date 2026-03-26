"""Tests for shared audio/visual timing constants."""

from src.ui.audio_sync import LOGO_AUDIO_VISUAL_LAG_MS, SPLASH_SOUND_SYNC_MS


def test_splash_sync_is_non_negative() -> None:
    assert SPLASH_SOUND_SYNC_MS >= 0


def test_logo_lag_is_non_negative() -> None:
    assert LOGO_AUDIO_VISUAL_LAG_MS >= 0


def test_splash_sync_reasonable_range() -> None:
    assert 0 <= SPLASH_SOUND_SYNC_MS <= 200


def test_logo_lag_reasonable_range() -> None:
    assert 0 <= LOGO_AUDIO_VISUAL_LAG_MS <= 200
