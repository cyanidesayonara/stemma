"""Empirical audio/visual alignment for splash and logo SFX.

Tuned on Windows with ``winsound`` (splash) and ``QSoundEffect`` (logo
clicks). Adjust if motion and sound drift on other setups.
"""

# Splash: letters vs ``winsound`` + mixer latency (lower if audio sounds early).
SPLASH_SOUND_SYNC_MS = 32

# Main/footer logos: animation clock vs Qt SFX output (raise if audio sounds late).
LOGO_AUDIO_VISUAL_LAG_MS = 48
