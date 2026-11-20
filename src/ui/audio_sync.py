"""Empirical audio/visual alignment for splash and logo SFX.

Tuned on Windows, where the splash and the logo clicks both play through
``winsound``. Adjust if motion and sound drift on other setups.
"""

# Splash: letters vs ``winsound`` + mixer latency (lower if audio sounds early).
SPLASH_SOUND_SYNC_MS = 32

# Main and arpeggio logos: animation clock vs click sound output
# (raise if audio sounds late).
LOGO_AUDIO_VISUAL_LAG_MS = 48
