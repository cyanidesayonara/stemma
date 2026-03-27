"""Small PySide6 helpers shared across UI and audio code."""

import warnings


def safe_disconnect(signal: object) -> None:
    """Disconnect all slots from *signal*, ignoring ``RuntimeError`` when none are connected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            signal.disconnect()
        except RuntimeError:
            pass
