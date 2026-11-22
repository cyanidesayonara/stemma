"""System motion preference lookup for UI animations."""

import ctypes
import sys

# SystemParametersInfo action for Windows Settings -> Accessibility ->
# Visual effects -> Animation effects (legacy "Show animations").
_SPI_GETCLIENTAREAANIMATION = 0x1042


def animations_enabled() -> bool:
    """Return False when Windows has client-area animations turned off.

    Defaults to True on non-Windows platforms or if the Win32 query fails.
    """
    if sys.platform != "win32":
        return True
    try:
        enabled = ctypes.c_int(1)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            _SPI_GETCLIENTAREAANIMATION,
            0,
            ctypes.byref(enabled),
            0,
        )
        if not ok:
            return True
        return bool(enabled.value)
    except OSError:
        return True
