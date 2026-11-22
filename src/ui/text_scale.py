"""Windows "Make text bigger" lookup for the application stylesheet.

Windows Settings -> Accessibility -> Text size stores its slider as
``TextScaleFactor`` (100-225, a percentage) under
``HKCU\\Software\\Microsoft\\Accessibility``. Qt does not apply it to a
widget app, and stemma's stylesheet pins every font in points, so the
stylesheet multiplies its font sizes by this factor instead (#188).

Only stdlib modules are imported here so ``main.py`` can read the factor
before the heavy imports.
"""

import sys

_KEY = r"Software\Microsoft\Accessibility"
_VALUE = "TextScaleFactor"

# The range the Windows slider offers.
MIN_TEXT_SCALE = 1.0
MAX_TEXT_SCALE = 2.25


def clamp_text_scale(scale: float) -> float:
    """Clamp *scale* to the range the Windows Text size slider offers."""
    return max(MIN_TEXT_SCALE, min(MAX_TEXT_SCALE, float(scale)))


def text_scale_factor() -> float:
    """Return the Windows text size as a factor (1.0 = 100 %).

    Defaults to 1.0 off Windows, when the value is unset (the slider was
    never moved), or when the registry read fails.
    """
    if sys.platform != "win32":
        return 1.0
    try:
        import winreg  # Windows-only stdlib module.

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY) as key:
            value, _kind = winreg.QueryValueEx(key, _VALUE)
        return clamp_text_scale(int(value) / 100)
    except (OSError, TypeError, ValueError):
        return 1.0


def scaled_pt(points: float, scale: float) -> str:
    """Return a QSS ``font-size`` value: *points* times *scale*.

    Whole numbers print without a decimal (``10pt``); others keep one
    (``12.5pt``), which the QSS parser reads as a fractional point size.
    """
    value = round(points * scale, 1)
    if value == int(value):
        return f"{int(value)}pt"
    return f"{value}pt"
