"""WCAG contrast of the theme tokens, so a colour change cannot regress it.

4.5:1 for text (WCAG 1.4.3), 3:1 for controls, focus, and state fills
(WCAG 1.4.11). The pre-release audit measured the light accent at about
2:1 and several light stem names under 2.5:1 (#186).
"""

import pytest

from src.ui.styles import (
    DARK_COLORS,
    LIGHT_COLORS,
    STEM_COLORS_DARK,
    STEM_COLORS_LIGHT,
    badge_html,
)

THEMES = {
    "dark": (DARK_COLORS, STEM_COLORS_DARK),
    "light": (LIGHT_COLORS, STEM_COLORS_LIGHT),
}


def _luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg, bg, minimum", [
    ("text", "base", 4.5),
    ("text", "mantle", 4.5),
    ("text", "surface0", 4.5),       # buttons and badges
    ("subtext", "base", 4.5),        # loop points, master %, hints
    ("subtext", "mantle", 4.5),      # the same inside cards
    ("on_accent", "accent", 4.5),    # checked buttons, selected rows
    ("accent", "base", 3.0),         # slider handles, focus borders
    ("accent", "mantle", 3.0),
    ("accent", "surface0", 3.0),     # checked vs unchecked fill
    ("recording", "mantle", 4.5),    # take names
    ("recording", "surface0", 3.0),  # the Record glyph on its button
])
def test_token_pairs_meet_wcag(theme, fg, bg, minimum):
    colors, _ = THEMES[theme]
    assert contrast(colors[fg], colors[bg]) >= minimum


@pytest.mark.parametrize("theme", THEMES)
def test_stem_names_are_readable(theme):
    colors, stems = THEMES[theme]
    for name, color in stems.items():
        assert contrast(color, colors["mantle"]) >= 4.5, name


def test_confidence_no_longer_colours_the_value():
    """Red read as an error, and the light amber and green were under 3:1:
    the value keeps the text colour and a low one gets a "?"."""
    html = badge_html("light", "Key:", "D major", "low")
    assert "D major?" in html
    assert html.count(LIGHT_COLORS["text"]) == 2
    assert "D major?" not in badge_html("dark", "Key:", "D major", "high")
    assert "--?" not in badge_html("dark", "Chord:", "--", "low")
