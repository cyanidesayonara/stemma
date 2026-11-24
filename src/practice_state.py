"""Per-song practice state: the loop region, speed, and pitch last used.

Kept in the settings store next to the per-song detection results rather
than in ``library.json``: it changes on every loop or speed adjustment, and
must not be blocked by (or add writes to) the durable library index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

from src.player import PITCH_MAX_SEMITONES, PITCH_MIN_SEMITONES, SPEED_PRESETS


def _loop_point(value) -> float | None:
    """Return *value* as a loop point in seconds, or None if unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = float(value)
    except OverflowError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _clock(seconds: float) -> str:
    """Format *seconds* as m:ss."""
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


@dataclass(frozen=True)
class PracticeState:
    """The practice settings a song was last left with."""

    loop_a: float | None = None
    loop_b: float | None = None
    looping: bool = False
    speed: float = 1.0
    pitch: int = 0

    @property
    def is_default(self) -> bool:
        """True when there is nothing worth storing or showing."""
        return self == PracticeState()

    @property
    def has_region(self) -> bool:
        """True when both loop points are set in order."""
        return (
            self.loop_a is not None
            and self.loop_b is not None
            and self.loop_b > self.loop_a
        )

    def to_json(self) -> str:
        """Serialize for the settings store."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw) -> PracticeState:
        """Parse stored text; anything malformed falls back to its default."""
        try:
            data = json.loads(raw) if isinstance(raw, str) and raw else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            return cls()
        speed = data.get("speed")
        if isinstance(speed, bool) or speed not in SPEED_PRESETS:
            speed = 1.0
        pitch = data.get("pitch")
        if isinstance(pitch, bool) or not isinstance(pitch, int):
            pitch = 0
        return cls(
            loop_a=_loop_point(data.get("loop_a")),
            loop_b=_loop_point(data.get("loop_b")),
            looping=data.get("looping") is True,
            speed=float(speed),
            pitch=max(PITCH_MIN_SEMITONES, min(PITCH_MAX_SEMITONES, pitch)),
        )

    def badge_text(self) -> str:
        """Compact form for a library row, e.g. ``A-B 0.75x +2``."""
        parts = []
        if self.has_region:
            parts.append("A-B")
        if self.speed != 1.0:
            parts.append(f"{self.speed:g}x")
        if self.pitch:
            parts.append(f"{self.pitch:+d}")
        return " ".join(parts)

    def description_lines(self) -> list[str]:
        """Spelled-out form for tooltips and assistive technology."""
        lines = []
        if self.has_region:
            lines.append(f"Loop {_clock(self.loop_a)} to {_clock(self.loop_b)}")
        if self.speed != 1.0:
            lines.append(f"Speed {self.speed:g}x")
        if self.pitch:
            unit = "semitone" if abs(self.pitch) == 1 else "semitones"
            lines.append(f"Pitch {self.pitch:+d} {unit}")
        return lines
