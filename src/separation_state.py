"""Shared completion-state handling for separated song stems."""

import json
import os


COMPLETION_MARKER = ".separation-complete.json"
COMPLETION_STATE_VERSION = 1

EXPECTED_STEMS: dict[str, tuple[str, ...]] = {
    "htdemucs": ("drums", "bass", "other", "vocals"),
    "htdemucs_6s": (
        "drums",
        "bass",
        "other",
        "vocals",
        "guitar",
        "piano",
    ),
    "mdx_inst_hq3": ("vocals", "other"),
}


def expected_stems(model_key: str) -> tuple[str, ...] | None:
    """Return the canonical stem set for *model_key*, if supported."""
    return EXPECTED_STEMS.get(model_key)


def _marker_path(song_dir: str) -> str:
    return os.path.join(song_dir, COMPLETION_MARKER)


def clear_completion_marker(song_dir: str) -> None:
    """Remove completion state before starting a new set of writes."""
    marker = _marker_path(song_dir)
    for path in (marker, marker + ".tmp"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def write_completion_marker(song_dir: str, model_key: str) -> None:
    """Atomically mark a complete canonical stem set for *model_key*."""
    stems = expected_stems(model_key)
    if stems is None:
        raise ValueError(f"Unknown separation model: {model_key}")

    missing = [
        stem
        for stem in stems
        if not os.path.isfile(os.path.join(song_dir, f"{stem}.wav"))
    ]
    if missing:
        raise OSError(
            "Cannot mark separation complete; missing stems: "
            + ", ".join(missing)
        )

    marker = _marker_path(song_dir)
    tmp_path = marker + ".tmp"
    state = {
        "version": COMPLETION_STATE_VERSION,
        "model": model_key,
        "stems": list(stems),
    }
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, marker)
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


def separation_is_complete(song_dir: str, model_used: str) -> bool:
    """Validate marker-based state, or a complete legacy persisted model."""
    marker = _marker_path(song_dir)
    if os.path.exists(marker):
        try:
            with open(marker, encoding="utf-8") as f:
                state = json.load(f)
            model_key = state["model"]
            stems = expected_stems(model_key)
            if (
                state.get("version") != COMPLETION_STATE_VERSION
                or stems is None
                or tuple(state.get("stems", ())) != stems
            ):
                return False
        except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError):
            return False
    else:
        stems = expected_stems(model_used)
        if stems is None:
            return False

    return all(
        os.path.isfile(os.path.join(song_dir, f"{stem}.wav"))
        for stem in stems
    )
