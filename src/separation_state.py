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


# separation_status results. Only INCOMPLETE justifies removing a song:
# UNKNOWN means the state cannot be judged, and user audio is never deleted
# on a guess.
COMPLETE = "complete"
INCOMPLETE = "incomplete"
UNKNOWN = "unknown"


def _has_stems(song_dir: str, stems: tuple[str, ...]) -> bool:
    return all(
        os.path.isfile(os.path.join(song_dir, f"{stem}.wav"))
        for stem in stems
    )


def infer_model_from_stems(song_dir: str) -> str:
    """Return the model whose full stem set is on disk, or ``""``.

    Larger sets win, so a six-stem folder is not mistaken for the four- or
    two-stem sets it contains.
    """
    for model_key, stems in sorted(
        EXPECTED_STEMS.items(), key=lambda item: -len(item[1]),
    ):
        if _has_stems(song_dir, stems):
            return model_key
    return ""


def _read_marker_model(song_dir: str) -> str | None:
    """Return the model a valid marker records; None if it can't be judged.

    Raises FileNotFoundError when there is no marker at all.
    """
    try:
        with open(_marker_path(song_dir), encoding="utf-8") as f:
            state = json.load(f)
        model_key = state["model"]
        stems = expected_stems(model_key)
        if (
            state.get("version") != COMPLETION_STATE_VERSION
            or stems is None
            or tuple(state.get("stems", ())) != stems
        ):
            return None
    except FileNotFoundError:
        raise
    except (OSError, TypeError, KeyError, ValueError, AttributeError):
        # Unreadable (locked, garbled) or from another build.
        return None
    return model_key


def recorded_or_inferred_model(song_dir: str) -> str:
    """Model from a valid marker, else inferred from the stems on disk."""
    try:
        model_key = _read_marker_model(song_dir)
    except FileNotFoundError:
        model_key = None
    return model_key or infer_model_from_stems(song_dir)


def separation_status(song_dir: str, model_used: str) -> str:
    """Return COMPLETE, INCOMPLETE, or UNKNOWN for a song's stem folder.

    A valid marker, or a markerless song whose persisted model is known,
    is judged by its canonical stem set. A marker that cannot be read or
    comes from another build, and a markerless song without a known model
    (a library rebuilt from disk), are UNKNOWN unless a full stem set is
    present.
    """
    try:
        model_key = _read_marker_model(song_dir)
    except FileNotFoundError:
        model_key = model_used if expected_stems(model_used) else None
        if model_key is None:
            return COMPLETE if infer_model_from_stems(song_dir) else UNKNOWN
    if model_key is None:
        return COMPLETE if infer_model_from_stems(song_dir) else UNKNOWN
    return (
        COMPLETE if _has_stems(song_dir, expected_stems(model_key))
        else INCOMPLETE
    )


def separation_is_complete(song_dir: str, model_used: str) -> bool:
    """Validate marker-based state, or a complete legacy persisted model."""
    return separation_status(song_dir, model_used) == COMPLETE
