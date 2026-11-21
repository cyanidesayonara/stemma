"""Make the website's mixer-demo songs from stemma's own separations.

Each song in ``SONGS`` is a free-to-use track (CC0 or CC BY) with vocals,
separated by stemma itself into six stems (the htdemucs_6s model) or four
(htdemucs), so the demo plays what the app really produces. For each one
this script:

1. cuts a clip of whole bars, starting on a downbeat (``start`` in seconds,
   or picked here: the loudest stretch with every part playing),
2. runs the app's detector (``src.beat_detector``) on the clip for the
   tempo, key, beats, downbeats and chords the demo shows,
3. writes its stems as MP3 to ``site/demo/<id>/`` and a waveform
   overview, and lists every song in ``site/demo/songs.json``.

    python scripts/make_demo_songs.py STEMS_DIR

``STEMS_DIR`` holds one folder per song, named like its ``folder`` below,
with the WAVs the app writes when it separates: ``vocals``, ``drums``,
``bass`` and ``other``, plus ``guitar`` and ``piano`` for six stems.
The sources and their licences are in ``SONGS`` and on the page.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.beat_detector import detect_bpm_and_key  # noqa: E402

OUT = ROOT / "site" / "demo"
# the app's stem order (ALL_STEM_NAMES); a 4-stem song has the first four
STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")
CLIP_SECONDS = 40.0
PEAKS = 480          # waveform overview points per stem
EDGE_FADE = 0.02     # seconds faded at each end of a clip

# Two songs in six stems, then four in four. ccMixter tracks are credited
# with the attribution name each page sets (handle in brackets) and the
# singer whose a cappella they build on, as the licence asks.
SONGS = [
    {
        "id": "makes-shift-salvation",
        "folder": "holizna-makes-shift-salvation",
        "title": "Makes-Shift Salvation",
        "artist": "HoliznaCC0",
        "genre": "Indie rock",
        "licence": "CC0 1.0",
        "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source": "https://freemusicarchive.org/music/holiznacc0/"
                  "makeshift-salvation/makes-shift-salvation",
        "start": 179.82,
    },
    {
        "id": "blue-part-ii",
        "folder": "jimmy233-blue-part-ii",
        "title": "Blue Part II",
        "artist": "JCrews",
        "credit_as": "JCrews (jimmy233) feat. Adisa McKenzie",
        "genre": "Soul",
        "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "source": "https://ccmixter.org/files/jimmy233/58818",
        "start": 192.0,
    },
    {
        "id": "free-music-and-free-beer",
        "folder": "admiralbob77-free-music-and-free-beer-4",
        "title": "Free Music and Free Beer",
        "artist": "Admiral Bob",
        "credit_as": "Admiral Bob (admiralbob77) feat. SackJo22",
        "genre": "Blues",
        "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "source": "https://ccmixter.org/files/admiralbob77/38104",
        "start": None,
    },
    {
        "id": "girls-like-you",
        "folder": "jeffspeed68-girls-like-you",
        "title": "Girls Like You",
        "artist": "Stefan Kartenberg",
        "credit_as": "Stefan Kartenberg (JeffSpeed68) feat. Thespinwires",
        "genre": "Garage rock",
        "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "source": "https://ccmixter.org/files/JeffSpeed68/44932",
        "start": None,
    },
    {
        "id": "let-it-in",
        "folder": "josh-woodward-let-it-in",
        "title": "Let It In",
        "artist": "Josh Woodward",
        "genre": "Funk rock",
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "source": "https://joshwoodward.com/song/LetItIn",
        "start": None,
    },
    {
        "id": "where-you-are-now",
        "folder": "alexberoza-where-you-are-now",
        "title": "Where You Are Now",
        "artist": "AlexBeroza",
        "credit_as": "AlexBeroza feat. WolfFly",
        "genre": "Hard rock",
        "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "source": "https://ccmixter.org/files/AlexBeroza/33065",
        "start": None,
    },
]


def beat_model() -> str | None:
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "stemma",
                        "models", "beat_this.onnx")
    return path if os.path.isfile(path) else None


def load_stems(folder: Path) -> tuple[dict[str, np.ndarray], int]:
    stems = {}
    rate = 0
    present = [n for n in STEMS if (folder / f"{n}.wav").is_file()]
    if present not in (list(STEMS[:4]), list(STEMS)):
        raise SystemExit(f"{folder}: expected the 4 or 6 stems the app "
                         f"writes, found {present or 'none'}")
    for name in present:
        data, rate = sf.read(folder / f"{name}.wav", dtype="float32",
                             always_2d=True)
        stems[name] = data[:, :2] if data.shape[1] >= 2 else \
            np.repeat(data, 2, axis=1)
    return stems, rate


def rms(x: np.ndarray, rate: int, window: float) -> np.ndarray:
    """RMS per *window* seconds of a mono signal."""
    size = int(rate * window)
    usable = len(x) // size * size
    return np.sqrt((x[:usable].reshape(-1, size) ** 2).mean(axis=1) + 1e-12)


def pick_start(stems: dict[str, np.ndarray], rate: int,
               downbeats: list[float], length: float) -> float:
    """The downbeat that starts the fullest stretch: every part playing,
    no quiet break, in the middle of the song."""
    mix = sum(s.mean(axis=1) for s in stems.values())
    total = len(mix) / rate
    loud = rms(mix, rate, 1.0)
    parts = {n: rms(s.mean(axis=1), rate, 1.0) for n, s in stems.items()}
    best, best_score = None, -1.0
    for start in downbeats:
        if start < total * 0.12 or start + length > total * 0.85:
            continue
        a, b = int(start), int(start + length)
        window = loud[a:b]
        if len(window) < b - a:
            continue
        score = float(window.min() / (loud.max() + 1e-9)) * 2
        for n, p in parts.items():
            active = p[a:b] > 0.1 * (p.max() + 1e-9)
            score += float(active.mean()) * (1.5 if n == "vocals" else 1)
        if score > best_score:
            best, best_score = start, score
    return best if best is not None else total * 0.3


def whole_bars(downbeats: list[float], start: float, length: float) -> float:
    """End time: the downbeat closest to start + length."""
    after = [d for d in downbeats if d > start + length * 0.75]
    if not after:
        return start + length
    return min(after, key=lambda d: abs(d - start - length))


def overview(clip: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Waveform overview per stem: RMS in PEAKS bins, 0-99, scaled to the
    song's loudest bin across all stems so the lanes compare as they do
    in the app, and a quiet stem looks quiet."""
    levels = {}
    for name, data in clip.items():
        mono = data.mean(axis=1)
        levels[name] = np.array([
            np.sqrt(np.mean(x ** 2)) for x in np.array_split(mono, PEAKS)
        ])
    top = max(float(v.max()) for v in levels.values()) or 1.0
    return {
        name: [int(round(min(1.0, float(x) / top) * 99)) for x in v]
        for name, v in levels.items()
    }


def make_song(song: dict, stems_dir: Path) -> dict:
    stems, rate = load_stems(stems_dir / song["folder"])
    model = beat_model()
    if model is None:
        raise SystemExit("beat_this.onnx is missing: run a detection in "
                         "stemma once so it downloads, then rerun")
    whole = detect_bpm_and_key(stems, rate, model_path=model)
    start = song["start"]
    if start is None:
        start = pick_start(stems, rate, whole.downbeat_times, CLIP_SECONDS)
    elif whole.downbeat_times:
        # a hand-picked start lands on the nearest bar line
        start = min(whole.downbeat_times, key=lambda d: abs(d - start))
    end = whole_bars(whole.downbeat_times, start, CLIP_SECONDS)
    a, b = int(round(start * rate)), int(round(end * rate))
    clip = {n: s[a:b].copy() for n, s in stems.items()}
    fade = int(EDGE_FADE * rate)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    for s in clip.values():
        s[:fade] *= ramp
        s[-fade:] *= ramp[::-1]

    found = detect_bpm_and_key(clip, rate, model_path=model)
    folder = OUT / song["id"]
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in clip.items():
        sf.write(folder / f"{name}.mp3", data, rate, format="MP3",
                 subtype="MPEG_LAYER_III", compression_level=0.55,
                 bitrate_mode="VARIABLE")
    peaks = overview(clip)
    duration = (b - a) / rate
    print(f"{song['id']}: {start:.2f}s to {end:.2f}s ({duration:.1f}s), "
          f"{found.bpm:.0f} BPM, {found.key}", flush=True)
    entry = {k: song[k] for k in ("id", "title", "artist", "genre",
                                   "licence", "licence_url", "source")}
    if song.get("credit_as"):
        # the attribution the artist asks for, shown in the credit line
        entry["credit_as"] = song["credit_as"]
    entry.update({
        "clip_start": round(start, 3),
        "duration": round(duration, 3),
        "bpm": round(found.bpm or whole.bpm, 1),
        "key": found.key,
        "beats": [round(t, 3) for t in found.beat_times],
        "downbeats": [round(t, 3) for t in found.downbeat_times],
        "chords": [[round(t, 3), c] for t, c in found.chord_sequence],
        "stems": list(clip),
        "peaks": peaks,
    })
    return entry


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    stems_dir = Path(argv[0])
    entries = [make_song(song, stems_dir) for song in SONGS]
    OUT.mkdir(parents=True, exist_ok=True)
    # one song per line: small, and a remade song is a one-line diff
    lines = ",\n".join(
        json.dumps(e, ensure_ascii=False, separators=(",", ":"))
        for e in entries
    )
    (OUT / "songs.json").write_text(
        '{"songs":[\n' + lines + "\n]}\n", encoding="utf-8", newline="\n",
    )
    print(f"Wrote {len(entries)} songs to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
