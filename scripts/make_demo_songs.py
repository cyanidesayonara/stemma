"""Make the website's mixer-demo songs from stemma's own separations.

Each song in ``SONGS`` is a free-to-use track (CC0 or CC BY) separated into
six stems by stemma itself (the htdemucs_6s model), so the demo plays what
the app really produces. For each one this script:

1. cuts a clip of whole bars, starting on a downbeat (``start`` in seconds,
   or picked here: the loudest stretch with every part playing),
2. runs the app's detector (``src.beat_detector``) on the clip for the
   tempo, key, beats, downbeats and chords the demo shows,
3. writes the six stems as MP3 to ``site/demo/<id>/`` and a waveform
   overview, and lists every song in ``site/demo/songs.json``.

    python scripts/make_demo_songs.py STEMS_DIR

``STEMS_DIR`` holds one folder per song, named like its ``folder`` below,
with ``vocals.wav``, ``drums.wav``, ``bass.wav``, ``guitar.wav``,
``piano.wav`` and ``other.wav`` (what the app writes when it separates).
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
STEMS = ("vocals", "drums", "bass", "other", "guitar", "piano")
CLIP_SECONDS = 40.0
PEAKS = 480          # waveform overview points per stem
EDGE_FADE = 0.02     # seconds faded at each end of a clip

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
        "id": "swansong",
        "folder": "josh-woodward-swansong",
        "title": "Swansong",
        "artist": "Josh Woodward",
        "genre": "Folk pop",
        "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "source": "https://freemusicarchive.org/music/Josh_Woodward/"
                  "Breadcrumbs/JoshWoodward-Breadcrumbs-01-Swansong/",
        "start": 127.38,
    },
    {
        "id": "bossa-antigua",
        "folder": "kevin-macleod-bossa-antigua",
        "title": "Bossa Antigua",
        "artist": "Kevin MacLeod",
        "genre": "Bossa nova",
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "source": "https://incompetech.com/music/royalty-free/"
                  "index.html?isrc=USUAN1700069",
        "start": 88.74,
    },
    {
        "id": "funkorama",
        "folder": "kevin-macleod-funkorama",
        "title": "Funkorama",
        "artist": "Kevin MacLeod",
        "genre": "Funk",
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "source": "https://incompetech.com/music/royalty-free/"
                  "index.html?isrc=USUAN1100474",
        "start": 38.04,
    },
]


def beat_model() -> str | None:
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "stemma",
                        "models", "beat_this.onnx")
    return path if os.path.isfile(path) else None


def load_stems(folder: Path) -> tuple[dict[str, np.ndarray], int]:
    stems = {}
    rate = 0
    for name in STEMS:
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
    whole = detect_bpm_and_key(stems, rate, model_path=model)
    start = song["start"]
    if start is None:
        start = pick_start(stems, rate, whole.downbeat_times, CLIP_SECONDS)
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
    entry.update({
        "clip_start": round(start, 3),
        "duration": round(duration, 3),
        "bpm": round(found.bpm or whole.bpm, 1),
        "key": found.key,
        "beats": [round(t, 3) for t in found.beat_times],
        "downbeats": [round(t, 3) for t in found.downbeat_times],
        "chords": [[round(t, 3), c] for t, c in found.chord_sequence],
        "stems": list(STEMS),
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
