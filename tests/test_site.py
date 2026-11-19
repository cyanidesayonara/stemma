"""The website: palette held to the app, templates filled from a release."""

import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


site_palette = _load("site_palette")
build_site = _load("build_site")

COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(")

# store/listing.yaml as v2.6.0 shipped it: no store identity block
OLD_LISTING = """\
listing_version: "2.6.0"

short_description: |
  Practice any song: split it into stems, mute the part you play, slow it
  down, and loop the hard bars.

description: |
  stemma turns any song into a practice tool.

features:
  - "Split any song into vocals, drums, bass, guitar, piano, and other"

search_terms:
  - stem splitter

whats_new:
  "2.6.0": |
    What's new in version 2.6.0

    Faster 2-stem separation -- with GPU acceleration when available.
"""


def _png(path: Path, width: int, height: int) -> None:
    """Write just enough of a PNG for png_size to read."""
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
        + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    )


def _release(tmp_path: Path, *, captions: bool = True) -> Path:
    """A fake released checkout: a listing and two screenshots."""
    source = tmp_path / "rel"
    (source / "store").mkdir(parents=True)
    shutil.copy2(ROOT / "store" / "listing.yaml", source / "store")
    shots = source / "assets" / "store_listing" / "screenshots"
    shots.mkdir(parents=True)
    _png(shots / "01_one.png", 1920, 1080)
    _png(shots / "02_two.png", 1366, 768)
    if captions:
        (shots / "captions.txt").write_text(
            "02_two.png: Second & last\n01_one.png: First\n", encoding="utf-8"
        )
    return source


def _version() -> str:
    return build_site.load_listing(
        ROOT / "store" / "listing.yaml"
    ).listing_version


def test_palette_matches_the_app_themes():
    current = (ROOT / "site" / "palette.css").read_text(encoding="utf-8")
    assert current == site_palette.render(), (
        "site/palette.css is stale; run python scripts/site_palette.py"
    )


@pytest.mark.parametrize("name", ["style.css", "demo.js", "site.js"])
def test_site_sources_use_palette_variables_only(name):
    text = (ROOT / "site" / name).read_text(encoding="utf-8")
    assert COLOUR.findall(text) == []


def test_build_fills_every_block_and_links_resolve(tmp_path):
    out = tmp_path / "site"
    build_site.build(ROOT, _version(), "2026-11-19", out)
    page = (out / "index.html").read_text(encoding="utf-8")
    for match in build_site.BLOCK.finditer(page):
        assert match.group(3).strip(), f"gen:{match.group(2)} is empty"
    assert "November 19, 2026" in page
    assert "What's new in version" not in page
    assert " -- " not in page
    for ref in re.findall(r'(?:src|href|srcset)="([^"#:]+)"', page):
        assert (out / ref).is_file(), f"index.html links a missing {ref}"
    assert (out / ".nojekyll").is_file()


def test_screenshots_follow_captions_order_with_real_sizes(tmp_path):
    out = tmp_path / "site"
    build_site.build(_release(tmp_path), _version(), "2026-11-19", out)
    page = (out / "index.html").read_text(encoding="utf-8")
    assert page.index("02_two.png") < page.index("01_one.png")
    assert 'alt="Second &amp; last" width="1366" height="768"' in page
    assert (out / "img" / "shots" / "01_one.png").is_file()


def test_release_without_captions_lists_every_screenshot(tmp_path):
    source = _release(tmp_path, captions=False)
    assert build_site.screenshots(source) == [
        ("01_one.png", ""), ("02_two.png", ""),
    ]


def test_builds_from_an_older_release_listing(tmp_path):
    source = _release(tmp_path, captions=False)
    (source / "store" / "listing.yaml").write_text(OLD_LISTING,
                                                   encoding="utf-8")
    out = tmp_path / "site"
    build_site.build(source, "2.6.0", "2026-07-31T19:48:21Z", out)
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "Practice any song: split it into stems" in page
    assert "Faster 2-stem separation – with GPU" in page
    assert "July 31, 2026" in page
    assert 'alt="stemma screenshot: one"' in page


def test_uncaptioned_alt_text_names_the_view_and_theme():
    assert build_site.screenshot_alt("02_player_light.png") == (
        "stemma screenshot: player, light theme"
    )
    assert build_site.screenshot_alt("01_empty_dark.png") == (
        "stemma screenshot: empty, dark theme"
    )


def test_unknown_version_fails(tmp_path):
    with pytest.raises(SystemExit, match="no whats_new for 0.0.1"):
        build_site.build(ROOT, "0.0.1", "2026-11-19", tmp_path / "site")


def test_unpaired_marker_fails():
    with pytest.raises(SystemExit, match="gen:pitch"):
        build_site.check_markers(
            "index.html", "<!-- gen:pitch --> no end", ["pitch"]
        )


def test_missing_marker_fails():
    with pytest.raises(SystemExit, match="no <!-- gen:shots --> marker"):
        build_site.check_markers("index.html", "", ["shots"])


# ---- the mixer demo's songs ------------------------------------------------

DEMO = ROOT / "site" / "demo"
NOTE_NAMES = {"C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"}
# licences that allow stems (derivatives) on a page promoting a Store app:
# no NoDerivatives, no NonCommercial
ALLOWED_LICENCES = {
    "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC BY 3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
}


def _songs():
    return json.loads((DEMO / "songs.json").read_text(encoding="utf-8"))[
        "songs"
    ]


def test_demo_songs_are_credited_and_freely_licensed():
    songs = _songs()
    assert len(songs) >= 2
    for song in songs:
        for field in ("id", "title", "artist", "genre", "source"):
            assert song[field], f"{song.get('id')}: no {field}"
        assert ALLOWED_LICENCES.get(song["licence"]) == song["licence_url"]
        assert song["source"].startswith("https://")


def test_demo_songs_have_every_stem_and_overview():
    for song in _songs():
        assert song["stems"] == [
            "vocals", "drums", "bass", "other", "guitar", "piano",
        ]
        for stem in song["stems"]:
            path = DEMO / song["id"] / f"{stem}.mp3"
            assert path.is_file(), f"missing {path}"
            assert path.stat().st_size < 1_500_000, f"{path} is too large"
            peaks = song["peaks"][stem]
            assert len(peaks) > 100
            assert all(0 <= p <= 99 for p in peaks)


def test_demo_song_timings_fit_the_clip():
    for song in _songs():
        duration = song["duration"]
        assert 20 <= duration <= 60
        for key in ("beats", "downbeats"):
            times = song[key]
            assert times == sorted(times)
            assert all(0 <= t <= duration for t in times)
        for start, chord in song["chords"]:
            assert 0 <= start <= duration
            # spelled as demo.js's NOTE_NAMES, which it transposes
            assert chord == "N" or chord.rstrip("m") in NOTE_NAMES, chord


def test_demo_copy_is_published_with_the_site(tmp_path):
    out = tmp_path / "site"
    build_site.build(ROOT, _version(), "2026-11-19", out)
    assert (out / "stretch-worklet.js").is_file()
    for song in _songs():
        assert (out / "demo" / song["id"] / "vocals.mp3").is_file()


# ---- the demo's audio engine, run in Node -----------------------------------

NODE = shutil.which("node")
HARNESS = ROOT / "tests" / "js" / "worklet_harness.js"
WORKLET = ROOT / "site" / "stretch-worklet.js"
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _scenario(name, *extra):
    done = subprocess.run(
        [NODE, str(HARNESS), str(WORKLET), name, *map(str, extra)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(done.stdout or "{}")


@needs_node
@pytest.mark.parametrize("speed", [0.75, 1.25])
def test_worklet_stretcher_matches_the_apps(tmp_path, speed):
    sys.path.insert(0, str(ROOT))
    from src.stretch import StreamingStretcher

    rate = 44100
    t = np.arange(rate * 3) / rate
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 331 * t)
    right = 0.25 * np.sin(2 * np.pi * 277 * t)
    signal = np.stack([left, right]).astype(np.float32)
    (tmp_path / "in.f32").write_bytes(signal.tobytes())
    _scenario("stretch", tmp_path / "in.f32", tmp_path / "out.f32", speed)
    js = np.frombuffer((tmp_path / "out.f32").read_bytes(), np.float32)
    js = js.reshape(2, -1).T

    stretcher = StreamingStretcher(rate, 2)
    stretcher.set_params(speed, 0)
    frames = signal.T
    position = [0]

    def source(count):
        chunk = frames[position[0]:position[0] + count]
        start = position[0]
        position[0] += len(chunk)
        return chunk, start

    py = stretcher.read(len(js) + 1, source)
    # the JS resampler starts one sample ahead (read = 1)
    assert np.max(np.abs(js[: len(js) - 1] - py[1:len(js)])) < 1e-4


@needs_node
def test_worklet_never_boosts_when_a_loop_moves():
    """A loop dragged later, or turned back on after a wrap, never plays
    louder than the mix (the seam fade once went negative: 1238x)."""
    mix_peak = 0.48
    for step, found in _scenario("loop-moves").items():
        assert not found["nan"], step
        assert found["peak"] < mix_peak * 1.1, (step, found["peak"])


@needs_node
def test_worklet_pause_survives_a_jump_during_its_fade():
    found = _scenario("pause-in-fade")
    assert found["playingAfterPause"] is False
    assert found["heard"] == pytest.approx(3.0, abs=0.01)


@needs_node
def test_worklet_path_follows_the_last_speed_in_a_fade():
    assert _scenario("switch-in-fade")["direct"] is True


@needs_node
def test_worklet_clicks_every_beat_of_every_loop_pass():
    for speed, found in _scenario("clicks").items():
        # a 1 s loop with two beats, heard for 4 s of song time
        passes = found["wraps"] + 1
        assert found["clicks"] == passes * 2, (speed, found)


@needs_node
def test_worklet_clamps_a_loop_past_the_end():
    found = _scenario("past-the-end")
    assert found["loopB"] == found["length"]
    assert not found["direct"]["nan"]
    assert not found["stretched"]["nan"]


@needs_node
def test_worklet_reports_carry_the_load_id():
    assert _scenario("gens")["gens"] == [1]
