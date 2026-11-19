"""The website: palette held to the app, templates filled from a release."""

import importlib.util
import re
import shutil
import struct
from pathlib import Path

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

HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


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
    assert HEX.findall(text) == []


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
