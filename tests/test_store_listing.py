"""Tests for Store listing-as-code helpers."""

import struct
import zlib
from pathlib import Path

import pytest

from src.store_listing import (
    ListingData,
    ValidationError,
    load_listing,
    outputs_match,
    png_size,
    read_app_version,
    render_markdown,
    render_skeleton,
    validate_listing,
    write_outputs,
)


def _minimal_png(width: int, height: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(
        b"\x00" + bytes([0, 0, 255, 255] * width) for _ in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _data(**overrides) -> ListingData:
    base = dict(
        listing_version="2.6.0",
        short_description="Short",
        description="Desc",
        features=["Feature one"],
        search_terms=["stem"],
        whats_new={"2.6.0": "What's new in version 2.6.0\n\nNotes."},
    )
    base.update(overrides)
    return ListingData(**base)


def test_load_listing_reads_required_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "listing.yaml"
    yaml_path.write_text(
        """
listing_version: "2.6.0"
short_description: Short text
description: |
  Longer description.
features:
  - Feature one
search_terms:
  - stem separation
whats_new:
  "2.5.0": |
    What's new in version 2.5.0
  "2.6.0": |
    What's new in version 2.6.0
""".lstrip(),
        encoding="utf-8",
    )

    data = load_listing(yaml_path)

    assert data.listing_version == "2.6.0"
    assert data.short_description == "Short text"
    assert "Longer description." in data.description
    assert data.features == ["Feature one"]
    assert data.search_terms == ["stem separation"]
    assert "2.6.0" in data.whats_new


def test_png_size_reads_ihdr(tmp_path: Path) -> None:
    path = tmp_path / "tiny.png"
    path.write_bytes(_minimal_png(8, 4))

    assert png_size(path) == (8, 4)


def test_read_app_version(tmp_path: Path) -> None:
    path = tmp_path / "version.py"
    path.write_text('__version__ = "2.6.0"\n', encoding="utf-8")
    assert read_app_version(path) == "2.6.0"


def test_validate_listing_requires_whats_new(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    with pytest.raises(ValidationError) as exc:
        validate_listing(
            _data(whats_new={"2.5.0": "old"}),
            version="2.6.0",
            screenshots_dir=shots,
        )
    assert any("whats_new" in err for err in exc.value.errors)


def test_validate_listing_rejects_too_many_features(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    with pytest.raises(ValidationError):
        validate_listing(
            _data(features=[f"f{i}" for i in range(21)]),
            version="2.6.0",
            screenshots_dir=shots,
        )


def test_validate_listing_rejects_undersized_screenshots(
    tmp_path: Path,
) -> None:
    shots = tmp_path / "shots"
    shots.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (shots / name).write_bytes(_minimal_png(8, 4))
    with pytest.raises(ValidationError) as exc:
        validate_listing(
            _data(),
            version="2.6.0",
            screenshots_dir=shots,
        )
    assert any(
        "1366" in err or "screenshot" in err.lower()
        for err in exc.value.errors
    )


def test_render_markdown_marks_generated_and_includes_sections() -> None:
    text = render_markdown(_data(), release_version="2.6.0")
    assert "generated from" in text.lower()
    assert "store/listing.yaml" in text
    assert "## Short description" in text
    assert "Short" in text
    assert "## Product features" in text
    assert "Feature one" in text
    assert "What's new in version 2.6.0" in text


def test_render_skeleton_includes_package_url_placeholder() -> None:
    payload = render_skeleton(_data(), release_version="2.6.0")
    assert "packages" in payload
    assert "v2.6.0/stemma.msix" in payload["packages"][0]["packageUrl"]
    assert payload["listing"]["shortDescription"] == "Short"


def test_build_check_detects_stale_markdown(tmp_path: Path) -> None:
    yaml_path = tmp_path / "listing.yaml"
    yaml_path.write_text(
        """
listing_version: "2.6.0"
short_description: Short text
description: |
  Longer description.
features:
  - Feature one
search_terms:
  - stem separation
whats_new:
  "2.6.0": |
    What's new in version 2.6.0

    Notes.
""".lstrip(),
        encoding="utf-8",
    )
    data = load_listing(yaml_path)
    markdown_path = tmp_path / "store-listing.md"
    skeleton_path = tmp_path / "product-update.skeleton.json"
    write_outputs(
        data,
        release_version="2.6.0",
        markdown_path=markdown_path,
        skeleton_path=skeleton_path,
    )
    assert outputs_match(
        data,
        release_version="2.6.0",
        markdown_path=markdown_path,
        skeleton_path=skeleton_path,
    )
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8") + "\nstale\n",
        encoding="utf-8",
    )
    assert not outputs_match(
        data,
        release_version="2.6.0",
        markdown_path=markdown_path,
        skeleton_path=skeleton_path,
    )
