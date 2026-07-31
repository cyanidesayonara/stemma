"""Tests for Store listing-as-code helpers."""

from pathlib import Path

from src.store_listing import (
    load_listing,
    png_size,
    read_app_version,
)


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
    # Minimal 8x4 RGBA PNG
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 8, 4, 8, 2, 0, 0, 0)
    raw = b"".join(
        b"\x00" + bytes([0, 0, 255, 255] * 8) for _ in range(4)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path = tmp_path / "tiny.png"
    path.write_bytes(png)

    assert png_size(path) == (8, 4)


def test_read_app_version(tmp_path: Path) -> None:
    path = tmp_path / "version.py"
    path.write_text('__version__ = "2.6.0"\n', encoding="utf-8")
    assert read_app_version(path) == "2.6.0"
