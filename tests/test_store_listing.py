"""Tests for Store listing-as-code helpers."""

import struct
import zlib
from pathlib import Path

import pytest

from src.store_listing import (
    DEFAULT_LISTING_YAML,
    DEFAULT_MARKDOWN,
    DEFAULT_SCREENSHOTS,
    DEFAULT_SKELETON,
    DEFAULT_VERSION_PY,
    ListingData,
    StoreIdentity,
    ValidationError,
    load_listing,
    outputs_match,
    png_size,
    read_app_version,
    render_markdown,
    render_metadata_update,
    merge_submission_listing_metadata,
    parse_msstore_submission_json,
    verify_submission_listing_metadata,
    render_product_update,
    render_skeleton,
    tag_to_version,
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
store:
  product_id: "9P2W12L8F381"
  package_family_name: "SanttuNyknen.stemma_rt9h3xsn8gsh8"
  url: "https://apps.microsoft.com/detail/9p2w12l8f381"
  publisher_display_name: "Santtu Nykänen"
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
    assert data.store == StoreIdentity(
        product_id="9P2W12L8F381",
        package_family_name="SanttuNyknen.stemma_rt9h3xsn8gsh8",
        url="https://apps.microsoft.com/detail/9p2w12l8f381",
        publisher_display_name="Santtu Nykänen",
    )


def test_load_listing_rejects_null_store_fields(tmp_path: Path) -> None:
    yaml_path = tmp_path / "listing.yaml"
    yaml_path.write_text(
        """
listing_version: "2.6.0"
store:
  product_id: "9P2W12L8F381"
  package_family_name: "SanttuNyknen.stemma_rt9h3xsn8gsh8"
  url: null
  publisher_display_name: "Santtu Nykänen"
short_description: Short text
description: Desc
features:
  - Feature one
search_terms:
  - stem
whats_new:
  "2.6.0": |
    What's new in version 2.6.0
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="url"):
        load_listing(yaml_path)


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
    assert "productId" not in payload


def test_render_skeleton_includes_store_identity() -> None:
    identity = StoreIdentity(
        product_id="9P2W12L8F381",
        package_family_name="SanttuNyknen.stemma_rt9h3xsn8gsh8",
        url="https://apps.microsoft.com/detail/9p2w12l8f381",
        publisher_display_name="Santtu Nykänen",
    )
    payload = render_skeleton(_data(store=identity), release_version="2.6.0")
    assert payload["productId"] == "9P2W12L8F381"
    assert payload["packageFamilyName"] == identity.package_family_name
    assert payload["storeUrl"] == identity.url
    assert payload["publisherDisplayName"] == identity.publisher_display_name


def test_tag_to_version_strips_prefix() -> None:
    assert tag_to_version("v2.6.0") == "2.6.0"
    assert tag_to_version("refs/tags/v2.6.0") == "2.6.0"


def test_render_product_update_uses_release_msix_url() -> None:
    payload = render_product_update(
        _data(),
        release_version="2.6.0",
        repository="cyanidesayonara/stemma",
    )
    assert payload["packages"][0]["architectures"] == ["X64"]
    assert (
        payload["packages"][0]["packageUrl"]
        == "https://github.com/cyanidesayonara/stemma/releases/download/v2.6.0/stemma.msix"
    )


def test_render_metadata_update_maps_listing_fields() -> None:
    payload = render_metadata_update(_data(), release_version="2.6.0")
    assert payload["language"] == "en-us"
    listing = payload["BaseListing"]
    assert listing["ShortDescription"] == "Short"
    assert listing["Description"] == "Desc"
    assert listing["Features"] == ["Feature one"]
    assert listing["Keywords"] == ["stem"]
    assert "What's new in version 2.6.0" in listing["ReleaseNotes"]


def test_parse_msstore_submission_json_repairs_multiline_strings() -> None:
    raw = """\
Retrieving Submission
{
  "Id": "1",
  "Listings": {
    "en-us": {
      "BaseListing": {
        "Description": "Line one
Line two"
      }
    }
  }
}
"""
    parsed = parse_msstore_submission_json(raw)
    assert parsed["Id"] == "1"
    assert parsed["Listings"]["en-us"]["BaseListing"]["Description"] == (
        "Line one\nLine two"
    )


def test_merge_submission_listing_metadata_updates_base_listing() -> None:
    submission = {
        "Id": "1152921505701556700",
        "Listings": {
            "en-us": {
                "BaseListing": {
                    "Description": "Old",
                    "ReleaseNotes": "Old notes",
                    "Features": ["old"],
                    "Keywords": ["old"],
                }
            }
        },
    }
    merged = merge_submission_listing_metadata(
        submission,
        _data(),
        release_version="2.6.0",
    )
    base = merged["Listings"]["en-us"]["BaseListing"]
    assert base["Description"] == "Desc"
    assert base["ShortDescription"] == "Short"
    assert base["Features"] == ["Feature one"]
    assert "What's new in version 2.6.0" in base["ReleaseNotes"]
    assert submission["Listings"]["en-us"]["BaseListing"]["Description"] == "Old"


def test_verify_submission_listing_metadata_passes_when_fields_match() -> None:
    data = _data()
    submission = merge_submission_listing_metadata(
        {
            "Listings": {
                "en-us": {
                    "BaseListing": {
                        "Description": "Old",
                        "ReleaseNotes": "Old notes",
                    }
                }
            }
        },
        data,
        release_version="2.6.0",
    )
    verify_submission_listing_metadata(
        submission,
        data,
        release_version="2.6.0",
    )


def test_verify_submission_listing_metadata_raises_on_mismatch() -> None:
    submission = {
        "Listings": {
            "en-us": {
                "BaseListing": {
                    "Description": "Wrong",
                    "ShortDescription": "Short",
                    "ReleaseNotes": "Old notes",
                    "Features": ["Feature one"],
                    "Keywords": ["stem separation"],
                }
            }
        }
    }
    with pytest.raises(ValueError, match="Description"):
        verify_submission_listing_metadata(
            submission,
            _data(),
            release_version="2.6.0",
        )


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


def test_committed_store_listing_is_valid_and_fresh() -> None:
    """Smoke test real repo paths; catches listing, screenshot, and drift regressions."""
    data = load_listing(DEFAULT_LISTING_YAML)
    version = read_app_version(DEFAULT_VERSION_PY)
    validate_listing(data, version=version, screenshots_dir=DEFAULT_SCREENSHOTS)
    assert outputs_match(
        data,
        release_version=version,
        markdown_path=DEFAULT_MARKDOWN,
        skeleton_path=DEFAULT_SKELETON,
    )
