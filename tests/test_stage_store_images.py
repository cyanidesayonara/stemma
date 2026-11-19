"""Staging the Store listing's screenshots and art in a Partner Center draft."""

import importlib.util
import json
import struct
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "stage_store_images", ROOT / "scripts" / "stage_store_images.py",
)
stage_store_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_store_images)


def _draft():
    """A draft shaped like msstore's submission get (trimmed)."""
    return {
        "Pricing": {"PriceId": "Free"},
        "ApplicationPackages": [
            {"FileName": "stemma.msix", "FileStatus": "PendingDelete",
             "Id": "2000000000096847452", "Version": "2.6.0.0"},
            {"FileName": "stemma.msix", "FileStatus": "PendingUpload"},
        ],
        "Listings": {"en-us": {"BaseListing": {
            "Description": "unchanged",
            "Images": [
                {"FileName": "Screen1.png", "FileStatus": "Uploaded",
                 "Id": "1", "ImageType": "Screenshot"},
                {"FileName": "box_1080x1080.png", "FileStatus": "Uploaded",
                 "Id": "2", "ImageType": "StoreLogoSquare"},
            ],
        }}},
    }


def test_listing_images_are_the_captioned_screenshots_then_the_art():
    images = stage_store_images.listing_images()
    shots = [name for name, _path, kind in images if kind == "Screenshot"]
    captions = (ROOT / "assets" / "store_listing" / "screenshots"
                / "captions.txt").read_text(encoding="utf-8")
    expected = [line.split(":")[0] for line in captions.splitlines() if line]
    assert shots == [f"listing/{name}" for name in expected]
    kinds = {name.split("/")[-1]: kind for name, _p, kind in images}
    assert kinds["tile_300x300.png"] == "Icon"
    assert kinds["box_1080x1080.png"] == "StoreLogoSquare"
    assert kinds["poster_720x1080.png"] == "StoreLogo9x16"


def test_stage_retires_the_stores_images_and_adds_the_new_ones():
    images = stage_store_images.listing_images()
    draft = stage_store_images.stage(_draft(), images)
    listing = draft["Listings"]["en-us"]["BaseListing"]
    statuses = [(i["FileName"], i["FileStatus"]) for i in listing["Images"]]
    assert statuses[:2] == [("Screen1.png", "PendingDelete"),
                            ("box_1080x1080.png", "PendingDelete")]
    staged = [i for i in listing["Images"] if i["FileStatus"] == "PendingUpload"]
    assert len(staged) == len(images)
    assert all("Id" not in i for i in staged)
    assert listing["Description"] == "unchanged"


def test_a_rerun_replaces_its_own_staged_images_instead_of_adding_more():
    images = stage_store_images.listing_images()
    once = stage_store_images.stage(_draft(), images)
    # the read-back gives staged images "Id": null, as Partner Center does
    reread = json.loads(json.dumps(once))
    for image in reread["Listings"]["en-us"]["BaseListing"]["Images"]:
        image.setdefault("Id", None)
    twice = stage_store_images.stage(reread, images)
    listing = twice["Listings"]["en-us"]["BaseListing"]["Images"]
    assert len(listing) == 2 + len(images)
    assert sum(i["FileStatus"] == "PendingDelete" for i in listing) == 2


def test_the_zip_holds_the_package_under_the_name_the_draft_expects(tmp_path):
    msix = tmp_path / "built.msix"
    msix.write_bytes(b"MSIX")
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(_draft()), encoding="utf-8")
    upload = tmp_path / "upload.zip"
    assert stage_store_images.main([
        "--submission", str(submission), "--msix", str(msix),
        "--zip", str(upload),
    ]) == 0
    with zipfile.ZipFile(upload) as archive:
        names = archive.namelist()
        assert archive.read("stemma.msix") == b"MSIX"
    staged = json.loads(submission.read_text(encoding="utf-8"))
    for image in staged["Listings"]["en-us"]["BaseListing"]["Images"]:
        if image["FileStatus"] == "PendingUpload":
            assert image["FileName"] in names


def test_a_draft_without_a_pending_package_is_refused():
    draft = _draft()
    draft["ApplicationPackages"] = draft["ApplicationPackages"][:1]
    with pytest.raises(SystemExit, match="no package pending upload"):
        stage_store_images.pending_package(draft)


def test_a_screenshot_below_the_store_minimum_is_refused(tmp_path):
    shots = tmp_path / "screenshots"
    shots.mkdir()
    (shots / "captions.txt").write_text("small.png: too small\n",
                                        encoding="utf-8")
    (shots / "small.png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
        + struct.pack(">II", 800, 600) + b"\x08\x02\x00\x00\x00"
    )
    with pytest.raises(SystemExit, match="800x600"):
        stage_store_images.listing_images(shots, tmp_path)


def test_images_of_other_types_are_left_alone():
    """A hero image added by hand in Partner Center survives a restage."""
    draft = _draft()
    hero = {"FileName": "hero.png", "FileStatus": "Uploaded", "Id": "9",
            "ImageType": "PromotionalArt16x9"}
    draft["Listings"]["en-us"]["BaseListing"]["Images"].append(hero)
    staged = stage_store_images.stage(draft, stage_store_images.listing_images())
    assert hero in staged["Listings"]["en-us"]["BaseListing"]["Images"]


def test_store_art_of_the_wrong_size_is_refused(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    for name in stage_store_images.ART:
        (art / name).write_bytes(
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">II", 64, 64) + b"\x08\x02\x00\x00\x00"
        )
    with pytest.raises(SystemExit, match="64x64; the Store needs"):
        stage_store_images.listing_images(art=art)
