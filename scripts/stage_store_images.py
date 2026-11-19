"""Stage the Store listing's screenshots and art in a Partner Center draft.

``update_draft`` stages the package and the listing text with msstore, but
msstore has no command for listing images. The submission API takes them
the same way it takes the package: each image is listed in the
submission's ``Listings.<lang>.BaseListing.Images`` with ``FileStatus``
``PendingUpload``, the files travel in the one zip uploaded to the
submission's ``FileUploadUrl``, and committing the draft ingests both.
This script does the two local halves of that:

1. rewrites the images in a submission JSON: every image already in the
   Store is marked ``PendingDelete``, and the repository's images are
   added as ``PendingUpload`` (images a previous run staged are replaced,
   so a rerun never duplicates them);
2. writes the upload zip: the release MSIX (under the file name the draft
   expects) plus the images.

The workflow then PUTs the zip to ``FileUploadUrl`` and sends the JSON with
``msstore submission updateMetadata``. msstore's image model has no
description field, so screenshot captions are not sent.

    python scripts/stage_store_images.py --submission submission-metadata.json \\
        --msix stemma.msix --zip upload.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import png_size  # noqa: E402

LISTING_ART = ROOT / "assets" / "store_listing"
SCREENSHOTS = LISTING_ART / "screenshots"
# the listing art and the Partner Center image type each one fills, as the
# current Store listing uses them
ART = {
    "tile_300x300.png": ("Icon", (300, 300)),
    "box_1080x1080.png": ("StoreLogoSquare", (1080, 1080)),
    "poster_720x1080.png": ("StoreLogo9x16", (720, 1080)),
}
# Partner Center's floor for desktop screenshots
MIN_SCREENSHOT = (1366, 768)
ZIP_FOLDER = "listing"


def listing_images(screenshots: Path = SCREENSHOTS,
                   art: Path = LISTING_ART) -> list[tuple[str, Path, str]]:
    """(name in the zip, local file, image type) for every listing image:
    the screenshots in captions.txt order, then the art."""
    names = [
        line.split(":", 1)[0].strip()
        for line in (screenshots / "captions.txt").read_text(
            encoding="utf-8").splitlines()
        if ":" in line
    ]
    if not names:
        raise SystemExit(f"{screenshots / 'captions.txt'} lists no screenshots")
    images = []
    for name in names:
        path = screenshots / name
        if not path.is_file():
            raise SystemExit(f"captions.txt names a missing file: {name}")
        width, height = png_size(path)
        if width < MIN_SCREENSHOT[0] or height < MIN_SCREENSHOT[1]:
            raise SystemExit(f"{name} is {width}x{height}; the Store needs "
                             f"at least {MIN_SCREENSHOT[0]}x{MIN_SCREENSHOT[1]}")
        images.append((f"{ZIP_FOLDER}/{name}", path, "Screenshot"))
    for name, (image_type, size) in ART.items():
        path = art / name
        if not path.is_file():
            raise SystemExit(f"missing Store art: {path}")
        if png_size(path) != size:
            width, height = png_size(path)
            raise SystemExit(f"{name} is {width}x{height}; the Store needs "
                             f"{size[0]}x{size[1]}")
        images.append((f"{ZIP_FOLDER}/{name}", path, image_type))
    return images


def replace_images(current: list[dict],
                   new: list[tuple[str, Path, str]]) -> list[dict]:
    """The listing's image list with the Store's images of the types being
    restaged retired, and the new ones pending upload. Images of any other
    type (a hero image added in Partner Center, say) are left alone."""
    restaged = {image_type for _name, _path, image_type in new}
    kept = []
    for image in current:
        if image.get("ImageType") not in restaged:
            kept.append(image)
            continue
        if not image.get("Id"):
            # staged by an earlier run and never ingested: replaced below
            continue
        kept.append({**image, "FileStatus": "PendingDelete"})
    staged = [
        {"FileName": name, "FileStatus": "PendingUpload",
         "ImageType": image_type}
        for name, _path, image_type in new
    ]
    return kept + staged


def pending_package(submission: dict) -> str:
    """The file name the draft expects for its package upload."""
    for package in submission.get("ApplicationPackages") or []:
        if package.get("FileStatus") == "PendingUpload":
            return package["FileName"]
    raise SystemExit("the draft has no package pending upload; run "
                     "msstore publish --noCommit first")


def stage(submission: dict, images: list[tuple[str, Path, str]]) -> dict:
    """Return *submission* with every listing's images replaced."""
    listings = submission.get("Listings") or {}
    if not listings:
        raise SystemExit("the submission has no listings")
    for listing in listings.values():
        base = listing.setdefault("BaseListing", {})
        base["Images"] = replace_images(base.get("Images") or [], images)
    return submission


def write_zip(path: Path, msix: Path, package_name: str,
              images: list[tuple[str, Path, str]]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        # the MSIX is already compressed; storing it saves a minute of CPU
        archive.write(msix, package_name, compress_type=zipfile.ZIP_STORED)
        for name, local, _type in images:
            archive.write(local, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submission", type=Path, required=True,
                        help="submission JSON, rewritten in place")
    parser.add_argument("--msix", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args(argv)
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    images = listing_images()
    package = pending_package(submission)
    stage(submission, images)
    args.submission.write_text(json.dumps(submission, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    write_zip(args.zip, args.msix, package, images)
    print(f"Staged {len(images)} listing images; wrote {args.zip} with "
          f"{package} and the images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
