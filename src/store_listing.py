"""Store listing load/validate/render helpers (issue #134 slice 1)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct

import yaml

MAX_FEATURES = 20
MAX_FEATURE_LENGTH = 200
MAX_SHORT_DESCRIPTION_LENGTH = 1000
MIN_SCREENSHOTS = 3
MIN_SCREENSHOT_WIDTH = 1366
MIN_SCREENSHOT_HEIGHT = 768

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LISTING_YAML = REPO_ROOT / "store" / "listing.yaml"
DEFAULT_MARKDOWN = REPO_ROOT / "docs" / "store-listing.md"
DEFAULT_SKELETON = REPO_ROOT / "store" / "product-update.skeleton.json"
DEFAULT_SCREENSHOTS = REPO_ROOT / "assets" / "store_listing" / "screenshots"
DEFAULT_VERSION_PY = REPO_ROOT / "src" / "version.py"


@dataclass(frozen=True)
class StoreIdentity:
    """Microsoft Store product identity for Partner Center / public listing."""

    product_id: str
    package_family_name: str
    url: str
    publisher_display_name: str


@dataclass(frozen=True)
class ListingData:
    listing_version: str
    short_description: str
    description: str
    features: list[str]
    search_terms: list[str]
    whats_new: dict[str, str]
    store: StoreIdentity | None = None


def _load_store_identity(raw: dict) -> StoreIdentity | None:
    block = raw.get("store")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("store must be a mapping")
    required = (
        "product_id",
        "package_family_name",
        "url",
        "publisher_display_name",
    )
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in required:
        value = block.get(key)
        if value is None or not isinstance(value, str) or not value.strip():
            missing.append(key)
        else:
            values[key] = value.strip()
    if missing:
        raise ValueError(f"store missing fields: {', '.join(missing)}")
    return StoreIdentity(
        product_id=values["product_id"],
        package_family_name=values["package_family_name"],
        url=values["url"],
        publisher_display_name=values["publisher_display_name"],
    )


def load_listing(path: str | Path) -> ListingData:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("listing YAML must be a mapping")
    whats_new = raw.get("whats_new") or {}
    if not isinstance(whats_new, dict):
        raise ValueError("whats_new must be a mapping")
    return ListingData(
        listing_version=str(raw["listing_version"]).strip(),
        short_description=str(raw["short_description"]).strip(),
        description=str(raw["description"]).strip(),
        features=[str(item).strip() for item in raw.get("features") or []],
        search_terms=[
            str(item).strip() for item in raw.get("search_terms") or []
        ],
        whats_new={
            str(key): str(value).strip() for key, value in whats_new.items()
        },
        store=_load_store_identity(raw),
    )


def png_size(path: str | Path) -> tuple[int, int]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    length = struct.unpack(">I", data[8:12])[0]
    tag = data[12:16]
    if tag != b"IHDR" or length < 8:
        raise ValueError(f"PNG missing IHDR: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def read_app_version(version_py: str | Path) -> str:
    text = Path(version_py).read_text(encoding="utf-8")
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"could not parse __version__ from {version_py}")
    return match.group(1)


class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def validate_listing(
    data: ListingData,
    *,
    version: str,
    screenshots_dir: str | Path,
) -> None:
    errors: list[str] = []
    notes = data.whats_new.get(version, "").strip()
    if not notes:
        errors.append(f"whats_new missing entry for version {version}")
    if not (1 <= len(data.features) <= MAX_FEATURES):
        errors.append(
            f"features count {len(data.features)} not in 1..{MAX_FEATURES}"
        )
    for feature in data.features:
        if not feature:
            errors.append("empty feature entry")
        elif len(feature) > MAX_FEATURE_LENGTH:
            errors.append(
                f"feature exceeds {MAX_FEATURE_LENGTH} chars: {feature!r}"
            )
    if not data.short_description:
        errors.append("short_description is empty")
    elif len(data.short_description) > MAX_SHORT_DESCRIPTION_LENGTH:
        errors.append("short_description exceeds max length")
    if not data.description.strip():
        errors.append("description is empty")

    shot_dir = Path(screenshots_dir)
    shots = sorted(shot_dir.glob("*.png")) if shot_dir.is_dir() else []
    if len(shots) < MIN_SCREENSHOTS:
        errors.append(
            f"need >= {MIN_SCREENSHOTS} screenshots, found {len(shots)}"
        )
    for shot in shots:
        try:
            width, height = png_size(shot)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if width < MIN_SCREENSHOT_WIDTH or height < MIN_SCREENSHOT_HEIGHT:
            errors.append(
                f"{shot.name} is {width}x{height}; "
                f"need >= {MIN_SCREENSHOT_WIDTH}x{MIN_SCREENSHOT_HEIGHT}"
            )
    if errors:
        raise ValidationError(errors)


def render_markdown(data: ListingData, *, release_version: str) -> str:
    """Render Partner Center paste-ready markdown from listing data."""
    whats_new = data.whats_new[release_version].strip()
    features = "\n".join(data.features)
    # Comma-separated for Partner Center paste; wrap roughly like the
    # historical hand-edited listing.
    search_terms = ", ".join(data.search_terms)
    if data.store is not None:
        store_line = (
            f"\nPublic Store listing: {data.store.url} "
            f"(product id `{data.store.product_id}`).\n"
        )
    else:
        store_line = "\n"
    return f"""# Microsoft Store listing copy

Generated from `store/listing.yaml`. Edit the YAML, then run
`python scripts/build_store_listing.py` to regenerate this file.
Do not edit this markdown by hand.

This copy reflects listing content for version **{release_version}**.
{store_line}
Fields map to Partner Center as follows:

| Partner Center field | Section below |
|---|---|
| Description | [Description](#description) |
| What's new in this version | [What's new](#whats-new-in-this-version) |
| Product features (max 20, one per line) | [Product features](#product-features) |
| Short description | [Short description](#short-description) |
| Search terms | [Search terms](#search-terms) |

Assets: `assets/store_listing/` (regenerate with
`python scripts/generate_brand.py` then
`python scripts/generate_store_listing_assets.py`).
Screenshots: `assets/store_listing/screenshots/` (regenerate with
`python scripts/generate_screenshots.py`).

---

## Short description

{data.short_description}

---

## Description

{data.description}

---

## What's new in this version

{whats_new}

---

## Product features

{features}

---

## Search terms

{search_terms}

---

## Notes for future submissions

- Update `What's new` for every Store submission; keep the version
  number in the first line (Partner Center shows it verbatim).
- The Description avoids naming specific model versions (HTDemucs,
  MDX-Net): those change, and the Store copy should not need a rewrite
  when they do. Attribution for the models lives in the README.
- Do not claim GPU acceleration for 4/6-stem separation: only the
  2-stem path runs on the GPU today (see issue #125).
"""


def render_skeleton(data: ListingData, *, release_version: str) -> dict:
    """Render Partner Center product-update skeleton JSON payload."""
    listing: dict = {
        "shortDescription": data.short_description,
        "description": data.description,
        "features": data.features,
        "searchTerms": data.search_terms,
    }
    if release_version in data.whats_new:
        listing["whatsNew"] = data.whats_new[release_version]
    payload: dict = {
        "packages": [
            {
                "packageUrl": (
                    "https://github.com/cyanidesayonara/stemma/releases/"
                    f"download/v{release_version}/stemma.msix"
                ),
                "languages": ["en-us"],
                "architectures": ["x64"],
                "installerParameters": "",
                "isSilentInstall": True,
            }
        ],
        "listing": listing,
    }
    if data.store is not None:
        payload["productId"] = data.store.product_id
        payload["packageFamilyName"] = data.store.package_family_name
        payload["storeUrl"] = data.store.url
        payload["publisherDisplayName"] = data.store.publisher_display_name
    return payload


def write_outputs(
    data: ListingData,
    *,
    release_version: str,
    markdown_path: str | Path,
    skeleton_path: str | Path,
) -> None:
    """Write generated markdown and skeleton JSON to disk."""
    markdown = render_markdown(data, release_version=release_version)
    skeleton = render_skeleton(data, release_version=release_version)
    md_path = Path(markdown_path)
    sk_path = Path(skeleton_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    sk_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8", newline="\n")
    sk_path.write_text(
        json.dumps(skeleton, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def outputs_match(
    data: ListingData,
    *,
    release_version: str,
    markdown_path: str | Path,
    skeleton_path: str | Path,
) -> bool:
    """Return True when committed outputs match a fresh render."""
    md_path = Path(markdown_path)
    sk_path = Path(skeleton_path)
    if not md_path.is_file() or not sk_path.is_file():
        return False
    expected_md = render_markdown(data, release_version=release_version)
    expected_sk = render_skeleton(data, release_version=release_version)
    actual_md = md_path.read_text(encoding="utf-8")
    try:
        actual_sk = json.loads(sk_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return actual_md == expected_md and actual_sk == expected_sk
