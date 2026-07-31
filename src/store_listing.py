"""Store listing load/validate/render helpers (issue #134 slice 1)."""

from __future__ import annotations

from dataclasses import dataclass
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
class ListingData:
    listing_version: str
    short_description: str
    description: str
    features: list[str]
    search_terms: list[str]
    whats_new: dict[str, str]


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
