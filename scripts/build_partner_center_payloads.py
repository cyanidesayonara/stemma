"""Build Partner Center product-update and metadata-update JSON payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import (  # noqa: E402
    DEFAULT_LISTING_YAML,
    load_listing,
    render_metadata_update,
    render_product_update,
    tag_to_version,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v2.6.0")
    parser.add_argument(
        "--listing",
        type=Path,
        default=DEFAULT_LISTING_YAML,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "store" / "payloads",
    )
    parser.add_argument(
        "--repository",
        default="cyanidesayonara/stemma",
        help="GitHub owner/repo for the MSIX packageUrl",
    )
    args = parser.parse_args(argv)

    version = tag_to_version(args.tag)
    data = load_listing(args.listing)
    product = render_product_update(
        data,
        release_version=version,
        repository=args.repository,
    )
    metadata = render_metadata_update(data, release_version=version)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    product_path = out_dir / "product-update.json"
    metadata_path = out_dir / "metadata-update.json"
    product_path.write_text(
        json.dumps(product, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {product_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
