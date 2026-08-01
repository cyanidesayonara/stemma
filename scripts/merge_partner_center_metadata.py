"""Merge store/listing.yaml fields into an msstore submission get JSON payload."""

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
    merge_submission_listing_metadata,
    parse_msstore_submission_json,
    tag_to_version,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v2.6.0")
    parser.add_argument("--listing", type=Path, default=DEFAULT_LISTING_YAML)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    version = tag_to_version(args.tag)
    submission = parse_msstore_submission_json(
        args.submission.read_text(encoding="utf-8"),
    )
    data = load_listing(args.listing)
    merged = merge_submission_listing_metadata(
        submission,
        data,
        release_version=version,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(merged, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
