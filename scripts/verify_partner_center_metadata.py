"""Verify Partner Center submission listing fields match store/listing.yaml."""

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
    parse_msstore_submission_json,
    tag_to_version,
    verify_submission_listing_metadata,
    verify_submission_listing_metadata_applied,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        help="Merged submission JSON sent to updateMetadata",
    )
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v2.6.0")
    parser.add_argument(
        "--listing",
        type=Path,
        default=DEFAULT_LISTING_YAML,
    )
    parser.add_argument(
        "--fields",
        default="ReleaseNotes",
        help="Comma-separated BaseListing fields to verify (default: ReleaseNotes)",
    )
    args = parser.parse_args(argv)

    version = tag_to_version(args.tag)
    submission = parse_msstore_submission_json(
        args.submission.read_text(encoding="utf-8"),
    )
    if args.expected:
        expected_submission = json.loads(args.expected.read_text(encoding="utf-8"))
        fields = tuple(
            part.strip()
            for part in args.fields.split(",")
            if part.strip()
        )
        verify_submission_listing_metadata_applied(
            submission,
            expected_submission,
            fields=fields,
        )
        print(
            "Verified Partner Center listing metadata matches the updateMetadata payload "
            f"for {', '.join(fields)}.",
        )
    else:
        data = load_listing(args.listing)
        verify_submission_listing_metadata(
            submission,
            data,
            release_version=version,
        )
        print(
            f"Verified Partner Center listing metadata for version {version} "
            f"against {args.listing.name}.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
