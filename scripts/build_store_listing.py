"""Generate docs/store-listing.md and store/product-update.skeleton.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import (  # noqa: E402
    DEFAULT_LISTING_YAML,
    DEFAULT_MARKDOWN,
    DEFAULT_SKELETON,
    DEFAULT_VERSION_PY,
    load_listing,
    outputs_match,
    read_app_version,
    write_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--listing", type=Path, default=DEFAULT_LISTING_YAML)
    args = parser.parse_args(argv)
    version = args.version or read_app_version(DEFAULT_VERSION_PY)
    data = load_listing(args.listing)
    if version not in data.whats_new:
        print(f"whats_new missing {version}", file=sys.stderr)
        return 1
    if args.check:
        if outputs_match(
            data,
            release_version=version,
            markdown_path=DEFAULT_MARKDOWN,
            skeleton_path=DEFAULT_SKELETON,
        ):
            print("Store listing outputs are up to date")
            return 0
        print(
            "Store listing outputs are stale; run build_store_listing.py",
            file=sys.stderr,
        )
        return 1
    write_outputs(
        data,
        release_version=version,
        markdown_path=DEFAULT_MARKDOWN,
        skeleton_path=DEFAULT_SKELETON,
    )
    print(f"Wrote {DEFAULT_MARKDOWN} and {DEFAULT_SKELETON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
