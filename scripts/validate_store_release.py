"""Validate Store listing metadata for a release version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.store_listing import (
    DEFAULT_LISTING_YAML,
    DEFAULT_SCREENSHOTS,
    DEFAULT_VERSION_PY,
    ValidationError,
    load_listing,
    read_app_version,
    validate_listing,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="X.Y.Z release version")
    parser.add_argument("--listing", type=Path, default=DEFAULT_LISTING_YAML)
    parser.add_argument(
        "--screenshots", type=Path, default=DEFAULT_SCREENSHOTS,
    )
    args = parser.parse_args(argv)
    version = args.version or read_app_version(DEFAULT_VERSION_PY)
    try:
        data = load_listing(args.listing)
        validate_listing(
            data, version=version, screenshots_dir=args.screenshots,
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Store listing OK for version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
