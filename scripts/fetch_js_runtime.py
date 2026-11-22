"""Fetch and verify the QuickJS-ng Windows executable bundled by PyInstaller."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

QUICKJS_VERSION = "v0.18.0"
QUICKJS_URL = (
    "https://github.com/quickjs-ng/quickjs/releases/download/"
    "v0.18.0/qjs-windows-x86_64.exe"
)
QUICKJS_SHA256 = (
    "ff03398d74ea6e6bd3a8e58058f7003225abcac229d7d57775b7aaae812a49c7"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "build" / "js-runtime" / "qjs.exe"
)


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def fetch_js_runtime(
    output: Path = DEFAULT_OUTPUT,
    *,
    url: str = QUICKJS_URL,
    expected_sha256: str = QUICKJS_SHA256,
) -> Path:
    """Ensure *output* exists and matches *expected_sha256*."""
    expected = expected_sha256.lower()
    if output.is_file() and sha256_file(output) == expected:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_name(f"{output.name}.{os.getpid()}.part")
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, part_path.open(
            "wb"
        ) as out_file:
            while chunk := response.read(65536):
                digest.update(chunk)
                out_file.write(chunk)
        actual = digest.hexdigest().lower()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected}, got {actual}"
            )
        os.replace(part_path, output)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for fetching QuickJS-ng."""
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify QuickJS-ng for Windows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for qjs.exe",
    )
    args = parser.parse_args(argv)
    try:
        path = fetch_js_runtime(
            args.output,
            url=QUICKJS_URL,
            expected_sha256=QUICKJS_SHA256,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Failed to fetch QuickJS-ng: {exc}", file=sys.stderr)
        return 1
    print(f"QuickJS-ng {QUICKJS_VERSION} ready at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
