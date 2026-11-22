"""Live YouTube download smoke check for scheduled CI and local testing."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_io import audio_duration  # noqa: E402
from src.diagnostics import collect_youtube_js_check  # noqa: E402
from src.downloader import download_audio, extract_metadata  # noqa: E402

DEFAULT_SMOKE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
_FORBIDDEN_WARNINGS = (
    "without a js runtime",
    "no supported javascript runtime",
)


class _WarningRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def run_smoke(url: str) -> int:
    """Exercise metadata + MP3 extraction against a real YouTube URL."""
    js_status = collect_youtube_js_check()
    print(f"YouTube JS runtime: {js_status}")
    if js_status.startswith("unavailable"):
        print(f"YouTube JS check failed: {js_status}", file=sys.stderr)
        return 1

    stemma_logger = logging.getLogger("stemma")
    recorder = _WarningRecorder()
    stemma_logger.addHandler(recorder)
    try:
        title, artist = extract_metadata(url)
        print(f"Metadata: title={title!r}, artist={artist!r}")
        if not title.strip():
            print("Extracted title is empty", file=sys.stderr)
            return 1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "smoke.mp3")
            downloaded = download_audio(url, output_path)
            if not os.path.isfile(downloaded):
                print(f"Downloaded file missing: {downloaded}", file=sys.stderr)
                return 1
            size = os.path.getsize(downloaded)
            if size <= 0:
                print("Downloaded MP3 is empty", file=sys.stderr)
                return 1
            duration_s = audio_duration(downloaded)
            print(f"Downloaded {size} bytes, duration={duration_s:.2f}s")
            if duration_s <= 1.0:
                print(
                    f"Downloaded MP3 duration too short: {duration_s:.2f}s",
                    file=sys.stderr,
                )
                return 1
    finally:
        stemma_logger.removeHandler(recorder)

    for warning in recorder.messages:
        print(f"Warning logged: {warning}")
        lowered = warning.lower()
        for forbidden in _FORBIDDEN_WARNINGS:
            if forbidden in lowered:
                print(
                    f"Forbidden JS runtime warning logged: {warning}",
                    file=sys.stderr,
                )
                return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the YouTube download smoke check."""
    parser = argparse.ArgumentParser(
        description="Smoke test real YouTube metadata + audio download.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("STEMMA_YOUTUBE_SMOKE_URL") or DEFAULT_SMOKE_URL,
        help="YouTube URL to smoke test",
    )
    args = parser.parse_args(argv)
    return run_smoke(args.url)


if __name__ == "__main__":
    raise SystemExit(main())
