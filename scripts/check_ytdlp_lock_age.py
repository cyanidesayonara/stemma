"""Warn when the release lock's yt-dlp falls behind the newest release.

The scheduled CI run calls this next to the YouTube smoke test (#260). It
compares the PyPI upload dates of the locked ``yt-dlp`` and of the newest
release. When the lock is more than ``--max-days`` behind it prints a
GitHub Actions warning, and with ``--open-issue`` it opens an issue, or
comments on the one already open, so the drift stays visible between runs.
See "Keeping YouTube import working" in docs/DEVELOPMENT.md.

    python scripts/check_ytdlp_lock_age.py [--max-days 60] [--open-issue]

Exits 0 whether or not the lock is stale; it exits 1 only when it cannot
check (PyPI unreachable, or the lock has no yt-dlp pin).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "requirements-release.txt"
PYPI_URL = "https://pypi.org/pypi/yt-dlp/json"
ISSUE_TITLE = "yt-dlp release lock is behind"
DEFAULT_MAX_DAYS = 60


@dataclass(frozen=True)
class LockAge:
    """The locked and newest yt-dlp versions and how far apart they are."""

    locked: str
    latest: str
    days_behind: int

    def is_stale(self, max_days: int) -> bool:
        return self.days_behind > max_days


def locked_version(lock_text: str) -> str:
    """Return the ``yt-dlp==`` version pinned in a pip-compile lock."""
    match = re.search(r"^yt-dlp==(\S+)", lock_text, re.MULTILINE)
    if match is None:
        raise ValueError("requirements-release.txt does not pin yt-dlp")
    return match.group(1)


def _upload_time(pypi: dict, version: str) -> datetime:
    files = pypi.get("releases", {}).get(version) or []
    times = [f["upload_time_iso_8601"] for f in files
             if f.get("upload_time_iso_8601")]
    if not times:
        raise ValueError(f"PyPI lists no files for yt-dlp {version}")
    return min(datetime.fromisoformat(t.replace("Z", "+00:00"))
               for t in times)


def lock_age(lock_text: str, pypi: dict) -> LockAge:
    """Compare the lock with PyPI's newest yt-dlp by upload date."""
    locked = locked_version(lock_text)
    latest = pypi["info"]["version"]
    if latest == locked:
        return LockAge(locked, latest, 0)
    delta = _upload_time(pypi, latest) - _upload_time(pypi, locked)
    return LockAge(locked, latest, max(0, delta.days))


def fetch_pypi(url: str = PYPI_URL) -> dict:
    """Fetch the yt-dlp project JSON from PyPI."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "stemma-ci-lock-age-check"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def issue_body(age: LockAge, max_days: int) -> str:
    return (
        f"The release lock pins yt-dlp {age.locked}. The newest release, "
        f"{age.latest}, came out {age.days_behind} days later (the limit is "
        f"{max_days}).\n\n"
        "Refresh the lock and ship a patch release: see step 1 of "
        '"Keeping YouTube import working" in docs/DEVELOPMENT.md. The '
        "weekly youtube-smoke-test canary step shows whether the newest "
        "yt-dlp works.\n\n"
        "Opened by scripts/check_ytdlp_lock_age.py (#260)."
    )


def report_issue(age: LockAge, max_days: int, run=subprocess.run) -> str:
    """Open the stale-lock issue, or comment on the open one.

    Returns "created" or "commented". *run* is subprocess.run, injectable
    for tests; it calls the ``gh`` CLI, which reads GH_TOKEN.
    """
    found = run(
        ["gh", "issue", "list", "--state", "open", "--search",
         f'"{ISSUE_TITLE}" in:title', "--json", "number,title"],
        capture_output=True, text=True, check=True,
    )
    numbers = [item["number"] for item in json.loads(found.stdout or "[]")
               if item.get("title", "").startswith(ISSUE_TITLE)]
    body = issue_body(age, max_days)
    if numbers:
        run(["gh", "issue", "comment", str(numbers[0]), "--body", body],
            check=True)
        return "commented"
    run(["gh", "issue", "create", "--title",
         f"{ISSUE_TITLE} ({age.locked} vs {age.latest})", "--body", body],
        check=True)
    return "created"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-days", type=int, default=DEFAULT_MAX_DAYS)
    parser.add_argument(
        "--open-issue", action="store_true",
        help="open or comment on an issue when the lock is stale (uses gh)",
    )
    args = parser.parse_args(argv)

    try:
        age = lock_age(LOCK_PATH.read_text(encoding="utf-8"), fetch_pypi())
    except (OSError, ValueError, KeyError) as exc:
        print(f"::warning title=yt-dlp lock age::could not check: {exc}")
        return 1

    print(f"yt-dlp locked {age.locked}, newest {age.latest}, "
          f"{age.days_behind} days behind")
    if not age.is_stale(args.max_days):
        return 0
    print(f"::warning title=yt-dlp lock age::The locked yt-dlp {age.locked} "
          f"is {age.days_behind} days behind {age.latest}. Refresh the lock.")
    if args.open_issue:
        print(f"Issue {report_issue(age, args.max_days)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
