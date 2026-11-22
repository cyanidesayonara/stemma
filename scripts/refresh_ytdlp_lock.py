"""Refresh the release lock's yt-dlp and the yt-dlp-ejs it expects.

Step 1 of "Keeping YouTube import working" in docs/DEVELOPMENT.md, run by
the "Refresh yt-dlp lock" workflow (#260) and usable by hand on Windows
with Python 3.12 and pip-tools:

    python scripts/refresh_ytdlp_lock.py [--yt-dlp 2026.11.20] [--dry-run]

It takes the newest yt-dlp from PyPI (or the given version) and reads the
exact yt-dlp-ejs version from that release's ``default`` extra, which
pip-compile cannot see. If that yt-dlp-ejs is outside the requirements.txt
range, the range moves to its minor version. Then it recompiles
requirements-release.txt with both pinned. QuickJS-ng is not touched:
check the yt-dlp-ejs release notes for that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# One reading of the lock and of yt-dlp's yt-dlp-ejs pin, shared with the
# canary, so the two never disagree on the version yt-dlp accepts.
from scripts.check_ytdlp_lock_age import ejs_pin, locked_version  # noqa: E402

REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements-release.txt"
LOCK_INPUT = "requirements-release.in"
PYPI_URL = "https://pypi.org/pypi/yt-dlp/{path}json"


@dataclass(frozen=True)
class Refresh:
    """What the refresh changes."""

    old_yt_dlp: str
    old_ejs: str
    yt_dlp: str
    ejs: str
    range_moved: bool

    @property
    def changed(self) -> bool:
        """True when the lock gets a different yt-dlp or yt-dlp-ejs."""
        return ((Version(self.old_yt_dlp), Version(self.old_ejs))
                != (Version(self.yt_dlp), Version(self.ejs)))

    @property
    def downgrade(self) -> bool:
        """True when the target yt-dlp is older than the locked one."""
        return Version(self.yt_dlp) < Version(self.old_yt_dlp)


def fetch_release(version: str | None) -> dict:
    """PyPI JSON for *version* of yt-dlp, or for the newest release."""
    path = f"{version}/" if version else ""
    request = urllib.request.Request(
        PYPI_URL.format(path=path),
        headers={"User-Agent": "stemma-ci-lock-refresh"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def expected_ejs(release: dict) -> str:
    """The exact yt-dlp-ejs version a yt-dlp release's default extra pins."""
    version = release["info"]["version"]
    try:
        specifier = SpecifierSet(
            ejs_pin(release["info"].get("requires_dist") or []),
        )
    except ValueError:
        specifier = SpecifierSet()
    pins = [spec.version for spec in specifier if spec.operator == "=="]
    if len(pins) != 1:
        raise ValueError(f"yt-dlp {version} does not pin one yt-dlp-ejs")
    return pins[0]


def parse_version(text: str | None) -> str | None:
    """Normalise a requested yt-dlp version, or None for the newest.

    Rejects anything that is not a version before it reaches the PyPI URL.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return str(Version(text))
    except InvalidVersion:
        raise ValueError(f"{text!r} is not a yt-dlp version") from None


def move_ejs_range(requirements_text: str, ejs: str) -> tuple[str, bool]:
    """Point the requirements.txt yt-dlp-ejs range at *ejs*'s minor.

    Returns the new text and whether it changed. The range stays on one
    minor version, as tests/test_dependency_locks.py requires.
    """
    pattern = re.compile(r"^yt-dlp-ejs[^\n#]*", re.MULTILINE)
    match = pattern.search(requirements_text)
    if match is None:
        raise ValueError("requirements.txt has no yt-dlp-ejs line")
    current = Requirement(match.group(0).strip())
    target = Version(ejs)
    if current.specifier.contains(target, prereleases=True):
        return requirements_text, False
    line = (f"yt-dlp-ejs>={target.major}.{target.minor}.0,"
            f"<{target.major}.{target.minor + 1}")
    return pattern.sub(line, requirements_text, count=1), True


def compile_command(yt_dlp: str, ejs: str) -> list[str]:
    """The pip-compile command of docs/DEVELOPMENT.md, both pins given."""
    return [
        sys.executable, "-m", "piptools", "compile", "--allow-unsafe",
        "--generate-hashes", "--index-url=https://pypi.org/simple",
        "--output-file=requirements-release.txt", "--strip-extras",
        "--upgrade-package", f"yt-dlp=={yt_dlp}",
        "--upgrade-package", f"yt-dlp-ejs=={ejs}",
        LOCK_INPUT,
    ]


def plan(version: str | None, fetch=None) -> Refresh:
    """Work out the target versions without changing any file.

    *fetch* defaults to fetch_release, looked up at call time.
    """
    release = (fetch or fetch_release)(version)
    lock_text = LOCK.read_text(encoding="utf-8")
    requirements_text = REQUIREMENTS.read_text(encoding="utf-8")
    ejs = expected_ejs(release)
    _, range_moved = move_ejs_range(requirements_text, ejs)
    return Refresh(
        old_yt_dlp=locked_version(lock_text, "yt-dlp"),
        old_ejs=locked_version(lock_text, "yt-dlp-ejs"),
        yt_dlp=release["info"]["version"],
        ejs=ejs,
        range_moved=range_moved,
    )


def write_outputs(refresh: Refresh) -> None:
    """Expose the plan to later workflow steps through GITHUB_OUTPUT."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as out:
        out.write(f"old_yt_dlp={refresh.old_yt_dlp}\n")
        out.write(f"old_ejs={refresh.old_ejs}\n")
        out.write(f"yt_dlp={refresh.yt_dlp}\n")
        out.write(f"ejs={refresh.ejs}\n")
        out.write(f"range_moved={str(refresh.range_moved).lower()}\n")
        out.write(f"changed={str(refresh.changed).lower()}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yt-dlp", dest="version",
                        help="yt-dlp version (default: newest on PyPI)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without changing files")
    parser.add_argument("--allow-downgrade", action="store_true",
                        help="allow a yt-dlp older than the locked one")
    args = parser.parse_args(argv)

    try:
        refresh = plan(parse_version(args.version))
    except (OSError, ValueError, KeyError) as exc:
        print(f"::error title=yt-dlp lock refresh::{exc}")
        return 1

    print(f"yt-dlp {refresh.old_yt_dlp} -> {refresh.yt_dlp}, "
          f"yt-dlp-ejs {refresh.old_ejs} -> {refresh.ejs}"
          + (" (requirements.txt range moves)" if refresh.range_moved
             else ""))
    if refresh.downgrade and not args.allow_downgrade:
        print(f"::error title=yt-dlp lock refresh::yt-dlp {refresh.yt_dlp} "
              f"is older than the locked {refresh.old_yt_dlp}. Pass "
              "--allow-downgrade to roll back on purpose.")
        return 1
    write_outputs(refresh)
    if args.dry_run or not refresh.changed:
        if not refresh.changed:
            print("The lock already has these versions.")
        return 0

    if refresh.range_moved:
        text, _ = move_ejs_range(
            REQUIREMENTS.read_text(encoding="utf-8"), refresh.ejs,
        )
        # LF, as the repository stores it, also on a Windows runner.
        REQUIREMENTS.write_text(text, encoding="utf-8", newline="\n")
    result = subprocess.run(
        compile_command(refresh.yt_dlp, refresh.ejs), cwd=ROOT, check=False,
    )
    if result.returncode != 0:
        print("::error title=yt-dlp lock refresh::pip-compile failed")
        return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
