"""Keep CI's floating requirements on the release lock's minor versions.

CI installs ``requirements.txt``, whose ranges float, while users get the
hash-pinned ``requirements-release.txt``. For packages whose minor releases
change what the tests measure, a float means CI tests a build users never
get. PySide6 6.12 narrowed ``QStyle.sizeFromContents(CT_SpinBox)`` by
16 px, which broke the widget snapshots on CI while the shipped 6.11
build was unchanged. librosa is capped for the same reason (issue #154).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_ROOT = Path(__file__).resolve().parent.parent

# Packages whose minor version must match the release lock in CI.
_LOCKED_MINOR = ("PySide6", "librosa")


def _floating_requirements() -> dict[str, Requirement]:
    """Parse requirements.txt into requirements keyed by lowercase name."""
    parsed = {}
    for line in (_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            parsed[requirement.name.lower()] = requirement
    return parsed


def _locked_version(name: str) -> Version:
    """Return the version requirements-release.txt pins for *name*."""
    text = (_ROOT / "requirements-release.txt").read_text()
    match = re.search(
        rf"^{re.escape(name.lower())}==(\S+)", text, re.MULTILINE
    )
    assert match, f"{name} is not pinned in requirements-release.txt"
    return Version(match.group(1))


@pytest.mark.parametrize("name", _LOCKED_MINOR)
def test_floating_range_stays_on_the_locked_minor(name):
    requirement = _floating_requirements().get(name.lower())
    assert requirement is not None, f"{name} missing from requirements.txt"
    locked = _locked_version(name)
    next_minor = Version(f"{locked.major}.{locked.minor + 1}.0")

    assert requirement.specifier.contains(locked, prereleases=True), (
        f"requirements.txt {requirement} excludes the locked {name} {locked}"
    )
    assert not requirement.specifier.contains(next_minor, prereleases=True), (
        f"requirements.txt {requirement} lets CI install {name} {next_minor}, "
        f"but users get the locked {locked}. Cap it below {next_minor} and "
        f"lift the cap together with the release lock."
    )
