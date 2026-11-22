"""Keep local requirements aligned with the release lock and verify CI workflows.

CI installs the hash-pinned ``requirements-release.txt`` plus
``requirements-dev.txt`` so it tests the dependency graph that releases ship,
while local development (``pip install -r requirements.txt``) uses
``requirements.txt``. For packages whose releases change what the tests
measure, any drift in ``requirements.txt`` would still diverge locally:
PySide6 6.12 narrowed ``QStyle.sizeFromContents(CT_SpinBox)`` by 16 px, which
broke widget snapshots while the shipped build was unchanged, and a
minor-version cap still let installs float to 6.11.3. So PySide6 is pinned
exactly and librosa is capped at the locked minor version (issue #154). The
workflow tests also verify the CI lock install order and Partner Center
workflow environment/input-handling guards.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

_ROOT = Path(__file__).resolve().parent.parent

# Packages pinned to exactly the locked version.
_LOCKED_EXACT = ("PySide6",)
# Packages whose range must stay on the locked minor version.
_LOCKED_MINOR = ("librosa", "yt-dlp-ejs")


def _floating_requirements() -> dict[str, Requirement]:
    """Parse requirements.txt into requirements keyed by lowercase name."""
    parsed = {}
    for line in (_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            parsed[requirement.name.lower()] = requirement
    return parsed


def _requirement(name: str) -> Requirement:
    requirement = _floating_requirements().get(name.lower())
    assert requirement is not None, f"{name} missing from requirements.txt"
    return requirement


def _locked_version(name: str) -> Version:
    """Return the version requirements-release.txt pins for *name*."""
    text = (_ROOT / "requirements-release.txt").read_text()
    match = re.search(
        rf"^{re.escape(name.lower())}==(\S+)", text, re.MULTILINE
    )
    assert match, f"{name} is not pinned in requirements-release.txt"
    return Version(match.group(1))


@pytest.mark.parametrize("name", _LOCKED_EXACT)
def test_requirement_pins_the_locked_version(name):
    requirement = _requirement(name)
    locked = _locked_version(name)
    assert str(requirement.specifier) == f"=={locked}", (
        f"requirements.txt {requirement} must be {name}=={locked}, the "
        f"version requirements-release.txt ships. Move both together."
    )


@pytest.mark.parametrize("name", _LOCKED_MINOR)
def test_floating_range_stays_on_the_locked_minor(name):
    requirement = _requirement(name)
    locked = _locked_version(name)
    next_minor = Version(f"{locked.major}.{locked.minor + 1}.0")
    beyond = (
        next_minor,
        Version(f"{locked.major}.{locked.minor + 1}.99"),
        Version(f"{locked.major + 1}.0.0"),
    )

    assert requirement.specifier.contains(locked, prereleases=True), (
        f"requirements.txt {requirement} excludes the locked {name} {locked}"
    )
    admitted = [
        str(v) for v in beyond
        if requirement.specifier.contains(v, prereleases=True)
    ]
    assert not admitted, (
        f"requirements.txt {requirement} lets CI install {name} "
        f"{', '.join(admitted)}, but users get the locked {locked}. Cap it "
        f"below {next_minor} and lift the cap together with the release lock."
    )


def test_ci_workflow_installs_release_lock_then_dev_extras():
    workflow = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    for job_name in ("test", "slow-model-test"):
        steps = workflow["jobs"][job_name]["steps"]
        setup_step = next(
            step for step in steps
            if str(step.get("uses", "")).startswith("actions/setup-python")
        )
        cache_paths = str(setup_step.get("with", {}).get("cache-dependency-path", ""))
        assert "requirements-release.txt" in cache_paths, (
            f"{job_name} should cache requirements-release.txt"
        )
        assert "requirements-dev.txt" in cache_paths, (
            f"{job_name} should cache requirements-dev.txt"
        )

        install_step = next(
            step for step in steps
            if step.get("name") == "Install dependencies"
        )
        run_script = install_step["run"]
        release_cmd = "pip install --require-hashes -r requirements-release.txt"
        dev_cmd = "pip install -r requirements-dev.txt"
        assert release_cmd in run_script, (
            f"{job_name} must install hashed requirements-release.txt"
        )
        assert dev_cmd in run_script, (
            f"{job_name} must install requirements-dev.txt"
        )
        assert run_script.index(release_cmd) < run_script.index(dev_cmd), (
            f"{job_name} must install release lock before dev extras"
        )
        assert "-r requirements.txt" not in run_script, (
            f"{job_name} must not install unlocked requirements.txt"
        )


def test_partner_center_workflow_uses_store_environment_and_env_tag():
    workflow = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "partner-center-submit.yml").read_text(
            encoding="utf-8",
        )
    )
    submit_job = workflow["jobs"]["submit"]
    env_spec = submit_job.get("environment")
    env_name = (
        env_spec.get("name") if isinstance(env_spec, dict) else env_spec
    )
    assert env_name == "store"
    assert "refs/heads/main" in str(submit_job.get("if", ""))

    job_env = submit_job.get("env") or {}
    assert job_env.get("TAG") == "${{ inputs.tag }}"
    assert job_env.get("MODE") == "${{ inputs.mode }}"

    input_interpolation = re.compile(r"\$\{\{\s*(?:github\.event\.)?inputs\.")
    assert input_interpolation.search("${{ inputs.tag }}")
    assert input_interpolation.search("${{ github.event.inputs.tag }}")

    for step in submit_job["steps"]:
        run_script = step.get("run")
        if run_script:
            assert not input_interpolation.search(run_script), (
                f"Step {step.get('name')!r} interpolates workflow inputs directly "
                "into bash instead of passing them through env:"
            )


def test_youtube_ejs_and_workflows_wired():
    assert "yt-dlp-ejs" in _floating_requirements()
    _locked_version("yt-dlp-ejs")

    ci_workflow = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    yt_job = ci_workflow["jobs"].get("youtube-smoke-test")
    assert yt_job is not None
    assert "schedule" in str(yt_job.get("if", ""))
    assert "workflow_dispatch" in str(yt_job.get("if", ""))
    yt_commands = "\n".join(str(s.get("run", "")) for s in yt_job["steps"])
    assert "pip install --require-hashes -r requirements-release.txt" in yt_commands
    assert "scripts/fetch_js_runtime.py" in yt_commands
    assert "scripts/smoke_youtube_download.py" in yt_commands
    # The lock-age check and the canary warn without failing the job, and
    # run even after the locked smoke test fails (#260).
    warn_steps = {
        name: step for step in yt_job["steps"]
        if (name := step.get("name", "")) in (
            "Check yt-dlp lock age", "Canary smoke test on the newest yt-dlp",
        )
    }
    assert len(warn_steps) == 2
    for step in warn_steps.values():
        assert step.get("continue-on-error") is True
        assert "!cancelled()" in str(step.get("if", ""))
    assert yt_job["steps"][-1]["name"] == (
        "Canary smoke test on the newest yt-dlp"
    )
    assert yt_job.get("permissions", {}).get("issues") == "write"

    release_workflow = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    build_steps = release_workflow["jobs"]["build"]["steps"]
    step_names = [s.get("name", "") for s in build_steps]
    fetch_idx = next(
        i for i, s in enumerate(build_steps)
        if "scripts/fetch_js_runtime.py" in str(s.get("run", ""))
    )
    pyinstaller_idx = step_names.index("Build executable")
    assert fetch_idx < pyinstaller_idx

    diag_step = next(
        s for s in build_steps if s.get("name") == "Validate frozen diagnostics"
    )
    assert "YouTube JS runtime:" in diag_step["run"]




def test_locked_yt_dlp_ejs_is_the_one_yt_dlp_expects():
    """yt-dlp hash-checks the solver scripts; another version is skipped.

    pip-compile cannot see yt-dlp's ``default`` extra pin, so the lock could
    pick a newer yt-dlp-ejs than yt-dlp accepts (#260). Checked when the
    installed yt-dlp is the locked one, as in CI.
    """
    from importlib import metadata

    try:
        installed = metadata.version("yt-dlp")
    except metadata.PackageNotFoundError:
        pytest.skip("yt-dlp is not installed")
    if Version(installed) != _locked_version("yt-dlp"):
        pytest.skip(f"installed yt-dlp {installed} is not the locked one")

    expected = None
    for line in metadata.requires("yt-dlp") or []:
        requirement = Requirement(line)
        if (requirement.name.lower() == "yt-dlp-ejs" and requirement.marker
                and requirement.marker.evaluate({"extra": "default"})):
            expected = requirement.specifier
    assert expected is not None, "yt-dlp no longer pins yt-dlp-ejs"
    locked = _locked_version("yt-dlp-ejs")
    assert locked in expected, (
        f"yt-dlp {installed} expects yt-dlp-ejs{expected}, but the lock has "
        f"{locked}. Pin it with --upgrade-package (docs/DEVELOPMENT.md)."
    )
