"""Tests for scripts/refresh_ytdlp_lock.py (#260)."""

from types import SimpleNamespace

import pytest

import scripts.refresh_ytdlp_lock as refresh_script

_LOCK = (
    "yt-dlp==2026.10.26 \\\n    --hash=sha256:aa\n"
    "yt-dlp-ejs==0.8.0 \\\n    --hash=sha256:bb\n"
)
_REQUIREMENTS = (
    "yt-dlp>=2024.1.0\n"
    "# YouTube JS challenge solver scripts for yt-dlp.\n"
    "yt-dlp-ejs>=0.8.0,<0.9\n"
    "soxr>=1.0\n"
)


def _release(version: str, ejs: str | None) -> dict:
    requires = ["requests>=2.32; extra == \"default\""]
    if ejs is not None:
        requires += [
            f"yt-dlp-ejs=={ejs}; extra == \"default\"",
            f"yt-dlp-ejs=={ejs}; extra == \"pin\"",
        ]
    return {"info": {"version": version, "requires_dist": requires}}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A requirements.txt and lock in a temporary directory."""
    lock = tmp_path / "requirements-release.txt"
    requirements = tmp_path / "requirements.txt"
    lock.write_text(_LOCK, encoding="utf-8")
    requirements.write_text(_REQUIREMENTS, encoding="utf-8")
    monkeypatch.setattr(refresh_script, "LOCK", lock)
    monkeypatch.setattr(refresh_script, "REQUIREMENTS", requirements)
    monkeypatch.setattr(refresh_script, "ROOT", tmp_path)
    return SimpleNamespace(lock=lock, requirements=requirements)


def test_expected_ejs_reads_the_default_extra():
    assert refresh_script.expected_ejs(_release("2026.11.20", "0.8.1")) == (
        "0.8.1"
    )


def test_expected_ejs_requires_a_pin():
    with pytest.raises(ValueError):
        refresh_script.expected_ejs(_release("2026.11.20", None))


def test_ejs_range_stays_when_the_minor_is_the_same():
    text, moved = refresh_script.move_ejs_range(_REQUIREMENTS, "0.8.3")
    assert (text, moved) == (_REQUIREMENTS, False)


def test_ejs_range_moves_to_a_new_minor():
    text, moved = refresh_script.move_ejs_range(_REQUIREMENTS, "0.9.0")
    assert moved
    assert "yt-dlp-ejs>=0.9.0,<0.10\n" in text
    # Only that line changes; its comment and the other lines stay.
    assert text.replace("yt-dlp-ejs>=0.9.0,<0.10", "") == (
        _REQUIREMENTS.replace("yt-dlp-ejs>=0.8.0,<0.9", "")
    )


def test_compile_command_pins_both_versions():
    command = refresh_script.compile_command("2026.11.20", "0.8.1")
    assert command[1:4] == ["-m", "piptools", "compile"]
    assert "--generate-hashes" in command
    assert "yt-dlp==2026.11.20" in command
    assert "yt-dlp-ejs==0.8.1" in command
    assert command[-1] == "requirements-release.in"


def test_plan_reports_the_change(repo):
    refresh = refresh_script.plan(
        None, fetch=lambda _v: _release("2026.11.20", "0.9.0"),
    )
    assert refresh == refresh_script.Refresh(
        old_yt_dlp="2026.10.26", old_ejs="0.8.0",
        yt_dlp="2026.11.20", ejs="0.9.0", range_moved=True,
    )
    assert refresh.changed
    # Planning changes no file.
    assert repo.requirements.read_text(encoding="utf-8") == _REQUIREMENTS


def test_up_to_date_lock_does_not_compile(repo, monkeypatch, capsys):
    monkeypatch.setattr(
        refresh_script, "fetch_release",
        lambda _v: _release("2026.10.26", "0.8.0"),
    )
    ran = []
    monkeypatch.setattr(refresh_script.subprocess, "run",
                        lambda *a, **k: ran.append(a))
    assert refresh_script.main([]) == 0
    assert ran == []
    assert "already has these versions" in capsys.readouterr().out


def test_refresh_moves_the_range_and_compiles(repo, monkeypatch, tmp_path):
    monkeypatch.setattr(
        refresh_script, "fetch_release",
        lambda _v: _release("2026.11.20", "0.9.0"),
    )
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    ran = []

    def fake_run(command, **kwargs):
        ran.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(refresh_script.subprocess, "run", fake_run)
    assert refresh_script.main([]) == 0

    assert "yt-dlp-ejs>=0.9.0,<0.10" in repo.requirements.read_text(
        encoding="utf-8",
    )
    command, kwargs = ran[0]
    assert "yt-dlp==2026.11.20" in command
    assert kwargs["cwd"] == tmp_path
    outputs = output.read_text(encoding="utf-8").splitlines()
    assert "yt_dlp=2026.11.20" in outputs
    assert "ejs=0.9.0" in outputs
    assert "range_moved=true" in outputs
    assert "changed=true" in outputs


def test_dry_run_changes_nothing(repo, monkeypatch):
    monkeypatch.setattr(
        refresh_script, "fetch_release",
        lambda _v: _release("2026.11.20", "0.9.0"),
    )
    monkeypatch.setattr(refresh_script.subprocess, "run",
                        lambda *a, **k: pytest.fail("compiled on a dry run"))
    assert refresh_script.main(["--dry-run"]) == 0
    assert repo.requirements.read_text(encoding="utf-8") == _REQUIREMENTS


def test_failed_compile_fails_the_refresh(repo, monkeypatch):
    monkeypatch.setattr(
        refresh_script, "fetch_release",
        lambda _v: _release("2026.11.20", "0.8.1"),
    )
    monkeypatch.setattr(refresh_script.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=2))
    assert refresh_script.main([]) == 2


def test_unreachable_pypi_is_an_error(repo, monkeypatch, capsys):
    def offline(_v):
        raise OSError("no network")

    monkeypatch.setattr(refresh_script, "fetch_release", offline)
    assert refresh_script.main([]) == 1
    assert "::error" in capsys.readouterr().out


def test_older_yt_dlp_is_refused_without_allow_downgrade(repo, monkeypatch,
                                                          capsys):
    monkeypatch.setattr(
        refresh_script, "fetch_release",
        lambda _v: _release("2026.8.19", "0.8.0"),
    )
    monkeypatch.setattr(refresh_script.subprocess, "run",
                        lambda *a, **k: pytest.fail("compiled a downgrade"))
    assert refresh_script.main(["--yt-dlp", "2026.8.19"]) == 1
    assert "--allow-downgrade" in capsys.readouterr().out


def test_allow_downgrade_rolls_back(repo, monkeypatch):
    monkeypatch.setattr(
        refresh_script, "fetch_release",
        lambda _v: _release("2026.8.19", "0.8.0"),
    )
    ran = []
    monkeypatch.setattr(
        refresh_script.subprocess, "run",
        lambda command, **_k: ran.append(command)
        or SimpleNamespace(returncode=0),
    )
    assert refresh_script.main(
        ["--yt-dlp", "2026.8.19", "--allow-downgrade"],
    ) == 0
    assert "yt-dlp==2026.8.19" in ran[0]


@pytest.mark.parametrize("text, expected", [
    (None, None), ("", None), ("   ", None),
    ("2026.08.19", "2026.8.19"), (" 2026.11.20 ", "2026.11.20"),
])
def test_parse_version_normalises(text, expected):
    assert refresh_script.parse_version(text) == expected


def test_parse_version_rejects_a_non_version(repo, capsys):
    with pytest.raises(ValueError):
        refresh_script.parse_version("v1; rm")
    assert refresh_script.main(["--yt-dlp", "latest!"]) == 1
    assert "is not a yt-dlp version" in capsys.readouterr().out


def test_versions_compare_as_versions_not_strings():
    refresh = refresh_script.Refresh(
        old_yt_dlp="2026.8.19", old_ejs="0.8.0",
        yt_dlp="2026.08.19", ejs="0.8.0", range_moved=False,
    )
    assert not refresh.changed
