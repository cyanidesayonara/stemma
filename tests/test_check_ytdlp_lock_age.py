"""Tests for scripts/check_ytdlp_lock_age.py (#260)."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import scripts.check_ytdlp_lock_age as age_script

_LOCK = "yt-dlp==2026.7.4 \\\n    --hash=sha256:abc\nyt-dlp-ejs==0.8.0 \\\n"


def _pypi(latest: str, uploads: dict[str, str]) -> dict:
    return {
        "info": {"version": latest},
        "releases": {
            version: [{"upload_time_iso_8601": stamp}]
            for version, stamp in uploads.items()
        },
    }


def test_locked_version_reads_the_pin():
    assert age_script.locked_version(_LOCK) == "2026.7.4"


def test_locked_version_does_not_match_yt_dlp_ejs():
    with pytest.raises(ValueError):
        age_script.locked_version("yt-dlp-ejs==0.8.0\n")


def test_up_to_date_lock_is_zero_days_behind():
    age = age_script.lock_age(_LOCK, _pypi("2026.7.4", {}))
    assert age.days_behind == 0
    assert not age.is_stale(60)


def test_days_behind_uses_upload_dates():
    pypi = _pypi("2026.10.26", {
        "2026.7.4": "2026-07-04T20:00:00.000000Z",
        "2026.10.26": "2026-10-26T01:00:00.000000Z",
    })
    age = age_script.lock_age(_LOCK, pypi)
    assert (age.locked, age.latest, age.days_behind) == (
        "2026.7.4", "2026.10.26", 113,
    )
    assert age.is_stale(60)
    assert not age.is_stale(113)


def test_missing_release_files_raise():
    with pytest.raises(ValueError):
        age_script.lock_age(_LOCK, _pypi("2026.10.26", {}))


def _fake_gh(open_issues):
    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[:3] == ["gh", "issue", "list"]:
            return SimpleNamespace(stdout=json.dumps(open_issues))
        return SimpleNamespace(stdout="")

    return run, calls


_STALE = age_script.LockAge("2026.7.4", "2026.10.26", 113)


def test_stale_lock_opens_an_issue():
    run, calls = _fake_gh([])
    assert age_script.report_issue(_STALE, 60, run=run) == "created"
    create = calls[-1]
    assert create[:3] == ["gh", "issue", "create"]
    title = create[create.index("--title") + 1]
    assert title.startswith(age_script.ISSUE_TITLE)
    assert "2026.7.4" in title


def test_stale_lock_comments_on_the_open_issue():
    """A newer release since the issue opened: retitle it and comment."""
    run, calls = _fake_gh([
        {"number": 7, "title": "Something else about yt-dlp"},
        {"number": 12, "title": f"{age_script.ISSUE_TITLE} (2026.7.4 vs x)"},
    ])
    assert age_script.report_issue(_STALE, 60, run=run) == "commented"
    assert calls[-2][:4] == ["gh", "issue", "edit", "12"]
    assert calls[-2][-1] == _STALE.title
    assert calls[-1][:4] == ["gh", "issue", "comment", "12"]


def test_unchanged_stale_lock_does_not_comment_again():
    run, calls = _fake_gh([{"number": 12, "title": _STALE.title}])
    assert age_script.report_issue(_STALE, 60, run=run) == "unchanged"
    assert [c[:3] for c in calls] == [["gh", "issue", "list"]]


def test_gh_failure_is_a_warning_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(age_script, "fetch_pypi", lambda: _pypi(
        "2026.10.26", {
            "2026.7.4": "2026-07-04T20:00:00Z",
            "2026.10.26": "2026-10-26T01:00:00Z",
        },
    ))
    monkeypatch.setattr(
        age_script.Path, "read_text", lambda *_a, **_k: _LOCK,
    )

    def failing_gh(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["gh"])

    monkeypatch.setattr(age_script, "report_issue", failing_gh)
    assert age_script.main(["--open-issue"]) == 1
    assert "could not file the issue" in capsys.readouterr().out


def test_print_expected_ejs_reads_the_default_extra(monkeypatch, capsys):
    monkeypatch.setattr(age_script.metadata, "requires", lambda _name: [
        "yt-dlp-ejs==0.9.1; extra == 'default'",
        "yt-dlp-ejs==0.9.1; extra == 'pin'",
        "requests>=2.32; extra == 'default'",
    ])
    assert age_script.main(["--print-expected-ejs"]) == 0
    assert capsys.readouterr().out.strip() == "==0.9.1"


def test_print_expected_ejs_fails_without_a_pin(monkeypatch):
    monkeypatch.setattr(age_script.metadata, "requires", lambda _name: [])
    assert age_script.main(["--print-expected-ejs"]) == 1


def test_main_warns_and_files_only_when_stale(monkeypatch, capsys):
    monkeypatch.setattr(age_script, "fetch_pypi", lambda: _pypi(
        "2026.10.26", {
            "2026.7.4": "2026-07-04T20:00:00Z",
            "2026.10.26": "2026-10-26T01:00:00Z",
        },
    ))
    monkeypatch.setattr(
        age_script.Path, "read_text", lambda *_a, **_k: _LOCK,
    )
    filed = []
    monkeypatch.setattr(
        age_script, "report_issue", lambda *a, **k: filed.append(a) or "created",
    )

    assert age_script.main(["--max-days", "200", "--open-issue"]) == 0
    assert "::warning" not in capsys.readouterr().out
    assert filed == []

    assert age_script.main(["--open-issue"]) == 0
    assert "::warning title=yt-dlp lock age::" in capsys.readouterr().out
    assert len(filed) == 1


def test_main_fails_when_pypi_is_unreachable(monkeypatch, capsys):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(age_script, "fetch_pypi", offline)
    assert age_script.main([]) == 1
    assert "could not check" in capsys.readouterr().out


def test_ejs_pin_reads_the_default_extra_only():
    assert age_script.ejs_pin([
        "yt-dlp-ejs==0.9.1; extra == 'pin'",
        "yt-dlp-ejs==0.9.0; extra == 'default'",
    ]) == "==0.9.0"
    with pytest.raises(ValueError):
        age_script.ejs_pin(["yt-dlp-ejs==0.9.1; extra == 'pin'"])


def test_locked_version_reads_any_package():
    assert age_script.locked_version(_LOCK, "yt-dlp-ejs") == "0.8.0"
