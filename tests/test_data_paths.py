"""Tests for user data directory resolution and legacy migration."""

import json
import os

import pytest
from PySide6.QtCore import QSettings

from src.data_paths import (
    consume_data_dir_reset_notice,
    legacy_repo_data_dir,
    location_reachable,
    platform_user_data_dir,
    resolve_data_dir,
)


@pytest.fixture
def settings_ini(tmp_path):
    path = tmp_path / "st.ini"
    return QSettings(str(path), QSettings.Format.IniFormat)


class TestResolveDataDir:
    def test_custom_path_skips_migration(self, tmp_path, settings_ini, monkeypatch):
        custom = tmp_path / "custom_data"
        settings_ini.setValue("paths/data_dir", str(custom))
        app_root = tmp_path / "repo"
        legacy = app_root / "data"
        legacy.mkdir(parents=True)
        (legacy / "library.json").write_text("[]", encoding="utf-8")

        resolved = resolve_data_dir(str(app_root), settings_ini)

        assert resolved == str(custom)
        assert os.path.isdir(custom)
        assert not (custom / "library.json").exists()

    def test_default_uses_platform_dir(self, tmp_path, settings_ini, monkeypatch):
        fake_user = tmp_path / "userdata"
        monkeypatch.setattr(
            "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
        )
        app_root = tmp_path / "repo"
        os.makedirs(app_root, exist_ok=True)

        resolved = resolve_data_dir(str(app_root), settings_ini)

        assert resolved == str(fake_user)

    def test_migrates_legacy_when_dest_empty(self, tmp_path, settings_ini, monkeypatch):
        fake_user = tmp_path / "userdata"
        monkeypatch.setattr(
            "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
        )
        app_root = tmp_path / "repo"
        legacy = os.path.join(app_root, "data")
        os.makedirs(os.path.join(legacy, "songs"), exist_ok=True)
        lib_path = os.path.join(legacy, "library.json")
        with open(lib_path, "w", encoding="utf-8") as f:
            json.dump([], f)

        resolved = resolve_data_dir(str(app_root), settings_ini)

        assert resolved == str(fake_user)
        assert os.path.isfile(os.path.join(fake_user, "library.json"))
        migrated = settings_ini.value("migration/repo_data_migrated", False, type=bool)
        assert migrated is True

    def test_no_migration_when_dest_has_library(
        self, tmp_path, settings_ini, monkeypatch
    ):
        fake_user = tmp_path / "userdata"
        monkeypatch.setattr(
            "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
        )
        os.makedirs(fake_user, exist_ok=True)
        with open(
            os.path.join(fake_user, "library.json"), "w", encoding="utf-8"
        ) as f:
            json.dump([{"id": "x"}], f)

        app_root = tmp_path / "repo"
        legacy = os.path.join(app_root, "data")
        os.makedirs(legacy, exist_ok=True)
        with open(os.path.join(legacy, "library.json"), "w", encoding="utf-8") as f:
            json.dump([], f)

        resolve_data_dir(str(app_root), settings_ini)

        with open(
            os.path.join(fake_user, "library.json"), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        assert data == [{"id": "x"}]

    def test_migration_runs_once(self, tmp_path, settings_ini, monkeypatch):
        fake_user = tmp_path / "userdata"
        monkeypatch.setattr(
            "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
        )
        app_root = tmp_path / "repo"
        legacy = os.path.join(app_root, "data")
        os.makedirs(legacy, exist_ok=True)
        with open(os.path.join(legacy, "library.json"), "w", encoding="utf-8") as f:
            json.dump([], f)
        with open(os.path.join(legacy, "note.txt"), "w", encoding="utf-8") as f:
            f.write("legacy")

        resolve_data_dir(str(app_root), settings_ini)
        os.remove(os.path.join(fake_user, "note.txt"))
        with open(os.path.join(legacy, "note2.txt"), "w", encoding="utf-8") as f:
            f.write("new")

        resolve_data_dir(str(app_root), settings_ini)

        assert not os.path.exists(os.path.join(str(fake_user), "note2.txt"))

    def test_invalid_custom_path_is_file_falls_back(
        self, tmp_path, settings_ini, monkeypatch
    ):
        fake_user = tmp_path / "userdata"
        monkeypatch.setattr(
            "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
        )
        bad = tmp_path / "not_a_dir"
        bad.write_text("x", encoding="utf-8")
        settings_ini.setValue("paths/data_dir", str(bad))
        app_root = tmp_path / "repo"
        os.makedirs(app_root, exist_ok=True)

        resolved = resolve_data_dir(str(app_root), settings_ini)

        assert resolved == str(fake_user)
        assert not str(settings_ini.value("paths/data_dir", "")).strip()
        assert settings_ini.value(
            "internal/data_dir_was_reset", False, type=bool
        ) is True


class TestUnreachableDataDir:
    """A custom folder on a disconnected drive is kept for next time."""

    @pytest.fixture
    def setup(self, tmp_path, settings_ini, monkeypatch):
        fake_user = tmp_path / "userdata"
        monkeypatch.setattr(
            "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
        )
        custom = os.path.join("X:\\", "music", "stemma")
        settings_ini.setValue("paths/data_dir", custom)
        monkeypatch.setattr(
            "src.data_paths.location_reachable", lambda path: False
        )
        app_root = tmp_path / "repo"
        app_root.mkdir()
        return fake_user, custom, str(app_root)

    def test_falls_back_for_this_session(self, setup, settings_ini):
        fake_user, custom, app_root = setup
        assert resolve_data_dir(app_root, settings_ini) == str(fake_user)

    def test_keeps_the_setting(self, setup, settings_ini):
        _fake_user, custom, app_root = setup
        resolve_data_dir(app_root, settings_ini)
        assert settings_ini.value("paths/data_dir", "") == custom
        assert not settings_ini.value(
            "internal/data_dir_was_reset", False, type=bool
        )

    def test_notice_names_the_folder_once(self, setup, settings_ini):
        _fake_user, custom, app_root = setup
        resolve_data_dir(app_root, settings_ini)
        msg = consume_data_dir_reset_notice(settings_ini)
        assert msg and custom in msg
        assert "next time" in msg
        assert consume_data_dir_reset_notice(settings_ini) is None

    def test_no_legacy_migration_into_fallback(self, setup, settings_ini):
        fake_user, _custom, app_root = setup
        legacy = os.path.join(app_root, "data")
        os.makedirs(legacy)
        with open(os.path.join(legacy, "library.json"), "w") as fh:
            fh.write("[]")
        resolve_data_dir(app_root, settings_ini)
        assert not os.path.exists(os.path.join(fake_user, "library.json"))
        assert not settings_ini.value(
            "migration/repo_data_migrated", False, type=bool
        )

    def test_reachable_again_uses_the_folder(
        self, tmp_path, settings_ini, monkeypatch,
    ):
        custom = tmp_path / "custom"
        settings_ini.setValue("paths/data_dir", str(custom))
        monkeypatch.setattr(
            "src.data_paths.location_reachable", lambda path: True
        )
        assert resolve_data_dir(str(tmp_path), settings_ini) == str(custom)


def _missing_drive_letter() -> str:
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        if not os.path.isdir(f"{letter}:\\"):
            return letter
    pytest.skip("every drive letter is in use")


def test_location_reachable_for_existing_roots(tmp_path):
    assert location_reachable(str(tmp_path / "not" / "yet" / "made"))
    # A relative path has no root to lose.
    assert location_reachable("relative/data")


@pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows-only")
def test_location_on_a_missing_drive_is_unreachable():
    letter = _missing_drive_letter()
    assert not location_reachable(f"{letter}:\\music\\stemma")


@pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows-only")
def test_missing_drive_keeps_the_setting_end_to_end(
    tmp_path, settings_ini, monkeypatch,
):
    fake_user = tmp_path / "userdata"
    monkeypatch.setattr(
        "src.data_paths.platform_user_data_dir", lambda: str(fake_user)
    )
    custom = f"{_missing_drive_letter()}:\\music\\stemma"
    settings_ini.setValue("paths/data_dir", custom)

    assert resolve_data_dir(str(tmp_path), settings_ini) == str(fake_user)
    assert settings_ini.value("paths/data_dir") == custom
    assert custom in consume_data_dir_reset_notice(settings_ini)


def test_consume_data_dir_reset_notice_once(settings_ini):
    settings_ini.setValue("internal/data_dir_was_reset", True)
    assert consume_data_dir_reset_notice(settings_ini)
    assert consume_data_dir_reset_notice(settings_ini) is None


def test_legacy_repo_data_dir():
    assert legacy_repo_data_dir("/app").replace("\\", "/") == "/app/data"


def test_platform_user_data_dir_contains_stemma():
    p = platform_user_data_dir()
    assert p.endswith("stemma")
