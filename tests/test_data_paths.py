"""Tests for user data directory resolution and legacy migration."""

import json
import os

import pytest
from PySide6.QtCore import QSettings

from src.data_paths import (
    consume_data_dir_reset_notice,
    legacy_repo_data_dir,
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


def test_consume_data_dir_reset_notice_once(settings_ini):
    settings_ini.setValue("internal/data_dir_was_reset", True)
    assert consume_data_dir_reset_notice(settings_ini)
    assert consume_data_dir_reset_notice(settings_ini) is None


def test_legacy_repo_data_dir():
    assert legacy_repo_data_dir("/app").replace("\\", "/") == "/app/data"


def test_platform_user_data_dir_contains_stemma():
    p = platform_user_data_dir()
    assert p.endswith("stemma")
