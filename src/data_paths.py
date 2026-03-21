"""User-writable data directory resolution and legacy migration.

Production data lives under the OS user data location (e.g. %LOCALAPPDATA%\\stemma
on Windows) so packaged installs and MSIX can keep a read-only program directory.
A repo-relative ``data/`` tree is migrated once when the user data folder is new.
"""

from __future__ import annotations

import os
import shutil
import sys

from PySide6.QtCore import QSettings


def platform_user_data_dir() -> str:
    """Return the default per-user data directory for stemma."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "stemma")
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"), "Library", "Application Support", "stemma"
        )
    xdg = os.environ.get(
        "XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share")
    )
    return os.path.join(xdg, "stemma")


def legacy_repo_data_dir(app_root: str) -> str:
    """Return the legacy repo-relative ``data`` directory path."""
    return os.path.join(app_root, "data")


def _legacy_has_user_data(legacy_dir: str) -> bool:
    """True if *legacy_dir* looks like an existing stemma data tree."""
    if not os.path.isdir(legacy_dir):
        return False
    if os.path.isfile(os.path.join(legacy_dir, "library.json")):
        return True
    songs = os.path.join(legacy_dir, "songs")
    if os.path.isdir(songs):
        try:
            return len(os.listdir(songs)) > 0
        except OSError:
            return False
    return False


def _merge_legacy_tree(legacy_dir: str, dest_dir: str) -> None:
    """Copy missing files and directories from *legacy_dir* into *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(legacy_dir):
        src = os.path.join(legacy_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _maybe_migrate_legacy(
    app_root: str, dest_dir: str, settings: QSettings
) -> None:
    """If appropriate, copy legacy ``data/`` into *dest_dir* once."""
    if settings.value("migration/repo_data_migrated", False, type=bool):
        return
    if os.path.isfile(os.path.join(dest_dir, "library.json")):
        settings.setValue("migration/repo_data_migrated", True)
        return
    legacy = legacy_repo_data_dir(app_root)
    if not _legacy_has_user_data(legacy):
        settings.setValue("migration/repo_data_migrated", True)
        return
    _merge_legacy_tree(legacy, dest_dir)
    settings.setValue("migration/repo_data_migrated", True)


def resolve_data_dir(app_root: str, settings: QSettings) -> str:
    """Resolve the active data directory and run one-time legacy migration.

    If ``paths/data_dir`` is set in *settings*, that path is used (created if
    needed) and no repo migration is performed. Otherwise the platform default
    user directory is used and legacy ``<app_root>/data`` may be merged in.
    """
    custom = settings.value("paths/data_dir", "")
    if isinstance(custom, str) and custom.strip():
        path = os.path.normpath(os.path.expanduser(custom.strip()))
        os.makedirs(path, exist_ok=True)
        return path

    dest = platform_user_data_dir()
    os.makedirs(dest, exist_ok=True)
    _maybe_migrate_legacy(app_root, dest, settings)
    return dest
