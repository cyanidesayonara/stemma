"""Tests for source and PyInstaller application-root resolution."""

import os
import sys

from src import paths


def _source_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(paths.__file__)))


def test_app_root_returns_source_root_normally(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert paths.app_root() == _source_root()


def test_app_root_returns_meipass_when_frozen(monkeypatch, tmp_path):
    frozen_root = str(tmp_path / "bundle")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", frozen_root, raising=False)

    assert paths.app_root() == frozen_root


def test_frozen_state_is_scoped_without_leaking(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    with monkeypatch.context() as frozen:
        frozen.setattr(sys, "frozen", True, raising=False)
        frozen.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
        assert paths.app_root() == str(tmp_path)

    assert paths.app_root() == _source_root()
    assert not hasattr(sys, "frozen")
    assert not hasattr(sys, "_MEIPASS")
