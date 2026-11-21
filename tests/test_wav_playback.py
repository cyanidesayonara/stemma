"""Tests for deferred WAV playback helpers."""

import os

import pytest
from PySide6.QtWidgets import QApplication

from src.paths import app_root
from src.ui.wav_playback import play_wav_async

# Importing Qt Multimedia (QSoundEffect) or calling play_impl on GitHub Actions
# Windows runners has been observed to block indefinitely; the rest of the
# suite does not cover this path on CI.
_SKIP_QT_MULTIMEDIA_CI = os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def test_play_missing_file_is_noop() -> None:
    play_wav_async("/nonexistent/stemma/arpeggio.wav")


def test_playback_module_is_visible_to_pyinstaller() -> None:
    """The packaged build shipped without the playback module.

    PyInstaller finds modules by their import statements and does not read
    ``importlib.import_module("...")`` strings, so the 3.0.0 bundle had no
    ``src.ui._wav_playback_impl``: every logo click raised inside the
    widgets' guard and animated in silence, while the splash, which calls
    winsound itself, still played. Any deferred import of a ``src`` module
    has to stay a plain import statement.
    """
    import ast
    import pathlib

    src_dir = pathlib.Path(app_root()) / "src"
    offenders = []
    for path in src_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if _hides_a_src_import(node):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _hides_a_src_import(node) -> bool:
    """A dynamic import PyInstaller cannot see that may name a src module.

    Covers ``importlib.import_module(...)``, a bare ``import_module(...)``
    (``from importlib import import_module``) and ``__import__(...)``. A
    constant name is flagged when it starts with ``src``; a computed one
    (f-string, variable) is always flagged, since it could name anything.
    """
    import ast

    if not isinstance(node, ast.Call) or not node.args:
        return False
    func = node.func
    name = (
        func.attr if isinstance(func, ast.Attribute)
        else func.id if isinstance(func, ast.Name)
        else None
    )
    if name not in _DYNAMIC_IMPORTERS:
        return False
    target = node.args[0]
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value.startswith("src")
    return True


@pytest.mark.parametrize("source, flagged", [
    ('importlib.import_module("src.ui.x")', True),
    ('import_module("src.ui.x")', True),
    ('__import__("src.ui.x")', True),
    ('import_module(f"src.ui.{name}")', True),
    ('importlib.import_module(name)', True),
    ('__import__(name)', True),
    ('importlib.import_module("numpy")', False),
    ('obj.load("src.ui.x")', False),
])
def test_dynamic_import_guard_shapes(source, flagged) -> None:
    import ast

    call = ast.parse(source, mode="eval").body
    assert _hides_a_src_import(call) is flagged


@pytest.mark.skipif(
    _SKIP_QT_MULTIMEDIA_CI,
    reason="Qt Multimedia hangs on GitHub Actions Windows runners",
)
def test_play_impl_module_loads() -> None:
    from src.ui import _wav_playback_impl

    assert hasattr(_wav_playback_impl, "play_impl")


@pytest.mark.skipif(
    _SKIP_QT_MULTIMEDIA_CI,
    reason="Qt Multimedia hangs on GitHub Actions Windows runners",
)
def test_play_real_wav_no_crash(app) -> None:
    path = os.path.join(app_root(), "assets", "audio", "arpeggio.wav")
    if not os.path.isfile(path):
        pytest.skip("arpeggio.wav not present")
    play_wav_async(path)


@pytest.mark.skipif(
    _SKIP_QT_MULTIMEDIA_CI,
    reason="Qt Multimedia hangs on GitHub Actions Windows runners",
)
def test_two_different_wavs_no_crash(app) -> None:
    root = app_root()
    a = os.path.join(root, "assets", "audio", "arpeggio.wav")
    b = os.path.join(root, "assets", "audio", "chord.wav")
    if not (os.path.isfile(a) and os.path.isfile(b)):
        pytest.skip("audio assets not present")
    play_wav_async(a)
    play_wav_async(b)


class TestQtMultimediaOptional:
    """The module must import even when QtMultimedia is unavailable.

    In the packaged (MSIX) build QtMultimedia's backend DLLs were not
    collected, so the top-level import raised. That ImportError escaped
    play_wav_async into the logo widgets' exception guard, which is why
    clicking a logo animated in silence while the splash -- which calls
    winsound directly -- still played.
    """

    def test_module_imports_without_qtmultimedia(self):
        import builtins
        import importlib
        import sys
        from unittest.mock import patch

        impl_name = "src.ui._wav_playback_impl"
        saved_impl = sys.modules.pop(impl_name, None)
        saved_qtmm = sys.modules.pop("PySide6.QtMultimedia", None)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PySide6.QtMultimedia":
                raise ImportError("QtMultimedia unavailable (simulated)")
            return real_import(name, *args, **kwargs)

        try:
            with patch.object(builtins, "__import__", fake_import):
                mod = importlib.import_module(impl_name)
                assert mod._sound_effect_class() is None
        finally:
            sys.modules.pop(impl_name, None)
            if saved_qtmm is not None:
                sys.modules["PySide6.QtMultimedia"] = saved_qtmm
            if saved_impl is not None:
                sys.modules[impl_name] = saved_impl
                # The re-import also rebound the package attribute, which is
                # what ``from src.ui import _wav_playback_impl`` reads; left
                # alone, later tests patch one module and play through the
                # other.
                import src.ui

                src.ui._wav_playback_impl = saved_impl

    def test_play_impl_uses_winsound_on_windows(self, tmp_path):
        """On Windows the Qt path is never reached, so a missing
        QtMultimedia cannot silence playback."""
        from unittest.mock import patch

        import src.ui._wav_playback_impl as impl

        wav = tmp_path / "s.wav"
        wav.write_bytes(b"RIFF....WAVEfmt ")
        if not impl._HAS_WINSOUND:
            import pytest
            pytest.skip("winsound unavailable")
        with patch.object(impl, "_play_winsound_fallback") as fallback:
            impl.play_impl(wav)
        fallback.assert_called_once()
