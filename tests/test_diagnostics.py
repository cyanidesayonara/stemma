"""Tests for release smoke diagnostics."""

import sys
from types import SimpleNamespace

from src.version import __version__


_DIAGNOSTICS = {
    "stemma_version": "2.6.0",
    "onnxruntime_version": "1.24.4",
    "librosa_version": "1.0.0",
    "soxr_version": "1.1.0",
    "youtube_js_runtime": "quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)",
    "available_providers": [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ],
    "model_providers": [],
}


def test_collect_diagnostics_reports_versions_and_providers(monkeypatch):
    from src.diagnostics import collect_diagnostics

    ort = SimpleNamespace(
        __version__="1.24.4",
        get_available_providers=lambda: [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    # Building a real session per cached model is the slow, machine-
    # dependent part; it has its own test below.
    monkeypatch.setattr(
        "src.diagnostics.collect_model_providers", lambda: [],
    )
    monkeypatch.setattr(
        "src.diagnostics.collect_librosa_version", lambda: "1.0.0",
    )
    monkeypatch.setattr(
        "src.diagnostics.collect_stretch_check", lambda: "1.1.0",
    )
    monkeypatch.setattr(
        "src.diagnostics.collect_youtube_js_check",
        lambda: "quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)",
    )

    diagnostics = collect_diagnostics()

    assert diagnostics == {
        "stemma_version": __version__,
        "onnxruntime_version": "1.24.4",
        "librosa_version": "1.0.0",
        "soxr_version": "1.1.0",
        "youtube_js_runtime": "quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)",
        "available_providers": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
        "model_providers": [],
    }


def test_format_diagnostics_is_human_readable():
    from src.diagnostics import format_diagnostics

    output = format_diagnostics({
        "stemma_version": "2.6.0",
        "onnxruntime_version": "1.24.4",
        "librosa_version": "1.0.0",
        "soxr_version": "1.1.0",
        "youtube_js_runtime": "quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)",
        "available_providers": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    })

    assert output.splitlines() == [
        "stemma version: 2.6.0",
        "ONNX Runtime version: 1.24.4",
        "librosa version: 1.0.0",
        "soxr version: 1.1.0 (live speed and pitch ran)",
        "YouTube JS runtime: quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)",
        "Available ONNX providers: "
        "DmlExecutionProvider, CPUExecutionProvider",
        "Provider selected per cached model: (no models cached)",
    ]


def test_diagnostics_request_requires_explicit_flag():
    from src.diagnostics import diagnostics_requested

    assert diagnostics_requested(["stemma.exe", "--diagnostics"])
    assert diagnostics_requested([
        "stemma.exe",
        "--diagnostics-file",
        "diagnostics.txt",
    ])
    assert not diagnostics_requested(["stemma.exe"])


def test_diagnostics_file_is_atomic_utf8_and_returns_success(
    monkeypatch, tmp_path
):
    from src import diagnostics

    output = tmp_path / "frozen diagnostics.txt"
    monkeypatch.setattr(
        diagnostics,
        "collect_diagnostics",
        lambda: _DIAGNOSTICS,
    )

    result = diagnostics.main([
        "stemma.exe",
        "--diagnostics-file",
        str(output),
    ])

    assert result == 0
    assert output.read_text(encoding="utf-8").splitlines() == [
        "stemma version: 2.6.0",
        "ONNX Runtime version: 1.24.4",
        "librosa version: 1.0.0",
        "soxr version: 1.1.0 (live speed and pitch ran)",
        "YouTube JS runtime: quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)",
        "Available ONNX providers: "
        "DmlExecutionProvider, CPUExecutionProvider",
        "Provider selected per cached model: (no models cached)",
    ]
    assert list(tmp_path.glob("*.tmp")) == []


def test_diagnostics_file_succeeds_when_stdout_is_none(
    monkeypatch, tmp_path
):
    from src import diagnostics

    output = tmp_path / "diagnostics.txt"
    monkeypatch.setattr(
        diagnostics,
        "collect_diagnostics",
        lambda: _DIAGNOSTICS,
    )
    monkeypatch.setattr(sys, "stdout", None)

    result = diagnostics.main([
        "stemma.exe",
        "--diagnostics-file",
        str(output),
    ])

    assert result == 0
    assert "DmlExecutionProvider" in output.read_text(encoding="utf-8")


def test_diagnostics_file_requires_a_path(monkeypatch):
    from src import diagnostics

    collected = []
    monkeypatch.setattr(
        diagnostics,
        "collect_diagnostics",
        lambda: collected.append(True),
    )

    assert diagnostics.main(["stemma.exe", "--diagnostics-file"]) == 2
    assert collected == []


def test_console_diagnostics_remain_available(monkeypatch, capsys):
    from src import diagnostics

    monkeypatch.setattr(
        diagnostics,
        "collect_diagnostics",
        lambda: _DIAGNOSTICS,
    )

    assert diagnostics.main(["stemma.exe", "--diagnostics"]) == 0
    assert "DmlExecutionProvider" in capsys.readouterr().out


def test_format_diagnostics_lists_provider_per_model():
    """The per-model line is the one that catches a silent CPU fallback."""
    from src.diagnostics import format_diagnostics

    output = format_diagnostics({
        "stemma_version": "2.6.0",
        "onnxruntime_version": "1.24.4",
        "librosa_version": "1.0.0",
        "available_providers": ["DmlExecutionProvider"],
        "model_providers": [
            "htdemucs.onnx: CPUExecutionProvider",
            "beat_this.onnx: DmlExecutionProvider",
        ],
    })

    assert output.splitlines()[-3:] == [
        "Provider selected per cached model:",
        "  htdemucs.onnx: CPUExecutionProvider",
        "  beat_this.onnx: DmlExecutionProvider",
    ]


def test_collect_model_providers_reports_each_cached_model(
    monkeypatch, tmp_path,
):
    """Each .onnx in the models dir is probed for its real provider."""
    from src import diagnostics as diagnostics_module

    models = tmp_path / "models"
    models.mkdir()
    (models / "htdemucs.onnx").write_bytes(b"stub")
    (models / "beat_this.onnx").write_bytes(b"stub")
    (models / "notes.txt").write_bytes(b"ignored")

    providers = {
        "htdemucs.onnx": "CPUExecutionProvider",
        "beat_this.onnx": "DmlExecutionProvider",
    }

    import src.onnx_session as onnx_session

    monkeypatch.setattr(
        "src.data_paths.platform_user_data_dir", lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        onnx_session, "create_onnx_session",
        lambda path: SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        onnx_session, "selected_session_provider",
        lambda session: providers[
            session.path.replace("\\", "/").rsplit("/", 1)[-1]
        ],
    )

    assert diagnostics_module.collect_model_providers() == [
        ("beat_this.onnx", "DmlExecutionProvider"),
        ("htdemucs.onnx", "CPUExecutionProvider"),
    ]


def test_collect_model_providers_skips_dml_sidecar_and_uses_prepared_path(
    monkeypatch, tmp_path,
):
    """Diagnostics report canonical filenames while probing HTDemucs via its sidecar."""
    from src import diagnostics as diagnostics_module
    import src.onnx_session as onnx_session

    models = tmp_path / "models"
    models.mkdir()
    orig = models / "htdemucs.onnx"
    sidecar = models / "htdemucs.dml.onnx"
    beat = models / "beat_this.onnx"
    orig.write_bytes(b"stub")
    sidecar.write_bytes(b"stub")
    beat.write_bytes(b"stub")

    opened_paths: list[str] = []

    def fake_prepare(path: str) -> str:
        if path == str(orig):
            return str(sidecar)
        return path

    def fake_session(path: str):
        opened_paths.append(path)
        return SimpleNamespace(path=path)

    monkeypatch.setattr(
        "src.data_paths.platform_user_data_dir", lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        diagnostics_module, "prepare_htdemucs_dml_path", fake_prepare,
    )
    monkeypatch.setattr(onnx_session, "create_onnx_session", fake_session)
    monkeypatch.setattr(
        onnx_session,
        "selected_session_provider",
        lambda _session: "DmlExecutionProvider",
    )

    assert diagnostics_module.collect_model_providers() == [
        ("beat_this.onnx", "DmlExecutionProvider"),
        ("htdemucs.onnx", "DmlExecutionProvider"),
    ]
    assert opened_paths == [str(beat), str(sidecar)]


def test_collect_model_providers_reports_session_failure(
    monkeypatch, tmp_path,
):
    """A model that cannot open is reported, not swallowed."""
    from src import diagnostics as diagnostics_module
    import src.onnx_session as onnx_session

    models = tmp_path / "models"
    models.mkdir()
    (models / "broken.onnx").write_bytes(b"stub")

    def boom(path):
        raise RuntimeError("bad graph")

    monkeypatch.setattr(
        "src.data_paths.platform_user_data_dir", lambda: str(tmp_path),
    )
    monkeypatch.setattr(onnx_session, "create_onnx_session", boom)

    assert diagnostics_module.collect_model_providers() == [
        ("broken.onnx", "unavailable (RuntimeError)"),
    ]


def test_collect_librosa_version_imports_the_modules_stemma_uses():
    """The frozen smoke check must fail on a bundle missing librosa's
    lazily loaded submodules, not the first Speed or Pitch render."""
    import librosa

    from src.diagnostics import collect_librosa_version

    assert collect_librosa_version() == librosa.__version__


def test_collect_librosa_version_reports_a_broken_bundle(monkeypatch):
    from src.diagnostics import collect_librosa_version

    # None in sys.modules makes the import raise, as a missing module would.
    monkeypatch.setitem(sys.modules, "librosa.effects", None)

    reported = collect_librosa_version()

    assert reported.startswith("unavailable (")


def test_collect_stretch_check_runs_the_live_stretcher(monkeypatch):
    """The frozen smoke check must exercise live Speed and Pitch (soxr's
    native extension included), not just import the package."""
    import soxr

    from src.diagnostics import collect_stretch_check

    real = soxr.ResampleStream.resample_chunk
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(soxr.ResampleStream, "resample_chunk", counted)

    assert collect_stretch_check() == soxr.__version__
    # a pitch change, so the native resampler really ran
    assert calls


def test_collect_stretch_check_reports_non_finite_output(monkeypatch):
    import numpy as np

    from src import stretch
    from src.diagnostics import collect_stretch_check

    monkeypatch.setattr(
        stretch.StreamingStretcher, "read",
        lambda self, count, source: np.full((count, 2), np.nan, np.float32),
    )

    assert collect_stretch_check() == "unavailable (non-finite output)"


def test_collect_stretch_check_reports_a_broken_bundle(monkeypatch):
    from src.diagnostics import collect_stretch_check

    # None in sys.modules makes the import raise, as a missing module would.
    monkeypatch.setitem(sys.modules, "soxr", None)

    assert collect_stretch_check().startswith("unavailable (")


def test_collect_stretch_check_reports_silent_output(monkeypatch):
    import numpy as np

    from src import stretch
    from src.diagnostics import collect_stretch_check

    monkeypatch.setattr(
        stretch.StreamingStretcher, "read",
        lambda self, count, source: np.zeros((count, 2), np.float32),
    )

    assert collect_stretch_check() == "unavailable (silent output)"


def test_format_diagnostics_marks_a_failed_stretch_check():
    from src.diagnostics import format_diagnostics

    output = format_diagnostics({
        "stemma_version": "3.0.0",
        "onnxruntime_version": "1.24.4",
        "librosa_version": "1.0.0",
        "soxr_version": "unavailable (ImportError: no soxr)",
        "available_providers": ["CPUExecutionProvider"],
    })

    assert "soxr version: unavailable (ImportError: no soxr)" in output
    # the release workflow requires a version number on this line
    assert "soxr version: 1" not in output


def test_collect_youtube_js_check_reports_supported_runtime(monkeypatch):
    from src.diagnostics import collect_youtube_js_check

    fake_solver = SimpleNamespace(core=lambda: "core-js", lib=lambda: "lib-js")
    fake_ejs = SimpleNamespace(
        version="0.8.0",
        yt=SimpleNamespace(solver=fake_solver),
    )
    monkeypatch.setitem(sys.modules, "yt_dlp_ejs", fake_ejs)
    monkeypatch.setitem(sys.modules, "yt_dlp_ejs.yt", fake_ejs.yt)
    monkeypatch.setitem(sys.modules, "yt_dlp_ejs.yt.solver", fake_solver)

    fake_info = SimpleNamespace(name="quickjs-ng", version="0.18.0", supported=True)
    fake_ydl = SimpleNamespace(
        _js_runtimes={"quickjs": SimpleNamespace(info=fake_info)},
    )
    monkeypatch.setitem(
        sys.modules,
        "yt_dlp",
        SimpleNamespace(YoutubeDL=lambda _opts: fake_ydl),
    )

    assert collect_youtube_js_check() == "quickjs-ng 0.18.0 (yt-dlp-ejs 0.8.0)"


def test_collect_youtube_js_check_reports_missing_ejs(monkeypatch):
    from src.diagnostics import collect_youtube_js_check

    monkeypatch.setitem(sys.modules, "yt_dlp_ejs", None)

    assert collect_youtube_js_check().startswith("unavailable (")


def test_collect_youtube_js_check_reports_missing_runtime(monkeypatch):
    from src.diagnostics import collect_youtube_js_check

    fake_solver = SimpleNamespace(core=lambda: "core-js", lib=lambda: "lib-js")
    fake_ejs = SimpleNamespace(
        version="0.8.0",
        yt=SimpleNamespace(solver=fake_solver),
    )
    monkeypatch.setitem(sys.modules, "yt_dlp_ejs", fake_ejs)
    monkeypatch.setitem(sys.modules, "yt_dlp_ejs.yt", fake_ejs.yt)
    monkeypatch.setitem(sys.modules, "yt_dlp_ejs.yt.solver", fake_solver)

    fake_ydl = SimpleNamespace(
        _js_runtimes={"quickjs": SimpleNamespace(info=None)},
    )
    monkeypatch.setitem(
        sys.modules,
        "yt_dlp",
        SimpleNamespace(YoutubeDL=lambda _opts: fake_ydl),
    )

    assert collect_youtube_js_check() == "unavailable (no supported JS runtime)"

