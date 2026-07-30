"""Tests for release smoke diagnostics."""

import sys
from types import SimpleNamespace


_DIAGNOSTICS = {
    "stemma_version": "2.6.0",
    "onnxruntime_version": "1.24.4",
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

    diagnostics = collect_diagnostics()

    assert diagnostics == {
        "stemma_version": "2.6.0",
        "onnxruntime_version": "1.24.4",
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
        "available_providers": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    })

    assert output.splitlines() == [
        "stemma version: 2.6.0",
        "ONNX Runtime version: 1.24.4",
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
