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

    diagnostics = collect_diagnostics()

    assert diagnostics == {
        "stemma_version": "2.6.0",
        "onnxruntime_version": "1.24.4",
        "available_providers": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
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
