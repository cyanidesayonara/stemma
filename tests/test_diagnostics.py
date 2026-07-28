"""Tests for release smoke diagnostics."""

import sys
from types import SimpleNamespace


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
        "stemma_version": "2.2.0",
        "onnxruntime_version": "1.24.4",
        "available_providers": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    }


def test_format_diagnostics_is_human_readable():
    from src.diagnostics import format_diagnostics

    output = format_diagnostics({
        "stemma_version": "2.2.0",
        "onnxruntime_version": "1.24.4",
        "available_providers": [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    })

    assert output.splitlines() == [
        "stemma version: 2.2.0",
        "ONNX Runtime version: 1.24.4",
        "Available ONNX providers: "
        "DmlExecutionProvider, CPUExecutionProvider",
    ]


def test_diagnostics_request_requires_explicit_flag():
    from src.diagnostics import diagnostics_requested

    assert diagnostics_requested(["stemma.exe", "--diagnostics"])
    assert not diagnostics_requested(["stemma.exe"])
