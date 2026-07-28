"""Tests for shared ONNX Runtime session construction."""

import sys
from types import SimpleNamespace


class _SessionOptions:
    def __init__(self):
        self.enable_mem_pattern = True
        self.execution_mode = None


def _fake_ort(available_providers, fail_dml=False):
    calls = []
    sequential = object()

    def inference_session(model_path, **kwargs):
        calls.append((model_path, kwargs))
        if fail_dml and kwargs["providers"][0] == "DmlExecutionProvider":
            raise RuntimeError("DML initialization failed")
        return SimpleNamespace(
            get_providers=lambda: list(kwargs["providers"]),
        )

    ort = SimpleNamespace(
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=sequential),
        InferenceSession=inference_session,
        SessionOptions=_SessionOptions,
        get_available_providers=lambda: list(available_providers),
    )
    return ort, calls, sequential


def test_dml_session_uses_required_options_and_provider_order(
    monkeypatch, tmp_path,
):
    from src.onnx_session import create_onnx_session

    ort, calls, sequential = _fake_ort(
        ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    session = create_onnx_session(str(model_path))

    assert session.get_providers()[0] == "DmlExecutionProvider"
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["providers"] == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert kwargs["sess_options"].enable_mem_pattern is False
    assert kwargs["sess_options"].execution_mode is sequential


def test_dml_initialization_failure_retries_with_cpu(monkeypatch, tmp_path):
    from src.onnx_session import create_onnx_session

    ort, calls, _ = _fake_ort(
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        fail_dml=True,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    session = create_onnx_session(str(model_path))

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert [call[1]["providers"] for call in calls] == [
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_cpu_session_is_used_when_dml_is_unavailable(monkeypatch, tmp_path):
    from src.onnx_session import create_onnx_session

    ort, calls, _ = _fake_ort(["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    session = create_onnx_session(str(model_path))

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert [call[1]["providers"] for call in calls] == [
        ["CPUExecutionProvider"],
    ]


def test_selected_provider_and_user_label_are_reported():
    from src.onnx_session import (
        selected_session_provider,
        session_provider_label,
    )

    dml_session = SimpleNamespace(
        get_providers=lambda: [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    cpu_session = SimpleNamespace(
        get_providers=lambda: ["CPUExecutionProvider"],
    )

    assert selected_session_provider(dml_session) == "DmlExecutionProvider"
    assert session_provider_label(dml_session) == "DirectML GPU"
    assert selected_session_provider(cpu_session) == "CPUExecutionProvider"
    assert session_provider_label(cpu_session) == "CPU fallback"
