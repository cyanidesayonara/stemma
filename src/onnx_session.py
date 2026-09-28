"""Shared ONNX Runtime session construction and provider reporting."""

import os

from src.import_messages import ModelDamagedError

_DML_PROVIDER = "DmlExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


def create_onnx_session(model_path: str):
    """Create a DML-first ONNX session with a safe CPU fallback."""
    # Deferred import: ONNX Runtime is heavy and is only needed for inference.
    import onnxruntime as ort

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"ONNX model file not found: {model_path}"
        )

    available = set(ort.get_available_providers())
    if _DML_PROVIDER in available:
        dml_options = ort.SessionOptions()
        dml_options.enable_mem_pattern = False
        dml_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            return ort.InferenceSession(
                model_path,
                sess_options=dml_options,
                providers=[_DML_PROVIDER, _CPU_PROVIDER],
            )
        except Exception:
            pass

    try:
        return ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=[_CPU_PROVIDER],
        )
    except Exception as exc:
        if is_damaged_model_error(exc):
            raise ModelDamagedError(model_path) from exc
        raise


# ONNX Runtime error types and codes that mean the file itself is bad
# (truncated, corrupt, or missing its external weights), not that the
# machine ran out of memory.
_DAMAGED_TYPES = frozenset({"InvalidProtobuf", "NoSuchFile", "InvalidGraph"})
_DAMAGED_TEXT = (
    "invalid_protobuf",
    "protobuf parsing failed",
    "no_suchfile",
    "external data",
)


def is_damaged_model_error(exc: BaseException) -> bool:
    """True if *exc* from ONNX Runtime means the model file is unusable."""
    if type(exc).__name__ in _DAMAGED_TYPES:
        return True
    low = str(exc).lower()
    return any(text in low for text in _DAMAGED_TEXT)


def selected_session_provider(session) -> str:
    """Return the provider selected first by an ONNX Runtime session."""
    providers = session.get_providers()
    return providers[0] if providers else ""


def session_provider_label(session) -> str:
    """Return a user-facing label for the active inference provider."""
    if selected_session_provider(session) == _DML_PROVIDER:
        return "DirectML GPU"
    return "CPU fallback"
