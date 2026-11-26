"""Tests for DirectML graph patching of HTDemucs ONNX models (issue #125)."""

from __future__ import annotations

import os
import shutil
import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from src import htdemucs_dml
from src.data_paths import platform_user_data_dir
from src.htdemucs_dml import (
    DML_SIDECAR_SUFFIX,
    PATCH_VERSION,
    dml_sidecar_path,
    ensure_dml_model,
    is_dml_sidecar,
    is_htdemucs_model_path,
    patch_onnx_bytes,
    prepare_htdemucs_dml_path,
)


# ---------------------------------------------------------------------------
# Minimal ONNX protobuf test builders (stdlib only, no onnx dependency)
# ---------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    if n < 0:
        n &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _vfield(field_no: int, val: int) -> bytes:
    return _varint(field_no << 3) + _varint(val)


def _bfield(field_no: int, payload: bytes) -> bytes:
    return _varint((field_no << 3) | 2) + _varint(len(payload)) + payload


def _attr_int(name: str, val: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2)
    return _bfield(1, name.encode()) + _vfield(3, val) + _vfield(20, 2)


def _attr_ints(name: str, vals: list[int]) -> bytes:
    # AttributeProto: name=1, ints=8, type=20 (INTS=7)
    out = bytearray(_bfield(1, name.encode()))
    for v in vals:
        out.extend(_vfield(8, v))
    out.extend(_vfield(20, 7))
    return bytes(out)


def _node(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    *,
    name: str = "",
    attrs: list[bytes] = (),
) -> bytes:
    out = bytearray()
    for inp in inputs:
        out.extend(_bfield(1, inp.encode()))
    for o in outputs:
        out.extend(_bfield(2, o.encode()))
    if name:
        out.extend(_bfield(3, name.encode()))
    out.extend(_bfield(4, op_type.encode()))
    for attr in attrs:
        out.extend(_bfield(5, attr))
    return bytes(out)


def _value_info(name: str, elem_type: int, dims: list[int | str | None]) -> bytes:
    shape = bytearray()
    for d in dims:
        dim = bytearray()
        if isinstance(d, int):
            dim.extend(_vfield(1, d))
        elif isinstance(d, str):
            dim.extend(_bfield(2, d.encode()))
        shape.extend(_bfield(1, bytes(dim)))
    tensor_type = _vfield(1, elem_type) + _bfield(2, bytes(shape))
    type_proto = _bfield(1, tensor_type)
    return _bfield(1, name.encode()) + _bfield(2, type_proto)


def _tensor_f32(name: str, arr: np.ndarray) -> bytes:
    out = bytearray()
    for d in arr.shape:
        out.extend(_vfield(1, int(d)))
    out.extend(_vfield(2, 1))  # FLOAT=1
    out.extend(_bfield(8, name.encode()))
    out.extend(_bfield(9, np.asarray(arr, dtype="<f4").tobytes()))
    return bytes(out)


def _tensor_i64(name: str, vals: list[int]) -> bytes:
    out = bytearray(_vfield(1, len(vals)) + _vfield(2, 7))  # INT64=7
    out.extend(_bfield(8, name.encode()))
    out.extend(_bfield(9, struct.pack(f"<{len(vals)}q", *vals)))
    return bytes(out)


def _tensor_external_f32(
    name: str,
    dims: tuple[int, ...],
    location: str,
    offset: int,
    length: int,
) -> bytes:
    out = bytearray()
    for d in dims:
        out.extend(_vfield(1, d))
    out.extend(_vfield(2, 1))  # FLOAT=1
    out.extend(_bfield(8, name.encode()))
    for k, v in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = _bfield(1, k.encode()) + _bfield(2, v.encode())
        out.extend(_bfield(13, entry))
    out.extend(_vfield(14, 1))  # EXTERNAL=1
    return bytes(out)


def _model(
    *,
    nodes: list[bytes],
    inputs: list[bytes],
    outputs: list[bytes],
    initializers: list[bytes] = (),
    value_infos: list[bytes] = (),
    opset: int = 18,
    ir_version: int = 9,
) -> bytes:
    graph = bytearray()
    for n in nodes:
        graph.extend(_bfield(1, n))
    graph.extend(_bfield(2, b"test_graph"))
    for init in initializers:
        graph.extend(_bfield(5, init))
    for vi in inputs:
        graph.extend(_bfield(11, vi))
    for vi in outputs:
        graph.extend(_bfield(12, vi))
    for vi in value_infos:
        graph.extend(_bfield(13, vi))

    opset_proto = _bfield(1, b"") + _vfield(2, opset)
    return (
        _vfield(1, ir_version)
        + _bfield(7, bytes(graph))
        + _bfield(8, opset_proto)
    )


def _initializer_names(raw_bytes: bytes) -> list[str]:
    names: list[str] = []
    buf = memoryview(raw_bytes)

    def read_varint(p: int) -> tuple[int, int]:
        val = 0
        shift = 0
        while True:
            b = buf[p]
            p += 1
            val |= (b & 0x7F) << shift
            if not (b & 0x80):
                return val, p
            shift += 7

    def fields(mv: memoryview):
        p = 0
        while p < len(mv):
            tag, p = read_varint(p)
            fno, wire = tag >> 3, tag & 0x07
            if wire == 0:
                val, p = read_varint(p)
                yield fno, wire, val
            elif wire == 1:
                yield fno, wire, bytes(mv[p : p + 8])
                p += 8
            elif wire == 2:
                ln, p = read_varint(p)
                yield fno, wire, mv[p : p + ln]
                p += ln
            elif wire == 5:
                yield fno, wire, bytes(mv[p : p + 4])
                p += 4

    for fno, wire, val in fields(buf):
        if fno == 7 and wire == 2:
            for gfno, gwire, gval in fields(val):
                if gfno == 5 and gwire == 2:
                    for tfno, twire, tval in fields(gval):
                        if tfno == 8 and twire == 2:
                            names.append(bytes(tval).decode())
    return names


# ---------------------------------------------------------------------------
# Path and model-detection helpers
# ---------------------------------------------------------------------------


def test_path_helpers():
    assert DML_SIDECAR_SUFFIX == ".dml.onnx"
    assert is_dml_sidecar("C:/models/htdemucs.dml.onnx") is True
    assert is_dml_sidecar("C:/models/htdemucs.onnx") is False
    assert is_dml_sidecar("C:/models/htdemucs.onnx.data") is False

    assert dml_sidecar_path("C:/models/htdemucs.onnx") == "C:/models/htdemucs.dml.onnx"
    assert dml_sidecar_path("C:/models/htdemucs.dml.onnx") == "C:/models/htdemucs.dml.onnx"
    assert dml_sidecar_path("C:/models/custom.bin") == "C:/models/custom.bin.dml.onnx"

    assert is_htdemucs_model_path("C:/models/htdemucs.onnx") is True
    assert is_htdemucs_model_path("/opt/models/htdemucs_6s.onnx") is True
    assert is_htdemucs_model_path("C:/models/htdemucs.dml.onnx") is False
    assert is_htdemucs_model_path("C:/models/beat_this.onnx") is False
    assert is_htdemucs_model_path("C:/models/UVR-MDX-NET-Inst_HQ_3.onnx") is False


# ---------------------------------------------------------------------------
# 1D ConvTranspose lifting + external data preservation
# ---------------------------------------------------------------------------


def test_lifts_1d_conv_transpose_to_2d_and_preserves_external_data(tmp_path):
    rng = np.random.default_rng(7)
    w_arr = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b_arr = rng.standard_normal((3,)).astype(np.float32)

    data_file = tmp_path / "weights.onnx.data"
    raw_weights = w_arr.astype("<f4").tobytes()
    data_file.write_bytes(raw_weights)

    raw_model = _model(
        nodes=[
            _node(
                "ConvTranspose",
                ["x", "w", "b"],
                ["y"],
                name="ct1d",
                attrs=[
                    _attr_ints("strides", [2]),
                    _attr_ints("pads", [1, 1]),
                    _attr_ints("output_padding", [0]),
                    _attr_ints("dilations", [1]),
                    _attr_ints("kernel_shape", [4]),
                    _attr_int("group", 1),
                ],
            )
        ],
        inputs=[_value_info("x", 1, [1, 2, 10])],
        outputs=[_value_info("y", 1, [1, 3, 20])],
        initializers=[
            _tensor_external_f32("w", (2, 3, 4), "weights.onnx.data", 0, len(raw_weights)),
            _tensor_f32("b", b_arr),
        ],
    )
    orig_path = tmp_path / "htdemucs.onnx"
    orig_path.write_bytes(raw_model)

    patched_bytes, changes = patch_onnx_bytes(raw_model, model_dir=tmp_path)
    assert changes == 1
    assert data_file.read_bytes() == raw_weights  # untouched external weights

    sidecar_path = tmp_path / "htdemucs.dml.onnx"
    sidecar_path.write_bytes(patched_bytes)

    x_in = rng.standard_normal((1, 2, 10)).astype(np.float32)
    cpu_orig = ort.InferenceSession(str(orig_path), providers=["CPUExecutionProvider"])
    cpu_patched = ort.InferenceSession(str(sidecar_path), providers=["CPUExecutionProvider"])
    ref_out = cpu_orig.run(None, {"x": x_in})[0]
    pat_out = cpu_patched.run(None, {"x": x_in})[0]
    np.testing.assert_allclose(pat_out, ref_out, atol=1e-6, rtol=1e-6)

    # Re-patching an already-lifted 2D ConvTranspose model is a no-op.
    _, changes_again = patch_onnx_bytes(patched_bytes, model_dir=tmp_path)
    assert changes_again == 0


def test_refuses_shared_1d_conv_transpose_weight():
    rng = np.random.default_rng(1)
    w_arr = rng.standard_normal((2, 2, 3)).astype(np.float32)
    raw_model = _model(
        nodes=[
            _node("ConvTranspose", ["x", "w"], ["y1"], name="ct"),
            _node("Conv", ["x", "w"], ["y2"], name="conv_sharing_w"),
        ],
        inputs=[_value_info("x", 1, [1, 2, 8])],
        outputs=[_value_info("y1", 1, [1, 2, 10]), _value_info("y2", 1, [1, 2, 6])],
        initializers=[_tensor_f32("w", w_arr)],
    )
    with pytest.raises(ValueError, match="shared"):
        patch_onnx_bytes(raw_model)


def test_refuses_opset_below_13_when_lifting_conv_transpose():
    w_arr = np.ones((2, 2, 3), dtype=np.float32)
    raw_model = _model(
        nodes=[_node("ConvTranspose", ["x", "w"], ["y"], name="ct")],
        inputs=[_value_info("x", 1, [1, 2, 8])],
        outputs=[_value_info("y", 1, [1, 2, 10])],
        initializers=[_tensor_f32("w", w_arr)],
        opset=12,
    )
    with pytest.raises(ValueError, match="opset"):
        patch_onnx_bytes(raw_model)


# ---------------------------------------------------------------------------
# Reshape(allowzero=1) attribute stripping & zero-dim refusal
# ---------------------------------------------------------------------------


def test_strips_reshape_allowzero_when_shape_has_no_zeros():
    raw_model = _model(
        nodes=[
            _node(
                "Reshape",
                ["x", "shape"],
                ["y"],
                name="r0",
                attrs=[_attr_int("allowzero", 1)],
            )
        ],
        inputs=[_value_info("x", 1, [1, 6])],
        outputs=[_value_info("y", 1, [2, 3])],
        initializers=[_tensor_i64("shape", [2, 3])],
    )
    patched_bytes, changes = patch_onnx_bytes(raw_model)
    assert changes == 1
    _, changes_again = patch_onnx_bytes(patched_bytes)
    assert changes_again == 0

    x_in = np.arange(6, dtype=np.float32).reshape(1, 6)
    s_orig = ort.InferenceSession(raw_model, providers=["CPUExecutionProvider"])
    s_pat = ort.InferenceSession(patched_bytes, providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(s_pat.run(None, {"x": x_in})[0], s_orig.run(None, {"x": x_in})[0])


def test_refuses_reshape_allowzero_with_zero_dim_or_dynamic_shape():
    zero_model = _model(
        nodes=[
            _node("Reshape", ["x", "shape"], ["y"], name="rz", attrs=[_attr_int("allowzero", 1)])
        ],
        inputs=[_value_info("x", 1, [0, 6])],
        outputs=[_value_info("y", 1, [0, 2, 3])],
        initializers=[_tensor_i64("shape", [0, 2, 3])],
    )
    with pytest.raises(ValueError, match="allowzero"):
        patch_onnx_bytes(zero_model)

    dynamic_model = _model(
        nodes=[
            _node("Reshape", ["x", "dyn_shape"], ["y"], name="rd", attrs=[_attr_int("allowzero", 1)])
        ],
        inputs=[_value_info("x", 1, [2, 3]), _value_info("dyn_shape", 7, [2])],
        outputs=[_value_info("y", 1, [3, 2])],
    )
    with pytest.raises(ValueError, match="allowzero"):
        patch_onnx_bytes(dynamic_model)


# ---------------------------------------------------------------------------
# Split -> Slice rewrite + negative axis + orphaned initializer pruning
# ---------------------------------------------------------------------------


def test_rewrites_split_with_explicit_sizes_and_prunes_initializer():
    raw_model = _model(
        nodes=[
            _node(
                "Split",
                ["x", "split_sizes"],
                ["a", "b"],
                name="sp",
                attrs=[_attr_int("axis", 1)],
            ),
            _node("Sigmoid", ["b"], ["sig"]),
            _node("Mul", ["a", "sig"], ["y"]),
        ],
        inputs=[_value_info("x", 1, [1, 6, 4])],
        outputs=[_value_info("y", 1, [1, 3, 4])],
        initializers=[_tensor_i64("split_sizes", [3, 3])],
    )
    patched_bytes, changes = patch_onnx_bytes(raw_model)
    assert changes == 1
    assert "split_sizes" not in _initializer_names(patched_bytes)

    x_in = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(1, 6, 4)
    s_orig = ort.InferenceSession(raw_model, providers=["CPUExecutionProvider"])
    s_pat = ort.InferenceSession(patched_bytes, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(s_pat.run(None, {"x": x_in})[0], s_orig.run(None, {"x": x_in})[0], atol=1e-6)


def test_rewrites_split_with_inferred_output_sizes_and_negative_axis():
    raw_model = _model(
        nodes=[
            _node(
                "Split",
                ["x"],
                ["a", "b"],
                name="sp_even",
                attrs=[_attr_int("axis", -1), _attr_int("num_outputs", 2)],
            ),
            _node("Add", ["a", "b"], ["y"]),
        ],
        inputs=[_value_info("x", 1, [1, 2, 6])],
        outputs=[_value_info("y", 1, [1, 2, 3])],
        value_infos=[
            _value_info("a", 1, [1, 2, 3]),
            _value_info("b", 1, [1, 2, 3]),
        ],
    )
    patched_bytes, changes = patch_onnx_bytes(raw_model)
    assert changes == 1

    x_in = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    s_orig = ort.InferenceSession(raw_model, providers=["CPUExecutionProvider"])
    s_pat = ort.InferenceSession(patched_bytes, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(s_pat.run(None, {"x": x_in})[0], s_orig.run(None, {"x": x_in})[0], atol=1e-6)


def test_refuses_split_without_known_sizes():
    raw_model = _model(
        nodes=[_node("Split", ["x"], ["a", "b"], name="sp_dyn", attrs=[_attr_int("axis", 1)])],
        inputs=[_value_info("x", 1, [1, "C", 4])],
        outputs=[_value_info("a", 1, [1, "C_half", 4]), _value_info("b", 1, [1, "C_half", 4])],
    )
    with pytest.raises(ValueError, match="Split"):
        patch_onnx_bytes(raw_model)


# ---------------------------------------------------------------------------
# Combined mini-graph, ensure_dml_model caching & fallbacks
# ---------------------------------------------------------------------------


def _build_combined_htdemucs_like_model() -> tuple[bytes, np.ndarray]:
    rng = np.random.default_rng(42)
    w = rng.standard_normal((4, 4, 3)).astype(np.float32)
    raw = _model(
        nodes=[
            _node(
                "ConvTranspose",
                ["x", "w"],
                ["ct_out"],
                name="ct",
                attrs=[_attr_ints("strides", [2]), _attr_ints("pads", [1, 1])],
            ),
            _node(
                "Reshape",
                ["ct_out", "shape"],
                ["rs_out"],
                name="rs",
                attrs=[_attr_int("allowzero", 1)],
            ),
            _node(
                "Split",
                ["rs_out", "splits"],
                ["half0", "half1"],
                name="sp",
                attrs=[_attr_int("axis", 1)],
            ),
            _node("Sigmoid", ["half1"], ["gate"]),
            _node("Mul", ["half0", "gate"], ["y"]),
        ],
        inputs=[_value_info("x", 1, [1, 4, 8])],
        outputs=[_value_info("y", 1, [1, 2, 15])],
        initializers=[
            _tensor_f32("w", w),
            _tensor_i64("shape", [1, 4, 15]),
            _tensor_i64("splits", [2, 2]),
        ],
    )
    x_in = rng.standard_normal((1, 4, 8)).astype(np.float32)
    return raw, x_in


def test_combined_graph_parity_and_dml_if_available(tmp_path):
    raw_model, x_in = _build_combined_htdemucs_like_model()
    orig_path = tmp_path / "htdemucs.onnx"
    orig_path.write_bytes(raw_model)

    sidecar = ensure_dml_model(str(orig_path))
    assert sidecar == str(tmp_path / "htdemucs.dml.onnx")
    assert os.path.isfile(sidecar)

    cpu_orig = ort.InferenceSession(str(orig_path), providers=["CPUExecutionProvider"])
    ref_out = cpu_orig.run(None, {"x": x_in})[0]

    cpu_sidecar = ort.InferenceSession(sidecar, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(cpu_sidecar.run(None, {"x": x_in})[0], ref_out, atol=1e-6, rtol=1e-6)

    if "DmlExecutionProvider" in ort.get_available_providers():
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        dml_sess = ort.InferenceSession(sidecar, sess_options=opts, providers=["DmlExecutionProvider"])
        dml_out = dml_sess.run(None, {"x": x_in})[0]
        np.testing.assert_allclose(dml_out, ref_out, atol=1e-4, rtol=1e-4)


def test_noop_graph_skips_sidecar(tmp_path):
    raw_model = _model(
        nodes=[_node("Relu", ["x"], ["y"], name="r")],
        inputs=[_value_info("x", 1, [1, 4])],
        outputs=[_value_info("y", 1, [1, 4])],
    )
    src = tmp_path / "htdemucs.onnx"
    src.write_bytes(raw_model)

    selected = ensure_dml_model(str(src))
    assert selected == str(src)
    assert not (tmp_path / "htdemucs.dml.onnx").exists()


def test_ensure_dml_model_caches_and_invalidates_on_source_or_version_change(tmp_path, monkeypatch):
    raw1, _ = _build_combined_htdemucs_like_model()
    src = tmp_path / "htdemucs.onnx"
    src.write_bytes(raw1)

    sidecar = ensure_dml_model(str(src))
    assert os.path.isfile(sidecar)

    # Cache hit: patch_onnx_bytes must not run again when source and version match.
    real_patch = htdemucs_dml.patch_onnx_bytes
    calls = 0

    def counting_patch(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_patch(*args, **kwargs)

    monkeypatch.setattr(htdemucs_dml, "patch_onnx_bytes", counting_patch)
    assert ensure_dml_model(str(src)) == sidecar
    assert calls == 0

    # Bumping PATCH_VERSION invalidates the cached sidecar.
    monkeypatch.setattr(htdemucs_dml, "PATCH_VERSION", PATCH_VERSION + 1)
    assert ensure_dml_model(str(src)) == sidecar
    assert calls == 1

    # Modifying source bytes (even preserving mtime) invalidates via source SHA-256.
    stat_before = src.stat()
    raw2 = _model(
        nodes=[
            _node(
                "Reshape",
                ["x", "shape"],
                ["y"],
                name="rs_only",
                attrs=[_attr_int("allowzero", 1)],
            )
        ],
        inputs=[_value_info("x", 1, [1, 4, 8])],
        outputs=[_value_info("y", 1, [1, 8, 4])],
        initializers=[_tensor_i64("shape", [1, 8, 4])],
    )
    src.write_bytes(raw2)
    os.utime(src, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))

    assert ensure_dml_model(str(src)) == sidecar
    assert calls == 2


def test_ensure_dml_model_fallbacks_on_corrupt_or_missing_file(tmp_path):
    missing = str(tmp_path / "missing.onnx")
    assert ensure_dml_model(missing) == missing

    # Corrupt bytes fallback cleanly
    corrupt = tmp_path / "htdemucs.onnx"
    corrupt.write_bytes(b"not-a-valid-onnx-protobuf-\xff\xff")
    assert ensure_dml_model(str(corrupt)) == str(corrupt)
    assert not (tmp_path / "htdemucs.dml.onnx").exists()


def test_lifts_1d_conv_transpose_when_weight_is_also_in_graph_input(tmp_path):
    """Legacy exporters duplicate initializers in graph.input; keep ORT load clean."""
    rng = np.random.default_rng(19)
    w_arr = rng.standard_normal((2, 3, 4)).astype(np.float32)
    raw_model = _model(
        nodes=[
            _node(
                "ConvTranspose",
                ["x", "w"],
                ["y"],
                name="ct1d",
                attrs=[_attr_ints("strides", [2]), _attr_ints("pads", [1, 1])],
            )
        ],
        inputs=[_value_info("x", 1, [1, 2, 6]), _value_info("w", 1, [2, 3, 4])],
        outputs=[_value_info("y", 1, [1, 3, 12])],
        initializers=[_tensor_f32("w", w_arr)],
    )
    patched, changes = patch_onnx_bytes(raw_model)
    assert changes == 1
    x_in = rng.standard_normal((1, 2, 6)).astype(np.float32)
    orig_sess = ort.InferenceSession(raw_model, providers=["CPUExecutionProvider"])
    pat_sess = ort.InferenceSession(patched, providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        pat_sess.run(None, {"x": x_in})[0],
        orig_sess.run(None, {"x": x_in})[0],
        atol=1e-6,
    )


def test_ensure_dml_model_logs_to_stemma_and_catches_struct_errors(
    tmp_path, monkeypatch, caplog,
):
    """Patch errors (including struct.error) log under 'stemma' and fall back safely."""
    import logging

    src = tmp_path / "htdemucs.onnx"
    src.write_bytes(b"valid-looking-bytes")

    def broken_patch(*_args, **_kwargs):
        raise struct.error("bad packed tensor")

    monkeypatch.setattr(htdemucs_dml, "patch_onnx_bytes", broken_patch)
    with caplog.at_level(logging.WARNING, logger="stemma"):
        assert ensure_dml_model(str(src)) == str(src)
    assert any("Failed to build DirectML sidecar" in r.message for r in caplog.records)


def test_prepare_htdemucs_dml_path_checks_provider_and_model_name(tmp_path, monkeypatch):
    raw_model, _ = _build_combined_htdemucs_like_model()
    ht_path = tmp_path / "htdemucs.onnx"
    other_path = tmp_path / "beat_this.onnx"
    ht_path.write_bytes(raw_model)
    other_path.write_bytes(raw_model)

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert prepare_htdemucs_dml_path(str(ht_path)) == str(ht_path)
    assert not (tmp_path / "htdemucs.dml.onnx").exists()

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    assert prepare_htdemucs_dml_path(str(other_path)) == str(other_path)
    assert not (tmp_path / "beat_this.dml.onnx").exists()

    prepared = prepare_htdemucs_dml_path(str(ht_path))
    assert prepared == str(tmp_path / "htdemucs.dml.onnx")
    assert os.path.isfile(prepared)


# ---------------------------------------------------------------------------
# Slow real-model DirectML integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["htdemucs.onnx", "htdemucs_6s.onnx"])
def test_real_htdemucs_models_load_and_match_cpu_on_directml(tmp_path, model_name):
    if "DmlExecutionProvider" not in ort.get_available_providers():
        pytest.skip("DmlExecutionProvider is not available on this host")

    models_dir = Path(platform_user_data_dir()) / "models"
    src_onnx = models_dir / model_name
    src_data = models_dir / f"{model_name}.data"
    if not (src_onnx.is_file() and src_data.is_file()):
        pytest.skip(f"Cached model {model_name} not found in {models_dir}")

    # Copy/link into tmp_path so the user's cached models directory is untouched.
    dst_onnx = tmp_path / model_name
    dst_data = tmp_path / f"{model_name}.data"
    shutil.copy2(src_onnx, dst_onnx)
    try:
        os.link(src_data, dst_data)
    except OSError:
        shutil.copy2(src_data, dst_data)

    sidecar = ensure_dml_model(str(dst_onnx))
    assert is_dml_sidecar(sidecar)

    rng = np.random.default_rng(123)
    feeds = {
        "input": (rng.standard_normal((1, 2, 343980)) * 0.1).astype(np.float32),
        "x": (rng.standard_normal((1, 4, 2048, 336)) * 0.1).astype(np.float32),
    }

    cpu_sess = ort.InferenceSession(str(dst_onnx), providers=["CPUExecutionProvider"])
    cpu_outs = cpu_sess.run(None, feeds)
    del cpu_sess

    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    dml_sess = ort.InferenceSession(
        sidecar,
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert "DmlExecutionProvider" in dml_sess.get_providers()
    dml_outs = dml_sess.run(None, feeds)

    for cpu_out, dml_out in zip(cpu_outs, dml_outs):
        assert not (np.isnan(dml_out).any() or np.isinf(dml_out).any())
        diff = dml_out.astype(np.float64) - cpu_out.astype(np.float64)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        ref_rms = float(np.sqrt(np.mean(cpu_out.astype(np.float64) ** 2)))
        snr_db = float(20.0 * np.log10(ref_rms / rmse)) if rmse > 0 else float("inf")
        denom = float(np.linalg.norm(cpu_out) * np.linalg.norm(dml_out))
        cos = float(np.dot(cpu_out.ravel().astype(np.float64), dml_out.ravel().astype(np.float64)) / denom)
        assert snr_db >= 60.0
        assert cos >= 0.99999
