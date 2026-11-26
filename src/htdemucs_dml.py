"""DirectML graph compatibility patcher for HTDemucs ONNX models.

The upstream HTDemucs v4 ONNX exports contain three graph constructs that
``DmlExecutionProvider`` in ONNX Runtime either rejects at compile time or
evaluates inaccurately:

1. Four 1D ``ConvTranspose`` layers (``tdecoder.0..3.conv_tr``), which fail
   DirectML validation with ``E_INVALIDARG`` (``0x80070057``). Lifting them
   to 2D ``ConvTranspose`` (via ``Unsqueeze(-1)`` / ``Squeeze(-1)``) with
   weight shape ``(C_in, C_out, K, 1)`` preserves external ``.onnx.data``
   offsets untouched.
2. Output ``Reshape`` nodes with ``allowzero=1``, which DirectML also rejects
   with ``0x80070057``. Stripping ``allowzero`` is exact when the target
   shape contains no ``0`` dimensions.
3. Internal ``Split`` nodes (most visibly in ``Split(axis=1) -> Sigmoid ->
   Mul`` GLU blocks), whose secondary DirectML output buffers suffer silent
   data corruption. Rewriting each ``Split`` to opset-18 ``Slice`` nodes with
   explicit constant bounds removes the corruption.

This module applies all three transformations directly on the ONNX protobuf
wire format (zero external dependencies beyond the standard library) and
caches a compact ``.dml.onnx`` sidecar next to the original model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import tempfile
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Bump whenever the graph rewrite logic changes so cached sidecars rebuild.
PATCH_VERSION = 1

DML_SIDECAR_SUFFIX = ".dml.onnx"
HTDEMUCS_MODEL_FILES = frozenset({"htdemucs.onnx", "htdemucs_6s.onnx"})

_META_PATCH_VERSION = "stemma_dml_patch_version"
_META_SOURCE_SHA256 = "stemma_dml_source_sha256"
_DEFAULT_DOMAINS = frozenset({"", "ai.onnx"})


def is_dml_sidecar(path: str | os.PathLike[str]) -> bool:
    """Return True if *path* names a generated DirectML sidecar graph."""
    return os.fspath(path).endswith(DML_SIDECAR_SUFFIX)


def dml_sidecar_path(model_path: str | os.PathLike[str]) -> str:
    """Return the companion ``.dml.onnx`` path for *model_path*."""
    path_str = os.fspath(model_path)
    if path_str.endswith(DML_SIDECAR_SUFFIX):
        return path_str
    if path_str.endswith(".onnx"):
        return path_str[:-5] + DML_SIDECAR_SUFFIX
    return path_str + DML_SIDECAR_SUFFIX


def is_htdemucs_model_path(model_path: str | os.PathLike[str]) -> bool:
    """Return True if *model_path* points to a canonical HTDemucs ONNX file."""
    return os.path.basename(os.fspath(model_path)) in HTDEMUCS_MODEL_FILES


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format helpers
# ---------------------------------------------------------------------------


def _read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    end = len(buf)
    while pos < end:
        byte = buf[pos]
        pos += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, pos
        shift += 7
        if shift > 70:
            raise ValueError("Malformed protobuf varint")
    raise ValueError("Truncated protobuf varint")


def _encode_varint(n: int) -> bytes:
    if n < 0:
        n &= 0xFFFFFFFFFFFFFFFF
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_varint_field(field_no: int, val: int) -> bytes:
    return _encode_varint(field_no << 3) + _encode_varint(val)


def _encode_bytes_field(field_no: int, payload: bytes | memoryview) -> bytes:
    return _encode_varint((field_no << 3) | 2) + _encode_varint(len(payload)) + bytes(payload)


def _encode_raw_field(field_no: int, wire: int, val: object) -> bytes:
    if wire == 0:
        return _encode_varint_field(field_no, int(val))  # type: ignore[arg-type]
    if wire == 2:
        return _encode_bytes_field(field_no, val)  # type: ignore[arg-type]
    if wire in (1, 5):
        return _encode_varint((field_no << 3) | wire) + bytes(val)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported protobuf wire type {wire}")


def _iter_fields(buf: memoryview):
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_no = tag >> 3
        wire = tag & 0x07
        if field_no == 0:
            raise ValueError("Malformed protobuf field number 0")
        if wire == 0:
            val, pos = _read_varint(buf, pos)
            yield field_no, wire, val
        elif wire == 1:
            if pos + 8 > end:
                raise ValueError("Truncated fixed64 field")
            val = bytes(buf[pos : pos + 8])
            pos += 8
            yield field_no, wire, val
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            if pos + length > end:
                raise ValueError("Truncated length-delimited field")
            val = buf[pos : pos + length]
            pos += length
            yield field_no, wire, val
        elif wire == 5:
            if pos + 4 > end:
                raise ValueError("Truncated fixed32 field")
            val = bytes(buf[pos : pos + 4])
            pos += 4
            yield field_no, wire, val
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")


def _to_signed64(val: int) -> int:
    return val - (1 << 64) if val >= (1 << 63) else val


# ---------------------------------------------------------------------------
# ONNX message parsers and builders
# ---------------------------------------------------------------------------


def _parse_attr(buf: memoryview) -> tuple[str, object]:
    name = ""
    atype = 0
    val_out: object = None
    ints: list[int] = []
    floats: list[float] = []
    for fno, wire, val in _iter_fields(buf):
        if fno == 1 and wire == 2:
            name = bytes(val).decode("utf-8", "replace")
        elif fno == 2 and wire == 5:
            val_out = struct.unpack("<f", val)[0]
        elif fno == 3 and wire == 0:
            val_out = _to_signed64(val)
        elif fno == 4 and wire == 2:
            val_out = bytes(val).decode("utf-8", "replace")
        elif fno == 7:
            if wire == 5:
                floats.append(struct.unpack("<f", val)[0])
            elif wire == 2:
                mv = memoryview(val)
                for off in range(0, len(mv), 4):
                    floats.append(struct.unpack("<f", mv[off : off + 4])[0])
        elif fno == 8:
            if wire == 0:
                ints.append(_to_signed64(val))
            elif wire == 2:
                pos = 0
                mv = memoryview(val)
                while pos < len(mv):
                    iv, pos = _read_varint(mv, pos)
                    ints.append(_to_signed64(iv))
        elif fno == 20 and wire == 0:
            atype = val
    if ints:
        val_out = tuple(ints)
    elif floats:
        val_out = tuple(floats)
    elif val_out is None and atype:
        val_out = f"<type={atype}>"
    return name, val_out


def _encode_ints_attr(name: str, values: tuple[int, ...] | list[int]) -> bytes:
    out = bytearray(_encode_bytes_field(1, name.encode("utf-8")))
    for v in values:
        out.extend(_encode_varint_field(8, v))
    out.extend(_encode_varint_field(20, 7))  # AttributeProto.INTS = 7
    return bytes(out)


def _parse_node(raw: memoryview) -> dict:
    inputs: list[str] = []
    outputs: list[str] = []
    attrs: dict[str, object] = {}
    name = ""
    op_type = ""
    domain = ""
    for fno, wire, val in _iter_fields(raw):
        if fno == 1 and wire == 2:
            inputs.append(bytes(val).decode("utf-8", "replace"))
        elif fno == 2 and wire == 2:
            outputs.append(bytes(val).decode("utf-8", "replace"))
        elif fno == 3 and wire == 2:
            name = bytes(val).decode("utf-8", "replace")
        elif fno == 4 and wire == 2:
            op_type = bytes(val).decode("utf-8", "replace")
        elif fno == 5 and wire == 2:
            k, v = _parse_attr(val)
            attrs[k] = v
        elif fno == 7 and wire == 2:
            domain = bytes(val).decode("utf-8", "replace")
    return {
        "raw": bytes(raw),
        "name": name,
        "op_type": op_type,
        "domain": domain,
        "inputs": inputs,
        "outputs": outputs,
        "attrs": attrs,
    }


def _parse_string_pair(raw: memoryview) -> tuple[str, str]:
    key = ""
    value = ""
    for fno, wire, val in _iter_fields(raw):
        if fno == 1 and wire == 2:
            key = bytes(val).decode("utf-8", "replace")
        elif fno == 2 and wire == 2:
            value = bytes(val).decode("utf-8", "replace")
    return key, value


def _encode_string_pair(key: str, value: str) -> bytes:
    return (
        _encode_bytes_field(1, key.encode("utf-8"))
        + _encode_bytes_field(2, value.encode("utf-8"))
    )


def _parse_tensor(raw: memoryview) -> dict:
    name = ""
    dims: list[int] = []
    dtype = 0
    int64_data: list[int] = []
    raw_data = b""
    ext: dict[str, str] = {}
    data_location = 0
    for fno, wire, val in _iter_fields(raw):
        if fno == 1:
            if wire == 0:
                dims.append(_to_signed64(val))
            elif wire == 2:
                pos = 0
                mv = memoryview(val)
                while pos < len(mv):
                    dv, pos = _read_varint(mv, pos)
                    dims.append(_to_signed64(dv))
        elif fno == 2 and wire == 0:
            dtype = val
        elif fno == 7:
            if wire == 0:
                int64_data.append(_to_signed64(val))
            elif wire == 2:
                pos = 0
                mv = memoryview(val)
                while pos < len(mv):
                    iv, pos = _read_varint(mv, pos)
                    int64_data.append(_to_signed64(iv))
        elif fno == 8 and wire == 2:
            name = bytes(val).decode("utf-8", "replace")
        elif fno == 9 and wire == 2:
            raw_data = bytes(val)
        elif fno == 13 and wire == 2:
            ek, ev = _parse_string_pair(val)
            if ek:
                ext[ek] = ev
        elif fno == 14 and wire == 0:
            data_location = val
    return {
        "raw": bytes(raw),
        "name": name,
        "dims": tuple(dims),
        "dtype": dtype,
        "int64_data": int64_data,
        "raw_data": raw_data,
        "ext": ext,
        "data_location": data_location,
    }


def _parse_value_info(raw: memoryview) -> dict:
    name = ""
    dims: list[int | str | None] = []
    for fno, wire, val in _iter_fields(raw):
        if fno == 1 and wire == 2:
            name = bytes(val).decode("utf-8", "replace")
        elif fno == 2 and wire == 2:
            for tfno, twire, tval in _iter_fields(val):
                if tfno != 1 or twire != 2:
                    continue
                for sfno, swire, sval in _iter_fields(tval):
                    if sfno != 2 or swire != 2:
                        continue
                    for dfno, dwire, dval in _iter_fields(sval):
                        if dfno != 1 or dwire != 2:
                            continue
                        dim_val: int | str | None = None
                        for dim_fno, dim_wire, dim_v in _iter_fields(dval):
                            if dim_fno == 1 and dim_wire == 0:
                                dim_val = _to_signed64(dim_v)
                            elif dim_fno == 2 and dim_wire == 2:
                                dim_val = bytes(dim_v).decode("utf-8", "replace")
                        dims.append(dim_val)
    return {"name": name, "dims": tuple(dims)}


def _parse_opset(raw: memoryview) -> tuple[str, int]:
    domain = ""
    version = 0
    for fno, wire, val in _iter_fields(raw):
        if fno == 1 and wire == 2:
            domain = bytes(val).decode("utf-8", "replace")
        elif fno == 2 and wire == 0:
            version = val
    return domain, version


def _read_int64_tensor_values(tensor: dict, model_dir: str | None) -> list[int] | None:
    if tensor["dtype"] != 7:  # TensorProto.INT64
        return None
    if tensor["int64_data"]:
        return list(tensor["int64_data"])
    raw = tensor["raw_data"]
    if not raw and tensor["data_location"] == 1 and "location" in tensor["ext"]:
        if not model_dir:
            return None
        base_dir = os.path.abspath(model_dir)
        ext_path = os.path.abspath(os.path.join(base_dir, tensor["ext"]["location"]))
        if os.path.commonpath([base_dir, ext_path]) != base_dir or not os.path.isfile(ext_path):
            raise ValueError(f"Invalid external tensor path {ext_path!r}")
        offset = int(tensor["ext"].get("offset", "0"))
        length = int(tensor["ext"].get("length", "0"))
        with open(ext_path, "rb") as f:
            f.seek(offset)
            raw = f.read(length) if length else f.read()
    if not raw or len(raw) % 8 != 0:
        return None
    count = len(raw) // 8
    return list(struct.unpack(f"<{count}q", raw))


def _rewrite_tensor_dims(raw_tensor: bytes, new_dims: tuple[int, ...]) -> bytes:
    out = bytearray()
    dims_written = False
    for fno, wire, val in _iter_fields(memoryview(raw_tensor)):
        if fno == 1:
            if not dims_written:
                for dim in new_dims:
                    out.extend(_encode_varint_field(1, dim))
                dims_written = True
            continue
        out.extend(_encode_raw_field(fno, wire, val))
    if not dims_written:
        for dim in new_dims:
            out.extend(_encode_varint_field(1, dim))
    return bytes(out)


def _make_simple_node(
    op_type: str,
    name: str,
    inputs: list[str],
    outputs: list[str],
) -> bytes:
    out = bytearray()
    for inp in inputs:
        out.extend(_encode_bytes_field(1, inp.encode("utf-8")))
    for output in outputs:
        out.extend(_encode_bytes_field(2, output.encode("utf-8")))
    out.extend(_encode_bytes_field(3, name.encode("utf-8")))
    out.extend(_encode_bytes_field(4, op_type.encode("utf-8")))
    return bytes(out)


def _make_int64_initializer(name: str, values: list[int]) -> bytes:
    out = bytearray()
    out.extend(_encode_varint_field(1, len(values)))
    out.extend(_encode_varint_field(2, 7))  # TensorProto.INT64 = 7
    out.extend(_encode_bytes_field(8, name.encode("utf-8")))
    out.extend(_encode_bytes_field(9, struct.pack(f"<{len(values)}q", *values)))
    return bytes(out)


def _strip_attribute(raw_node: bytes, attr_name: str) -> bytes:
    out = bytearray()
    for fno, wire, val in _iter_fields(memoryview(raw_node)):
        if fno == 5 and wire == 2:
            k, _ = _parse_attr(val)
            if k == attr_name:
                continue
        out.extend(_encode_raw_field(fno, wire, val))
    return bytes(out)


def _unique_name(used: set[str], candidate: str) -> str:
    name = candidate
    suffix = 0
    while name in used:
        suffix += 1
        name = f"{candidate}_{suffix}"
    used.add(name)
    return name


# ---------------------------------------------------------------------------
# Graph rewrite rules
# ---------------------------------------------------------------------------


def _lift_conv_transpose_1d_to_2d(
    node: dict,
    idx: int,
    axis_init: str,
    used_names: set[str],
) -> tuple[list[bytes], list[str]]:
    base = node["name"] or f"conv_tr_{idx}"
    orig_x = node["inputs"][0]
    orig_y = node["outputs"][0]
    in4d = _unique_name(used_names, f"{base}__dml_in4d")
    out4d = _unique_name(used_names, f"{base}__dml_out4d")

    unsq_inputs = [orig_x, axis_init]
    unsq = _make_simple_node(
        "Unsqueeze",
        _unique_name(used_names, f"{base}__unsqueeze"),
        unsq_inputs,
        [in4d],
    )
    sq_inputs = [out4d, axis_init]
    sq = _make_simple_node(
        "Squeeze",
        _unique_name(used_names, f"{base}__squeeze"),
        sq_inputs,
        [orig_y],
    )

    ct_inputs = [in4d, *node["inputs"][1:]]
    ct = bytearray()
    for inp in ct_inputs:
        ct.extend(_encode_bytes_field(1, inp.encode("utf-8")))
    ct.extend(_encode_bytes_field(2, out4d.encode("utf-8")))
    ct.extend(_encode_bytes_field(3, (node["name"] or base).encode("utf-8")))
    ct.extend(_encode_bytes_field(4, b"ConvTranspose"))
    if node["domain"]:
        ct.extend(_encode_bytes_field(7, node["domain"].encode("utf-8")))

    for fno, wire, val in _iter_fields(memoryview(node["raw"])):
        if fno != 5 or wire != 2:
            continue
        k, v = _parse_attr(val)
        if k == "strides":
            s0 = v[0] if isinstance(v, tuple) else int(v)  # type: ignore[arg-type]
            ct.extend(_encode_bytes_field(5, _encode_ints_attr("strides", (s0, 1))))
        elif k == "pads":
            if isinstance(v, tuple) and len(v) == 2:
                ct.extend(_encode_bytes_field(5, _encode_ints_attr("pads", (v[0], 0, v[1], 0))))
            else:
                ct.extend(_encode_bytes_field(5, val))
        elif k == "output_padding":
            op0 = v[0] if isinstance(v, tuple) else int(v)  # type: ignore[arg-type]
            ct.extend(_encode_bytes_field(5, _encode_ints_attr("output_padding", (op0, 0))))
        elif k == "dilations":
            d0 = v[0] if isinstance(v, tuple) else int(v)  # type: ignore[arg-type]
            ct.extend(_encode_bytes_field(5, _encode_ints_attr("dilations", (d0, 1))))
        elif k == "kernel_shape" and isinstance(v, tuple) and len(v) == 1:
            ct.extend(_encode_bytes_field(5, _encode_ints_attr("kernel_shape", (v[0], 1))))
        else:
            ct.extend(_encode_bytes_field(5, val))

    return [unsq, bytes(ct), sq], [*unsq_inputs, *ct_inputs, *sq_inputs]


def _resolve_split_sizes(
    node: dict,
    axis: int,
    inits: dict[str, dict],
    vi_map: dict[str, dict],
    model_dir: str | None,
) -> tuple[list[int], str | None]:
    dropped_init: str | None = None
    inputs = node["inputs"]
    if len(inputs) >= 2 and inputs[1]:
        split_name = inputs[1]
        if split_name not in inits:
            raise ValueError(
                f"Split node {node['name']!r} uses non-constant split sizes {split_name!r}"
            )
        sizes = _read_int64_tensor_values(inits[split_name], model_dir)
        if sizes is None:
            raise ValueError(
                f"Split node {node['name']!r} has unreadable split tensor {split_name!r}"
            )
        dropped_init = split_name
    elif isinstance(node["attrs"].get("split"), tuple):
        sizes = [int(x) for x in node["attrs"]["split"]]  # type: ignore[union-attr]
    else:
        sizes = []
        for out_name in node["outputs"]:
            if not out_name or out_name not in vi_map:
                raise ValueError(
                    f"Split node {node['name']!r} has missing output shape metadata for {out_name!r}"
                )
            dims = vi_map[out_name]["dims"]
            rank = len(dims)
            norm_axis = axis + rank if axis < 0 else axis
            if norm_axis < 0 or norm_axis >= rank:
                raise ValueError(
                    f"Split node {node['name']!r} axis {axis} out of range for rank {rank}"
                )
            dim = dims[norm_axis]
            if not isinstance(dim, int) or dim < 0:
                raise ValueError(
                    f"Split node {node['name']!r} has non-static output dim {dim!r}"
                )
            sizes.append(dim)

    if len(sizes) != len(node["outputs"]):
        raise ValueError(
            f"Split node {node['name']!r} size count ({len(sizes)}) "
            f"!= outputs ({len(node['outputs'])})"
        )
    return sizes, dropped_init


def _plan_rewrites(
    nodes: list[dict],
    inits: dict[str, dict],
    vi_map: dict[str, dict],
    default_opset: int | None,
    model_dir: str | None,
) -> tuple[set[int], set[int], set[str], dict[int, tuple[int, list[int]]], set[str]]:
    reshape_indices: set[int] = set()
    conv_tr_indices: set[int] = set()
    weights_to_4d: set[str] = set()
    split_specs: dict[int, tuple[int, list[int]]] = {}
    dropped_inits: set[str] = set()

    for idx, node in enumerate(nodes):
        if node["domain"] not in _DEFAULT_DOMAINS:
            continue
        op = node["op_type"]
        if op == "Reshape" and int(node["attrs"].get("allowzero", 0)) != 0:  # type: ignore[call-overload]
            shape_name = node["inputs"][1] if len(node["inputs"]) >= 2 else ""
            shape_vals = (
                _read_int64_tensor_values(inits[shape_name], model_dir)
                if shape_name in inits
                else None
            )
            if shape_vals is None or 0 in shape_vals:
                raise ValueError(
                    f"Reshape {node['name']!r} has allowzero!=0 with zero or dynamic target shape"
                )
            reshape_indices.add(idx)
        elif op == "ConvTranspose" and len(node["inputs"]) >= 2:
            wname = node["inputs"][1]
            if wname in inits and len(inits[wname]["dims"]) == 3:
                conv_tr_indices.add(idx)
                weights_to_4d.add(wname)
        elif op == "Split":
            axis = int(node["attrs"].get("axis", 0))  # type: ignore[call-overload]
            sizes, dropped_init = _resolve_split_sizes(
                node, axis, inits, vi_map, model_dir
            )
            split_specs[idx] = (axis, sizes)
            if dropped_init:
                dropped_inits.add(dropped_init)

    if conv_tr_indices and default_opset is not None and default_opset < 13:
        raise ValueError(
            f"Unsupported ONNX opset {default_opset}; opset >= 13 is required for 1D ConvTranspose lifting"
        )

    if weights_to_4d:
        for idx, node in enumerate(nodes):
            if idx in conv_tr_indices:
                continue
            for inp in node["inputs"]:
                if inp in weights_to_4d:
                    raise ValueError(
                        f"1D ConvTranspose weight {inp!r} is shared with node {node['name'] or node['op_type']!r}"
                    )

    return reshape_indices, conv_tr_indices, weights_to_4d, split_specs, dropped_inits


def _expand_split_to_slices(
    node: dict,
    idx: int,
    axis: int,
    sizes: list[int],
    ensure_i64: Callable[[str, list[int]], str],
    used_names: set[str],
) -> tuple[list[bytes], list[str]]:
    data_inp = node["inputs"][0]
    ax_init = ensure_i64(f"__stemma_sl_ax_{axis}", [axis])
    step_init = ensure_i64("__stemma_sl_step_1", [1])
    base = node["name"] or f"split_{idx}"

    slice_nodes: list[bytes] = []
    consumed: list[str] = []
    offset = 0
    for out_idx, (out_name, size) in enumerate(zip(node["outputs"], sizes)):
        if out_name:
            st_init = ensure_i64(f"__stemma_sl_val_{offset}", [offset])
            en_init = ensure_i64(f"__stemma_sl_val_{offset + size}", [offset + size])
            sl_inputs = [data_inp, st_init, en_init, ax_init, step_init]
            consumed.extend(sl_inputs)
            slice_nodes.append(
                _make_simple_node(
                    "Slice",
                    _unique_name(used_names, f"{base}__slice_{out_idx}"),
                    sl_inputs,
                    [out_name],
                )
            )
        offset += size
    return slice_nodes, consumed


def patch_onnx_bytes(
    raw_bytes: bytes,
    *,
    model_dir: str | os.PathLike[str] | None = None,
) -> tuple[bytes, int]:
    """Patch raw ONNX protobuf bytes for DirectML compatibility.

    Returns ``(patched_bytes, change_count)``. If ``change_count == 0``, the
    original bytes are returned unchanged. Raises ``ValueError`` if *raw_bytes*
    is malformed or uses an unsupported pattern.
    """
    mv = memoryview(raw_bytes)
    top_other: list[tuple[int, int, object]] = []
    graph_fields: list[tuple[int, int, object]] = []
    nodes: list[dict] = []
    inits: dict[str, dict] = {}
    vi_map: dict[str, dict] = {}
    default_opset: int | None = None
    has_graph = False

    for fno, wire, val in _iter_fields(mv):
        if fno == 7 and wire == 2:
            has_graph = True
            for gfno, gwire, gval in _iter_fields(val):
                if gfno == 1 and gwire == 2:
                    nodes.append(_parse_node(gval))
                elif gfno == 5 and gwire == 2:
                    tensor = _parse_tensor(gval)
                    inits[tensor["name"]] = tensor
                    graph_fields.append((gfno, gwire, bytes(gval)))
                elif gfno in (11, 12, 13) and gwire == 2:
                    vi = _parse_value_info(gval)
                    vi_map[vi["name"]] = vi
                    graph_fields.append((gfno, gwire, bytes(gval)))
                else:
                    graph_fields.append((gfno, gwire, bytes(gval) if gwire == 2 else gval))
        else:
            if fno == 8 and wire == 2:
                domain, ver = _parse_opset(val)
                if domain in _DEFAULT_DOMAINS:
                    default_opset = ver
            top_other.append((fno, wire, bytes(val) if wire == 2 else val))

    if not has_graph:
        raise ValueError("Invalid ONNX model: missing graph")

    model_dir_str = os.fspath(model_dir) if model_dir is not None else None
    reshape_indices, conv_tr_indices, weights_to_4d, split_specs, dropped_inits = (
        _plan_rewrites(nodes, inits, vi_map, default_opset, model_dir_str)
    )

    change_count = len(reshape_indices) + len(conv_tr_indices) + len(split_specs)
    if change_count == 0:
        return raw_bytes, 0

    used_names: set[str] = set(inits) | set(vi_map)
    for node in nodes:
        if node["name"]:
            used_names.add(node["name"])
        used_names.update(inp for inp in node["inputs"] if inp)
        used_names.update(out for out in node["outputs"] if out)

    extra_inits: list[bytes] = []
    extra_init_names: dict[str, str] = {}

    def ensure_i64(key: str, vals: list[int]) -> str:
        if key not in extra_init_names:
            name = _unique_name(used_names, key)
            extra_init_names[key] = name
            extra_inits.append(_make_int64_initializer(name, vals))
        return extra_init_names[key]

    axis_neg1 = ensure_i64("__stemma_axis_neg1", [-1]) if conv_tr_indices else ""

    new_node_raws: list[bytes] = []
    used_node_inputs: set[str] = set()
    for idx, node in enumerate(nodes):
        if idx in conv_tr_indices:
            emitted, consumed = _lift_conv_transpose_1d_to_2d(
                node, idx, axis_neg1, used_names
            )
            new_node_raws.extend(emitted)
            used_node_inputs.update(consumed)
        elif idx in reshape_indices:
            new_node_raws.append(_strip_attribute(node["raw"], "allowzero"))
            used_node_inputs.update(inp for inp in node["inputs"] if inp)
        elif idx in split_specs:
            axis, sizes = split_specs[idx]
            emitted, consumed = _expand_split_to_slices(
                node, idx, axis, sizes, ensure_i64, used_names
            )
            new_node_raws.extend(emitted)
            used_node_inputs.update(consumed)
        else:
            new_node_raws.append(node["raw"])
            used_node_inputs.update(inp for inp in node["inputs"] if inp)

    graph_out = bytearray()
    for raw_node in new_node_raws:
        graph_out.extend(_encode_bytes_field(1, raw_node))
    for gfno, gwire, gval in graph_fields:
        if gfno == 5 and gwire == 2:
            tensor = _parse_tensor(memoryview(gval))  # type: ignore[arg-type]
            t_name = tensor["name"]
            if t_name in dropped_inits and t_name not in used_node_inputs:
                continue
            if t_name in weights_to_4d:
                gval = _rewrite_tensor_dims(gval, tensor["dims"] + (1,))  # type: ignore[arg-type]
            graph_out.extend(_encode_bytes_field(5, gval))  # type: ignore[arg-type]
        elif gfno == 13 and gwire == 2:
            vi = _parse_value_info(memoryview(gval))  # type: ignore[arg-type]
            if vi["name"] in weights_to_4d:
                continue
            graph_out.extend(_encode_bytes_field(13, gval))  # type: ignore[arg-type]
        else:
            graph_out.extend(_encode_raw_field(gfno, gwire, gval))
    for init_bytes in extra_inits:
        graph_out.extend(_encode_bytes_field(5, init_bytes))

    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    model_out = bytearray()
    for fno, wire, val in top_other:
        if fno == 14 and wire == 2:
            mk, _ = _parse_string_pair(memoryview(val))  # type: ignore[arg-type]
            if mk in (_META_PATCH_VERSION, _META_SOURCE_SHA256):
                continue
        model_out.extend(_encode_raw_field(fno, wire, val))
    model_out.extend(_encode_bytes_field(7, graph_out))
    model_out.extend(
        _encode_bytes_field(14, _encode_string_pair(_META_PATCH_VERSION, str(PATCH_VERSION)))
    )
    model_out.extend(
        _encode_bytes_field(14, _encode_string_pair(_META_SOURCE_SHA256, source_sha256))
    )
    return bytes(model_out), change_count


# ---------------------------------------------------------------------------
# Sidecar cache management and worker integration
# ---------------------------------------------------------------------------


def _read_metadata_props(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        data = f.read()
    props: dict[str, str] = {}
    for fno, wire, val in _iter_fields(memoryview(data)):
        if fno == 14 and wire == 2:
            k, v = _parse_string_pair(val)
            if k:
                props[k] = v
    return props


def ensure_dml_model(model_path: str) -> str:
    """Return a cached DirectML-patched ``.dml.onnx`` path for *model_path*.

    Falls back cleanly to *model_path* when no graph rewrites are needed, the
    source file does not exist, or patching/writing fails.
    """
    if is_dml_sidecar(model_path) or not os.path.isfile(model_path):
        return model_path

    try:
        with open(model_path, "rb") as f:
            raw_bytes = f.read()
    except OSError as exc:
        logger.warning("Failed to read ONNX model %s for DML patch: %s", model_path, exc)
        return model_path

    src_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    sidecar = dml_sidecar_path(model_path)

    if os.path.isfile(sidecar):
        try:
            meta = _read_metadata_props(sidecar)
            if (
                meta.get(_META_PATCH_VERSION) == str(PATCH_VERSION)
                and meta.get(_META_SOURCE_SHA256) == src_sha256
            ):
                return sidecar
        except (OSError, ValueError):
            pass

    model_dir = os.path.dirname(os.path.abspath(model_path)) or None
    try:
        patched_bytes, change_count = patch_onnx_bytes(raw_bytes, model_dir=model_dir)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to build DirectML sidecar for %s: %s", model_path, exc)
        return model_path

    if change_count == 0:
        return model_path

    sidecar_dir = os.path.dirname(os.path.abspath(sidecar))
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".dml_tmp_",
            suffix=".onnx",
            dir=sidecar_dir,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(patched_bytes)
        os.replace(tmp_path, sidecar)
        return sidecar
    except OSError as exc:
        logger.warning("Failed to write DirectML sidecar %s: %s", sidecar, exc)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return model_path


def prepare_htdemucs_dml_path(model_path: str) -> str:
    """Return the DirectML sidecar for HTDemucs when DirectML is available."""
    if not is_htdemucs_model_path(model_path):
        return model_path

    # Deferred import keeps cold app startup free of ONNX Runtime load time.
    import onnxruntime as ort

    if "DmlExecutionProvider" not in ort.get_available_providers():
        return model_path
    return ensure_dml_model(model_path)
