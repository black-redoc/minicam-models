"""Checks that an .onnx file is a model minicam can load and run.

This is the gate every published asset passes. It checks what minicam's
upload/fetch endpoints and OnnxDetectionBackend rely on, using the same
onnxruntime version minicam pins (see contract.MINICAM_ONNXRUNTIME_VERSION).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from minicam_models import contract

_ONNX_FLOAT = onnx.TensorProto.FLOAT


@dataclass(slots=True)
class ValidationReport:
    """Outcome of validating one model file."""

    path: Path
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the file satisfies the whole contract."""
        return not self.errors


def _tensor_signature(value: onnx.ValueInfoProto) -> tuple[int, tuple[object, ...]]:
    tensor_type = value.type.tensor_type
    dims = tuple(
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in tensor_type.shape.dim
    )
    return tensor_type.elem_type, dims


def _check_graph(model: onnx.ModelProto, errors: list[str]) -> None:
    inputs = [
        i
        for i in model.graph.input
        if i.name not in {init.name for init in model.graph.initializer}
    ]
    if len(inputs) != 1:
        errors.append(f"expected exactly 1 input, found {len(inputs)}")
    else:
        elem, dims = _tensor_signature(inputs[0])
        if elem != _ONNX_FLOAT or dims != contract.INPUT_SHAPE:
            errors.append(
                f"input must be float32 {contract.INPUT_SHAPE}, got "
                f"{onnx.TensorProto.DataType.Name(elem)} {dims}"
            )

    if not model.graph.output:
        errors.append("model has no outputs")
    else:
        elem, dims = _tensor_signature(model.graph.output[0])
        if elem != _ONNX_FLOAT or dims != contract.OUTPUT_SHAPE:
            errors.append(
                f"output[0] must be float32 {contract.OUTPUT_SHAPE} (no NMS), got "
                f"{onnx.TensorProto.DataType.Name(elem)} {dims}"
            )


def _check_metadata(model: onnx.ModelProto, errors: list[str]) -> None:
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    if metadata.get("end2end", "False") != "False":
        errors.append(
            "model was exported end2end (built-in NMS); minicam needs raw output"
        )
    names_raw = metadata.get("names")
    if names_raw is None:
        errors.append("missing 'names' metadata; cannot prove COCO class order")
        return
    try:
        names = ast.literal_eval(names_raw)
    except ValueError, SyntaxError:
        errors.append("'names' metadata is not a valid class mapping")
        return
    ordered = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else ()
    if ordered != contract.COCO_CLASSES:
        errors.append("class names/order differ from minicam's COCO_CLASSES")


def _check_inference(path: Path, errors: list[str]) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    tensor, _, _, _ = contract.preprocess(frame)
    raw = session.run(None, {input_name: tensor})[0]
    if raw.shape != contract.OUTPUT_SHAPE:
        errors.append(f"runtime output shape {raw.shape} != {contract.OUTPUT_SHAPE}")
        return
    if not np.isfinite(raw).all():
        errors.append("runtime output contains NaN/inf")
    class_scores = raw[0, contract.NUM_BOX_COORDS :]
    if class_scores.min() < 0.0 or class_scores.max() > 1.0:
        errors.append("class scores are outside [0, 1]; expected sigmoid outputs")


def validate_model(path: Path) -> ValidationReport:
    """Validate one model file against minicam's contract."""
    report = ValidationReport(path=path)
    errors = report.errors

    if ort.__version__ != contract.MINICAM_ONNXRUNTIME_VERSION:
        errors.append(
            f"validator runs onnxruntime {ort.__version__}, but minicam pins "
            f"{contract.MINICAM_ONNXRUNTIME_VERSION}"
        )
    if path.suffix != contract.MODEL_SUFFIX:
        errors.append(f"filename must end with {contract.MODEL_SUFFIX}")
    if not path.is_file():
        errors.append("file does not exist")
        return report
    size = path.stat().st_size
    if size > contract.MAX_MODEL_BYTES:
        errors.append(
            f"file is {size} bytes, over minicam's {contract.MAX_MODEL_BYTES}"
        )

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - any load failure is a contract failure
        errors.append(f"not a valid ONNX model: {exc}")
        return report

    _check_graph(model, errors)
    _check_metadata(model, errors)
    if errors:
        return report

    try:
        _check_inference(path, errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"onnxruntime CPU inference failed: {exc}")
    return report
