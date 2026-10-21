"""Exports catalog models to ONNX files that satisfy minicam's contract.

FP32 comes straight from Ultralytics' exporter with the options minicam
needs (static 640x640, batch 1, no NMS, no half). INT8 is produced from that
FP32 file with onnxruntime static QDQ quantization, calibrated on images
preprocessed exactly as minicam does at runtime, and gated by an agreement
check against FP32 on held-out images.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

from minicam_models import contract
from minicam_models.catalog import Catalog, ModelSpec, Precision
from minicam_models.fetch import fetch_pinned

ONNX_OPSET = 18
# minicam's default MINICAM_DETECTION_CONFIDENCE_THRESHOLD.
AGREEMENT_CONFIDENCE = 0.5
AGREEMENT_IOU = 0.5
MIN_INT8_AGREEMENT = 0.80
# Ultralytics stamps the export time; dropping it keeps exports byte-stable.
_VOLATILE_METADATA_KEYS = frozenset({"date"})


class AgreementTooLowError(RuntimeError):
    """The INT8 model disagrees with its FP32 source beyond the allowed gate."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    """One exported asset and the facts recorded about it in the manifest."""

    model: ModelSpec
    precision: Precision
    path: Path
    int8_agreement: float | None = None


def _configure_ultralytics(work_dir: Path) -> None:
    """Keep Ultralytics from auto-installing packages or writing user config."""
    os.environ["YOLO_CONFIG_DIR"] = str(work_dir / "ultralytics-config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "1"


def _stamp_metadata(path: Path, model: ModelSpec, precision: Precision) -> None:
    onnx_model = onnx.load(str(path))
    kept = [
        p for p in onnx_model.metadata_props if p.key not in _VOLATILE_METADATA_KEYS
    ]
    kept = [p for p in kept if not p.key.startswith("minicam_")]
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(kept)
    onnx.helper.set_model_props(
        onnx_model,
        {
            **{p.key: p.value for p in kept},
            "minicam_model_id": model.id,
            "minicam_precision": precision,
            "minicam_source_sha256": model.source.sha256,
        },
    )
    onnx.save(onnx_model, str(path))


def export_fp32(
    model: ModelSpec, weights: Path, out_dir: Path, cache_dir: Path
) -> Path:
    """Export `weights` to `out_dir/<id>-fp32.onnx` in minicam's format."""
    _configure_ultralytics(cache_dir)
    from ultralytics import YOLO  # type: ignore[attr-defined]  # heavy, export-only

    work = cache_dir / "work" / model.id
    work.mkdir(parents=True, exist_ok=True)
    local_weights = work / weights.name
    shutil.copyfile(weights, local_weights)

    exported = YOLO(str(local_weights)).export(
        format="onnx",
        imgsz=contract.MODEL_INPUT_SIZE,
        batch=1,
        dynamic=False,
        nms=False,
        simplify=True,
        opset=ONNX_OPSET,
        device="cpu",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / model.asset_name("fp32")
    shutil.move(str(exported), destination)
    _stamp_metadata(destination, model, "fp32")
    return destination


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image  # pulled in by ultralytics (export extra)

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def calibration_images(
    catalog: Catalog, cache_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Return (calibration, evaluation) image paths from the pinned dataset."""
    archive = fetch_pinned(catalog.calibration, cache_dir)
    extract_dir = cache_dir / "calibration"
    if not extract_dir.is_dir():
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                target = (extract_dir / member).resolve()
                if not target.is_relative_to(extract_dir.resolve()):
                    raise ValueError(f"Unsafe path in calibration archive: {member}")
            bundle.extractall(extract_dir)
    images = sorted(extract_dir.rglob("*.jpg"))
    split = catalog.calibration_images
    if len(images) <= split:
        raise ValueError("Calibration dataset leaves no images for evaluation")
    return images[:split], images[split:]


class _CalibrationReader:
    """onnxruntime CalibrationDataReader over minicam-preprocessed images."""

    def __init__(self, input_name: str, images: list[Path]) -> None:
        self._input_name = input_name
        self._tensors: Iterator[np.ndarray] = (
            contract.preprocess(_load_rgb(path))[0] for path in images
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        tensor = next(self._tensors, None)
        return None if tensor is None else {self._input_name: tensor}


def _head_node_names(onnx_model: onnx.ModelProto) -> list[str]:
    """Nodes of the final Detect module ("/model.<last>/..."), kept in FP32.

    Quantizing the box/class head (DFL, concat, sigmoid) costs most of the
    accuracy loss for little speed-up, so it stays in float.
    """
    indices = {
        int(node.name.split("/")[1].split(".")[1])
        for node in onnx_model.graph.node
        if node.name.startswith("/model.")
    }
    head_prefix = f"/model.{max(indices)}/"
    return [n.name for n in onnx_model.graph.node if n.name.startswith(head_prefix)]


def _detections(session: object, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]  # type: ignore[attr-defined]
    boxes, _, class_ids = contract.postprocess(raw, AGREEMENT_CONFIDENCE)
    return boxes, class_ids


def _matches(
    reference: tuple[np.ndarray, np.ndarray], candidate: tuple[np.ndarray, np.ndarray]
) -> int:
    ref_boxes, ref_classes = reference
    cand_boxes, cand_classes = candidate
    used = np.zeros(len(cand_boxes), dtype=bool)
    matched = 0
    for box, class_id in zip(ref_boxes, ref_classes, strict=True):
        eligible = (cand_classes == class_id) & ~used
        if not eligible.any():
            continue
        ious = np.where(eligible, contract.box_iou(box, cand_boxes), 0.0)
        best = int(np.argmax(ious))
        if ious[best] >= AGREEMENT_IOU:
            used[best] = True
            matched += 1
    return matched


def int8_agreement(fp32_path: Path, int8_path: Path, images: list[Path]) -> float:
    """F1 of INT8 detections against FP32 detections over `images`."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    fp32 = ort.InferenceSession(str(fp32_path), providers=providers)
    int8 = ort.InferenceSession(str(int8_path), providers=providers)
    matched = total = 0
    for path in images:
        tensor = contract.preprocess(_load_rgb(path))[0]
        reference, candidate = _detections(fp32, tensor), _detections(int8, tensor)
        matched += _matches(reference, candidate)
        total += len(reference[0]) + len(candidate[0])
    return 1.0 if total == 0 else 2 * matched / total


def export_int8(
    model: ModelSpec,
    fp32_path: Path,
    calibration: list[Path],
    evaluation: list[Path],
    out_dir: Path,
    cache_dir: Path,
) -> tuple[Path, float]:
    """Quantize an FP32 export to INT8 QDQ and gate it on FP32 agreement."""
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    work = cache_dir / "work" / model.id
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / f"{model.id}-prepared.onnx"
    quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=True)

    prepared_model = onnx.load(str(prepared))
    input_name = prepared_model.graph.input[0].name
    destination = out_dir / model.asset_name("int8")
    quantize_static(
        str(prepared),
        str(destination),
        _CalibrationReader(input_name, calibration),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=_head_node_names(prepared_model),
    )
    _stamp_metadata(destination, model, "int8")

    agreement = int8_agreement(fp32_path, destination, evaluation)
    if agreement < MIN_INT8_AGREEMENT:
        raise AgreementTooLowError(
            f"{destination.name}: INT8/FP32 agreement {agreement:.3f} "
            f"< {MIN_INT8_AGREEMENT}"
        )
    return destination, agreement


def export_model(
    catalog: Catalog, model: ModelSpec, cache_dir: Path, out_dir: Path
) -> list[ExportResult]:
    """Export every precision listed for `model`."""
    weights = fetch_pinned(model.source, cache_dir / "weights")
    fp32_path = export_fp32(model, weights, out_dir, cache_dir)
    results = []
    if "fp32" in model.precisions:
        results.append(ExportResult(model, "fp32", fp32_path))
    if "int8" in model.precisions:
        calibration, evaluation = calibration_images(catalog, cache_dir)
        int8_path, agreement = export_int8(
            model, fp32_path, calibration, evaluation, out_dir, cache_dir
        )
        results.append(ExportResult(model, "int8", int8_path, agreement))
    if "fp32" not in model.precisions:
        fp32_path.unlink()
    return results
