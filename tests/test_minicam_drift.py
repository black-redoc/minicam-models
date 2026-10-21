"""Detects drift between contract.py and a minicam checkout.

Skipped unless MINICAM_SRC points at a minicam repository root, e.g.
`MINICAM_SRC=~/dev/minicam uv run pytest tests/test_minicam_drift.py`.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from minicam_models import contract

MINICAM_SRC = os.environ.get("MINICAM_SRC")
pytestmark = pytest.mark.skipif(not MINICAM_SRC, reason="MINICAM_SRC not set")


def _minicam() -> Path:
    assert MINICAM_SRC is not None
    return Path(MINICAM_SRC).expanduser()


def _module_constants(relative: str) -> dict[str, object]:
    tree = ast.parse((_minicam() / relative).read_text())
    constants: dict[str, object] = {}
    for node in tree.body:
        target: ast.expr
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and value is not None:
            try:
                constants[target.id] = ast.literal_eval(value)
            except ValueError:
                continue
    return constants


def test_coco_classes_match() -> None:
    minicam = _module_constants("app/services/detection/coco_classes.py")
    assert minicam["COCO_CLASSES"] == contract.COCO_CLASSES


def test_preprocessing_constants_match() -> None:
    minicam = _module_constants("app/services/detection/onnx_backend.py")
    assert minicam["MODEL_INPUT_SIZE"] == contract.MODEL_INPUT_SIZE
    assert minicam["LETTERBOX_PAD_VALUE"] == contract.LETTERBOX_PAD_VALUE
    nms = _module_constants("app/services/detection/nms.py")
    assert nms["DEFAULT_IOU_THRESHOLD"] == contract.DEFAULT_IOU_THRESHOLD


def test_upload_limits_match() -> None:
    source = (_minicam() / "app/routers/setup.py").read_text()
    match = re.search(r"_MAX_MODEL_UPLOAD_BYTES = (.+?)\s+#", source)
    assert match is not None
    assert eval(match.group(1), {}) == contract.MAX_MODEL_BYTES  # noqa: S307 - literal arithmetic from a trusted checkout


def test_onnxruntime_pin_matches() -> None:
    pyproject = tomllib.loads((_minicam() / "pyproject.toml").read_text())
    pins = [
        d for d in pyproject["project"]["dependencies"] if d.startswith("onnxruntime")
    ]
    assert pins == [f"onnxruntime=={contract.MINICAM_ONNXRUNTIME_VERSION}"]


def test_preprocess_matches_minicam_bit_for_bit() -> None:
    sys.path.insert(0, str(_minicam()))
    try:
        onnx_backend = importlib.import_module("app.services.detection.onnx_backend")
    finally:
        sys.path.pop(0)
    frame = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    ours = contract.preprocess(frame)
    theirs = onnx_backend._preprocess(frame)
    assert np.array_equal(ours[0], theirs[0])
    assert ours[1:] == theirs[1:]
