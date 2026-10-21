"""Shared fixtures: tiny synthetic ONNX models that do or don't fit the contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from minicam_models import contract

COCO_NAMES = repr(dict(enumerate(contract.COCO_CLASSES)))

ModelFactory = Callable[..., Path]


def _build(
    path: Path,
    output_shape: tuple[int, ...] = contract.OUTPUT_SHAPE,
    names: str | None = COCO_NAMES,
    sigmoid: bool = True,
    bias: float = 0.0,
    extra_metadata: dict[str, str] | None = None,
) -> Path:
    """Output = [Sigmoid](bias + mean(input)), broadcast to output_shape."""
    zeros = numpy_helper.from_array(
        np.full(output_shape, bias, dtype=np.float32), name="bias"
    )
    axes = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), name="axes")
    nodes = [
        helper.make_node("ReduceMean", ["images", "axes"], ["mean"], keepdims=0),
        helper.make_node("Add", ["bias", "mean"], ["pre" if sigmoid else "output0"]),
    ]
    if sigmoid:
        nodes.append(helper.make_node("Sigmoid", ["pre"], ["output0"]))
    graph = helper.make_graph(
        nodes,
        "fake",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, contract.INPUT_SHAPE
            )
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)],
        initializer=[zeros, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    metadata = {"end2end": "False", **(extra_metadata or {})}
    if names is not None:
        metadata["names"] = names
    helper.set_model_props(model, metadata)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def make_model(tmp_path: Path) -> ModelFactory:
    def factory(name: str = "model.onnx", **kwargs: object) -> Path:
        return _build(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return factory
