"""INT8 helpers, plus a full end-to-end export (marker: export)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper

from minicam_models import export
from minicam_models.catalog import load_catalog
from minicam_models.validate import validate_model


def test_head_nodes_are_last_model_module() -> None:
    nodes = [
        helper.make_node("Relu", ["a"], ["b"], name="/model.0/conv/Conv"),
        helper.make_node("Relu", ["b"], ["c"], name="/model.22/dfl/Conv"),
        helper.make_node("Relu", ["c"], ["d"], name="/model.22/Concat"),
        helper.make_node("Relu", ["d"], ["e"], name="/model.3/act"),
        helper.make_node("Relu", ["e"], ["f"], name="Reshape_1"),
    ]
    graph = helper.make_graph(nodes, "g", [], [])
    names = export._head_node_names(helper.make_model(graph))
    assert names == ["/model.22/dfl/Conv", "/model.22/Concat"]


def test_matches_requires_same_class_and_iou() -> None:
    reference = (np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), np.array([0, 1]))
    candidate = (np.array([[1, 1, 10, 10], [50, 50, 60, 60]], float), np.array([0, 2]))
    assert export._matches(reference, candidate) == 1


@pytest.mark.export
def test_end_to_end_export_produces_valid_reproducible_assets(tmp_path: Path) -> None:
    catalog = load_catalog()
    cache = Path(".cache")
    model = catalog.get("yolov8n")

    first = export.export_model(catalog, model, cache, tmp_path / "a")
    second = export.export_model(catalog, model, cache, tmp_path / "b")

    for result in first:
        assert validate_model(result.path).ok
        metadata = {p.key: p.value for p in onnx.load(str(result.path)).metadata_props}
        assert metadata["minicam_precision"] == result.precision
        assert "date" not in metadata
    int8 = next(r for r in first if r.precision == "int8")
    assert int8.int8_agreement is not None
    assert int8.int8_agreement >= export.MIN_INT8_AGREEMENT
    for a, b in zip(first, second, strict=True):
        assert a.path.read_bytes() == b.path.read_bytes()
