"""validate_model accepts contract-compliant models and rejects the rest."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
import pytest

from minicam_models import contract
from minicam_models.validate import validate_model

from .conftest import ModelFactory


def test_accepts_compliant_model(make_model: ModelFactory) -> None:
    report = validate_model(make_model())
    assert report.ok, report.errors


def test_rejects_end2end_output_shape(make_model: ModelFactory) -> None:
    report = validate_model(make_model(output_shape=(1, 300, 6)))
    assert any("output[0]" in error for error in report.errors)


def test_rejects_end2end_metadata(make_model: ModelFactory) -> None:
    report = validate_model(make_model(extra_metadata={"end2end": "True"}))
    assert any("end2end" in error for error in report.errors)


def test_rejects_missing_class_names(make_model: ModelFactory) -> None:
    report = validate_model(make_model(names=None))
    assert any("names" in error for error in report.errors)


def test_rejects_wrong_class_order(make_model: ModelFactory) -> None:
    swapped = list(contract.COCO_CLASSES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    report = validate_model(make_model(names=repr(dict(enumerate(swapped)))))
    assert any("COCO" in error for error in report.errors)


def test_rejects_scores_outside_unit_interval(make_model: ModelFactory) -> None:
    report = validate_model(make_model(sigmoid=False, bias=5.0))
    assert any("[0, 1]" in error for error in report.errors)


def test_rejects_wrong_suffix(make_model: ModelFactory) -> None:
    report = validate_model(make_model(name="model.bin"))
    assert any(".onnx" in error for error in report.errors)


def test_rejects_file_over_minicam_size_cap(
    make_model: ModelFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "MAX_MODEL_BYTES", 10)
    report = validate_model(make_model())
    assert any("over minicam" in error for error in report.errors)


def test_rejects_garbage_file(tmp_path: Path) -> None:
    path = tmp_path / "garbage.onnx"
    path.write_bytes(b"not a model")
    assert not validate_model(path).ok


def test_rejects_missing_file(tmp_path: Path) -> None:
    report = validate_model(tmp_path / "missing.onnx")
    assert "file does not exist" in report.errors


def test_rejects_other_onnxruntime_version(
    make_model: ModelFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ort, "__version__", "0.0.1")
    report = validate_model(make_model())
    assert any("minicam pins" in error for error in report.errors)
