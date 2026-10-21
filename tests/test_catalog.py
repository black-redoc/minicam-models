"""models.toml parsing and its safety rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from minicam_models.catalog import CatalogError, load_catalog

GOOD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/x.pt"
SHA = "a" * 64


def _write(tmp_path: Path, models: str, calibration_url: str = GOOD_URL) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(
        f'[calibration]\nurl = "{calibration_url}"\nsha256 = "{SHA}"\n'
        f"calibration_images = 10\n{models}"
    )
    return path


def _model(
    model_id: str = "yolov8n",
    url: str = GOOD_URL,
    sha: str = SHA,
    precisions: str = '["fp32", "int8"]',
) -> str:
    return (
        f'[[models]]\nid = "{model_id}"\nurl = "{url}"\nsha256 = "{sha}"\n'
        f"precisions = {precisions}\n"
    )


def test_repository_catalog_is_valid() -> None:
    catalog = load_catalog()
    ids = {model.id for model in catalog.models}
    assert {"yolov8n", "yolov8s", "yolo11n", "yolo11s"} <= ids
    assert catalog.get("yolov8n").asset_name("int8") == "yolov8n-int8.onnx"


@pytest.mark.parametrize(
    "models",
    [
        _model(url="https://example.com/x.pt"),
        _model(url="http://github.com/ultralytics/assets/releases/download/x.pt"),
        _model(sha="ABC"),
        _model(precisions='["fp16"]'),
        _model(precisions='["fp32", "fp32"]'),
        _model(model_id="../evil"),
        _model() + _model(),
        "",
    ],
)
def test_rejects_invalid_catalogs(tmp_path: Path, models: str) -> None:
    with pytest.raises(CatalogError):
        load_catalog(_write(tmp_path, models))


def test_rejects_untrusted_calibration_url(tmp_path: Path) -> None:
    with pytest.raises(CatalogError):
        load_catalog(
            _write(tmp_path, _model(), calibration_url="https://evil.test/a.zip")
        )


def test_unknown_model_id(tmp_path: Path) -> None:
    with pytest.raises(CatalogError):
        load_catalog(_write(tmp_path, _model())).get("nope")
