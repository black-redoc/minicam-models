"""Loads and validates models.toml, the list of models this repo publishes."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Precision = Literal["fp32", "int8"]

PRECISIONS: tuple[Precision, ...] = ("fp32", "int8")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[a-z0-9]+$")
_TRUSTED_URL_PREFIX = "https://github.com/ultralytics/assets/releases/download/"

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "models.toml"


class CatalogError(ValueError):
    """models.toml is malformed or violates a catalog rule."""


@dataclass(frozen=True, slots=True)
class PinnedFile:
    """A remote file pinned by its sha256 digest."""

    url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One source model and the precisions exported from it."""

    id: str
    family: str
    source: PinnedFile
    precisions: tuple[Precision, ...]
    profile: str

    def asset_name(self, precision: Precision) -> str:
        """Release asset filename, e.g. "yolov8n-int8.onnx"."""
        return f"{self.id}-{precision}.onnx"


@dataclass(frozen=True, slots=True)
class Catalog:
    """The parsed models.toml."""

    calibration: PinnedFile
    calibration_images: int
    models: tuple[ModelSpec, ...]

    def get(self, model_id: str) -> ModelSpec:
        """Return the model with this id, or raise CatalogError."""
        for model in self.models:
            if model.id == model_id:
                return model
        raise CatalogError(f"Unknown model id {model_id!r}")


def _pinned(table: dict[str, object], where: str) -> PinnedFile:
    url = table.get("url")
    sha256 = table.get("sha256")
    if not isinstance(url, str) or not url.startswith(_TRUSTED_URL_PREFIX):
        raise CatalogError(f"{where}: url must start with {_TRUSTED_URL_PREFIX}")
    if not isinstance(sha256, str) or not _SHA256_PATTERN.match(sha256):
        raise CatalogError(f"{where}: sha256 must be 64 lowercase hex chars")
    return PinnedFile(url=url, sha256=sha256)


def _model(table: dict[str, object]) -> ModelSpec:
    model_id = table.get("id")
    if not isinstance(model_id, str) or not _ID_PATTERN.match(model_id):
        raise CatalogError(f"Invalid model id {model_id!r}")
    precisions = table.get("precisions")
    if (
        not isinstance(precisions, list)
        or not precisions
        or any(p not in PRECISIONS for p in precisions)
        or len(set(precisions)) != len(precisions)
    ):
        raise CatalogError(f"{model_id}: precisions must be a subset of {PRECISIONS}")
    return ModelSpec(
        id=model_id,
        family=str(table.get("family", "")),
        source=_pinned(table, model_id),
        precisions=tuple(cast(list[Precision], precisions)),
        profile=str(table.get("profile", "")),
    )


def load_catalog(path: Path = DEFAULT_CATALOG) -> Catalog:
    """Parse and validate models.toml."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    calibration = data.get("calibration")
    if not isinstance(calibration, dict):
        raise CatalogError("Missing [calibration] table")
    images = calibration.get("calibration_images")
    if not isinstance(images, int) or images < 1:
        raise CatalogError("calibration_images must be a positive integer")

    models = tuple(_model(entry) for entry in data.get("models", []))
    if not models:
        raise CatalogError("Catalog has no [[models]]")
    ids = [model.id for model in models]
    if len(set(ids)) != len(ids):
        raise CatalogError("Duplicate model ids in catalog")

    return Catalog(
        calibration=_pinned(calibration, "calibration"),
        calibration_images=images,
        models=models,
    )
