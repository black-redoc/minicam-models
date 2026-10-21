"""Release bundle contents."""

from __future__ import annotations

import json
from pathlib import Path

from minicam_models.catalog import load_catalog
from minicam_models.export import ExportResult
from minicam_models.fetch import sha256_of
from minicam_models.release import ReleaseInfo, write_release

from .conftest import ModelFactory


def test_writes_manifest_sums_and_notes(
    tmp_path: Path, make_model: ModelFactory
) -> None:
    spec = load_catalog().get("yolov8n")
    fp32 = make_model("yolov8n-fp32.onnx")
    int8 = make_model("yolov8n-int8.onnx", bias=1.0)
    results = [ExportResult(spec, "fp32", fp32), ExportResult(spec, "int8", int8, 0.9)]
    release = ReleaseInfo("acme/minicam-models", "v1.0.0")

    write_release(results, release, tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["contract"]["output"]["shape"] == [1, 84, 8400]
    assert [a["name"] for a in manifest["assets"]] == [fp32.name, int8.name]
    assert manifest["assets"][0]["sha256"] == sha256_of(fp32)
    assert manifest["assets"][1]["int8_agreement"] == 0.9
    assert manifest["toolchain"]["onnxruntime"] == "1.27.0"

    sums = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert sums[0] == f"{sha256_of(fp32)}  {fp32.name}"

    notes = (tmp_path / "RELEASE_NOTES.md").read_text()
    url = "https://github.com/acme/minicam-models/releases/download/v1.0.0/yolov8n-int8.onnx"
    assert url in notes
    assert "Fetch from URL" in notes


def test_latest_url() -> None:
    release = ReleaseInfo("acme/minicam-models", "v1.0.0")
    assert release.latest_url("a.onnx") == (
        "https://github.com/acme/minicam-models/releases/latest/download/a.onnx"
    )
