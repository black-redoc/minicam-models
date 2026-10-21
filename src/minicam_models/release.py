"""Builds the release bundle: manifest.json, SHA256SUMS and release notes.

The notes are what a minicam admin reads: one row per asset with the URL to
paste into minicam's "Fetch from URL" (POST /api/detection/model/fetch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from minicam_models import contract
from minicam_models.export import ExportResult
from minicam_models.fetch import sha256_of

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """Where the release will live, used to build download URLs."""

    repository: str  # "owner/name"
    tag: str

    def download_url(self, asset: str) -> str:
        """Tag-pinned URL, stable forever once the release exists."""
        return (
            f"https://github.com/{self.repository}/releases/download/{self.tag}/{asset}"
        )

    def latest_url(self, asset: str) -> str:
        """Always resolves to the newest release's asset of this name."""
        return f"https://github.com/{self.repository}/releases/latest/download/{asset}"


def _toolchain() -> dict[str, str]:
    packages = ("ultralytics", "torch", "onnx", "onnxslim", "onnxruntime")
    return {name: version(name) for name in packages}


def build_manifest(
    results: list[ExportResult], release: ReleaseInfo
) -> dict[str, object]:
    """Machine-readable record of every asset and how it was produced."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release": release.tag,
        "toolchain": _toolchain(),
        "contract": {
            "input": {"dtype": "float32", "shape": list(contract.INPUT_SHAPE)},
            "output": {"dtype": "float32", "shape": list(contract.OUTPUT_SHAPE)},
            "classes": "coco80",
            "nms": False,
            "minicam_onnxruntime": contract.MINICAM_ONNXRUNTIME_VERSION,
        },
        "assets": [
            {
                "name": result.path.name,
                "model": result.model.id,
                "family": result.model.family,
                "precision": result.precision,
                "profile": result.model.profile,
                "bytes": result.path.stat().st_size,
                "sha256": sha256_of(result.path),
                "source_url": result.model.source.url,
                "source_sha256": result.model.source.sha256,
                "int8_agreement": result.int8_agreement,
                "url": release.download_url(result.path.name),
            }
            for result in results
        ],
    }


def _notes(manifest: dict[str, object], release: ReleaseInfo) -> str:
    assets = manifest["assets"]
    assert isinstance(assets, list)
    rows = "\n".join(
        f"| `{a['name']}` | {a['profile']} | {a['precision']} | "
        f"{a['bytes'] / 1_000_000:.1f} MB | `{a['sha256'][:12]}…` |"
        for a in assets
    )
    urls = "\n".join(release.download_url(a["name"]) for a in assets)
    return f"""# minicam-models {release.tag}

YOLO detection models in the exact format minicam's ONNX backend expects
(input float32 1x3x640x640, output float32 1x84x8400, COCO-80, no NMS).

## Install on a minicam device

1. In minicam, open **AI Detection** and use **Fetch from URL**.
2. Paste one of the URLs below (Pi 5: an `n` model, `int8` is fastest;
   x86 PC: an `s` model).
3. Copy the returned `model_path` into the detection settings and save.

| Asset | Profile | Precision | Size | sha256 |
|---|---|---|---|---|
{rows}

## URLs

```
{urls}
```

Full checksums are in `SHA256SUMS`; provenance in `manifest.json`.
Weights are Ultralytics models under AGPL-3.0.
"""


def write_release(
    results: list[ExportResult], release: ReleaseInfo, out_dir: Path
) -> None:
    """Write manifest.json, SHA256SUMS and RELEASE_NOTES.md into out_dir."""
    manifest = build_manifest(results, release)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sums = "".join(f"{sha256_of(r.path)}  {r.path.name}\n" for r in results)
    (out_dir / "SHA256SUMS").write_text(sums)
    (out_dir / "RELEASE_NOTES.md").write_text(_notes(manifest, release))
