# Architecture

## Flow
```
models.toml (pinned .pt + sha256)
  -> fetch.py      download, verify sha256
  -> export.py     Ultralytics -> FP32 ONNX; onnxruntime QDQ -> INT8 ONNX
                   (calibrated with minicam's preprocessing, gated vs FP32)
  -> validate.py   minicam contract checks (every asset)
  -> release.py    manifest.json, SHA256SUMS, RELEASE_NOTES.md
  -> .github/workflows/release.yml   GitHub Release assets
  -> minicam admin "Fetch from URL"  (POST /api/detection/model/fetch)
```

## Components (`src/minicam_models/`)
- `contract.py` — constants, preprocessing, decoding and NMS copied from
  minicam (`app/services/detection/*`, `app/routers/setup.py`).
- `catalog.py` — parses/validates `models.toml`.
- `fetch.py` — pinned downloads (temp file, verify, then move).
- `export.py` — FP32 export, INT8 quantization, agreement gate.
- `validate.py` — the publish gate; also usable on any third-party `.onnx`.
- `release.py`, `cli.py` — bundle and CLI (`minicam-models export|validate`).

## Contract invariants (what minicam requires)
Derived from minicam commit 5e5c028 (`OnnxDetectionBackend`, `factory.py`,
`setup.py`). `tests/test_minicam_drift.py` re-checks these against a checkout.

| Requirement | Value | Why (minicam) |
|---|---|---|
| Format | ONNX, filename ends `.onnx` | upload/fetch reject other suffixes |
| Size | <= 200 MB | `_MAX_MODEL_UPLOAD_BYTES`, buffered in RAM |
| Input | 1 tensor, float32, `[1,3,640,640]`, static | fixed letterbox to 640, batch 1 |
| Input data | RGB, /255, nearest letterbox, pad 114 | `_preprocess` |
| Output | `output[0]` float32 `[1,84,8400]` | `_decode_output` transposes to (8400,84) |
| Output layout | 4 box coords (cx,cy,w,h in 640 px) + 80 class scores, sigmoid, no objectness | argmax over scores |
| NMS | none in graph (not end2end) | minicam runs its own NMS (IoU 0.45) |
| Classes | COCO-80 in official order | `COCO_CLASSES` is hardcoded |
| Runtime | onnxruntime 1.27.0, `CPUExecutionProvider` | minicam pin; opset <= what it supports |
| Distribution | HTTPS URL whose path ends with the file name | `_require_https_url`, name from URL path |

YOLOv8 and YOLO11 Detect heads both satisfy this; YOLO26 does not (ADR 0001).
Asset filenames are unique per model+precision because minicam stores
downloads in `data/models/<filename>` (ADR 0003).

## ADRs
[0001 stack and model families](adr/0001-stack-and-model-families.md) ·
[0002 INT8 quantization](adr/0002-int8-quantization.md) ·
[0003 distribution via GitHub Releases](adr/0003-github-releases-distribution.md)
