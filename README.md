# minicam-models

YOLO object-detection models published in the exact format minicam's ONNX
detection backend expects, so a minicam admin can install one by pasting a
GitHub Release URL into **AI Detection → Fetch from URL**.

| Asset | Best for | Notes |
|---|---|---|
| `yolov8n-int8.onnx`, `yolo11n-int8.onnx` | Raspberry Pi 5 (CPU) | Fastest, ~5–6 MB |
| `yolov8n-fp32.onnx`, `yolo11n-fp32.onnx` | Raspberry Pi 5 (CPU) | Reference accuracy |
| `yolov8s-*.onnx`, `yolo11s-*.onnx` | x86 PC (CPU) | Larger, more accurate |

Every asset is checked against minicam's contract before it is published:
float32 input `1x3x640x640`, float32 output `1x84x8400`, COCO-80 class
order, no built-in NMS, under 200 MB, loads with minicam's onnxruntime.

## Install on a minicam device

1. Pick an asset from the [latest release](../../releases/latest).
2. In minicam: **AI Detection → Fetch from URL**, paste the asset URL.
3. Copy the returned `model_path` into the detection settings and save.

## Build locally

```bash
uv sync --locked --extra export
uv run minicam-models export            # all models -> dist/
uv run minicam-models validate dist/*.onnx
```

See [docs/PLAN.md](docs/PLAN.md) for the full documentation.

## License

The code in this repository is MIT licensed (see [LICENSE](LICENSE)). The
published model files are derived from Ultralytics YOLO weights and remain
under AGPL-3.0; see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) (KI-001).
