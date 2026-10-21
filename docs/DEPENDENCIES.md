# Dependencies

## Policy (decided 2026-10-21)
- Exact pins (`==x.y.z`) in `pyproject.toml`; `uv.lock` always committed;
  CI installs with `--locked`.
- Wheels only: `[tool.uv] no-build = true`. Any package that needs a source
  build must be added explicitly (with justification) and recorded here.
- New versions must be at least two weeks old.
- `onnxruntime` must equal minicam's pin (validated by the drift test).
- Linux installs `torch`/`torchvision` from the PyTorch CPU index (no CUDA).

## Runtime (validation)
| Package | Version | Why |
|---|---|---|
| numpy | 2.5.0 | preprocessing/decoding, same as minicam |
| onnx | 1.23.2 | load/check graphs, metadata |
| onnxruntime | 1.27.0 | minicam's runtime; validation + INT8 quantization |

## Extra `export`
| Package | Version | Why |
|---|---|---|
| ultralytics | 8.4.183 | official YOLOv8/YOLO11 ONNX exporter (AGPL-3.0, see KI-001) |
| torch / torchvision | 2.14.1 / 0.29.1 | required by ultralytics |
| onnxslim | 0.1.98 | graph simplification during export |

## Dev
ruff 0.15.20, mypy 2.1.0, pytest 9.1.1.

## CI actions (pinned by SHA)
actions/checkout v7.0.1, astral-sh/setup-uv v10.2.0.
