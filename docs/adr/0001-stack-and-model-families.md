# 0001 Stack and model families — Status: Accepted
Date: 2026-10-21
## Context
minicam's ONNX backend expects a raw YOLOv8-style Detect output `(1,84,8400)`
with COCO-80 classes and does its own NMS. We need a reproducible way to
produce such files.
## Decision
Python 3.14 + uv (same as minicam). Export with Ultralytics' official ONNX
exporter (opset 18, static 640, batch 1, `nms=False`, simplified). Publish
YOLOv8 n/s and YOLO11 n/s, whose heads produce exactly that output. Validate
with the same onnxruntime version minicam pins.
## Alternatives considered
- YOLOv8 only: works, but YOLO11 is more accurate at similar cost and fits the contract.
- YOLO26: NMS-free end-to-end output; needs a minicam backend change (Backlog).
- Hand-written torch.onnx export: more code, same result, loses Ultralytics fixes.
## Consequences
Heavy export toolchain (torch) lives in an optional `export` extra. Ultralytics
is AGPL-3.0 (KI-001).
