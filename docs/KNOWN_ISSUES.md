# Known issues

## KI-001 Licensing of published weights — Status: Open
Repository code is MIT (decided 2026-10-21). The published `.onnx` assets are
derived from Ultralytics YOLO weights, which stay AGPL-3.0 (stated in each
file's `license` metadata); MIT does not relicense them. Distributing them to
customer devices may require AGPL compliance or an Ultralytics Enterprise
license. Owner decision pending before making the repo/releases public.

## KI-002 minicam docs reference outdated names — Status: Open
minicam's `app/docs/detection.md` names the companion repo `minicam_weights`,
assets `yolov8n-cpu.onnx`/`yolov8s-cpu.onnx` and `ultralytics==8.4.95`. They
should point to this repo, the `<id>-<precision>.onnx` names and 8.4.183.
Change belongs in the minicam repo.

## KI-003 INT8 quality gate is a proxy — Status: Open
The gate measures FP32/INT8 detection agreement (F1, IoU 0.5, conf 0.5) on 28
COCO128 images, not mAP. Current values: 0.92–0.95. See Backlog.

## KI-004 Legacy TorchScript ONNX exporter — Status: Open
Ultralytics 8.4.183 uses torch's legacy exporter (DeprecationWarning). A future
torch upgrade may change or break the export; re-run `pytest -m export`.

## KI-005 Pi 5 latency not measured — Status: Open
INT8 was ~28% faster than FP32 for yolov8n on an Apple M4 (onnxruntime CPU,
4 threads). Needs measurement on real Pi 5 hardware.
