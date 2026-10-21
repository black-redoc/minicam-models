# 0002 INT8 quantization — Status: Accepted
Date: 2026-10-21
## Context
Raspberry Pi 5 runs inference on CPU only; smaller/faster models help. minicam
feeds float32 and reads float32, so the IO contract must not change.
## Decision
onnxruntime static quantization, QDQ format, uint8 activations, int8
per-channel weights, MinMax calibration on 100 COCO128 images preprocessed
with minicam's exact preprocessing. The final Detect module stays FP32. Gate:
F1 agreement with FP32 >= 0.80 on the 28 held-out images.
## Alternatives considered
- Dynamic quantization: poor results on conv-heavy models.
- Ultralytics INT8 export targets (OpenVINO/TFLite): not loadable by minicam's ONNX backend.
- Quantizing the head too: larger accuracy loss for little speed gain.
## Consequences
INT8 assets are ~40% of FP32 size and ~28% faster on ARM (M4 measurement);
quality gate is a proxy (KI-003).
