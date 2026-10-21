# Roadmap

## M1 - First release (in progress)
- [x] Contract module mirroring minicam's ONNX backend (spec 001)
- [x] Pinned catalog: YOLOv8 n/s, YOLO11 n/s (models.toml)
- [x] FP32 export + INT8 QDQ quantization with FP32 agreement gate
- [x] Validator, release bundle (manifest.json, SHA256SUMS, notes)
- [x] CI and tag-triggered release workflows
- [ ] GitHub repo created, first tag `v0.1.0` published
- [ ] Verified install on a real Pi 5 via "Fetch from URL"

## Scope
- Detection models that satisfy minicam's current ONNX contract (CPU).
- Reproducible, checksummed, validated GitHub Release assets.

## Non-goals
- Training or fine-tuning models.
- Custom (non-COCO) classes: minicam hardcodes COCO-80 today.
- Changing minicam itself (contract changes start in minicam, then here).

## Backlog
- YOLO26 (end-to-end/NMS-free output) — needs a minicam backend change first.
- Hailo `.hef` exports — needs minicam's Hailo backend first.
- Real mAP evaluation (COCO val) instead of the FP32-agreement proxy.
- Benchmark latency on a Raspberry Pi 5 and record it in release notes.
- Sigstore / GitHub artifact attestations for release assets.
