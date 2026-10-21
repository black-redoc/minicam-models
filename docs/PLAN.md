# minicam-models
A release pipeline that turns pinned Ultralytics YOLO weights into ONNX files
that minicam's ONNX detection backend can load unchanged. Its single job: let a
minicam admin download a model in the required format from a GitHub Release
URL ("Fetch from URL" in minicam's AI Detection page).

## Current focus
Milestone: M1 - First release · Spec: specs/001-model-release-pipeline.md · Next step: create the GitHub repo and push tag v0.1.0

## Docs
[ROADMAP](ROADMAP.md) · [ARCHITECTURE](ARCHITECTURE.md) · [SECURITY](SECURITY.md) ·
[VALIDATION](VALIDATION.md) · [DEVELOPMENT](DEVELOPMENT.md) ·
[DEPENDENCIES](DEPENDENCIES.md) · [PROGRESS](PROGRESS.md) ·
[KNOWN_ISSUES](KNOWN_ISSUES.md) · [adr/](adr/)
