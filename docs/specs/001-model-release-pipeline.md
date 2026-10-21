# 001 Model release pipeline — Status: In progress
## Goal
Publish YOLO detection models that a minicam admin can install via
"Fetch from URL" and that minicam's ONNX backend loads with no changes.

## Acceptance criteria
- [x] AC1: Only contract-compliant files are accepted (shape, dtype, NMS, COCO order, size, suffix, runtime) → tests: `test_validate.py::*`
- [x] AC2: contract.py matches minicam (constants, classes, size cap, onnxruntime pin, preprocessing bit-for-bit) → tests: `test_minicam_drift.py::*`
- [x] AC3: Source files are used only if their sha256 matches the pin; catalog rejects untrusted hosts and malformed entries → tests: `test_fetch.py::*`, `test_catalog.py::*`
- [x] AC4: Export produces valid FP32 and INT8 assets, byte-reproducible, INT8 agreement >= 0.80 → test: `test_export.py::test_end_to_end_export_produces_valid_reproducible_assets`
- [x] AC5: Release bundle has manifest.json, SHA256SUMS and admin notes with per-asset URLs → tests: `test_release.py::*`
- [ ] AC6: A tag push publishes the 8 assets as a GitHub Release → manual: first `v0.1.0` run
- [ ] AC7: A Pi 5 running minicam installs an asset through "Fetch from URL" and draws boxes → manual

## Design
See [ARCHITECTURE](../ARCHITECTURE.md) and ADRs 0001–0003.

## Security notes
Pinned hashes, trusted host prefix, wheels-only installs, SHA-pinned actions,
least-privilege workflow permissions. See [SECURITY](../SECURITY.md).

## Tasks
- [x] contract, catalog, fetch, export, validate, release, CLI
- [x] Tests and CI/release workflows
- [x] Local export of all 8 assets; all pass validation and minicam's real backend
- [ ] Create GitHub repo, push, tag v0.1.0
- [ ] Resolve KI-001 (license) before publishing

## Out of scope
YOLO26, Hailo, custom classes, training.
