# Progress

## 2026-10-21 — 001 model release pipeline
- Done: analyzed minicam's detection contract (commit 5e5c028); built contract,
  catalog, pinned fetch, FP32/INT8 export, validator, release bundle, CLI,
  CI + release workflows, docs and ADRs 0001–0003.
- Local run: 8 assets (yolov8n/s, yolo11n/s × fp32/int8) all pass validation
  and load in minicam's real `OnnxDetectionBackend`; INT8 agreement 0.92–0.95.
- Commit: initial commit
- Tests: 38 passed (incl. drift vs minicam) + export e2e passed; ruff, mypy clean
- Next: create GitHub repo, decide license (KI-001), tag v0.1.0

## 2026-10-21 — 001 first release
- Done: private repo black-redoc/minicam-models, MIT license for code, tag
  v0.1.0 released by CI with 8 assets + manifest + SHA256SUMS.
- Commit: 8f1dc1e (tag v0.1.0)
- Tests: CI checks and release workflow green; INT8 agreement 0.85–0.91 (KI-006)
- Next: KI-001 decision, make public, Pi 5 install test (AC7)

## 2026-10-21 — 001 minicam integration
- Done: repo public; minicam docs/placeholder updated (black-redoc/minicam#3);
  CI drift check enabled against minicam master, plus weekly schedule.
- Verified: minicam's current fetch code downloads `latest/download` assets
  (2 redirects) and `OnnxDetectionBackend` loads them.
- Tests: drift 5/5 against minicam ec5b619
- Next: Pi 5 install test (AC7)

## 2026-10-21 — minicam docs follow-up
- Done: minicam#3 (model docs → this repo) merged, resolving KI-002; minicam#4–#7
  (JWT secret + cloudflared setup, README cleanup, ignore app/data/models/) merged.
- Commit: see below
- Tests: minicam master CI green at 2a1d2a6
- Next: Pi 5 install test (AC7)
