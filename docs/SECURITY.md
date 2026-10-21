# Security

## Threat model
The assets are executed (as ONNX graphs) on customer devices. The main risks
are supply-chain: a tampered source checkpoint, a tampered dependency, or a
tampered release asset.

| Threat | Control |
|---|---|
| Tampered `.pt` / calibration zip | `models.toml` pins sha256; `fetch.py` verifies before use and never keeps a mismatched file |
| Untrusted source host | catalog only accepts `https://github.com/ultralytics/assets/releases/download/` |
| `.pt` is a pickle (code execution on load) | only load pinned, hash-verified official weights; never accept user `.pt` files |
| Malicious dependency install scripts | wheels only (`no-build = true`), exact pins, committed `uv.lock`, `--locked` in CI |
| Ultralytics auto-installing packages / telemetry | `YOLO_AUTOINSTALL=false`, `YOLO_OFFLINE=1`, config dir inside `.cache/` |
| Zip-slip in calibration archive | members resolved and checked to stay inside the extract dir |
| Tampered release asset | `SHA256SUMS` + per-asset sha256 in `manifest.json` and release notes |
| CI token abuse | workflow `contents: read` by default, `write` only for the release job; actions pinned by commit SHA; `persist-credentials: false` |

## Secrets
None required. Release uses the built-in `GITHUB_TOKEN`. The optional drift
job may use `MINICAM_READ_TOKEN` (read-only, minicam repo only).

## Per-feature checklist
- [ ] New source files pinned by sha256 from a trusted host
- [ ] No new code path loads unpinned pickles or archives
- [ ] Input validation on any new catalog/CLI field
- [ ] Workflow permissions remain least-privilege; new actions pinned by SHA
- [ ] No secrets or local paths in logs, manifests or release notes
