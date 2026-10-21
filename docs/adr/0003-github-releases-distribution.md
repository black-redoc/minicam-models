# 0003 Distribution via GitHub Releases — Status: Accepted
Date: 2026-10-21
## Context
minicam installs models from an HTTPS URL whose path ends in the file name
(`POST /api/detection/model/fetch`, follows redirects, <= 200 MB) and stores
them as `data/models/<filename>`.
## Decision
Publish assets on GitHub Releases from a tag-triggered workflow. Asset names
are `<model>-<precision>.onnx` (e.g. `yolov8n-int8.onnx`). Release notes list
tag-pinned URLs; `/releases/latest/download/<name>` also works.
## Alternatives considered
- Hugging Face Hub / S3: extra accounts and credentials for no benefit.
- Committing models to git (LFS): bloats the repo, no checksums/notes.
- Version in the filename: avoids overwrites on the device but breaks stable `latest` URLs.
## Consequences
Fetching the same asset name from a newer release overwrites the file on the
device (same path). minicam only reloads the backend when detection settings
are saved (`PUT /api/detection/settings`), so admins must save again after
re-fetching an asset.
