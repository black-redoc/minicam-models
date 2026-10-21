# Development

## Setup
Requires `uv` 0.12.21+. Python 3.14 is installed by uv.
```bash
uv sync --locked --extra export
```

## Workflow
- Add or change a model: edit `models.toml` (url + sha256 from
  `shasum -a 256`), run the export and validation commands in VALIDATION.md.
- Release: bump nothing in code; push a tag `vYYYY.MM.N`. The `release`
  workflow exports, validates and publishes assets with notes.
- If minicam changes its detection backend: update `contract.py` first, then
  run the drift test against the new minicam checkout.

## Layout
`src/minicam_models/` code · `tests/` · `models.toml` catalog ·
`.cache/` downloads and work dirs (ignored) · `dist/` build output (ignored).

## Env vars
- `MINICAM_SRC` — path to a minicam checkout for the drift test (optional).

## Conventions
ruff (line length 88), mypy strict, Conventional Commits, docs in English.
