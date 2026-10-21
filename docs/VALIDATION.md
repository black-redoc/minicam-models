# Validation

## Commands
```bash
uv sync --locked --extra export
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest                                   # fast suite (no network)
MINICAM_SRC=~/dev/minicam uv run pytest tests/test_minicam_drift.py
uv run pytest -m export                         # full export, needs network
uv run minicam-models export && uv run minicam-models validate dist/*.onnx
```

## Test strategy
- Unit: contract helpers, catalog rules, pinned fetch, validator (synthetic
  ONNX models built in `tests/conftest.py`), release bundle.
- Drift: `test_minicam_drift.py` compares constants, COCO order, size cap,
  onnxruntime pin and preprocessing output bit-for-bit with a minicam checkout.
  CI runs it on every push/PR and weekly (Mondays 06:00 UTC) against
  `black-redoc/minicam@master` (repo variable `MINICAM_REPOSITORY`).
- End-to-end (`-m export`): exports yolov8n twice, validates, checks the INT8
  gate and byte-for-byte reproducibility.
- Release gate: every asset passes `minicam-models validate` in CI.

## Definition of Done
- [ ] All acceptance criteria have passing tests
- [ ] ruff, mypy, pytest (incl. drift and `-m export`) pass
- [ ] SECURITY.md checklist reviewed
- [ ] Spec, ROADMAP, PLAN, PROGRESS updated; commit done; graphs refreshed
