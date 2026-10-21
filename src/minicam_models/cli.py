"""Command line entry point: `minicam-models export|validate`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minicam_models.catalog import load_catalog
from minicam_models.validate import validate_model


def _validate(paths: list[Path]) -> int:
    failed = False
    for path in paths:
        report = validate_model(path)
        print(f"{'OK  ' if report.ok else 'FAIL'} {path}")
        for error in report.errors:
            print(f"     - {error}")
        failed |= not report.ok
    return 1 if failed else 0


def _export(args: argparse.Namespace) -> int:
    from minicam_models.export import export_model
    from minicam_models.release import ReleaseInfo, write_release

    catalog = load_catalog()
    models = (
        [catalog.get(m) for m in args.model] if args.model else list(catalog.models)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for model in models:
        results.extend(export_model(catalog, model, args.cache, args.out))

    exit_code = _validate([r.path for r in results])
    if exit_code == 0:
        write_release(results, ReleaseInfo(args.repository, args.tag), args.out)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minicam-models")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export, validate and bundle models")
    export.add_argument("--model", action="append", help="model id (default: all)")
    export.add_argument("--out", type=Path, default=Path("dist"))
    export.add_argument("--cache", type=Path, default=Path(".cache"))
    export.add_argument("--repository", default="local/minicam-models")
    export.add_argument("--tag", default="dev")

    validate = commands.add_parser("validate", help="check .onnx files against minicam")
    validate.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.paths)
    return _export(args)


if __name__ == "__main__":
    sys.exit(main())
