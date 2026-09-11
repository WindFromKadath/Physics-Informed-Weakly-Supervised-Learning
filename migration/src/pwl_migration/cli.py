"""Command-line entry point for the heat-conduction migration experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from pwl_repro.cli import load_config

MIGRATION_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PWL heat-conduction migration experiments."
    )
    parser.add_argument(
        "--config",
        default=str(MIGRATION_ROOT / "configs" / "heat_default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--output",
        default=str(MIGRATION_ROOT / "results" / "latest"),
        help="Directory for CSV, metadata, and plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    from .heat_experiments import (
        run_heat_sample_size_experiment,
        save_heat_artifacts,
    )

    artifacts = run_heat_sample_size_experiment(config)
    output = save_heat_artifacts(artifacts, config, args.output)
    print(
        f"Wrote {len(artifacts.metrics)} metric rows and "
        f"{len(artifacts.predictions)} predictions to {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
