"""Command-line entry point for reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .experiments import run_experiments, save_artifacts

REPRODUCTION_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PWL synthetic reproduction experiments."
    )
    parser.add_argument(
        "--config",
        default=str(REPRODUCTION_ROOT / "configs" / "default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--output",
        default=str(REPRODUCTION_ROOT / "results" / "latest"),
        help="Directory for CSV, metadata, and plots.",
    )
    parser.add_argument(
        "--mode",
        choices=("sample-size", "physics-accuracy", "label-savings", "all"),
        default="sample-size",
        help="Synthetic experiment family to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    artifacts = run_experiments(config, args.mode)
    output = save_artifacts(artifacts, config, args.output)
    print(
        f"Wrote {len(artifacts.metrics)} metric rows and "
        f"{len(artifacts.predictions)} predictions to {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
