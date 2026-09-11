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
    parser.add_argument(
        "--scenario",
        choices=("simulation", "case_b"),
        default="simulation",
        help="Experiment scenario: paper simulation or case B. "
        "The heat migration scenario has its own entry point "
        "(migration/run_migration.py).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.scenario == "case_b":
        if args.mode != "sample-size":
            raise ValueError(
                "The case-B scenario currently supports --mode sample-size only."
            )
        from .scenarios.case_b import CaseBSurrogate, load_case_b_frames
        from .case_b_experiments import (
            run_case_b_experiment,
            save_case_b_artifacts,
        )

        data_root = config.get("case_b", {}).get(
            "data_root", "datasets/case_b_spotweld"
        )
        _, model_frame = load_case_b_frames(data_root)
        surrogate = CaseBSurrogate().fit(model_frame)
        artifacts = run_case_b_experiment(config, surrogate)
        output = save_case_b_artifacts(artifacts, config, args.output, surrogate)
    else:
        artifacts = run_experiments(config, args.mode)
        output = save_artifacts(artifacts, config, args.output)
    print(
        f"Wrote {len(artifacts.metrics)} metric rows and "
        f"{len(artifacts.predictions)} predictions to {output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
