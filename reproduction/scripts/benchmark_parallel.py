"""Benchmark deterministic serial and parallel PWL candidate evaluation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import yaml

from pwl_repro.experiments import run_sample_size_experiment


def run_once(base: dict, n_jobs: int) -> dict[str, object]:
    config = deepcopy(base)
    config["experiment"]["repeats"] = 1
    config["experiment"]["sample_sizes"] = [40]
    config["baselines"] = []
    config["lambda_grid"] = {
        "physics": [0.1, 1.0, 10.0],
        "l1": [0.01, 0.1, 1.0],
        "group": [0.01, 0.1, 1.0],
    }
    config["model"]["n_jobs"] = n_jobs
    config["model"]["parallel_verbose"] = 0
    started = perf_counter()
    artifacts = run_sample_size_experiment(
        config,
        sample_sizes=[40],
        pwl_sizes={40},
        baseline_sizes=set(),
    )
    elapsed = perf_counter() - started
    row = artifacts.metrics.iloc[0]
    return {
        "n_jobs": n_jobs,
        "seconds": elapsed,
        "hyperparameters": json.loads(row["hyperparameters"]),
        "validation_mse": float(row["validation_mse"]),
        "test_rmse": float(row["rmse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="reproduction/configs/paper_single_run.yaml",
    )
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    serial = run_once(base, 1)
    parallel = run_once(base, args.jobs)
    output = {
        "serial": serial,
        "parallel": parallel,
        "speedup": serial["seconds"] / parallel["seconds"],
        "selection_matches": (
            serial["hyperparameters"] == parallel["hyperparameters"]
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
