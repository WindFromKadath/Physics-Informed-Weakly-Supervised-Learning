"""M3 smoke gate: run every ablation arm at 2 batches x 3 sizes.

Loads each ablation config, overrides only `heat.batches` and
`experiment.sample_sizes` in memory (the override is recorded in each run's
metadata.json), and writes to `migration/results/<arm>_smoke/`.
Full 10-batch runs use the yaml configs unchanged.

Run:  uv run python migration/run_ablation_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))

from pwl_migration.heat_experiments import (  # noqa: E402
    run_heat_sample_size_experiment,
    save_heat_artifacts,
)

ARMS = (
    "heat_v3_ablate_no_b",
    "heat_v3_ablate_no_weak",
    "heat_v3_ablate_no_qint",
    "heat_v3_nweak_50",
    "heat_v3_nweak_100",
)


def main() -> int:
    for arm in ARMS:
        config_path = ROOT / "migration" / "configs" / f"{arm}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["heat"]["batches"] = [1, 2]
        config["experiment"]["sample_sizes"] = [10, 60, 120]
        output = ROOT / "migration" / "results" / f"{arm}_smoke"
        artifacts = run_heat_sample_size_experiment(config)
        save_heat_artifacts(artifacts, config, output)
        print(f"[smoke] {arm}: {len(artifacts.metrics)} metric rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
