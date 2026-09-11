"""Alignment-arms smoke gate: run A1/A2/A3 at 2 batches x 3 sizes.

Same convention as run_ablation_smoke.py: load each arm config, override
only `heat.batches` and `experiment.sample_sizes` in memory, and write to
`migration/results/<arm>_smoke/`.  Full 20-batch runs use the yaml configs
unchanged.

Run:  uv run python migration/run_alignment_smoke.py
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
    "heat_a1_linear_qint_b20",
    "heat_a2_linear_no_b_b20",
    "heat_a3_linear_no_qq_b20",
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
