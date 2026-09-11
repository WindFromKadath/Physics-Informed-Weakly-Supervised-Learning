"""Alignment arms A1-A3: small-scale 5-batch test run.

Same convention as run_ablation_smoke.py: load each arm yaml, override only
`heat.batches` in memory (recorded in each run's metadata.json), keep the
full 12-size curve, and write to `migration/results/<arm>_b5/`.  The yaml
configs remain the canonical 20-batch protocol (report §6); this runner is
the interim small-scale check requested before committing to the long run.

Run:  uv run python migration/run_alignment_b5.py
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
BATCHES = [1, 2, 3, 4, 5]


def main() -> int:
    for arm in ARMS:
        config_path = ROOT / "migration" / "configs" / f"{arm}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["heat"]["batches"] = BATCHES
        output = ROOT / "migration" / "results" / f"{arm.replace('_b20', '_b5')}"
        artifacts = run_heat_sample_size_experiment(config)
        save_heat_artifacts(artifacts, config, output)
        print(f"[b5] {arm}: {len(artifacts.metrics)} metric rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
