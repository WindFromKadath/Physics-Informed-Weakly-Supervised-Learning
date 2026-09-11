"""M3 ablation analysis: compare each arm against the frozen A0 baseline.

Reads migration/results/<arm>/summary.csv + results.csv + quality_checks.csv
and prints markdown tables for the ablation report.  A0 is the frozen
baseline in migration/results/heat_v3_qint_min/ (BASELINE.md); the
n_weak=200 arm is exactly A0 by construction (no separate run).

Run:  uv run python migration/analyze_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration" / "results"

ARMS = {
    "A0 基线 (v3_qint_min)": "heat_v3_qint_min",
    "A1 无 B": "heat_v3_ablate_no_b",
    "A2 无弱标签": "heat_v3_ablate_no_weak",
    "A3 去 qint 两列": "heat_v3_ablate_no_qint",
    "A4 n_weak=50": "heat_v3_nweak_50",
    "A4 n_weak=100": "heat_v3_nweak_100",
    "A4 n_weak=200 (=A0)": "heat_v3_qint_min",
}
NOISE_VAR = 20.19  # sigma^2 ~= 20.2 K^2 (dataset spec v1.3)


def md_table(df: pd.DataFrame) -> str:
    """Dependency-free markdown table (pandas.to_markdown needs tabulate)."""

    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def load_summary(directory: Path) -> pd.DataFrame:
    df = pd.read_csv(directory / "summary.csv", header=[0, 1])
    df.columns = [
        "experiment", "model", "source_model", "n_labeled",
        "mse_mean", "mse_std", "rmse_mean", "rmse_std", "mae_mean", "mae_std",
    ]
    return df


def arm_row(name: str, directory: Path) -> dict[str, object]:
    summary = load_summary(directory)
    pwl = summary[summary["model"] == "PWL"].set_index("n_labeled")
    physics = summary[summary["model"] == "Physics"].set_index("n_labeled")
    results = pd.read_csv(directory / "results.csv")
    pwl120 = results[(results["model"] == "PWL") & (results["n_labeled"] == 120)]
    lambdas = pwl120["lambda_physics"].astype(float)
    boundary, spread = [], []
    for _, row in pwl120.iterrows():
        details = json.loads(row["details"])
        boundary.append(bool(details["theta_boundary_hit"]))
        spread.append(float(details["theta_estimate"][0]))
    qc = pd.read_csv(directory / "quality_checks.csv").set_index("check")
    invalid = float(qc.loc["invalid_pwl_candidate_fraction", "value"])
    rmse120 = float(pwl.loc[120, "rmse_mean"])
    return {
        "臂": name,
        "@10": float(pwl.loc[10, "rmse_mean"]),
        "@120": rmse120,
        "MSE/σ²@120": float(pwl.loc[120, "mse_mean"]) / NOISE_VAR,
        "Physics@120": float(physics.loc[120, "rmse_mean"]),
        "Δ vs Physics": float(physics.loc[120, "rmse_mean"]) - rmse120,
        "λ_physics 均值@120": float(lambdas.mean()),
        "无效候选率": invalid,
        "θ撞界率": float(np.mean(boundary)),
    }


def main() -> None:
    rows = [arm_row(name, RESULTS / directory) for name, directory in ARMS.items()]
    df = pd.DataFrame(rows)
    print("## 消融对照表（10 批均值测试 RMSE，K）\n")
    show = df.copy()
    for col in ("@10", "@120", "MSE/σ²@120", "Physics@120", "Δ vs Physics",
                "λ_physics 均值@120", "θ撞界率"):
        show[col] = show[col].map(lambda v: f"{v:.3f}")
    show["无效候选率"] = show["无效候选率"].map(lambda v: f"{v:.3f}")
    print(md_table(show))

    print("\n## A4 弱标签数量曲线（PWL RMSE）\n")
    curve = []
    for n_weak, directory in ((50, "heat_v3_nweak_50"),
                              (100, "heat_v3_nweak_100"),
                              (200, "heat_v3_qint_min")):
        summary = load_summary(RESULTS / directory)
        pwl = summary[summary["model"] == "PWL"].set_index("n_labeled")
        curve.append({
            "n_weak": n_weak,
            "@10": round(float(pwl.loc[10, "rmse_mean"]), 3),
            "@30": round(float(pwl.loc[30, "rmse_mean"]), 3),
            "@60": round(float(pwl.loc[60, "rmse_mean"]), 3),
            "@120": round(float(pwl.loc[120, "rmse_mean"]), 3),
        })
    print(md_table(pd.DataFrame(curve)))


if __name__ == "__main__":
    main()
