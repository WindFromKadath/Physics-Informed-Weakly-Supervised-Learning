"""G0/G1/G2 + A4 single-panel comparison figure (dev batches 21-25).

Reporting figure for the S2 pre-study report
``migration/reports/迁移后续工作/PWL分层G0G1开发测试报告.md``:
mean +/- std RMSE curves over the 12 label sizes on dev batches 21-25.
Chinese labels (Microsoft YaHei) for presentation use.

Run:  uv run python migration/plot_gtiers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration" / "results"
OUTPUT = RESULTS / "gtiers_dev"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

TIERS = (
    ("G0", "G0：通用字典 PWL（无机理结构）", "tab:gray", "^", "--"),
    ("G1", "G1：工程先验 PWL（指数网格，无 0.7）", "tab:green", "d", "--"),
    ("G2", "G2：oracle 指数 PWL（精确 −0.7）", "tab:blue", "o", "-"),
    ("A4", "A4：机理对齐回归（OLS）", "tab:red", "D", "-"),
)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_csv(RESULTS / "heat_blind_s1" / "results.csv")
    s1 = s1[s1["repeat"] <= 4]  # batches 21-25
    frames = {
        "G0": pd.read_csv(RESULTS / "heat_g0_generic_dev" / "results.csv"),
        "G1": pd.read_csv(RESULTS / "heat_g1_prior_dev" / "results.csv"),
        "G2": s1[s1["model"] == "PWL"],
        "A4": s1[s1["model"] == "MechanismAligned"],
    }

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for name, label, color, marker, linestyle in TIERS:
        summary = frames[name].groupby("n_labeled")["rmse"].agg(["mean", "std"])
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0.0),
            marker=marker,
            color=color,
            linestyle=linestyle,
            capsize=2,
            label=label,
        )
    axis.axhline(
        4.494, color="black", linewidth=1.0, linestyle="-.", label="噪声下限 σ"
    )
    axis.set(
        xlabel="标签样本数",
        ylabel="测试 RMSE（K）",
        title="G0–G2 分层 PWL 与 A4（批 21–25，开发数据，方向性）",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    destination = OUTPUT / "figure_gtiers_rmse.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
