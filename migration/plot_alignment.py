"""Alignment arms A0-A4 (b5): cross-arm comparison figures.

House-style figures (cf. reporting.py) for the small-scale alignment report
``migration/reports/迁移后续工作/PWL输出对齐A0-A4小范围实验报告.md``:

  1. RMSE sample-size curves (mean +/- std across batches) for A0-A4 with
     the Physics/GP references from the frozen A0 result set;
  2. per-arm RMSE boxplots across the 12 label sizes (batch spread);
  3. paired-difference boxplots (per-batch RMSE difference vs A0 / A1),
     the decision-relevant view for report §7.

A0 rows are the frozen b20 results subset to the arms' batches (1-5).
English axis labels follow the existing figure convention (CJK fonts are
not guaranteed on this toolchain).

Run:  uv run python migration/plot_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration" / "results"
OUTPUT = RESULTS / "alignment_b5"

ARMS = {
    "A0": "heat_v3_qint_min_b20",   # frozen identity baseline (subset)
    "A1": "heat_a1_linear_qint_b5",
    "A2": "heat_a2_linear_no_b_b5",
    "A3": "heat_a3_linear_no_qq_b5",
}
ARM_LABELS = {
    "A0": "A0: PWL identity + qint B",
    "A1": "A1: PWL linear phi + qint B",
    "A2": "A2: PWL linear phi, no B",
    "A3": "A3: PWL linear phi, no qint",
    "A4": "A4: mechanism-aligned OLS",
}
PAIRS = (
    ("A1", "A0"),
    ("A4", "A0"),
    ("A4", "A1"),
    ("A1", "A3"),
)


def load_model_sets() -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.read_csv(RESULTS / directory / "results.csv")
        for name, directory in ARMS.items()
    }
    sets = {
        name: frame[frame["model"] == "PWL"] for name, frame in frames.items()
    }
    repeats = sorted(sets["A1"]["repeat"].unique())
    sets = {
        name: frame[frame["repeat"].isin(repeats)]
        for name, frame in sets.items()
    }
    sets["A4"] = frames["A1"][
        (frames["A1"]["model"] == "MechanismAligned")
        & (frames["A1"]["repeat"].isin(repeats))
    ]
    a0 = frames["A0"][frames["A0"]["repeat"].isin(repeats)]
    for rival in ("Physics", "GP"):
        sets[rival] = a0[a0["model"] == rival]
    return sets


def plot_curves(sets: dict[str, pd.DataFrame], sizes: list[int]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    styles = {
        "A0": ("tab:blue", "o", "-"),
        "A1": ("tab:orange", "s", "-"),
        "A2": ("tab:gray", "^", "--"),
        "A3": ("tab:green", "d", "--"),
        "A4": ("tab:red", "D", "-"),
        "Physics": ("tab:purple", None, ":"),
        "GP": ("tab:brown", None, ":"),
    }
    for name in ("A0", "A1", "A2", "A3", "A4", "Physics", "GP"):
        frame = sets[name]
        summary = frame.groupby("n_labeled")["rmse"].agg(["mean", "std"])
        color, marker, linestyle = styles[name]
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"].fillna(0.0),
            marker=marker,
            color=color,
            linestyle=linestyle,
            capsize=2,
            label=ARM_LABELS.get(name, name),
        )
    axis.axhline(4.494, color="black", linewidth=1.0, linestyle="-.",
                 label="noise floor sigma")
    axis.set(xlabel="Number of labeled samples", ylabel="Test RMSE (K)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_alignment_rmse.png", dpi=180)
    plt.close(fig)


def plot_arm_boxplots(sets: dict[str, pd.DataFrame], sizes: list[int]) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for axis, name in zip(axes, ("A0", "A1", "A2", "A3", "A4")):
        frame = sets[name]
        values = [
            frame[frame["n_labeled"] == size]["rmse"].to_numpy()
            for size in sizes
        ]
        axis.boxplot(values, tick_labels=sizes, showfliers=False)
        axis.set_ylabel(name, rotation=0, labelpad=28)
        axis.grid(alpha=0.2)
        if name == "A4":
            axis.axhline(4.494, color="black", linewidth=0.8,
                         linestyle="-.", alpha=0.6)
    axes[-1].set_xlabel("Number of labeled samples")
    fig.suptitle("Per-batch test RMSE (K), batches 1-5")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_alignment_boxplots.png", dpi=180)
    plt.close(fig)


def plot_paired_boxplots(sets: dict[str, pd.DataFrame], sizes: list[int]) -> None:
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(10, 10), sharex=True)
    for axis, (left, right) in zip(axes, PAIRS):
        diffs = []
        for size in sizes:
            l_frame = sets[left][sets[left]["n_labeled"] == size].set_index(
                "repeat"
            )
            r_frame = sets[right][sets[right]["n_labeled"] == size].set_index(
                "repeat"
            )
            joined = l_frame[["rmse"]].join(
                r_frame[["rmse"]], lsuffix="_l", rsuffix="_r", how="inner"
            )
            diffs.append((joined["rmse_l"] - joined["rmse_r"]).to_numpy())
        axis.boxplot(diffs, tick_labels=sizes, showfliers=False)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylabel(f"{left} - {right}", rotation=0, labelpad=42)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Number of labeled samples")
    fig.suptitle("Paired per-batch RMSE difference (K), negative = left better")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_alignment_paired_diff.png", dpi=180)
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sets = load_model_sets()
    sizes = sorted(int(value) for value in sets["A0"]["n_labeled"].unique())
    plot_curves(sets, sizes)
    plot_arm_boxplots(sets, sizes)
    plot_paired_boxplots(sets, sizes)
    print(f"Wrote 3 figures to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
