"""Blind test S1: pre-registered analysis (frozen before data generation).

Protocol freeze: migration/reports/迁移后续工作/PWL盲测S1预注册.md

Comparison family (fixed in advance; negative difference = left better):
  C1  MechanismAligned < PWL      (R4 confirmation: oracle grey-box vs PWL)
  C2  PWL < GP                    (framework vs strong supervised)
  C3  PWL < Physics               (framework vs two-step calibration + GP)
  C4  PWL < AffineAligned         (structured PWL vs pure global alignment)
  C5  PWL < PWL-noWeak            (weak-label channel value; audit §5.2.3)
  C6  MechanismAligned < AffineAligned  (mechanism terms vs pure alignment)

Endpoints (fixed):
  primary   : @120 RMSE, one-sided paired t per comparison, Holm across the
              6-comparison family;
  secondary : @120 MSE (same family/Holm), small-end (10-60) and large-end
              (70-120) segment mean RMSE, MSE/sigma^2, sign consistency,
              Cohen dz.  Per-size curve p-values are descriptive only.

Decision rules (fixed):
  D1  C1 significant (either direction, Holm) -> settles "A4 vs PWL" off the
      development data;
  D2  C2 and C3 both significant -> PWL keeps its anchor-6 claim on fresh
      batches;
  D3  C5 significant -> the weak-label channel carries measurable value;
  D4  C4 significant -> PWL's structured model is justified over pure affine
      alignment on this scenario.

Run (after unsealing):  uv run python migration/analyze_blind_s1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration" / "results"
SIGMA = 4.493844638867134
NOISE_VAR = SIGMA**2

SIZES = list(range(10, 121, 10))
SMALL_END = [s for s in SIZES if s <= 60]
LARGE_END = [s for s in SIZES if s >= 70]

COMPARISONS = (
    ("C1", "MechanismAligned", "PWL"),
    ("C2", "PWL", "GP"),
    ("C3", "PWL", "Physics"),
    ("C4", "PWL", "AffineAligned"),
    ("C5", "PWL", "PWL-noWeak"),
    ("C6", "MechanismAligned", "AffineAligned"),
)


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def paired(diff: np.ndarray) -> dict[str, float]:
    """One-sided paired t on ``left - right`` (negative = left better)."""

    n = diff.size
    mean = float(np.mean(diff))
    std = float(np.std(diff, ddof=1))
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, n - 1)
    ci = (mean - t_crit * se, mean + t_crit * se)
    t_stat, p_two = stats.ttest_1samp(diff, 0.0)
    p_one = float(p_two / 2.0) if t_stat < 0 else float(1.0 - p_two / 2.0)
    dz = mean / std if std > 0 else float("inf")
    return {
        "n": n, "mean": mean, "std": std,
        "ci_low": ci[0], "ci_high": ci[1],
        "t": float(t_stat), "p_one_sided": p_one, "cohens_dz": dz,
        "frac_negative": float(np.mean(diff < 0)),
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def main() -> None:
    main_frame = pd.read_csv(RESULTS / "heat_blind_s1" / "results.csv")
    no_weak = pd.read_csv(RESULTS / "heat_blind_s1_no_weak" / "results.csv")
    no_weak_pwl = no_weak[no_weak["model"] == "PWL"].copy()
    no_weak_pwl["model"] = "PWL-noWeak"
    frame = pd.concat([main_frame, no_weak_pwl], ignore_index=True)

    models = ("PWL", "PWL-noWeak", "MechanismAligned", "AffineAligned",
              "GP", "Physics", "PhysicsDirect")
    model_sets = {name: frame[frame["model"] == name] for name in models}

    print("== 盲测 S1：批 21-40 预注册分析（一次性解封）==\n")

    # ---- 12-size mean curves -------------------------------------------
    print("### 1. 12 档 × 20 批均值测试 RMSE（K）\n")
    rows = []
    for name in models:
        means = model_sets[name].groupby("n_labeled")["rmse"].mean()
        row = {"model": name}
        row.update({f"@{s}": round(float(means[s]), 3) for s in SIZES})
        rows.append(row)
    print(md_table(pd.DataFrame(rows)))

    # ---- primary endpoint @120 ------------------------------------------
    def paired_diff(left: str, right: str, size: int, metric: str) -> np.ndarray:
        wide_l = model_sets[left].pivot_table(
            index="repeat", columns="n_labeled", values=metric)
        wide_r = model_sets[right].pivot_table(
            index="repeat", columns="n_labeled", values=metric)
        joined = wide_l[size].to_frame("left").join(
            wide_r[size].to_frame("right"), how="inner")
        return (joined["left"] - joined["right"]).to_numpy()

    print("\n### 2. 主终点 @120（负值 = 前者更优；Holm 族 = 6 对照）\n")
    for metric in ("rmse", "mse"):
        family = [
            paired(paired_diff(left, right, 120, metric))
            for _, left, right in COMPARISONS
        ]
        p_adj = holm([item["p_one_sided"] for item in family])
        table = []
        for (label, left, right), item, p_h in zip(COMPARISONS, family, p_adj):
            table.append({
                "对照": f"{label} {left}<{right} @{metric}",
                "n": item["n"],
                "均值差": round(item["mean"], 4),
                "95% CI": f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
                "符号一致": f"{item['frac_negative']:.2f}",
                "dz": round(item["cohens_dz"], 3),
                "p(单侧)": f"{item['p_one_sided']:.4f}",
                "p(Holm6)": f"{p_h:.4f}",
            })
        print(md_table(pd.DataFrame(table)))
        print()

    # ---- segment endpoints ----------------------------------------------
    print("### 3. 小样本端（10-60）与大样本端（70-120）分段（RMSE）\n")
    table = []
    for label, left, right in COMPARISONS:
        for end_name, end_sizes in (("小样本端", SMALL_END),
                                    ("大样本端", LARGE_END)):
            wide_l = model_sets[left].pivot_table(
                index="repeat", columns="n_labeled", values="rmse"
            )[end_sizes].mean(axis=1)
            wide_r = model_sets[right].pivot_table(
                index="repeat", columns="n_labeled", values="rmse"
            )[end_sizes].mean(axis=1)
            joined = wide_l.to_frame("left").join(
                wide_r.to_frame("right"), how="inner")
            item = paired((joined["left"] - joined["right"]).to_numpy())
            table.append({
                "对照": f"{label} {left}<{right}", "样本端": end_name,
                "均值差": round(item["mean"], 3),
                "95% CI": f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
                "符号一致": f"{item['frac_negative']:.2f}",
                "p(单侧)": f"{item['p_one_sided']:.4f}",
            })
    print(md_table(pd.DataFrame(table)))

    # ---- MSE / sigma^2 ----------------------------------------------------
    print(f"\n### 4. MSE/sigma^2（sigma^2 = {NOISE_VAR:.3f}）\n")
    table = []
    for name in models:
        means = model_sets[name].groupby("n_labeled")["mse"].mean()
        table.append({
            "model": name,
            "@10": round(float(means[10]) / NOISE_VAR, 3),
            "@60": round(float(means[60]) / NOISE_VAR, 3),
            "@120": round(float(means[120]) / NOISE_VAR, 3),
        })
    print(md_table(pd.DataFrame(table)))

    # ---- phi audit on fresh batches --------------------------------------
    print("\n### 5. phi 系数（新批，@120，20 批 mean±std）\n")
    table = []
    for name in ("MechanismAligned", "AffineAligned"):
        subset = model_sets[name][model_sets[name]["n_labeled"] == 120]
        slopes, intercepts = [], []
        for _, row in subset.iterrows():
            details = json.loads(row["details"])
            slopes.append(float(details["phi_slope"]))
            intercepts.append(float(details["phi_intercept"]))
        table.append({
            "model": name,
            "截距 mean": round(float(np.mean(intercepts)), 2),
            "截距 std": round(float(np.std(intercepts)), 2),
            "斜率 mean": round(float(np.mean(slopes)), 4),
            "斜率 std": round(float(np.std(slopes)), 4),
        })
    print(md_table(pd.DataFrame(table)))

    # ---- quality gates -----------------------------------------------------
    print("\n### 6. 质量门槛\n")
    table = []
    for name, source in (("PWL", model_sets["PWL"]),
                         ("PWL-noWeak", model_sets["PWL-noWeak"])):
        converged, invalid, evaluated = [], 0, 0
        for _, row in source.iterrows():
            details = json.loads(row["details"])
            converged.append(bool(details["converged"]))
            tuning = details.get("tuning", {})
            evaluated += int(tuning.get("evaluated_candidates", 0))
            invalid += int(tuning.get("invalid_candidates", 0))
        table.append({
            "臂": name,
            "收敛率": round(float(np.mean(converged)), 4),
            "无效候选率": round(invalid / max(1, evaluated), 4),
        })
    print(md_table(pd.DataFrame(table)))

    # ---- pre-registered decision rules ------------------------------------
    print("\n### 7. 预注册判定（D1-D4）\n")
    rmse_family = [
        (label, left, right, paired(paired_diff(left, right, 120, "rmse")))
        for label, left, right in COMPARISONS
    ]
    rmse_holm = holm([item[3]["p_one_sided"] for item in rmse_family])
    verdict = {label: (item, p_h)
               for (label, _, _, item), p_h in zip(rmse_family, rmse_holm)}

    c1, c1_h = verdict["C1"]
    if c1["mean"] < 0 and c1_h < 0.05:
        d1 = (f"A4 显著优于 PWL（{c1['mean']:+.3f} K, p_holm={c1_h:.4f}）——"
              "R4 在新批上成立")
    elif c1["mean"] > 0:
        reverse_p = 1.0 - c1["p_one_sided"]
        d1 = (f"PWL 方向占优（{c1['mean']:+.3f} K；反向 p={reverse_p:.4f}）"
              "——R4 不成立")
    else:
        d1 = f"无显著差异（{c1['mean']:+.3f} K, p_holm={c1_h:.4f}）——不能定案"
    print(f"- D1（A4 vs PWL）: {d1}")

    c2, c2_h = verdict["C2"]
    c3, c3_h = verdict["C3"]
    d2 = (c2_h < 0.05 and c2["mean"] < 0) and (c3_h < 0.05 and c3["mean"] < 0)
    print(f"- D2（PWL 同时优于 GP 与 Physics）: {'成立' if d2 else '不成立'}"
          f"（C2 p_holm={c2_h:.4f}, C3 p_holm={c3_h:.4f}）")

    c5, c5_h = verdict["C5"]
    d3 = c5["mean"] < 0 and c5_h < 0.05
    print(f"- D3（弱标签通道有可测价值）: {'成立' if d3 else '不成立'}"
          f"（{c5['mean']:+.3f} K, p_holm={c5_h:.4f}）")

    c4, c4_h = verdict["C4"]
    d4 = c4["mean"] < 0 and c4_h < 0.05
    print(f"- D4（PWL 优于纯仿射对齐）: {'成立' if d4 else '不成立'}"
          f"（{c4['mean']:+.3f} K, p_holm={c4_h:.4f}）")


if __name__ == "__main__":
    main()
