"""Alignment arms A0-A4: formal 20-batch paired analysis (report §6.3).

Implements the acceptance metrics of
``migration/reports/迁移后续工作/PWL物理模型输出对齐与适用性说明.md`` §6.3 and
the decision rules of §7:

  * arms: A0 = frozen identity baseline (heat_v3_qint_min_b20, read-only),
    A1 = PWL linear + v3 qint, A2 = PWL linear + intercept B,
    A3 = PWL linear + no_qq, A4 = MechanismAligned OLS (stored in every new
    arm directory; identical rows, read from the A1 directory);
  * pairing unit: per-batch paired difference on the same split and the same
    acceptance set (batch == repeat), exactly as the M4 closure protocol;
  * primary endpoints: @120 RMSE and MSE per comparison, one-sided paired
    t test, Holm correction inside each comparison's {rmse, mse} family;
  * curve view: per-size paired differences with unadjusted p and Holm
    across the 12-size family;
  * small/large ends: per-batch mean RMSE within sizes 10-60 vs 70-120,
    then paired across batches;
  * phi audit: intercept/slope batch distributions for A1 (fitted at the
    calibrated theta) and A4 (fitted at k_nominal), plus the selected
    lambda_physics shift and the theta_hat-phi_slope correlation (§6.2);
  * MSE/sigma^2 ratios; the three frozen mechanism ablations (10 batches,
    identity mapping) are quoted as context.

Run:  uv run python migration/analyze_alignment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration" / "results"
SIGMA = 4.493844638867134
NOISE_VAR = SIGMA**2

# Result-directory suffix of the new arms: "b5" for the small-scale test
# (run_alignment_b5.py), "b20" for the canonical protocol run.  A0 is always
# the frozen 20-batch baseline, subset to the arms' batches for pairing.
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "b20"
ARMS = {
    "A0": "heat_v3_qint_min_b20",   # frozen identity baseline
    "A1": f"heat_a1_linear_qint_{SUFFIX}",
    "A2": f"heat_a2_linear_no_b_{SUFFIX}",
    "A3": f"heat_a3_linear_no_qq_{SUFFIX}",
}
ABLATIONS = {
    "no_weak": "heat_v3_ablate_no_weak",
    "no_b": "heat_v3_ablate_no_b",
    "no_qint": "heat_v3_ablate_no_qint",
}
SIZES = list(range(10, 121, 10))
SMALL_END = [s for s in SIZES if s <= 60]
LARGE_END = [s for s in SIZES if s >= 70]


def md_table(df: pd.DataFrame) -> str:
    """Dependency-free markdown table (pandas.to_markdown needs tabulate)."""

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


def load_arm(directory: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / directory / "results.csv")


def model_frame(arm: pd.DataFrame, model: str) -> pd.DataFrame:
    return arm[arm["model"] == model]


def main() -> None:
    frames = {name: load_arm(directory) for name, directory in ARMS.items()}
    a0 = frames["A0"]
    # A4 rows are identical across the new arm directories; read A1's copy.
    a4 = model_frame(frames["A1"], "MechanismAligned")
    arms_pwl = {name: model_frame(frame, "PWL") for name, frame in frames.items()}
    model_sets = {**arms_pwl, "A4": a4}

    # Pairing set: the new arms define the participating batches; every frame
    # (including frozen A0 and its rival rows) is subset to the same batches
    # so all means and differences are paired on identical data.
    repeats = sorted(int(value) for value in arms_pwl["A1"]["repeat"].unique())
    n_batches = len(repeats)
    model_sets = {
        name: frame[frame["repeat"].isin(repeats)]
        for name, frame in model_sets.items()
    }
    a0 = a0[a0["repeat"].isin(repeats)]

    print(f"== A0-A4 对齐实验臂：{n_batches} 批配对分析（报告 §6.3 协议，"
          f"批次 {repeats[0]}-{repeats[-1]}）==\n")

    # ---- 1. 12-size mean RMSE/MSE curves -------------------------------
    print(f"### 1. 12 档 × {n_batches} 批均值测试 RMSE（K）"
          "（A0 及其基线已限同批）\n")
    curve_rows = []
    for name, frame in model_sets.items():
        means = frame.groupby("n_labeled")["rmse"].mean()
        row = {"arm": name}
        row.update({f"@{s}": round(float(means[s]), 3) for s in SIZES})
        curve_rows.append(row)
    for rival in ("Physics", "GP", "PhysicsDirect"):
        frame = model_frame(a0, rival)
        means = frame.groupby("n_labeled")["rmse"].mean()
        row = {"arm": f"A0:{rival}"}
        row.update({f"@{s}": round(float(means[s]), 3) for s in SIZES})
        curve_rows.append(row)
    print(md_table(pd.DataFrame(curve_rows)))

    print(f"\n### 1b. 12 档 × {n_batches} 批均值测试 MSE（K^2）\n")
    curve_rows = []
    for name, frame in model_sets.items():
        means = frame.groupby("n_labeled")["mse"].mean()
        row = {"arm": name}
        row.update({f"@{s}": round(float(means[s]), 2) for s in SIZES})
        curve_rows.append(row)
    print(md_table(pd.DataFrame(curve_rows)))

    # ---- 2. Paired comparisons ------------------------------------------
    # (label, left, right): negative difference = left better.
    comparisons = [
        ("A1<A0", "A1", "A0"),
        ("A2<A0", "A2", "A0"),
        ("A1<A2", "A1", "A2"),
        ("A1<A3", "A1", "A3"),
        ("A4<A0", "A4", "A0"),
        ("A4<A1", "A4", "A1"),
    ]

    def paired_diff(left: str, right: str, size: int, metric: str) -> np.ndarray:
        wide_left = model_sets[left].pivot_table(
            index="repeat", columns="n_labeled", values=metric
        )
        wide_right = model_sets[right].pivot_table(
            index="repeat", columns="n_labeled", values=metric
        )
        joined = wide_left[size].to_frame("left").join(
            wide_right[size].to_frame("right"), how="inner"
        )
        return (joined["left"] - joined["right"]).to_numpy()

    print("\n### 2. 主终点配对统计（@120，负值 = 前者更优）\n")
    table = []
    for label, left, right in comparisons:
        family = [paired(paired_diff(left, right, 120, metric))
                  for metric in ("rmse", "mse")]
        p_adj = holm([item["p_one_sided"] for item in family])
        for metric, item, p_h in zip(("rmse", "mse"), family, p_adj):
            table.append({
                "对照": f"{label} @120 {metric}",
                "n": item["n"],
                "均值差": round(item["mean"], 4),
                "std": round(item["std"], 4),
                "95% CI": f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
                "符号一致": f"{item['frac_negative']:.2f}",
                "dz": round(item["cohens_dz"], 3),
                "p(单侧)": f"{item['p_one_sided']:.4f}",
                "p(Holm)": f"{p_h:.4f}",
            })
    print(md_table(pd.DataFrame(table)))

    print("\n### 2b. 全曲线配对差（RMSE；p 未校正 / Holm 跨 12 档）\n")
    for label, left, right in comparisons:
        per_size = [paired(paired_diff(left, right, s, "rmse")) for s in SIZES]
        p_adj = holm([item["p_one_sided"] for item in per_size])
        table = []
        for s, item, p_h in zip(SIZES, per_size, p_adj):
            table.append({
                "n_labeled": s,
                "均值差": round(item["mean"], 3),
                "95% CI": f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
                "符号一致": f"{item['frac_negative']:.2f}",
                "p(单侧)": f"{item['p_one_sided']:.4f}",
                "p(Holm12)": f"{p_h:.4f}",
            })
        print(f"对照 {label}（负值 = {left} 更优）:")
        print(md_table(pd.DataFrame(table)))
        print()

    # ---- 3. Small vs large sample ends ----------------------------------
    print("### 3. 小样本端（10-60）与大样本端（70-120）分段配对（RMSE）\n")
    table = []
    for label, left, right in comparisons:
        for end_name, end_sizes in (("小样本端", SMALL_END),
                                    ("大样本端", LARGE_END)):
            wide_left = model_sets[left].pivot_table(
                index="repeat", columns="n_labeled", values="rmse"
            )[end_sizes].mean(axis=1)
            wide_right = model_sets[right].pivot_table(
                index="repeat", columns="n_labeled", values="rmse"
            )[end_sizes].mean(axis=1)
            joined = wide_left.to_frame("left").join(
                wide_right.to_frame("right"), how="inner"
            )
            item = paired((joined["left"] - joined["right"]).to_numpy())
            table.append({
                "对照": label, "样本端": end_name,
                "均值差": round(item["mean"], 3),
                "95% CI": f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
                "符号一致": f"{item['frac_negative']:.2f}",
                "dz": round(item["cohens_dz"], 3),
                "p(单侧)": f"{item['p_one_sided']:.4f}",
            })
    print(md_table(pd.DataFrame(table)))

    # ---- 4. phi audit (report §6.2/§6.3) ---------------------------------
    print("\n### 4. phi 系数审计\n")

    def phi_details(frame: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in frame.iterrows():
            details = json.loads(row["details"])
            records.append({
                "repeat": int(row["repeat"]),
                "n_labeled": int(row["n_labeled"]),
                # Frozen A0 rows predate phi recording; identity mapping is
                # exactly intercept 0 / slope 1.
                "phi_intercept": float(details.get("phi_intercept", 0.0)),
                "phi_slope": float(details.get("phi_slope", 1.0)),
                "theta_hat": float(details.get("theta_estimate", [np.nan])[0]),
                "lambda_physics": float(
                    json.loads(row["hyperparameters"]).get(
                        "lambda_physics", np.nan)
                ),
            })
        return pd.DataFrame(records)

    a1_phi = phi_details(arms_pwl["A1"])
    a4_phi = phi_details(a4)
    a0_theta = phi_details(arms_pwl["A0"])
    for name, phi in (("A1 (PWL linear, phi@校准theta)", a1_phi),
                      ("A4 (OLS, phi@k_nominal)", a4_phi)):
        grouped = phi.groupby("n_labeled")[["phi_intercept", "phi_slope"]]
        summary = grouped.agg(["mean", "std"])
        print(f"{name}：按标签档的 phi 截距/斜率（{n_batches} 批 mean±std）")
        table = []
        for s in SIZES:
            table.append({
                "n_labeled": s,
                "截距 mean": round(float(summary.loc[s, ("phi_intercept", "mean")]), 2),
                "截距 std": round(float(summary.loc[s, ("phi_intercept", "std")]), 2),
                "斜率 mean": round(float(summary.loc[s, ("phi_slope", "mean")]), 4),
                "斜率 std": round(float(summary.loc[s, ("phi_slope", "std")]), 4),
            })
        print(md_table(pd.DataFrame(table)))
        print()

    merged = a1_phi.merge(
        a0_theta[["repeat", "n_labeled", "theta_hat", "lambda_physics"]],
        on=["repeat", "n_labeled"], suffixes=("_a1", "_a0"),
    )
    corr_theta_phi = float(
        np.corrcoef(merged["theta_hat_a1"], merged["phi_slope"])[0, 1]
    )
    print(f"- A1 theta_hat 与 phi_slope 逐条件相关：{corr_theta_phi:.3f}"
          "（§6.2 不可识别性检查；|r| 接近 1 才告警）")
    lam = merged.groupby("n_labeled")[["lambda_physics_a0",
                                       "lambda_physics_a1"]].median()
    print("- 选中 lambda_physics 中位数（A0 vs A1，按档）：")
    print(md_table(pd.DataFrame({
        "n_labeled": SIZES,
        "A0 中位": lam["lambda_physics_a0"].to_numpy(),
        "A1 中位": lam["lambda_physics_a1"].to_numpy(),
    })))

    # ---- 5. MSE / sigma^2 ------------------------------------------------
    print("\n### 5. MSE/sigma^2 比值（sigma^2 = "
          f"{NOISE_VAR:.3f}）\n")
    table = []
    for name, frame in model_sets.items():
        means = frame.groupby("n_labeled")["mse"].mean()
        table.append({
            "arm": name,
            "@10": round(float(means[10]) / NOISE_VAR, 3),
            "@60": round(float(means[60]) / NOISE_VAR, 3),
            "@120": round(float(means[120]) / NOISE_VAR, 3),
        })
    print(md_table(pd.DataFrame(table)))

    # ---- 6. Frozen mechanism ablations (context, 10 batches, identity) ---
    print("\n### 6. 机制消融（既有冻结结果，10 批、identity 口径，仅作背景）\n")
    a0_10 = pd.read_csv(RESULTS / "heat_v3_qint_min" / "results.csv")
    base = model_frame(a0_10, "PWL").groupby("n_labeled")["rmse"].mean()
    table = [{
        "臂": "A0(10批)", "@10": round(float(base[10]), 3),
        "@60": round(float(base[60]), 3), "@120": round(float(base[120]), 3),
    }]
    for name, directory in ABLATIONS.items():
        frame = pd.read_csv(RESULTS / directory / "results.csv")
        means = model_frame(frame, "PWL").groupby("n_labeled")["rmse"].mean()
        table.append({
            "臂": name, "@10": round(float(means[10]), 3),
            "@60": round(float(means[60]), 3),
            "@120": round(float(means[120]), 3),
        })
    print(md_table(pd.DataFrame(table)))

    # ---- 7. Quality gates -------------------------------------------------
    print("\n### 7. 质量门槛（新臂）\n")
    table = []
    for name in ("A1", "A2", "A3"):
        frame = arms_pwl[name]
        converged, invalid, evaluated = [], 0, 0
        for _, row in frame.iterrows():
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

    # ---- 8. Decision rules (report §7) ------------------------------------
    print("\n### 8. §7 决策规则逐条判定\n")
    a1_a0_120 = paired(paired_diff("A1", "A0", 120, "rmse"))
    a4_a0_120 = paired(paired_diff("A4", "A0", 120, "rmse"))
    a1_a3_120 = paired(paired_diff("A1", "A3", 120, "rmse"))
    a4_a0_sizes = [paired(paired_diff("A4", "A0", s, "rmse")) for s in SIZES]
    small = paired((model_sets["A4"].pivot_table(
        index="repeat", columns="n_labeled", values="rmse")[SMALL_END].mean(axis=1)
        - model_sets["A0"].pivot_table(
        index="repeat", columns="n_labeled", values="rmse")[SMALL_END].mean(axis=1)
    ).dropna().to_numpy())
    large = paired((model_sets["A4"].pivot_table(
        index="repeat", columns="n_labeled", values="rmse")[LARGE_END].mean(axis=1)
        - model_sets["A0"].pivot_table(
        index="repeat", columns="n_labeled", values="rmse")[LARGE_END].mean(axis=1)
    ).dropna().to_numpy())
    print(f"- R1（A1 稳定优于 A0 → identity 不是合理主配置）: "
          f"A1-A0 @120 = {a1_a0_120['mean']:+.3f} K, "
          f"p={a1_a0_120['p_one_sided']:.4f}, "
          f"符号一致 {a1_a0_120['frac_negative']:.2f}")
    print(f"- R2（A1 只改善小样本端 → phi 起弱标签校准/稳定作用）: 见 §3 分段")
    print(f"- R3（A3 明显差于 A1 → 对齐后 qint 仍必要）: "
          f"A1-A3 @120 = {a1_a3_120['mean']:+.3f} K, "
          f"p={a1_a3_120['p_one_sided']:.4f}")
    print(f"- R4（A4 稳定优于 PWL → 改结论为机理对齐回归更适合）: "
          f"A4-A0 @120 = {a4_a0_120['mean']:+.3f} K, "
          f"p={a4_a0_120['p_one_sided']:.4f}, "
          f"12 档全胜={all(item['mean'] < 0 for item in a4_a0_sizes)}")
    print(f"- R5（PWL 小样本端优、大样本端被追平 → 价值定位为标签效率）: "
          f"A4-A0 小端 {small['mean']:+.3f} K (p={small['p_one_sided']:.4f}), "
          f"大端 {large['mean']:+.3f} K (p={large['p_one_sided']:.4f})")
    print("- R6（完整强基线集合 + 多重检验通过 → 才可称 PWL 统计优势）: "
          "见 §2 Holm 列")


if __name__ == "__main__":
    main()
