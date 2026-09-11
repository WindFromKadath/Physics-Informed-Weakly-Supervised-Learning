"""M4 statistical closure: 20-batch paired analysis of the v3 baseline.

Pre-registered protocol (implementation plan M4.2, fixed before batches 11-20
were inspected):

  * pairing unit: per-batch paired PWL-vs-Physics difference on the same test
    set (same batch, same split, same acceptance set);
  * primary endpoint: @120 RMSE, comparison PWL < Physics (one-sided paired
    t test, Holm correction across the RMSE primary and MSE secondary);
  * secondary endpoints: full 12-size curve, small-sample advantage @10,
    MSE/noise ratio @120;
  * report: mean, std, 95% CI, paired p, Cohen's dz, Holm p;
  * terminal wording follows the M4.3 table verbatim.

Run:  uv run python migration/analyze_b20.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration" / "results" / "heat_v3_qint_min_b20"
NOISE_VAR = 20.19


def md_table(df: pd.DataFrame) -> str:
    """Dependency-free markdown table (pandas.to_markdown needs tabulate)."""

    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def paired(diff: np.ndarray) -> dict[str, float]:
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
    results = pd.read_csv(RESULTS / "results.csv")
    summary = pd.read_csv(RESULTS / "summary.csv", header=[0, 1])
    summary.columns = [
        "experiment", "model", "source_model", "n_labeled",
        "mse_mean", "mse_std", "rmse_mean", "rmse_std", "mae_mean", "mae_std",
    ]
    sizes = sorted(summary["n_labeled"].unique())

    print("== M4: 20-batch statistical closure (pre-registered protocol) ==\n")
    print("### 12 档均值 RMSE 曲线（20 批）\n")
    pivot = summary.pivot_table(index="model", columns="n_labeled", values="rmse_mean")
    key_models = ["PWL", "Physics", "GP", "GBDT", "Ridge", "PhysicsDirect"]
    print(md_table(pivot.loc[key_models].round(3)))

    print("\n### 配对统计（PWL − Physics，负值 = PWL 更优）\n")
    comparisons = []
    for metric in ("rmse", "mse"):
        wide = results.pivot_table(
            index=["repeat", "n_labeled"], columns="model", values=metric
        )
        diff120 = (wide.xs(120, level="n_labeled")["PWL"]
                   - wide.xs(120, level="n_labeled")["Physics"]).to_numpy()
        comparisons.append((f"PWL<Physics @120 {metric}", paired(diff120)))
    p_holm = holm([c[1]["p_one_sided"] for c in comparisons])
    table = []
    for (name, stats_dict), p_adj in zip(comparisons, p_holm):
        table.append({
            "对照": name, "n": stats_dict["n"],
            "均值差": round(stats_dict["mean"], 4),
            "std": round(stats_dict["std"], 4),
            "95% CI": f"[{stats_dict['ci_low']:.3f}, {stats_dict['ci_high']:.3f}]",
            "dz": round(stats_dict["cohens_dz"], 3),
            "p(单侧)": f"{stats_dict['p_one_sided']:.4f}",
            "p(Holm)": f"{p_adj:.4f}",
        })
    print(md_table(pd.DataFrame(table)))

    print("\n### 次要终点\n")
    pwl = summary[summary["model"] == "PWL"].set_index("n_labeled")
    physics = summary[summary["model"] == "Physics"].set_index("n_labeled")
    supervised = summary[summary["model"].isin(
        ["Ridge", "SVR", "DT", "RF", "GBDT", "GP"]
    )]
    best_sup = supervised.groupby("n_labeled")["rmse_mean"].min()
    print(f"- 小样本优势：gap@10 = {best_sup[10] - pwl.loc[10, 'rmse_mean']:.3f},"
          f" gap@120 = {best_sup[120] - pwl.loc[120, 'rmse_mean']:.3f}")
    print(f"- MSE/σ² @120 = {pwl.loc[120, 'mse_mean'] / NOISE_VAR:.3f}"
          f"（σ² ≈ {NOISE_VAR}）")
    print(f"- PWL 全档最优档数："
          f"{int(sum(pwl.loc[s, 'rmse_mean'] <= best_sup[s] and pwl.loc[s, 'rmse_mean'] <= physics.loc[s, 'rmse_mean'] for s in sizes))}/{len(sizes)}")

    print("\n### 终态表述（M4.3 规则表）\n")
    primary = comparisons[0][1]
    p_primary_holm = p_holm[0]
    if primary["mean"] < 0 and p_primary_holm < 0.05:
        verdict = ("在当前协议下，PWL 对 Physics-GP 具有统计显著优势"
                   f"（均值差 {primary['mean']:.3f} K，p_holm={p_primary_holm:.4f}）")
    elif primary["mean"] < 0:
        verdict = (f"PWL 保持均值优势趋势（均值差 {primary['mean']:.3f} K），"
                   f"但尚无充分显著性证据（p_holm={p_primary_holm:.4f}）")
    elif primary["ci_low"] < 0 < primary["ci_high"] and abs(primary["cohens_dz"]) < 0.2:
        verdict = ("PWL 与 Physics-GP 性能相当"
                   f"（CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}] 跨越零，"
                   f"dz={primary['cohens_dz']:.3f}）")
    else:
        verdict = ("PWL 均值不再领先——撤回锚点 6 的优势表述，"
                   f"报告批次扩展后的真实结论（均值差 {primary['mean']:.3f} K）")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
