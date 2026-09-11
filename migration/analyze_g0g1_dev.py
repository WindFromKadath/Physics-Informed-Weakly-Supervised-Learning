"""S2 pre-study (DEVELOPMENT scale): G0/G1 tiers on batches 21-25.

Directional analysis of the G0 (generic dictionary) and G1 (engineering
prior, no oracle exponent) PWL tiers against the frozen references from the
S1 blind run (same batches, identical split protocol):

  G2  = PWL from heat_blind_s1 (v3 qint, oracle exponent)   [L2]
  G1  = PWL from heat_g1_prior_dev (power grid without 0.7) [L1+]
  G0  = PWL from heat_g0_generic_dev (generic quadratic)    [L1]
  A4  = MechanismAligned, AA = AffineAligned, GP, Physics   [references]

Batches 21-25 are post-S1 development data: every number here is a
development-stage, directional figure (n=5 pairs), NOT a confirmation.
The formal S2 requires fresh batches and a frozen pre-registration.

Run:  uv run python migration/analyze_g0g1_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))
RESULTS = ROOT / "migration" / "results"
SIGMA = 4.493844638867134
NOISE_VAR = SIGMA**2
SIZES = list(range(10, 121, 10))
SMALL_END = [s for s in SIZES if s <= 60]
LARGE_END = [s for s in SIZES if s >= 70]

COMPARISONS = (
    ("G1<G0", "G1", "G0"),        # value of the engineering prior
    ("G2<G1", "G2", "G1"),        # value of the oracle exponent
    ("G2<G0", "G2", "G0"),        # total mechanism-knowledge value
    ("G0<GP", "G0", "GP"),        # framework vs strong supervised
    ("G1<GP", "G1", "GP"),
    ("G0<AA", "G0", "AffineAligned"),   # generic PWL vs pure alignment
    ("A4<G1", "A4", "G1"),        # oracle grey-box vs engineering tier
)


def md_table(df: pd.DataFrame) -> str:
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
    return {
        "n": n, "mean": mean, "ci_low": ci[0], "ci_high": ci[1],
        "p_one_sided": p_one,
        "frac_negative": float(np.mean(diff < 0)),
        "cohens_dz": mean / std if std > 0 else float("inf"),
    }


def main() -> None:
    s1 = pd.read_csv(RESULTS / "heat_blind_s1" / "results.csv")
    s1 = s1[s1["repeat"] <= 4]  # batches 21-25
    g0 = pd.read_csv(RESULTS / "heat_g0_generic_dev" / "results.csv")
    g1 = pd.read_csv(RESULTS / "heat_g1_prior_dev" / "results.csv")

    model_sets = {
        "G2": s1[s1["model"] == "PWL"],
        "G1": g1[g1["model"] == "PWL"],
        "G0": g0[g0["model"] == "PWL"],
        "A4": s1[s1["model"] == "MechanismAligned"],
        "AffineAligned": s1[s1["model"] == "AffineAligned"],
        "GP": s1[s1["model"] == "GP"],
        "Physics": s1[s1["model"] == "Physics"],
    }

    # Split-consistency guard: the dev arms must share the S1 splits.
    cond_ref = pd.read_csv(RESULTS / "heat_blind_s1" / "conditions.csv")
    cond_ref = cond_ref[cond_ref["condition_type"] == "split"]
    key_cols = ["repeat", "n_labeled", "train_indices", "validation_indices"]
    ref_keys = set(map(tuple, cond_ref[key_cols].to_numpy()))
    for name, directory in (("G0", "heat_g0_generic_dev"),
                            ("G1", "heat_g1_prior_dev")):
        cond = pd.read_csv(RESULTS / directory / "conditions.csv")
        cond = cond[cond["condition_type"] == "split"]
        shared = set(map(tuple, cond[key_cols].to_numpy())) & ref_keys
        print(f"split consistency {name} vs S1: {len(shared)}/"
              f"{len(cond)} splits identical")
    print()

    print("== G0/G1 开发期方向性分析（批 21-25，n=5 对；非确认性证据）==\n")
    print("### 1. 12 档均值测试 RMSE（K）\n")
    rows = []
    for name in ("G0", "G1", "G2", "A4", "AffineAligned", "GP", "Physics"):
        means = model_sets[name].groupby("n_labeled")["rmse"].mean()
        row = {"model": name}
        row.update({f"@{s}": round(float(means[s]), 3) for s in SIZES})
        rows.append(row)
    print(md_table(pd.DataFrame(rows)))

    def paired_diff(left: str, right: str, size: int) -> np.ndarray:
        wide_l = model_sets[left].pivot_table(
            index="repeat", columns="n_labeled", values="rmse")
        wide_r = model_sets[right].pivot_table(
            index="repeat", columns="n_labeled", values="rmse")
        joined = wide_l[size].to_frame("left").join(
            wide_r[size].to_frame("right"), how="inner")
        return (joined["left"] - joined["right"]).to_numpy()

    print("\n### 2. 配对差 @120（负值 = 前者更优；n=5，仅方向参考）\n")
    table = []
    for label, left, right in COMPARISONS:
        item = paired(paired_diff(left, right, 120))
        table.append({
            "对照": label,
            "均值差": round(item["mean"], 3),
            "95% CI": f"[{item['ci_low']:.3f}, {item['ci_high']:.3f}]",
            "符号一致": f"{item['frac_negative']:.2f}",
            "dz": round(item["cohens_dz"], 2),
            "p(单侧)": f"{item['p_one_sided']:.4f}",
        })
    print(md_table(pd.DataFrame(table)))

    print("\n### 3. 分段（RMSE 配对差）\n")
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
                "对照": label, "样本端": end_name,
                "均值差": round(item["mean"], 3),
                "p(单侧)": f"{item['p_one_sided']:.4f}",
            })
    print(md_table(pd.DataFrame(table)))

    print(f"\n### 4. MSE/sigma^2（sigma^2 = {NOISE_VAR:.3f}）\n")
    table = []
    for name in ("G0", "G1", "G2", "A4"):
        means = model_sets[name].groupby("n_labeled")["mse"].mean()
        table.append({
            "model": name,
            "@10": round(float(means[10]) / NOISE_VAR, 3),
            "@60": round(float(means[60]) / NOISE_VAR, 3),
            "@120": round(float(means[120]) / NOISE_VAR, 3),
        })
    print(md_table(pd.DataFrame(table)))

    # Which G1 columns does the sparse selector actually use?
    print("\n### 5. G1 差异系数使用面貌（@120，5 批 mean |d|，未标准化列序）\n")
    from pwl_migration.heat import HEAT_B_NAMES_G1_PRIOR

    b_names = HEAT_B_NAMES_G1_PRIOR
    coeffs = []
    for _, row in model_sets["G1"][model_sets["G1"]["n_labeled"] == 120].iterrows():
        coeffs.append(np.abs(json.loads(row["details"])["d_coefficients"]))
    mean_abs = np.mean(np.asarray(coeffs), axis=0)
    order = np.argsort(-mean_abs)
    table = [
        {"列": b_names[i], "mean|d|": round(float(mean_abs[i]), 4)}
        for i in order
    ]
    print(md_table(pd.DataFrame(table)))

    # Quality gates
    print("\n### 6. 质量门槛\n")
    table = []
    for name in ("G0", "G1"):
        source = model_sets[name]
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


if __name__ == "__main__":
    main()
