"""阶段 0：把任意结果目录与论文 Section IV 数值对照。

[PAPER] 参考值取自论文 Section IV 的公开数值（见 PAPER_REFERENCE 注释）。
[ENGINEERING] 本脚本只读取 results.csv / conditions.csv / quality_checks.csv，
默认额外写出 paper_comparison.csv 与 normalized_metrics.csv 两个新文件，
绝不修改结果目录中的既有文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# [PAPER] 论文 Section IV 公开数值。
PAPER_REFERENCE: dict[str, Any] = {
    "iv_a_pwl_rmse": {10: 2.298, 60: 1.619, 120: 1.378},
    "iv_b_pwl_rmse": {"H": 1.392, "M": 2.054, "L": 2.404},
    # IV-C：30 标签 PWL MSE，论文交点约 100 标签（监督相对 PWL 标签需求 +233.3%）。
    "iv_c_pwl_mse_30": 3.745,
    "iv_c_crossover_labels": 100,
}

DEFAULT_RESULTS = ROOT / "results" / "section_iv_single_full_parallel"


def _mean_noise_sigma(conditions: pd.DataFrame) -> float:
    """按 conditions.csv 的 dataset 行取 noise_sigma 均值（缺失则 NaN）。"""

    if conditions.empty or "noise_sigma" not in conditions:
        return float("nan")
    dataset = conditions
    if "condition_type" in conditions:
        dataset = conditions[conditions["condition_type"] == "dataset"]
        if dataset.empty:
            dataset = conditions
    values = pd.to_numeric(dataset["noise_sigma"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def compare_results(results_dir: str | Path) -> dict[str, Any]:
    """读取结果目录并计算论文对照、归一化指标与选参审计（不写文件）。"""

    results_dir = Path(results_dir)
    metrics = pd.read_csv(results_dir / "results.csv")
    conditions_path = results_dir / "conditions.csv"
    conditions = (
        pd.read_csv(conditions_path) if conditions_path.exists() else pd.DataFrame()
    )
    noise_sigma = _mean_noise_sigma(conditions)
    noise_variance = noise_sigma**2

    comparison_rows: list[dict[str, Any]] = []

    # --- IV-A：PWL 在 10/60/120 标签的 RMSE ---
    iv_a = metrics[
        (metrics["experiment"] == "sample_size") & (metrics["model"] == "PWL")
    ]
    iv_a_rmse = (
        iv_a.groupby("n_labeled")["rmse"].mean().to_dict() if not iv_a.empty else {}
    )
    iv_a_mse = (
        iv_a.groupby("n_labeled")["mse"].mean().to_dict() if not iv_a.empty else {}
    )
    for n_labeled, paper_rmse in PAPER_REFERENCE["iv_a_pwl_rmse"].items():
        current = float(iv_a_rmse.get(n_labeled, float("nan")))
        comparison_rows.append(
            {
                "section": "IV-A",
                "condition": f"pwl_rmse_{n_labeled}_labels",
                "metric": "rmse",
                "current": current,
                "paper": paper_rmse,
                "abs_diff": current - paper_rmse,
                "rel_diff": (current - paper_rmse) / paper_rmse,
            }
        )

    # --- IV-B：PWL-H/M/L 的 RMSE ---
    iv_b = metrics[
        (metrics["experiment"] == "physics_accuracy")
        & metrics["model"].astype(str).str.startswith("PWL-")
    ]
    iv_b_rmse: dict[str, float] = {}
    if not iv_b.empty:
        for level, paper_rmse in PAPER_REFERENCE["iv_b_pwl_rmse"].items():
            subset = iv_b[iv_b["model"] == f"PWL-{level}"]
            current = (
                float(subset["rmse"].mean()) if not subset.empty else float("nan")
            )
            iv_b_rmse[level] = current
            comparison_rows.append(
                {
                    "section": "IV-B",
                    "condition": f"pwl_rmse_level_{level}",
                    "metric": "rmse",
                    "current": current,
                    "paper": paper_rmse,
                    "abs_diff": current - paper_rmse,
                    "rel_diff": (current - paper_rmse) / paper_rmse,
                }
            )

    # --- IV-C：30 标签 PWL MSE 与监督交点状态 ---
    iv_c = metrics[metrics["experiment"] == "label_savings"]
    pwl_fixed = iv_c[iv_c["model"] == "PWL-fixed"]
    supervised = iv_c[iv_c["model"] == "Best-supervised"]
    iv_c_pwl_mse = (
        float(pwl_fixed["mse"].mean()) if not pwl_fixed.empty else float("nan")
    )
    best_supervised_by_size = (
        supervised.groupby("curve_n_labeled")["mse"].mean().sort_index()
        if not supervised.empty
        else pd.Series(dtype=float)
    )
    max_labels = int(best_supervised_by_size.index.max()) if len(best_supervised_by_size) else None
    best_supervised_max_mse = (
        float(best_supervised_by_size.loc[max_labels])
        if max_labels is not None
        else float("nan")
    )
    crossed = [
        int(size)
        for size, value in best_supervised_by_size.items()
        if value <= iv_c_pwl_mse
    ]
    crossover_at = min(crossed) if crossed else None
    already_below = bool(
        np.isfinite(iv_c_pwl_mse)
        and np.isfinite(best_supervised_max_mse)
        and iv_c_pwl_mse < best_supervised_max_mse
    )
    comparison_rows.append(
        {
            "section": "IV-C",
            "condition": "pwl_mse_30_labels",
            "metric": "mse",
            "current": iv_c_pwl_mse,
            "paper": PAPER_REFERENCE["iv_c_pwl_mse_30"],
            "abs_diff": iv_c_pwl_mse - PAPER_REFERENCE["iv_c_pwl_mse_30"],
            "rel_diff": (iv_c_pwl_mse - PAPER_REFERENCE["iv_c_pwl_mse_30"])
            / PAPER_REFERENCE["iv_c_pwl_mse_30"],
        }
    )
    comparison_rows.append(
        {
            "section": "IV-C",
            "condition": f"best_supervised_mse_{max_labels}_labels",
            "metric": "mse",
            "current": best_supervised_max_mse,
            "paper": float("nan"),
            "abs_diff": float("nan"),
            "rel_diff": float("nan"),
        }
    )
    comparison = pd.DataFrame(comparison_rows)

    # --- 归一化指标：mse/noise_variance、rmse/noise_sigma、mse/mse@120 ---
    normalized_rows: list[dict[str, Any]] = []
    mse_at_120 = float(iv_a_mse.get(120, float("nan")))
    for n_labeled in sorted(iv_a_mse):
        mse = float(iv_a_mse[n_labeled])
        normalized_rows.append(
            {
                "section": "IV-A",
                "condition": f"pwl_{n_labeled}_labels",
                "mse": mse,
                "rmse": float(iv_a_rmse[n_labeled]),
                "noise_sigma": noise_sigma,
                "noise_variance": noise_variance,
                "mse_over_noise_variance": mse / noise_variance,
                "rmse_over_noise_sigma": float(iv_a_rmse[n_labeled]) / noise_sigma,
                "mse_over_mse_at_120_labels": mse / mse_at_120,
            }
        )
    for level, rmse in iv_b_rmse.items():
        subset = iv_b[iv_b["model"] == f"PWL-{level}"]
        mse = float(subset["mse"].mean()) if not subset.empty else float("nan")
        normalized_rows.append(
            {
                "section": "IV-B",
                "condition": f"pwl_level_{level}",
                "mse": mse,
                "rmse": rmse,
                "noise_sigma": noise_sigma,
                "noise_variance": noise_variance,
                "mse_over_noise_variance": mse / noise_variance,
                "rmse_over_noise_sigma": rmse / noise_sigma,
                "mse_over_mse_at_120_labels": float("nan"),
            }
        )
    normalized = pd.DataFrame(normalized_rows)

    # --- 选参审计：解析 PWL 行 details JSON（tuning 键可能缺失，容错） ---
    pwl_rows = metrics[metrics["source_model"] == "PWL"]
    lambda_counts: dict[str, dict[str, int]] = {
        key: {} for key in ("lambda_physics", "lambda_l1", "lambda_group")
    }
    boundary_hits = 0
    theta_rows = 0
    selection_gaps: list[float] = []
    invalid_total = 0
    evaluated_total = 0
    for index, raw in pwl_rows.get("details", pd.Series(dtype=str)).dropna().items():
        details = json.loads(raw)
        raw_hyper = pwl_rows.loc[index, "hyperparameters"]
        hyper = json.loads(raw_hyper) if isinstance(raw_hyper, str) else {}
        for key in lambda_counts:
            value = hyper.get(key)
            if value is not None:
                label = str(float(value))
                lambda_counts[key][label] = lambda_counts[key].get(label, 0) + 1
        theta = details.get("theta_estimate")
        if theta is not None:
            theta_rows += 1
            boundary_hits += int(
                any(abs(component) < 1e-3 or abs(component - 1.0) < 1e-3
                    for component in theta)
            )
        tuning = details.get("tuning") or {}
        minimum = tuning.get("minimum_validation_mse")
        selected = pwl_rows.loc[index, "validation_mse"]
        if minimum is not None and pd.notna(selected):
            selection_gaps.append(float(selected) - float(minimum))
        invalid_total += int(tuning.get("invalid_candidates", 0))
        evaluated_total += int(tuning.get("evaluated_candidates", 0))
    audit = {
        "pwl_rows": int(len(pwl_rows)),
        "lambda_selection_counts": lambda_counts,
        "theta_boundary_hit_rate": (
            boundary_hits / theta_rows if theta_rows else float("nan")
        ),
        "validation_mse_minus_minimum": {
            "mean": float(np.mean(selection_gaps)) if selection_gaps else float("nan"),
            "max": float(np.max(selection_gaps)) if selection_gaps else float("nan"),
        },
        "invalid_candidate_fraction": (
            invalid_total / evaluated_total if evaluated_total else float("nan")
        ),
    }

    # --- 质量门槛 fail 数（若目录含 quality_checks.csv） ---
    quality_path = results_dir / "quality_checks.csv"
    quality_fail_count = None
    if quality_path.exists():
        checks = pd.read_csv(quality_path)
        quality_fail_count = int((checks["status"] == "fail").sum())

    key_metrics = {
        "iv_a_pwl_rmse": {int(k): float(v) for k, v in iv_a_rmse.items()},
        "iv_b_pwl_rmse": {k: float(v) for k, v in iv_b_rmse.items()},
        "iv_c_pwl_mse_30": iv_c_pwl_mse,
        "iv_c_best_supervised_mse_at_max_labels": best_supervised_max_mse,
        "iv_c_max_supervised_labels": max_labels,
        "iv_c_crossover_at_labels": crossover_at,
        "iv_c_pwl_below_supervised_at_max": already_below,
        "noise_sigma": noise_sigma,
        "noise_variance": noise_variance,
        "iv_a_mse_over_noise_variance_at_120": (
            mse_at_120 / noise_variance if np.isfinite(mse_at_120) else float("nan")
        ),
        "quality_fail_count": quality_fail_count,
    }
    return {
        "paper_comparison": comparison,
        "normalized_metrics": normalized,
        "selection_audit": audit,
        "key_metrics": key_metrics,
    }


def _print_summary(report: dict[str, Any], results_dir: Path) -> None:
    comparison = report["paper_comparison"]
    normalized = report["normalized_metrics"]
    audit = report["selection_audit"]
    keys = report["key_metrics"]

    print(f"== 论文对照（{results_dir}）==")
    for _, row in comparison.iterrows():
        paper = row["paper"]
        paper_text = f"{paper:.3f}" if np.isfinite(paper) else "-"
        diff_text = (
            f"abs={row['abs_diff']:+.3f} rel={row['rel_diff']:+.1%}"
            if np.isfinite(row["abs_diff"])
            else ""
        )
        print(
            f"  {row['section']} {row['condition']}: "
            f"current={row['current']:.3f} paper={paper_text} {diff_text}"
        )
    print(
        "  IV-C 交点状态: "
        f"30 标签 PWL MSE={keys['iv_c_pwl_mse_30']:.3f}, "
        f"{keys['iv_c_max_supervised_labels']} 标签最佳监督 MSE="
        f"{keys['iv_c_best_supervised_mse_at_max_labels']:.3f}, "
        f"PWL 已低于最大标签监督={keys['iv_c_pwl_below_supervised_at_max']}, "
        f"监督追平 PWL 的首个标签数={keys['iv_c_crossover_at_labels']}"
    )

    print("== 归一化指标 ==")
    for _, row in normalized.iterrows():
        print(
            f"  {row['section']} {row['condition']}: "
            f"mse/noise_variance={row['mse_over_noise_variance']:.3f} "
            f"rmse/noise_sigma={row['rmse_over_noise_sigma']:.3f} "
            f"mse/mse@120={row['mse_over_mse_at_120_labels']:.3f}"
        )

    print("== 选参审计 ==")
    print(f"  PWL 行数: {audit['pwl_rows']}")
    for key, counts in audit["lambda_selection_counts"].items():
        print(f"  {key} 选中分布: {counts}")
    print(f"  theta 边界命中率: {audit['theta_boundary_hit_rate']:.3f}")
    gap = audit["validation_mse_minus_minimum"]
    print(f"  选中验证 MSE - 最低验证 MSE: mean={gap['mean']:.4f} max={gap['max']:.4f}")
    print(f"  无效候选比例: {audit['invalid_candidate_fraction']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="结果目录（含 results.csv 等），默认当前冻结基线。",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="只打印摘要，不写出 paper_comparison.csv / normalized_metrics.csv。",
    )
    args = parser.parse_args()
    results_dir = Path(args.results)
    report = compare_results(results_dir)
    _print_summary(report, results_dir)
    if not args.no_write:
        report["paper_comparison"].to_csv(
            results_dir / "paper_comparison.csv", index=False, encoding="utf-8"
        )
        report["normalized_metrics"].to_csv(
            results_dir / "normalized_metrics.csv", index=False, encoding="utf-8"
        )
        print(
            "已写出: "
            f"{results_dir / 'paper_comparison.csv'}, "
            f"{results_dir / 'normalized_metrics.csv'}"
        )


if __name__ == "__main__":
    main()
