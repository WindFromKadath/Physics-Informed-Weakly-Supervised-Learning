"""Paired statistical tests and multiple-comparison adjustment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel


def _paired_frame(
    metrics: pd.DataFrame,
    left_filter: pd.Series,
    right_filter: pd.Series,
    metric: str,
) -> pd.DataFrame:
    left = metrics[left_filter][["repeat", metric]].rename(columns={metric: "left"})
    right = metrics[right_filter][["repeat", metric]].rename(columns={metric: "right"})
    return left.merge(right, on="repeat", how="inner")

def _sample_size_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for n_labeled in sorted(metrics["n_labeled"].unique()):
        models = sorted(
            set(metrics.loc[metrics["n_labeled"] == n_labeled, "model"]) - {"PWL"}
        )
        for model in models:
            for metric in ("mse", "rmse", "mae"):
                paired = _paired_frame(
                    metrics,
                    (metrics["model"] == "PWL") & (metrics["n_labeled"] == n_labeled),
                    (metrics["model"] == model) & (metrics["n_labeled"] == n_labeled),
                    metric,
                )
                if len(paired) < 2:
                    continue
                test = ttest_rel(paired["left"], paired["right"], alternative="less")
                records.append(
                    {
                        "experiment": "sample_size",
                        "comparison": f"PWL < {model}",
                        "n_labeled": n_labeled,
                        "metric": metric,
                        "n_pairs": len(paired),
                        "mean_difference": float((paired["left"] - paired["right"]).mean()),
                        "statistic": float(test.statistic),
                        "p_value": float(test.pvalue),
                    }
                )
    result = pd.DataFrame(records)
    return _add_holm_adjustment(result)

def _physics_accuracy_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if metrics.empty or "PWL-L" not in set(metrics["model"]):
        return pd.DataFrame()
    baselines = sorted(
        model
        for model in metrics["model"].unique()
        if not str(model).startswith("PWL-")
    )
    for model in baselines:
        for metric in ("mse", "rmse", "mae"):
            paired = _paired_frame(
                metrics,
                metrics["model"] == "PWL-L",
                metrics["model"] == model,
                metric,
            )
            if len(paired) < 2:
                continue
            test = ttest_rel(paired["left"], paired["right"], alternative="less")
            records.append(
                {
                    "experiment": "physics_accuracy",
                    "comparison": f"PWL-L < {model}",
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_difference": float((paired["left"] - paired["right"]).mean()),
                    "statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )
    return _add_holm_adjustment(pd.DataFrame(records))

def _label_savings_tests(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    margin_fraction = float(config["experiment"].get("equivalence_margin_fraction", 0.10))
    for size in sorted(metrics["curve_n_labeled"].unique()):
        paired = _paired_frame(
            metrics,
            (metrics["model"] == "PWL-fixed")
            & (metrics["curve_n_labeled"] == size),
            (metrics["model"] == "Best-supervised")
            & (metrics["curve_n_labeled"] == size),
            "mse",
        )
        if len(paired) < 2:
            continue
        difference = paired["right"] - paired["left"]
        paper_test = ttest_rel(
            paired["right"], paired["left"], alternative="greater"
        )
        margin = margin_fraction * float(paired["left"].mean())
        lower = ttest_1samp(difference, popmean=-margin, alternative="greater")
        upper = ttest_1samp(difference, popmean=margin, alternative="less")
        tost_p = float(max(lower.pvalue, upper.pvalue))
        records.append(
            {
                "experiment": "label_savings",
                "comparison": "Best-supervised vs fixed PWL",
                "curve_n_labeled": size,
                "metric": "mse",
                "n_pairs": len(paired),
                "mean_difference": float(difference.mean()),
                "paper_one_sided_statistic": float(paper_test.statistic),
                "paper_one_sided_p": float(paper_test.pvalue),
                "equivalence_margin": margin,
                "tost_p_value": tost_p,
                "tost_equivalent_0_05": bool(tost_p < 0.05),
            }
        )
    return pd.DataFrame(records)

def _add_holm_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "p_value" not in frame:
        return frame
    result = frame.copy()
    adjusted = np.empty(len(result), dtype=float)
    order = np.argsort(result["p_value"].to_numpy())
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(result.iloc[index]["p_value"]))
        running = max(running, value)
        adjusted[index] = running
    result["p_value_holm"] = adjusted
    return result
