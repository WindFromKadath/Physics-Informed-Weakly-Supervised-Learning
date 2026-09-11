"""Artifact summaries, plots, quality checks, and persistence."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import __version__
from .types import ExperimentArtifacts


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    group_candidates = (
        "experiment",
        "model",
        "source_model",
        "n_labeled",
        "curve_n_labeled",
        "accuracy_level",
        "target_physics_correlation",
    )
    groups = [column for column in group_candidates if column in metrics.columns]
    return (
        metrics.groupby(groups, dropna=False)[["mse", "rmse", "mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )

def _plot_sample_size(metrics: pd.DataFrame, destination: Path) -> None:
    data = metrics[metrics["experiment"] == "sample_size"]
    if data.empty:
        return
    summary = data.groupby(["model", "n_labeled"])["rmse"].agg(["mean", "std"]).reset_index()
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for model, rows in summary.groupby("model"):
        axis.errorbar(
            rows["n_labeled"],
            rows["mean"],
            yerr=rows["std"].fillna(0.0),
            marker="o",
            capsize=2,
            label=model,
        )
    axis.set(xlabel="Number of labeled samples", ylabel="Test RMSE")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)

def _plot_sample_boxplots(metrics: pd.DataFrame, destination: Path) -> None:
    data = metrics[metrics["experiment"] == "sample_size"]
    if data.empty:
        return
    sizes = sorted(data["n_labeled"].unique())
    pwl_values = []
    baseline_values = []
    baseline_sizes = []
    for size in sizes:
        subset = data[data["n_labeled"] == size]
        pwl_values.append(subset[subset["model"] == "PWL"]["mse"].to_numpy())
        means = subset[subset["model"] != "PWL"].groupby("model")["mse"].mean()
        if means.empty:
            # [ENGINEERING] PWL-only arms (blind ablation runs carry no
            # baseline rows): skip the baseline panel instead of failing.
            continue
        best = str(means.idxmin())
        baseline_values.append(subset[subset["model"] == best]["mse"].to_numpy())
        baseline_sizes.append(size)
    if baseline_values:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].boxplot(pwl_values, tick_labels=sizes, showfliers=False)
        axes[0].set_ylabel("PWL MSE")
        axes[1].boxplot(baseline_values, tick_labels=baseline_sizes, showfliers=False)
        axes[1].set(xlabel="Number of labeled samples", ylabel="Best baseline MSE")
        for axis in axes:
            axis.grid(alpha=0.2)
    else:
        fig, axis = plt.subplots(figsize=(10, 3.5))
        axis.boxplot(pwl_values, tick_labels=sizes, showfliers=False)
        axis.set(xlabel="Number of labeled samples", ylabel="PWL MSE")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)

def _plot_accuracy(metrics: pd.DataFrame, destination: Path) -> None:
    data = metrics[
        (metrics["experiment"] == "physics_accuracy")
        & metrics["model"].astype(str).str.startswith("PWL-")
    ]
    if data.empty:
        return
    x_column = (
        "calibration_physics_correlation"
        if "calibration_physics_correlation" in data
        and data["calibration_physics_correlation"].notna().all()
        else "target_physics_correlation"
    )
    summary = (
        data.groupby(["model", x_column])["rmse"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(x_column)
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(
        summary[x_column],
        summary["mean"],
        yerr=summary["std"].fillna(0.0),
        marker="o",
        capsize=3,
    )
    for _, row in summary.iterrows():
        axis.annotate(row["model"], (row[x_column], row["mean"]))
    axis.set(
        xlabel=(
            "Calibrated physics correlation"
            if x_column == "calibration_physics_correlation"
            else "Target physics correlation"
        ),
        ylabel="Test RMSE",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)

def _plot_label_savings(metrics: pd.DataFrame, destination: Path) -> None:
    data = metrics[metrics["experiment"] == "label_savings"]
    if data.empty:
        return
    summary = data.groupby(["model", "curve_n_labeled"])["mse"].agg(["mean", "std"]).reset_index()
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, rows in summary.groupby("model"):
        axis.errorbar(
            rows["curve_n_labeled"],
            rows["mean"],
            yerr=rows["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=model,
        )
    axis.set(xlabel="Number of labeled samples", ylabel="Test MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)

def _quality_checks(
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create [ENGINEERING] gates separating file completion from validity.

    These checks do not alter training.  They prevent finite files, failed
    convergence, or wrong IV-A/B/C trends from being reported as a successful
    paper reproduction.
    """

    records: list[dict[str, Any]] = []
    metrics = artifacts.metrics
    finite_metrics = (
        not metrics.empty
        and np.isfinite(metrics[["mse", "rmse", "mae"]].to_numpy()).all()
    )
    records.append(
        {
            "check": "finite_test_metrics",
            "status": "pass" if finite_metrics else "fail",
            "value": bool(finite_metrics),
            "threshold": True,
        }
    )

    pwl = metrics[metrics["source_model"] == "PWL"] if not metrics.empty else metrics
    convergence_values: list[bool] = []
    invalid_candidates = 0
    evaluated_candidates = 0
    for raw in pwl.get("details", pd.Series(dtype=str)).dropna():
        details = json.loads(raw)
        convergence_values.append(bool(details.get("converged", False)))
        tuning = details.get("tuning", {})
        invalid_candidates += int(tuning.get("invalid_candidates", 0))
        evaluated_candidates += int(tuning.get("evaluated_candidates", 0))
    convergence_rate = (
        float(np.mean(convergence_values)) if convergence_values else float("nan")
    )
    records.append(
        {
            "check": "selected_pwl_convergence_rate",
            "status": (
                "pass"
                if convergence_values and convergence_rate >= 0.95
                else "fail"
            ),
            "value": convergence_rate,
            "threshold": 0.95,
        }
    )
    records.append(
        {
            "check": "invalid_pwl_candidate_fraction",
            "status": (
                "pass"
                if evaluated_candidates == 0
                or invalid_candidates / evaluated_candidates <= 0.25
                else "warn"
            ),
            "value": (
                0.0
                if evaluated_candidates == 0
                else invalid_candidates / evaluated_candidates
            ),
            "threshold": 0.25,
        }
    )

    calibration = artifacts.conditions[
        artifacts.conditions.get("condition_type", pd.Series(dtype=str))
        == "accuracy_calibration"
    ]
    if not calibration.empty and "correlation_error" in calibration:
        maximum_error = float(calibration["correlation_error"].max())
        tolerance = float(
            config["experiment"].get("accuracy_correlation_tolerance", 0.02)
        )
        records.append(
            {
                "check": "maximum_calibration_correlation_error",
                "status": "pass" if maximum_error <= tolerance else "fail",
                "value": maximum_error,
                "threshold": tolerance,
            }
        )

    sample_pwl = metrics[
        (metrics["experiment"] == "sample_size")
        & (metrics["model"] == "PWL")
    ]
    sample_means = (
        sample_pwl.groupby("n_labeled")["rmse"].mean().sort_index()
        if not sample_pwl.empty
        else pd.Series(dtype=float)
    )
    if len(sample_means) >= 2:
        endpoint_change = float(
            sample_means.iloc[-1] - sample_means.iloc[0]
        )
        records.append(
            {
                "check": "iv_a_pwl_endpoint_rmse_change",
                "status": "pass" if endpoint_change < 0.0 else "fail",
                "value": endpoint_change,
                "threshold": "< 0",
            }
        )

    accuracy_pwl = metrics[
        (metrics["experiment"] == "physics_accuracy")
        & metrics["model"].isin(["PWL-H", "PWL-M", "PWL-L"])
    ]
    accuracy_means = accuracy_pwl.groupby("model")["rmse"].mean()
    if {"PWL-H", "PWL-M", "PWL-L"}.issubset(accuracy_means.index):
        ordered = bool(
            accuracy_means["PWL-H"]
            < accuracy_means["PWL-M"]
            < accuracy_means["PWL-L"]
        )
        records.append(
            {
                "check": "iv_b_rmse_orders_with_physics_accuracy",
                "status": "pass" if ordered else "fail",
                "value": ordered,
                "threshold": "PWL-H < PWL-M < PWL-L",
            }
        )

    profile = str(config.get("protocol", {}).get("profile", ""))
    savings = metrics[metrics["experiment"] == "label_savings"]
    if profile in {"diagnostic", "paper", "paper_single"} and not savings.empty:
        supervised = savings[savings["model"] == "Best-supervised"]
        fixed = savings[savings["model"] == "PWL-fixed"]
        if not supervised.empty and not fixed.empty:
            maximum_size = float(supervised["curve_n_labeled"].max())
            supervised_at_maximum = float(
                supervised[
                    supervised["curve_n_labeled"] == maximum_size
                ]["mse"].mean()
            )
            fixed_mse = float(fixed["mse"].mean())
            gap = supervised_at_maximum - fixed_mse
            records.append(
                {
                    "check": "iv_c_supervised_minus_pwl_mse_at_max_labels",
                    "status": "pass" if gap <= 0.0 else "fail",
                    "value": gap,
                    "threshold": "<= 0",
                }
            )
    return pd.DataFrame(records)

def save_artifacts(
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.metrics.to_csv(output / "results.csv", index=False, encoding="utf-8")
    artifacts.predictions.to_csv(
        output / "predictions.csv.gz", index=False, compression="gzip", encoding="utf-8"
    )
    artifacts.conditions.to_csv(output / "conditions.csv", index=False, encoding="utf-8")
    artifacts.statistical_tests.to_csv(
        output / "statistical_tests.csv", index=False, encoding="utf-8"
    )
    summary = _summary(artifacts.metrics)
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8")
    _quality_checks(artifacts, config).to_csv(
        output / "quality_checks.csv", index=False, encoding="utf-8"
    )

    sample = artifacts.metrics[artifacts.metrics["experiment"] == "sample_size"]
    if not sample.empty:
        available = sorted(int(value) for value in sample["n_labeled"].unique())
        paper_selection = [10, 60, 120]
        if set(paper_selection).issubset(available):
            selected = paper_selection
        else:
            selected = sorted(
                {
                    available[0],
                    available[len(available) // 2],
                    available[-1],
                }
            )
        table = _summary(sample[sample["n_labeled"].isin(selected)])
        table.to_csv(output / "table_iv_a.csv", index=False, encoding="utf-8")
    accuracy = artifacts.metrics[artifacts.metrics["experiment"] == "physics_accuracy"]
    if not accuracy.empty:
        _summary(accuracy).to_csv(
            output / "table_iv_b.csv", index=False, encoding="utf-8"
        )
    savings = artifacts.metrics[artifacts.metrics["experiment"] == "label_savings"]
    if not savings.empty:
        _summary(savings).to_csv(
            output / "table_iv_c.csv", index=False, encoding="utf-8"
        )

    _plot_sample_size(artifacts.metrics, output / "figure_iv_a_rmse.png")
    _plot_sample_boxplots(artifacts.metrics, output / "figure_iv_a_boxplots.png")
    _plot_accuracy(artifacts.metrics, output / "figure_iv_b_accuracy.png")
    _plot_label_savings(artifacts.metrics, output / "figure_iv_c_label_savings.png")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pwl_repro_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
