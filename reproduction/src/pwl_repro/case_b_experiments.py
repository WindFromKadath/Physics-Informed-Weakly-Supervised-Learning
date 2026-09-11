"""[ENGINEERING] Case-B (spot weld) experiment driver and anchors.

Protocol (09 doc §2): two labeled scenarios (10 / 100) x five random splits
of the 120 field samples, shared kriging surrogate (fit once on the 35
simulations), weak labels refreshed at the current theta (M1).  Anchors
compare against paper Table V: MSE@100 ~ 0.206 (noise variance ~0.2),
RMSE@10 ~ 0.5566, PWL best across models, Physics baseline worst.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scenarios.case_b import (
    NOISE_VAR_REFERENCE,
    CaseBSurrogate,
    build_case_b_scenario,
    load_case_b_frames,
    save_loo_diagnostics,
)
from .experiment_api import (
    ExperimentArtifacts,
    fit_baseline_condition,
    fit_pwl_condition,
    labeled_dataset_id,
    nested_split_plan,
    sample_size_tests,
    save_artifacts,
)

# Paper Table V reference values (mean RMSE unless noted).
PAPER_REFERENCES = {
    "pwl_mse_at_100": 0.206,
    "pwl_rmse_at_10": 0.5566,
    "physics_rmse_at_10": 0.9021,
    "physics_rmse_at_100": 0.8279,
}
SUPERVISED_BASELINES = ("Ridge", "SVR", "DT", "RF", "GBDT", "GP")


def run_case_b_experiment(
    config: dict[str, Any],
    surrogate: CaseBSurrogate | None = None,
) -> ExperimentArtifacts:
    """[ENGINEERING] Case-B protocol: scenarios x random splits."""

    experiment = config["experiment"]
    case_b = config["case_b"]
    data_root = case_b.get("data_root", "datasets/case_b_spotweld")
    scenarios = [int(value) for value in experiment["label_scenarios"]]
    n_splits = int(experiment.get("n_splits", 5))
    base_seed = int(experiment.get("seed", 2026))
    train_fraction = float(config.get("train_fraction", 0.7))
    n_weak = int(case_b.get("n_weak", 120))
    weak_seed = int(case_b.get("weak_seed", 9102026))

    if surrogate is None:
        _, model_frame = load_case_b_frames(data_root)
        surrogate = CaseBSurrogate().fit(model_frame)
    # [ENGINEERING] The surrogate is fit once on the 35 simulations and
    # shared by every split and scenario (09 doc layer 1).
    field, _ = load_case_b_frames(data_root)

    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for n_labeled in scenarios:
        for split in range(n_splits):
            seed = base_seed + 1000 * n_labeled + split
            # [INFERRED] The paper does not disclose case-B tuning splits;
            # reuse the 70/30 nested train/validation plan inside the labels.
            split_plan = nested_split_plan(
                n_labeled, [n_labeled], train_fraction=train_fraction, seed=seed + 17
            )
            train, validation = split_plan[n_labeled]
            data = build_case_b_scenario(
                field,
                surrogate,
                n_labeled=n_labeled,
                split_seed=seed,
                n_weak=n_weak,
                weak_seed=weak_seed + split,
                theta_calibration_indices=train,
            )
            conditions.append(
                {
                    "experiment": "sample_size",
                    "condition_type": "dataset",
                    "repeat": split,
                    "seed": seed,
                    "dataset_id": labeled_dataset_id(data),
                    "theta_true": json.dumps(data.theta_true.tolist()),
                    "noise_sigma": data.noise_sigma,
                    "physics_noise_sigma": data.physics_noise_sigma,
                    "sampling_diagnostics": json.dumps(
                        dict(data.sampling_diagnostics), ensure_ascii=False
                    ),
                }
            )
            conditions.append(
                {
                    "experiment": "sample_size",
                    "condition_type": "split",
                    "repeat": split,
                    "seed": seed,
                    "dataset_id": labeled_dataset_id(data),
                    "n_labeled": n_labeled,
                    "train_indices": json.dumps(train.tolist()),
                    "validation_indices": json.dumps(validation.tolist()),
                }
            )
            row, pred = fit_pwl_condition(
                data,
                train,
                validation,
                config,
                experiment="sample_size",
                model_name="PWL",
                repeat=split,
                seed=seed,
                n_labeled=n_labeled,
                extra={"theta_hat": float(data.theta_true[0])},
            )
            rows.append(row)
            predictions.extend(pred)
            for name in config.get("baselines", []):
                row, pred = fit_baseline_condition(
                    name,
                    data,
                    train,
                    validation,
                    config,
                    experiment="sample_size",
                    repeat=split,
                    seed=seed,
                    n_labeled=n_labeled,
                    extra={"theta_hat": float(data.theta_true[0])},
                )
                rows.append(row)
                predictions.extend(pred)
    metrics = pd.DataFrame(rows)
    return ExperimentArtifacts(
        metrics=metrics,
        predictions=pd.DataFrame(predictions),
        conditions=pd.DataFrame(conditions),
        statistical_tests=sample_size_tests(metrics),
    )


def _case_b_anchor_checks(
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
) -> pd.DataFrame:
    """[ENGINEERING] Case-B anchors vs paper Table V (09 doc §4.6)."""

    records: list[dict[str, Any]] = []
    metrics = artifacts.metrics

    def add(check: str, status: str, value: Any, threshold: Any) -> None:
        records.append(
            {
                "check": check,
                "status": status,
                "value": value,
                "threshold": threshold,
            }
        )

    if metrics.empty:
        return pd.DataFrame(records)

    noise_var = NOISE_VAR_REFERENCE
    for n_labeled in sorted(int(v) for v in metrics["n_labeled"].unique()):
        subset = metrics[metrics["n_labeled"] == n_labeled]
        pwl = subset[subset["model"] == "PWL"]
        if pwl.empty:
            continue
        pwl_mse = float(pwl["mse"].mean())
        pwl_rmse = float(pwl["rmse"].mean())
        # Anchor 1: MSE approaches the replicate noise variance (paper 0.206
        # at 100 labels; ratio reported for both scenarios).
        ratio = pwl_mse / noise_var
        add(
            f"case_b_mse_noise_ratio@{n_labeled}",
            "pass" if ratio <= 1.5 else "warn",
            f"{pwl_mse:.4f} (ratio {ratio:.3f})",
            f"<= 1.5; paper {PAPER_REFERENCES['pwl_mse_at_100']} @100",
        )
        # Anchor 2: PWL vs every baseline (paper: PWL best across models).
        worst_rival = ""
        worst_margin = float("inf")
        for rival in subset["model"].unique():
            if rival == "PWL":
                continue
            margin = float(
                subset[subset["model"] == rival]["rmse"].mean() - pwl_rmse
            )
            if margin < worst_margin:
                worst_margin = margin
                worst_rival = rival
        add(
            f"case_b_pwl_best@{n_labeled}",
            "pass" if worst_margin >= 0.0 else "fail",
            f"min margin {worst_margin:+.4f} vs {worst_rival}",
            ">= 0 (paper: PWL best)",
        )
        # Anchor 3: Physics is (near-)worst (paper Table V).
        physics_rmse = float(
            subset[subset["model"] == "Physics"]["rmse"].mean()
        )
        supervised = subset[subset["model"].isin(SUPERVISED_BASELINES)]
        supervised_max = (
            float(supervised["rmse"].max()) if not supervised.empty else float("nan")
        )
        add(
            f"case_b_physics_near_worst@{n_labeled}",
            "pass" if physics_rmse >= pwl_rmse else "fail",
            f"{physics_rmse:.4f} (max supervised {supervised_max:.4f})",
            f"paper {PAPER_REFERENCES['physics_rmse_at_10']} @10 / {PAPER_REFERENCES['physics_rmse_at_100']} @100",
        )
        # Anchor 4 (info): RMSE vs paper reference at 10 labels.
        if n_labeled == 10:
            add(
                "case_b_rmse_vs_paper@10",
                "info",
                f"{pwl_rmse:.4f}",
                f"paper {PAPER_REFERENCES['pwl_rmse_at_10']}",
            )
    return pd.DataFrame(records)


def save_case_b_artifacts(
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
    output_directory: str | Path,
    surrogate: CaseBSurrogate | None = None,
) -> Path:
    """[ENGINEERING] Shared artifacts plus case-B anchors and the LOO audit."""

    output = save_artifacts(artifacts, config, output_directory)
    generic = pd.read_csv(output / "quality_checks.csv")
    combined = pd.concat(
        [generic, _case_b_anchor_checks(artifacts, config)],
        ignore_index=True,
    )
    combined.to_csv(output / "quality_checks.csv", index=False, encoding="utf-8")
    if surrogate is not None:
        save_loo_diagnostics(surrogate, output / "surrogate_loo.json")
    return output
