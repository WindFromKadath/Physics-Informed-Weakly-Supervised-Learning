"""[ENGINEERING] Heat-conduction scenario experiment driver and anchors.

The driver reuses the shared tuning/metrics machinery from ``experiments.py``;
only data loading (per-batch CSVs), the untrained PhysicsDirect baseline, and
the migration anchors are scenario-specific.  Anchors follow
``migration/reports/COMSOL三维热传导场景迁移分析.md`` §8 and the No-Go
conditions of ``migration/reports/legacy/COMSOL迁移检查清单.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pwl_repro.experiment_api import (
    ExperimentArtifacts,
    fit_baseline_condition,
    fit_pwl_condition,
    labeled_dataset_id,
    metric_row,
    nested_split_plan,
    prediction_rows,
    sample_size_tests,
    save_artifacts,
)
from pwl_repro.core.features import get_feature_spec
from .heat import (
    K_NOMINAL,
    L,
    T0,
    THETA_LOWER,
    THETA_NOMINAL,
    THETA_UPPER,
    HeatScenarioData,
    load_heat_batch,
    low_fidelity_interface_flux,
    subsample_weak_labels,
)

SUPERVISED_BASELINES = ("Ridge", "SVR", "DT", "RF", "GBDT", "GP")
_B_PHYSICAL_TERMS = frozenset(
    {
        "P",
        "1/P",
        "P^-1/2",
        "Q/P",
        "Q*P^-1/2",
        "q_int*P^-0.7",
        "q_int/(1+hL/k)*P^-0.7",
    }
)


def _physics_direct_condition(
    data: HeatScenarioData,
    *,
    experiment: str,
    repeat: int,
    seed: int,
    n_labeled: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """[ENGINEERING] Untrained baseline: low-fidelity model at k_nominal.

    The migration checklist requires PWL to beat direct physics predictions;
    this row makes the comparison explicit without any fitting.
    """

    prediction = data.physics_model(data.test_x_ph, THETA_NOMINAL)
    row = metric_row(
        experiment=experiment,
        model="PhysicsDirect",
        source_model="PhysicsDirect",
        data=data,
        prediction=prediction,
        repeat=repeat,
        seed=seed,
        n_labeled=n_labeled,
        n_train=0,
        n_validation=0,
        validation_mse=None,
        hyperparameters={},
        details={"theta_fixed": [1.0 / K_NOMINAL]},
        extra=extra,
    )
    return row, prediction_rows(row, data.test_y, prediction)


def _affine_aligned_condition(
    data: HeatScenarioData,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    experiment: str,
    repeat: int,
    seed: int,
    n_labeled: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """[ENGINEERING] Blind-test arm: pure affine output alignment (the
    diagnostic M1 formalized), OLS on the train fold.

    ``y - T0 = alpha + beta*(y_LF - T0)`` with y_LF from the low-fidelity
    closed form at k_nominal; no mechanism columns, no weak labels, no
    hyperparameters.  Pre-registered in the S1 blind protocol as the
    "global alignment only" reference between PhysicsDirect and A4.
    """

    def design(x_ph: np.ndarray) -> np.ndarray:
        y_lf = data.physics_model(x_ph, THETA_NOMINAL)
        return np.column_stack((np.ones(len(y_lf)), y_lf - T0))

    coefficients, *_ = np.linalg.lstsq(
        design(data.x_ph[train]), data.y[train] - T0, rcond=None
    )
    prediction = T0 + design(data.test_x_ph) @ coefficients
    alpha, beta = float(coefficients[0]), float(coefficients[1])
    row = metric_row(
        experiment=experiment,
        model="AffineAligned",
        source_model="AffineAligned",
        data=data,
        prediction=prediction,
        repeat=repeat,
        seed=seed,
        n_labeled=n_labeled,
        n_train=len(train),
        n_validation=len(validation),
        validation_mse=None,
        hyperparameters={},
        details={
            "phi_intercept": T0 + alpha - beta * T0,
            "phi_slope": beta,
            "phi_rise_intercept": alpha,
            "theta_fixed": [1.0 / K_NOMINAL],
        },
        extra=extra,
    )
    return row, prediction_rows(row, data.test_y, prediction)


def _mechanism_aligned_condition(
    data: HeatScenarioData,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    experiment: str,
    repeat: int,
    seed: int,
    n_labeled: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """[ENGINEERING] A4 strong physics baseline (alignment report §6):
    temperature-rise affine output alignment plus the two q_int mechanism
    columns, fitted by plain OLS on the train fold.

    Transparent by design: no hyperparameters, no weak labels, no iterative
    solver; the validation fold is unused.  ``y_LF`` and ``q_int`` come from
    the low-fidelity closed form at k_nominal, so the arm touches neither
    acceptance labels nor high-fidelity information.  The fit uses the
    temperature-rise form ``y - T0 = alpha + beta*(y_LF - T0) + c1*q_int1
    + c2*q_int2`` (report §6.2); predictions are identical to the plain
    affine parametrization.
    """

    def design(x_ph: np.ndarray, x_pr: np.ndarray) -> np.ndarray:
        y_lf = data.physics_model(x_ph, THETA_NOMINAL)
        s = x_pr[:, 0] ** -0.7
        q_int = low_fidelity_interface_flux(x_ph, K_NOMINAL)
        modulation = 1.0 + x_ph[:, 1] * L / K_NOMINAL
        return np.column_stack(
            (np.ones(len(y_lf)), y_lf - T0, q_int * s, q_int / modulation * s)
        )

    coefficients, *_ = np.linalg.lstsq(
        design(data.x_ph[train], data.x_pr[train]),
        data.y[train] - T0,
        rcond=None,
    )
    prediction = T0 + design(data.test_x_ph, data.test_x_pr) @ coefficients
    alpha, beta = float(coefficients[0]), float(coefficients[1])
    row = metric_row(
        experiment=experiment,
        model="MechanismAligned",
        source_model="MechanismAligned",
        data=data,
        prediction=prediction,
        repeat=repeat,
        seed=seed,
        n_labeled=n_labeled,
        n_train=len(train),
        n_validation=len(validation),
        validation_mse=None,
        hyperparameters={},
        details={
            # Absolute-temperature form, parallel to the PWL phi schema.
            "phi_intercept": T0 + alpha - beta * T0,
            "phi_slope": beta,
            # [ENGINEERING] §6.2 interpretable rise form: dT = alpha + beta*dT_LF.
            "phi_rise_intercept": alpha,
            "q_int_coefficients": coefficients[2:].tolist(),
            "theta_fixed": [1.0 / K_NOMINAL],
        },
        extra=extra,
    )
    return row, prediction_rows(row, data.test_y, prediction)


def run_heat_sample_size_experiment(config: dict[str, Any]) -> ExperimentArtifacts:
    """[ENGINEERING] Sample-size curve over dataset batches as repeats."""

    experiment = config["experiment"]
    heat = config["heat"]
    sizes = sorted(int(value) for value in experiment["sample_sizes"])
    if not sizes:
        raise ValueError("experiment.sample_sizes must be nonempty.")
    datasets_root = heat.get("datasets_root", "datasets")
    batches = [int(value) for value in heat["batches"]]
    if not batches:
        raise ValueError("heat.batches must be nonempty.")
    base_seed = int(experiment.get("seed", 2026))
    train_fraction = float(config.get("train_fraction", 0.7))
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []

    for repeat, batch_index in enumerate(batches):
        seed = base_seed + repeat
        data = load_heat_batch(datasets_root, batch_index)
        n_weak = heat.get("n_weak")
        if n_weak is not None:
            # [ENGINEERING] M3 ablation A4: deterministic nested subsample of
            # the weak pool (same seed for every n_weak keeps 50 ⊂ 100 ⊂ 200).
            data = subsample_weak_labels(data, int(n_weak), seed=base_seed)
        max_size = max(sizes)
        if max_size > len(data.y):
            raise ValueError(
                f"sample_sizes exceed the labeled pool ({len(data.y)})."
            )
        # [ENGINEERING] A batch plays the repeat role: its k_true is the
        # system-level draw, and all sizes share the nested labeled pool,
        # weak labels, test set, and noise scale for paired comparisons.
        split = nested_split_plan(
            max_size, sizes, train_fraction=train_fraction, seed=seed + 17
        )
        conditions.append(
            {
                "experiment": "sample_size",
                "condition_type": "dataset",
                "repeat": repeat,
                "seed": seed,
                "batch_id": data.batch_id,
                "k_true": data.k_true,
                "dataset_id": labeled_dataset_id(data),
                "theta_true": json.dumps(data.theta_true.tolist()),
                "noise_sigma": data.noise_sigma,
                "physics_noise_sigma": data.physics_noise_sigma,
                "sampling_diagnostics": json.dumps(
                    dict(data.sampling_diagnostics), ensure_ascii=False
                ),
            }
        )
        for n_labeled in sizes:
            train, validation = split[n_labeled]
            conditions.append(
                {
                    "experiment": "sample_size",
                    "condition_type": "split",
                    "repeat": repeat,
                    "seed": seed,
                    "batch_id": data.batch_id,
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
                repeat=repeat,
                seed=seed,
                n_labeled=n_labeled,
                extra={"k_true": data.k_true},
            )
            rows.append(row)
            predictions.extend(pred)
            for name in config.get("baselines", []):
                if name == "PhysicsDirect":
                    row, pred = _physics_direct_condition(
                        data,
                        experiment="sample_size",
                        repeat=repeat,
                        seed=seed,
                        n_labeled=n_labeled,
                        extra={"k_true": data.k_true},
                    )
                elif name == "MechanismAligned":
                    # [ENGINEERING] Alignment arm A4 (report §6): one OLS per
                    # condition on the same train fold the PWL arm uses.
                    row, pred = _mechanism_aligned_condition(
                        data,
                        train,
                        validation,
                        experiment="sample_size",
                        repeat=repeat,
                        seed=seed,
                        n_labeled=n_labeled,
                        extra={"k_true": data.k_true},
                    )
                elif name == "AffineAligned":
                    # [ENGINEERING] Blind-test S1: pure affine output
                    # alignment (M1 formalized), OLS on the train fold.
                    row, pred = _affine_aligned_condition(
                        data,
                        train,
                        validation,
                        experiment="sample_size",
                        repeat=repeat,
                        seed=seed,
                        n_labeled=n_labeled,
                        extra={"k_true": data.k_true},
                    )
                else:
                    row, pred = fit_baseline_condition(
                        name,
                        data,
                        train,
                        validation,
                        config,
                        experiment="sample_size",
                        repeat=repeat,
                        seed=seed,
                        n_labeled=n_labeled,
                        extra={"k_true": data.k_true},
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


def _heat_anchor_checks(
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
) -> pd.DataFrame:
    """[ENGINEERING] Migration anchors (analysis §8) as quality-check rows."""

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

    # Anchor 1: label/low-fidelity correlation per batch within [0.7, 0.9].
    correlations = metrics.groupby("repeat")["physics_correlation"].first().dropna()
    if not correlations.empty:
        in_range = bool(((correlations >= 0.7) & (correlations <= 0.9)).all())
        add(
            "heat_fidelity_correlation_in_range",
            "pass" if in_range else "fail",
            f"[{correlations.min():.3f}, {correlations.max():.3f}]",
            "[0.7, 0.9]",
        )

    sample = metrics[metrics["experiment"] == "sample_size"]
    sizes = sorted(int(value) for value in sample["n_labeled"].unique())
    if len(sizes) < 2:
        return pd.DataFrame(records)
    min_size, max_size = sizes[0], sizes[-1]
    pwl_means = (
        sample[sample["model"] == "PWL"].groupby("n_labeled")["rmse"].mean()
    )
    supervised = sample[sample["model"].isin(SUPERVISED_BASELINES)]
    best_supervised = (
        supervised.groupby(["model", "n_labeled"])["rmse"]
        .mean()
        .reset_index()
        .groupby("n_labeled")["rmse"]
        .min()
    )

    # Anchor 3: the PWL advantage over the best supervised baseline is
    # largest at the small-sample end.  Skipped for arm-only result sets
    # (alignment arms A1-A3 carry no supervised baselines; the frozen A0
    # result set covers this anchor).
    if not best_supervised.empty:
        gap_min = float(best_supervised[min_size] - pwl_means[min_size])
        gap_max = float(best_supervised[max_size] - pwl_means[max_size])
        add(
            "heat_small_sample_advantage",
            "pass" if gap_min >= gap_max else "fail",
            f"gap@{min_size}={gap_min:.3f}, gap@{max_size}={gap_max:.3f}",
            "gap at min size >= gap at max size",
        )

    pwl_max = sample[
        (sample["model"] == "PWL") & (sample["n_labeled"] == max_size)
    ]
    theta_nominal = 1.0 / K_NOMINAL
    # [ENGINEERING] Read the active spec's B names so variants with fewer
    # columns (heat_no_qq) are indexed correctly.
    b_names = get_feature_spec(
        str(config.get("model", {}).get("feature_spec", "heat"))
    ).b_names
    closer: list[bool] = []
    boundary_hits: list[bool] = []
    theta_hats: list[float] = []
    top_physical = 0
    for _, record in pwl_max.iterrows():
        details = json.loads(record["details"])
        theta_hat = float(details["theta_estimate"][0])
        theta_true = float(json.loads(record["theta_true"])[0])
        theta_hats.append(theta_hat)
        closer.append(
            abs(theta_hat - theta_true) < abs(theta_nominal - theta_true)
        )
        boundary_hits.append(bool(details["theta_boundary_hit"]))
        d_coefficients = np.asarray(details["d_coefficients"], dtype=float)
        top3 = np.argsort(-np.abs(d_coefficients))[:3]
        if any(b_names[index] in _B_PHYSICAL_TERMS for index in top3):
            top_physical += 1

    # Anchor 4 (revised, M2): theta is a nuisance/calibration parameter, not
    # a recoverable 1/k.  R_c(P) is in series with the material resistance,
    # so the data constrain total resistance only; the noise-free effective
    # theta sits at 0.094-0.131, above the box upper 1/15 in all 10 batches
    # (migration/verify_theta_effective.py), and after the v3 B fix the
    # discrepancy term carries the contact mechanism, so theta_hat need not
    # chase any physical constant.  All theta indicators are info-level;
    # "closer to 1/k_true" is kept only for legacy traceability.
    if closer:
        fraction = float(np.mean(closer))
        add(
            "heat_theta_learning_fraction",
            "info",
            fraction,
            "legacy criterion (theta -> 1/k_true); not a success standard (M2)",
        )
        hit_rate = float(np.mean(boundary_hits))
        add(
            "heat_theta_boundary_hit_rate",
            "info",
            hit_rate,
            "structural: theta_eff outside box (M2); reported, not gated",
        )
        add(
            "heat_theta_batch_spread",
            "info",
            float(np.std(theta_hats)),
            "std of theta_hat across batches (nuisance parameter, M2)",
        )

    # Anchor 4 (revised, M2), prediction channel: how much the weak-label
    # model output moves across the whole theta box, relative to sigma.
    # A large value means the box position of theta carries predictive
    # weight even though the exact value is not physically interpretable.
    heat_config = config.get("heat", {})
    sensitivity_ratios: list[float] = []
    for batch_index in heat_config.get("batches", []):
        batch_data = load_heat_batch(
            heat_config.get("datasets_root", "datasets"), int(batch_index)
        )
        spread = batch_data.physics_model(
            batch_data.test_x_ph, np.asarray(THETA_UPPER)
        ) - batch_data.physics_model(
            batch_data.test_x_ph, np.asarray(THETA_LOWER)
        )
        sensitivity_ratios.append(
            float(np.sqrt(np.mean(spread**2))) / batch_data.noise_sigma
        )
    if sensitivity_ratios:
        add(
            "heat_theta_prediction_sensitivity",
            "info",
            f"[{min(sensitivity_ratios):.3f}, {max(sensitivity_ratios):.3f}]",
            "rms weak-label prediction change across theta box / sigma (M2)",
        )

    # Anchor 5: large-sample test MSE approaches the noise floor sigma^2.
    dataset_rows = artifacts.conditions[
        artifacts.conditions["condition_type"] == "dataset"
    ]
    if not dataset_rows.empty and not pwl_max.empty:
        noise_floor = float(np.mean(dataset_rows["noise_sigma"] ** 2))
        ratio = float(pwl_max["mse"].mean() / noise_floor)
        add(
            "heat_mse_noise_ratio_at_max",
            "pass" if ratio <= 2.0 else "warn",
            ratio,
            "<= 2.0",
        )

    # Anchor 6 (No-Go gate): PWL beats the two physics-only references.
    pwl_rmse = float(pwl_max["rmse"].mean()) if not pwl_max.empty else np.nan
    for rival in ("PhysicsDirect", "Physics"):
        rival_rows = sample[
            (sample["model"] == rival) & (sample["n_labeled"] == max_size)
        ]
        if rival_rows.empty:
            continue
        rival_rmse = float(rival_rows["rmse"].mean())
        add(
            f"heat_pwl_beats_{rival}",
            "pass" if pwl_rmse < rival_rmse else "fail",
            f"{pwl_rmse:.3f} vs {rival_rmse:.3f}",
            "PWL RMSE < rival RMSE",
        )

    # Anchor 7: d coefficients concentrate on Q/P^alpha terms (info level).
    if len(pwl_max):
        add(
            "heat_b_coefficient_physical",
            "info",
            top_physical / len(pwl_max),
            "fraction with Q/P^alpha term in top-3 |d|",
        )
    return pd.DataFrame(records)


def save_heat_artifacts(
    artifacts: ExperimentArtifacts,
    config: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    """[ENGINEERING] Shared artifacts plus scenario anchor rows."""

    output = save_artifacts(artifacts, config, output_directory)
    generic = pd.read_csv(output / "quality_checks.csv")
    combined = pd.concat(
        [generic, _heat_anchor_checks(artifacts, config)],
        ignore_index=True,
    )
    combined.to_csv(output / "quality_checks.csv", index=False, encoding="utf-8")
    return output
