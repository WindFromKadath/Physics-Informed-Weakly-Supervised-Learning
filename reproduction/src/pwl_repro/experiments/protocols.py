"""Section IV experiment protocols and configuration-driven dispatch."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..scenarios.simulation import (
    SimulationData,
    discrepancy,
    eta_true,
    generate_simulation,
)
from .statistics import (
    _label_savings_tests,
    _physics_accuracy_tests,
    _sample_size_tests,
)
from .tuning import _fit_baseline_condition, _fit_pwl_condition
from .types import (
    ExperimentArtifacts,
    _empty_artifacts,
    _labeled_dataset_id,
    nested_split_plan,
)


def _generate_pool(
    config: dict[str, Any],
    n_labeled: int,
    seed: int,
    scale: float,
    *,
    n_weak: int | None = None,
    n_test: int | None = None,
    theta_true: np.ndarray | None = None,
    noise_reference_seed: int | None = None,
) -> SimulationData:
    simulation = config["simulation"]
    stability = simulation.get("stability", {})
    covariance = simulation.get("covariance")
    return generate_simulation(
        n_labeled=n_labeled,
        n_weak=int(simulation["n_weak"] if n_weak is None else n_weak),
        n_test=int(simulation["n_test"] if n_test is None else n_test),
        seed=seed,
        snr=float(simulation.get("snr", 5.0)),
        correlation=float(simulation.get("correlation", 0.5)),
        covariance=None if covariance is None else np.asarray(covariance, dtype=float),
        discrepancy_scale=scale,
        physics_noise_fraction=float(simulation.get("physics_noise_fraction", 0.25)),
        theta_true=theta_true,
        singularity_policy=str(
            simulation.get("singularity_policy", "reject_near_pole")
        ),
        min_abs_x3=float(stability.get("min_abs_x3", 0.15)),
        min_abs_x4=float(stability.get("min_abs_x4", 0.15)),
        min_abs_x5=float(stability.get("min_abs_x5", 0.15)),
        min_abs_eta_denominator=float(
            stability.get("min_abs_eta_denominator", 1.0)
        ),
        min_abs_discrepancy_denominator=float(
            stability.get("min_abs_discrepancy_denominator", 0.75)
        ),
        max_abs_component=float(stability.get("max_abs_component", 25.0)),
        noise_reference_size=int(simulation.get("noise_reference_size", 5000)),
        noise_reference_seed=noise_reference_seed,
    )

def run_sample_size_experiment(
    config: dict[str, Any],
    *,
    sample_sizes: Iterable[int] | None = None,
    pwl_sizes: set[int] | None = None,
    baseline_sizes: set[int] | None = None,
) -> ExperimentArtifacts:
    """[PAPER] Section IV-A with an [INFERRED] shared pool per repeat."""

    experiment = config["experiment"]
    sizes = sorted(
        int(value)
        for value in (
            experiment["sample_sizes"] if sample_sizes is None else sample_sizes
        )
    )
    max_size = max(sizes)
    pwl_sizes = set(sizes) if pwl_sizes is None else set(pwl_sizes)
    baseline_sizes = set(sizes) if baseline_sizes is None else set(baseline_sizes)
    base_seed = int(experiment.get("seed", 2026))
    train_fraction = float(config.get("train_fraction", 0.7))
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []

    for repeat in range(int(experiment["repeats"])):
        seed = base_seed + repeat
        data = _generate_pool(
            config,
            max_size,
            seed,
            float(config["simulation"].get("discrepancy_scale", 1.0)),
        )
        # [INFERRED] All label counts share physics data, test data, theta,
        # noise scale, and nested labeled observations for paired comparisons.
        split = nested_split_plan(
            max_size, sizes, train_fraction=train_fraction, seed=seed + 17
        )
        conditions.append(
            {
                "experiment": "sample_size",
                "condition_type": "dataset",
                "repeat": repeat,
                "seed": seed,
                "dataset_id": _labeled_dataset_id(data),
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
                    "dataset_id": _labeled_dataset_id(data),
                    "n_labeled": n_labeled,
                    "train_indices": json.dumps(train.tolist()),
                    "validation_indices": json.dumps(validation.tolist()),
                }
            )
            if n_labeled in pwl_sizes:
                row, pred = _fit_pwl_condition(
                    data,
                    train,
                    validation,
                    config,
                    experiment="sample_size",
                    model_name="PWL",
                    repeat=repeat,
                    seed=seed,
                    n_labeled=n_labeled,
                )
                rows.append(row)
                predictions.extend(pred)
            if n_labeled in baseline_sizes:
                for name in config.get("baselines", []):
                    row, pred = _fit_baseline_condition(
                        name,
                        data,
                        train,
                        validation,
                        config,
                        experiment="sample_size",
                        repeat=repeat,
                        seed=seed,
                        n_labeled=n_labeled,
                    )
                    rows.append(row)
                    predictions.extend(pred)
    metrics = pd.DataFrame(rows)
    return ExperimentArtifacts(
        metrics=metrics,
        predictions=pd.DataFrame(predictions),
        conditions=pd.DataFrame(conditions),
        statistical_tests=_sample_size_tests(metrics),
    )

def _calibrate_accuracy_scales(
    config: dict[str, Any],
    base_data: SimulationData,
    *,
    repeat: int,
    seed: int,
) -> pd.DataFrame:
    """Calibrate IV-B discrepancy multipliers for one system.

    [PAPER] requests correlations 0.85/0.70/0.50.  [INFERRED] The independent
    Monte Carlo pool, log grid, inclusion of physics noise, and per-repeat
    calibration are the reproduction's way to attain those targets without
    leaking the formal training or test samples.
    """

    experiment = config["experiment"]
    targets = [float(value) for value in experiment.get("physics_accuracy_targets", [0.85, 0.7, 0.5])]
    size = int(experiment.get("accuracy_calibration_size", 5000))
    calibration_seed = int(
        experiment.get("accuracy_calibration_seed", 9_102_026)
    ) + repeat
    reference = _generate_pool(
        config,
        size,
        calibration_seed,
        0.0,
        n_weak=1,
        n_test=1,
        theta_true=base_data.theta_true,
        noise_reference_seed=seed + 7_000_003,
    )
    physics_core = (
        eta_true(reference.x_ph, reference.theta_true)
        + reference.labeled_physics_noise
    )
    delta = discrepancy(reference.x_ph)
    grid_config = experiment.get("accuracy_scale_log10", [-3.0, 3.0, 601])
    candidates = np.concatenate(
        ([0.0], np.logspace(float(grid_config[0]), float(grid_config[1]), int(grid_config[2])))
    )
    correlations = np.array(
        [
            np.corrcoef(reference.y, physics_core + scale * delta)[0, 1]
            for scale in candidates
        ]
    )
    records = []
    for target in targets:
        index = int(np.nanargmin(np.abs(correlations - target)))
        records.append(
            {
                "experiment": "physics_accuracy",
                "condition_type": "accuracy_calibration",
                "repeat": repeat,
                "seed": seed,
                "target_physics_correlation": target,
                "discrepancy_scale": float(candidates[index]),
                "calibration_correlation": float(correlations[index]),
                "correlation_error": float(abs(correlations[index] - target)),
                "calibration_seed": calibration_seed,
                "calibration_size": size,
                "theta_true": json.dumps(base_data.theta_true.tolist()),
                "zero_discrepancy_correlation": float(correlations[0]),
                "minimum_grid_correlation": float(np.nanmin(correlations)),
                "maximum_grid_correlation": float(np.nanmax(correlations)),
            }
        )
    calibrated = pd.DataFrame(records)
    tolerance = float(experiment.get("accuracy_correlation_tolerance", 0.02))
    if bool(experiment.get("accuracy_require_tolerance", True)):
        failures = calibrated[calibrated["correlation_error"] > tolerance]
        if not failures.empty:
            details = failures[
                [
                    "target_physics_correlation",
                    "calibration_correlation",
                    "zero_discrepancy_correlation",
                ]
            ].to_dict("records")
            raise RuntimeError(
                "Physics-correlation targets are not attainable under the current "
                f"data-generating assumptions (tolerance={tolerance}): {details}"
            )
    return calibrated

def run_physics_accuracy_experiment(config: dict[str, Any]) -> ExperimentArtifacts:
    """[PAPER] Section IV-B; only discrepancy magnitude changes across H/M/L.

    [INFERRED] Within a repeat, all levels deliberately share observations and
    noise so that the calibrated discrepancy is the only varying condition.
    """

    experiment = config["experiment"]
    n_labeled = int(experiment.get("physics_accuracy_n_labeled", 40))
    repeats = int(experiment["repeats"])
    base_seed = int(experiment.get("seed", 2026)) + 2_000_000
    level_names = {0: "H", 1: "M", 2: "L"}
    baseline_names = experiment.get(
        "physics_accuracy_baselines",
        [name for name in config.get("baselines", []) if name != "Physics"],
    )
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []

    for repeat in range(repeats):
        seed = base_seed + repeat
        base_data = _generate_pool(config, n_labeled, seed, 0.0)
        scales = _calibrate_accuracy_scales(
            config,
            base_data,
            repeat=repeat,
            seed=seed,
        )
        conditions.extend(scales.to_dict("records"))
        split = nested_split_plan(
            n_labeled,
            [n_labeled],
            train_fraction=float(config.get("train_fraction", 0.7)),
            seed=seed + 17,
        )
        train, validation = split[n_labeled]
        conditions.append(
            {
                "experiment": "physics_accuracy",
                "condition_type": "split",
                "repeat": repeat,
                "seed": seed,
                "n_labeled": n_labeled,
                "train_indices": json.dumps(train.tolist()),
                "validation_indices": json.dumps(validation.tolist()),
                "sampling_diagnostics": json.dumps(
                    dict(base_data.sampling_diagnostics), ensure_ascii=False
                ),
            }
        )
        data_by_level: list[tuple[str, float, float, SimulationData]] = []
        for index, scale_row in scales.reset_index(drop=True).iterrows():
            level = level_names.get(index, f"C{index + 1}")
            data = base_data.with_discrepancy_scale(
                float(scale_row["discrepancy_scale"])
            )
            data_by_level.append(
                (
                    level,
                    float(scale_row["target_physics_correlation"]),
                    float(scale_row["calibration_correlation"]),
                    data,
                )
            )

        # Supervised baselines see exactly the same labels and test set and run once.
        baseline_data = data_by_level[0][3]
        for name in baseline_names:
            row, pred = _fit_baseline_condition(
                name,
                baseline_data,
                train,
                validation,
                config,
                experiment="physics_accuracy",
                repeat=repeat,
                seed=seed,
                n_labeled=n_labeled,
                extra={
                    "accuracy_level": "shared_baseline",
                    "target_physics_correlation": np.nan,
                },
            )
            rows.append(row)
            predictions.extend(pred)

        labeled_ids = set()
        for level, target, calibration_correlation, data in data_by_level:
            labeled_ids.add(_labeled_dataset_id(data))
            row, pred = _fit_pwl_condition(
                data,
                train,
                validation,
                config,
                experiment="physics_accuracy",
                model_name=f"PWL-{level}",
                repeat=repeat,
                seed=seed,
                n_labeled=n_labeled,
                extra={
                    "accuracy_level": level,
                    "target_physics_correlation": target,
                    "calibration_physics_correlation": calibration_correlation,
                },
            )
            rows.append(row)
            predictions.extend(pred)
            conditions.append(
                {
                    "experiment": "physics_accuracy",
                    "condition_type": "accuracy_realized",
                    "repeat": repeat,
                    "seed": seed,
                    "accuracy_level": level,
                    "target_physics_correlation": target,
                    "calibration_physics_correlation": calibration_correlation,
                    "realized_physics_correlation": row["physics_correlation"],
                    "discrepancy_scale": data.discrepancy_scale,
                    "dataset_id": row["dataset_id"],
                }
            )
        if len(labeled_ids) != 1:
            raise RuntimeError("Physics-accuracy levels do not share labeled/test data.")

    metrics = pd.DataFrame(rows)
    return ExperimentArtifacts(
        metrics=metrics,
        predictions=pd.DataFrame(predictions),
        conditions=pd.DataFrame(conditions),
        statistical_tests=_physics_accuracy_tests(metrics),
    )

def derive_label_savings_experiment(
    sample_artifacts: ExperimentArtifacts,
    config: dict[str, Any],
) -> ExperimentArtifacts:
    """[PAPER] Section IV-C, derived from IV-A with PWL fixed at 30 labels.

    Reusing the paired IV-A predictions prevents the plotted fixed PWL line
    from accidentally retraining on the increasing supervised-label counts.
    """

    experiment = config["experiment"]
    fixed_n = int(experiment.get("label_savings_pwl_n_labeled", 30))
    sizes = [int(value) for value in experiment.get("label_savings_sizes", range(30, 111, 10))]
    supervised_names = set(
        experiment.get(
            "label_savings_baselines",
            [name for name in config.get("baselines", []) if name != "Physics"],
        )
    )
    source = sample_artifacts.metrics
    source_predictions = sample_artifacts.predictions
    pwl = source[(source["model"] == "PWL") & (source["n_labeled"] == fixed_n)]
    if len(pwl) != int(experiment["repeats"]):
        raise RuntimeError("IV-C requires one IV-A PWL result at 30 labels per repeat.")

    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    conditions: list[dict[str, Any]] = []
    for size in sizes:
        candidates = source[
            source["source_model"].isin(supervised_names)
            & (source["n_labeled"] == size)
        ]
        if candidates.empty:
            raise RuntimeError(f"No supervised baseline results for {size} labels.")
        model_means = candidates.groupby("source_model")["mse"].mean()
        best_model = str(model_means.idxmin())
        best = candidates[candidates["source_model"] == best_model].copy()
        best["experiment"] = "label_savings"
        best["model"] = "Best-supervised"
        best["curve_n_labeled"] = size
        metric_parts.append(best)

        fixed = pwl.copy()
        fixed["experiment"] = "label_savings"
        fixed["model"] = "PWL-fixed"
        fixed["source_model"] = "PWL"
        fixed["curve_n_labeled"] = size
        metric_parts.append(fixed)

        best_pred = source_predictions[
            (source_predictions["source_model"] == best_model)
            & (source_predictions["n_labeled"] == size)
        ].copy()
        best_pred["experiment"] = "label_savings"
        best_pred["model"] = "Best-supervised"
        best_pred["curve_n_labeled"] = size
        prediction_parts.append(best_pred)

        fixed_pred = source_predictions[
            (source_predictions["source_model"] == "PWL")
            & (source_predictions["n_labeled"] == fixed_n)
        ].copy()
        fixed_pred["experiment"] = "label_savings"
        fixed_pred["model"] = "PWL-fixed"
        fixed_pred["curve_n_labeled"] = size
        prediction_parts.append(fixed_pred)
        conditions.append(
            {
                "experiment": "label_savings",
                "condition_type": "best_supervised_selection",
                "curve_n_labeled": size,
                "fixed_pwl_n_labeled": fixed_n,
                "best_supervised_model": best_model,
                "best_supervised_mean_mse": float(model_means.loc[best_model]),
                "fixed_pwl_mean_mse": float(pwl["mse"].mean()),
            }
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return ExperimentArtifacts(
        metrics=metrics,
        predictions=predictions,
        conditions=pd.DataFrame(conditions),
        statistical_tests=_label_savings_tests(metrics, config),
    )

def validate_experiment_config(config: dict[str, Any], mode: str) -> None:
    """Fail early when a configuration cannot execute the requested protocol."""

    for section in ("experiment", "simulation", "lambda_grid", "model"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration mapping: {section}.")
    experiment = config["experiment"]
    simulation = config["simulation"]
    sizes = sorted(set(int(value) for value in experiment["sample_sizes"]))
    if not sizes or min(sizes) < 2:
        raise ValueError("sample_sizes must contain values >= 2.")
    if int(experiment["repeats"]) < 1:
        raise ValueError("repeats must be positive.")
    if min(int(simulation["n_weak"]), int(simulation["n_test"])) < 1:
        raise ValueError("n_weak and n_test must be positive.")
    for name in ("physics", "l1", "group"):
        values = config["lambda_grid"].get(name, [])
        if not values or any(float(value) < 0 for value in values):
            raise ValueError(f"lambda_grid.{name} must contain nonnegative values.")

    if mode in {"label-savings", "all"}:
        fixed = int(experiment.get("label_savings_pwl_n_labeled", 30))
        savings = set(
            int(value)
            for value in experiment.get(
                "label_savings_sizes", range(30, 111, 10)
            )
        )
        if mode == "all" and not ({fixed} | savings).issubset(sizes):
            missing = sorted(({fixed} | savings) - set(sizes))
            raise ValueError(
                "mode=all requires IV-A results for every IV-C size; "
                f"missing sample_sizes={missing}."
            )

    profile = str(config.get("protocol", {}).get("profile", "custom"))
    if profile == "paper":
        expected_sizes = list(range(10, 121, 10))
        requirements = {
            "experiment.repeats": int(experiment["repeats"]) == 20,
            "experiment.sample_sizes": sizes == expected_sizes,
            "simulation.n_weak": int(simulation["n_weak"]) == 100,
            "simulation.n_test": int(simulation["n_test"]) == 200,
            "experiment.label_savings_sizes": [
                int(value)
                for value in experiment.get("label_savings_sizes", [])
            ]
            == list(range(30, 111, 10)),
        }
        failures = [name for name, passed in requirements.items() if not passed]
        if failures:
            raise ValueError(
                "Paper profile does not satisfy disclosed Section IV protocol: "
                + ", ".join(failures)
            )

def run_experiments(config: dict[str, Any], mode: str) -> ExperimentArtifacts:
    validate_experiment_config(config, mode)
    if mode == "sample-size":
        return run_sample_size_experiment(config)
    if mode == "physics-accuracy":
        return run_physics_accuracy_experiment(config)
    if mode == "label-savings":
        sizes = [
            int(value)
            for value in config["experiment"].get(
                "label_savings_sizes", range(30, 111, 10)
            )
        ]
        fixed = int(
            config["experiment"].get("label_savings_pwl_n_labeled", 30)
        )
        internal = run_sample_size_experiment(
            config,
            sample_sizes=sorted(set(sizes + [fixed])),
            pwl_sizes={fixed},
            baseline_sizes=set(sizes),
        )
        return derive_label_savings_experiment(internal, config)
    if mode == "all":
        sample = run_sample_size_experiment(config)
        accuracy = run_physics_accuracy_experiment(config)
        label = derive_label_savings_experiment(sample, config)
        return ExperimentArtifacts.combine((sample, accuracy, label))
    raise ValueError(f"Unknown experiment mode: {mode}")
