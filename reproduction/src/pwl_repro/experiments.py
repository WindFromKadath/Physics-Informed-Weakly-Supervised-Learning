"""Audited orchestration for the three experiments in paper Section IV.

[PAPER] defines the IV-A/B/C questions, public sample sizes, metrics, and
repeat structure.  [INFERRED] Nested pools, per-repeat accuracy calibration,
grids, and model-selection details complete unpublished protocol choices.
[STABILITY] Nonconverged candidate rejection prevents invalid fits from being
selected.  [ENGINEERING] Parallelism, statistics, artifacts, and quality gates
make the reproduction inspectable but are not part of the PWL objective.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import ttest_1samp, ttest_rel

from . import __version__
from .baselines import PhysicsGPRegressor, tune_baseline
from .model import PWLRegressor
from .simulation import SimulationData, discrepancy, eta_true, generate_simulation


@dataclass(frozen=True)
class ExperimentArtifacts:
    """Raw metrics, predictions, conditions, and statistical tests."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    conditions: pd.DataFrame
    statistical_tests: pd.DataFrame

    @classmethod
    def combine(cls, parts: Iterable["ExperimentArtifacts"]) -> "ExperimentArtifacts":
        parts = list(parts)

        def combine_field(name: str) -> pd.DataFrame:
            frames = [getattr(part, name) for part in parts if not getattr(part, name).empty]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return cls(
            metrics=combine_field("metrics"),
            predictions=combine_field("predictions"),
            conditions=combine_field("conditions"),
            statistical_tests=combine_field("statistical_tests"),
        )


@dataclass(frozen=True)
class _PWLCandidateResult:
    """Small, picklable result returned by one hyperparameter worker."""

    params: dict[str, float]
    start_index: int
    theta_init: np.ndarray | None
    mse: float
    squared_errors: np.ndarray


def _empty_artifacts() -> ExperimentArtifacts:
    return ExperimentArtifacts(
        metrics=pd.DataFrame(),
        predictions=pd.DataFrame(),
        conditions=pd.DataFrame(),
        statistical_tests=pd.DataFrame(),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_true) - np.asarray(y_pred)
    mse = float(np.mean(error**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
    }


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """[ENGINEERING] Diagnostic R^2; NaN when the target has zero variance."""

    truth = np.asarray(y_true, dtype=float)
    estimate = np.asarray(y_pred, dtype=float)
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    if total <= 0.0:
        return float("nan")
    residual = float(np.sum((truth - estimate) ** 2))
    return 1.0 - residual / total


def nested_split_plan(
    max_samples: int,
    sample_sizes: Iterable[int],
    *,
    train_fraction: float,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Create [INFERRED] nested train/validation sets at the paper's ratio.

    Each newly added block is independently split, so earlier training and
    validation observations never change roles as sample size grows.  The
    paper reports a 70/30 split but does not disclose this nesting policy.
    """

    sizes = sorted(set(int(size) for size in sample_sizes))
    if not sizes or sizes[-1] > max_samples:
        raise ValueError("sample_sizes must be nonempty and <= max_samples.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(max_samples)
    train: list[int] = []
    validation: list[int] = []
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    previous = 0
    for size in sizes:
        block = order[previous:size]
        n_train = int(round(train_fraction * len(block)))
        n_train = min(max(1, n_train), len(block) - 1)
        train.extend(int(value) for value in block[:n_train])
        validation.extend(int(value) for value in block[n_train:])
        result[size] = (
            np.asarray(train, dtype=int).copy(),
            np.asarray(validation, dtype=int).copy(),
        )
        previous = size
    return result


def _labeled_dataset_id(data: SimulationData) -> str:
    digest = hashlib.sha256()
    for array in (
        data.x_ph,
        data.x_pr,
        data.y,
        data.test_x_ph,
        data.test_x_pr,
        data.test_y,
        data.theta_true,
    ):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()[:16]


def _scenario_indices(train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    return np.concatenate((train, validation))


def _make_pwl(
    model_config: dict[str, Any],
    lambdas: dict[str, float],
    seed: int,
) -> PWLRegressor:
    return PWLRegressor(
        **lambdas,
        mapping=model_config.get("mapping", "identity"),
        standardize=bool(model_config.get("standardize", True)),
        b_profile=str(model_config.get("b_profile", "expanded")),
        admm_rho=float(model_config.get("admm_rho", 1.0)),
        admm_adaptive_rho=bool(model_config.get("admm_adaptive_rho", True)),
        admm_tolerance=float(model_config.get("admm_tolerance", 1e-5)),
        admm_max_iter=int(model_config.get("admm_max_iter", 1000)),
        bcd_tolerance=float(model_config.get("bcd_tolerance", 1e-4)),
        bcd_parameter_tolerance=float(
            model_config.get(
                "bcd_parameter_tolerance",
                np.sqrt(float(model_config.get("bcd_tolerance", 1e-4))),
            )
        ),
        bcd_max_iter=int(model_config.get("bcd_max_iter", 50)),
        d_ridge=float(model_config.get("d_ridge", 0.0)),
        random_state=seed,
    )


def _evaluate_pwl_candidate(
    data: SimulationData,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    model_config: dict[str, Any],
    params: dict[str, float],
    start_index: int,
    theta_init: np.ndarray | None,
    seed: int,
    require_convergence: bool,
    maximum_validation_mse: float,
) -> _PWLCandidateResult | None:
    """Fit and score one independent PWL candidate without retaining the model."""

    model = _make_pwl(model_config, params, seed + start_index)
    try:
        model.fit(
            data.x_ph[train_index],
            data.x_pr[train_index],
            data.y[train_index],
            data.weak_x_ph,
            data.weak_y,
            theta_init=theta_init,
            physics_model=data.physics_model,
        )
        prediction = model.predict(
            data.x_ph[validation_index], data.x_pr[validation_index]
        )
        squared_errors = (data.y[validation_index] - prediction) ** 2
        mse = float(np.mean(squared_errors))
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        return None
    valid = (
        np.isfinite(mse)
        and mse <= maximum_validation_mse
        and np.all(np.isfinite(prediction))
        and (model.converged_ or not require_convergence)
    )
    if not valid:
        return None
    return _PWLCandidateResult(
        params=params,
        start_index=start_index,
        theta_init=(
            None if theta_init is None else np.asarray(theta_init).copy()
        ),
        mse=mse,
        squared_errors=squared_errors,
    )


def tune_pwl(
    data: SimulationData,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[PWLRegressor, dict[str, float], float]:
    """Tune all three lambdas by validation MSE.

    [PAPER] uses validation performance for tuning.  [INFERRED] The concrete
    grid and physics-preferred one-standard-error rule are reproduction
    choices.  [ENGINEERING] Candidate-level parallelism is result-preserving.
    """

    model_config = config["model"]
    grid = config["lambda_grid"]
    require_convergence = bool(model_config.get("require_convergence", True))
    theta_starts = max(1, int(model_config.get("theta_starts", 1)))
    maximum_validation_mse = float(
        model_config.get("maximum_validation_mse", float("inf"))
    )
    selection_rule = str(model_config.get("selection_rule", "minimum"))
    if selection_rule not in {"minimum", "physics_one_standard_error"}:
        raise ValueError(
            "model.selection_rule must be 'minimum' or "
            "'physics_one_standard_error'."
        )
    n_jobs = int(model_config.get("n_jobs", 1))
    if n_jobs == 0:
        raise ValueError("model.n_jobs must not be zero.")
    parallel_verbose = int(model_config.get("parallel_verbose", 0))
    rng = np.random.default_rng(seed + 31_337)
    specifications: list[
        tuple[dict[str, float], int, np.ndarray | None]
    ] = []
    for lambda_physics, lambda_l1, lambda_group in product(
        grid["physics"], grid["l1"], grid["group"]
    ):
        params = {
            "lambda_physics": float(lambda_physics),
            "lambda_l1": float(lambda_l1),
            "lambda_group": float(lambda_group),
        }
        for start_index in range(theta_starts):
            theta_init = (
                None
                if start_index == 0
                else rng.uniform(
                    np.asarray((0.0, 0.0)),
                    np.asarray((1.0, 1.0)),
                )
            )
            specifications.append(
                (
                    params.copy(),
                    start_index,
                    None if theta_init is None else theta_init.copy(),
                )
            )

    evaluated = len(specifications)
    if parallel_verbose:
        print(
            "[PWL tune] "
            f"seed={seed} train={len(train_index)} "
            f"validation={len(validation_index)} candidates={evaluated} "
            f"n_jobs={n_jobs}",
            flush=True,
        )

    def jobs() -> Iterable[Any]:
        for params, start_index, theta_init in specifications:
            yield delayed(_evaluate_pwl_candidate)(
                data,
                train_index,
                validation_index,
                model_config,
                params,
                start_index,
                theta_init,
                seed,
                require_convergence,
                maximum_validation_mse,
            )

    if n_jobs == 1:
        evaluations = [
            _evaluate_pwl_candidate(
                data,
                train_index,
                validation_index,
                model_config,
                params,
                start_index,
                theta_init,
                seed,
                require_convergence,
                maximum_validation_mse,
            )
            for params, start_index, theta_init in specifications
        ]
    else:
        # [ENGINEERING] Independent lambda candidates are distributed across
        # processes.  One inner BLAS thread avoids nested CPU oversubscription;
        # the ASCII workspace path avoids Windows non-ASCII temp-path failures.
        parallel_temp_folder = Path(
            model_config.get(
                "parallel_temp_folder",
                "reproduction/.joblib",
            )
        ).resolve()
        parallel_temp_folder.mkdir(parents=True, exist_ok=True)
        with parallel_config(backend="loky", inner_max_num_threads=1):
            evaluations = Parallel(
                n_jobs=n_jobs,
                verbose=parallel_verbose,
                pre_dispatch="2*n_jobs",
                temp_folder=str(parallel_temp_folder),
            )(jobs())
    candidates = [
        candidate for candidate in evaluations if candidate is not None
    ]
    invalid = evaluated - len(candidates)

    if not candidates:
        raise RuntimeError(
            "No valid PWL candidate. "
            f"evaluated={evaluated}, invalid={invalid}, "
            f"require_convergence={require_convergence}, seed={seed}, "
            f"n_train={len(train_index)}, n_validation={len(validation_index)}, "
            f"n_weak={len(data.weak_y)}."
        )
    minimum_candidate = min(candidates, key=lambda item: item.mse)
    selection_threshold = minimum_candidate.mse
    eligible = [minimum_candidate]
    if selection_rule == "physics_one_standard_error":
        # [INFERRED] Not a published author selection rule.  When a tiny
        # validation set cannot distinguish candidates within one standard
        # error, prefer stronger physics and sparsity regularization.
        best_errors = minimum_candidate.squared_errors
        standard_error = (
            float(np.std(best_errors, ddof=1) / np.sqrt(len(best_errors)))
            if len(best_errors) > 1
            else 0.0
        )
        selection_threshold += standard_error
        eligible = [
            candidate
            for candidate in candidates
            if candidate.mse <= selection_threshold
        ]
        selected = max(
            eligible,
            key=lambda item: (
                item.params["lambda_physics"],
                item.params["lambda_l1"],
                item.params["lambda_group"],
                -item.mse,
            ),
        )
    else:
        selected = minimum_candidate
    best_params = selected.params
    best_mse = selected.mse
    tuning_diagnostics = {
        "evaluated_candidates": evaluated,
        "invalid_candidates": invalid,
        "valid_candidates": evaluated - invalid,
        "theta_starts": theta_starts,
        "require_convergence": require_convergence,
        "n_jobs": n_jobs,
        "selection_rule": selection_rule,
        "minimum_validation_mse": minimum_candidate.mse,
        "selection_threshold": selection_threshold,
        "eligible_candidates": len(eligible),
        # [ENGINEERING] Stage-1 selection audit: rank of the selected
        # candidate among valid candidates ordered by validation MSE (1-based).
        "selected_rank_by_validation_mse": 1 + sum(
            1 for candidate in candidates if candidate.mse < selected.mse
        ),
        "n_valid_candidates": len(candidates),
    }
    if parallel_verbose:
        print(
            "[PWL tune] "
            f"seed={seed} valid={len(candidates)}/{evaluated} "
            f"selected={best_params} validation_mse={best_mse:.6g}",
            flush=True,
        )
    if not bool(config.get("refit_on_train_validation", False)):
        best_model = _reconstruct_selected(
            data,
            train_index,
            model_config,
            best_params,
            seed,
            selected,
            require_convergence,
        )
        best_model.tuning_diagnostics_ = tuning_diagnostics
        return best_model, best_params, best_mse

    final_index = _scenario_indices(train_index, validation_index)
    # [ENGINEERING] Warm-start the refit from the selected candidate, matching
    # the reconstruction branch above; a cold start intermittently fails the
    # convergence criteria on the merged train+validation data.
    final = _make_pwl(model_config, best_params, seed + selected.start_index)
    final.fit(
        data.x_ph[final_index],
        data.x_pr[final_index],
        data.y[final_index],
        data.weak_x_ph,
        data.weak_y,
        theta_init=selected.theta_init,
        physics_model=data.physics_model,
    )
    if require_convergence and not final.converged_:
        if _refit_objective_stagnated(final):
            # [ENGINEERING] BCD can enter a 2-cycle between equivalent minima
            # (typically theta at the box boundary): the objective flatlines
            # while parameters keep oscillating above the parameter tolerance.
            # Accept the refit only when the objective has genuinely stagnated
            # and both inner ADMM solves converged; record the override.
            tuning_diagnostics["refit_convergence_override"] = (
                "objective_stagnation"
            )
        else:
            # [ENGINEERING] Tiny merged refits can converge too slowly to meet
            # the strict criteria within the iteration budget.  Fall back to
            # the converged train-only reconstruction so the protocol keeps a
            # valid model; the deviation is recorded per condition.
            tuning_diagnostics["refit_convergence_override"] = (
                "fallback_train_only"
            )
            final = _reconstruct_selected(
                data,
                train_index,
                model_config,
                best_params,
                seed,
                selected,
                require_convergence,
            )
    final.tuning_diagnostics_ = tuning_diagnostics
    return final, best_params, best_mse


def _reconstruct_selected(
    data: SimulationData,
    train_index: np.ndarray,
    model_config: dict[str, Any],
    best_params: dict[str, float],
    seed: int,
    selected: "_PWLCandidateResult",
    require_convergence: bool,
) -> PWLRegressor:
    """Rebuild the selected candidate on the training split only."""

    model = _make_pwl(model_config, best_params, seed + selected.start_index)
    model.fit(
        data.x_ph[train_index],
        data.x_pr[train_index],
        data.y[train_index],
        data.weak_x_ph,
        data.weak_y,
        theta_init=selected.theta_init,
        physics_model=data.physics_model,
    )
    if require_convergence and not model.converged_:
        raise RuntimeError(
            "Selected PWL candidate did not converge when reconstructed."
        )
    return model


def _refit_objective_stagnated(model: PWLRegressor, window: int = 5) -> bool:
    """Return True when the refit objective flatlined without drift.

    Requires the last iteration's inner solves to have converged, the final
    relative objective change to be below the BCD tolerance, and the
    objective range over the trailing ``window`` iterations to stay below the
    same tolerance (oscillation without progress).
    """

    history = model.history_
    if not history:
        return False
    last = history[-1]
    if not (last.g_admm_converged and last.theta_admm_converged):
        return False
    tolerance = model.bcd_tolerance
    if not (np.isfinite(last.objective) and last.relative_change < tolerance):
        return False
    tail = [record.objective for record in history[-window:]]
    if not all(np.isfinite(tail)):
        return False
    spread = (max(tail) - min(tail)) / max(1.0, abs(float(np.mean(tail))))
    return bool(spread < tolerance)


def _metric_row(
    *,
    experiment: str,
    model: str,
    source_model: str,
    data: SimulationData,
    prediction: np.ndarray,
    repeat: int,
    seed: int,
    n_labeled: int,
    n_train: int,
    n_validation: int,
    validation_mse: float | None,
    hyperparameters: dict[str, Any],
    details: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": experiment,
        "model": model,
        "source_model": source_model,
        "repeat": repeat,
        "seed": seed,
        "dataset_id": _labeled_dataset_id(data),
        "n_labeled": n_labeled,
        "n_train": n_train,
        "n_validation": n_validation,
        "n_weak": len(data.weak_y),
        "validation_mse": validation_mse,
        "hyperparameters": json.dumps(hyperparameters, ensure_ascii=False),
        "theta_true": json.dumps(data.theta_true.tolist()),
        "discrepancy_scale": data.discrepancy_scale,
        "physics_correlation": data.labeled_physics_correlation(include_noise=True),
        "details": json.dumps(details, ensure_ascii=False),
        **regression_metrics(data.test_y, prediction),
    }
    if extra:
        row.update(extra)
    for key in ("lambda_physics", "lambda_l1", "lambda_group"):
        if key in hyperparameters:
            row[key] = float(hyperparameters[key])
    return row


def _prediction_rows(
    metric_row: dict[str, Any],
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    keys = (
        "experiment",
        "model",
        "source_model",
        "repeat",
        "seed",
        "dataset_id",
        "n_labeled",
    )
    shared = {key: metric_row.get(key) for key in keys}
    for optional in ("target_physics_correlation", "accuracy_level", "curve_n_labeled"):
        if optional in metric_row:
            shared[optional] = metric_row[optional]
    return [
        {
            **shared,
            "test_index": index,
            "y_true": float(truth),
            "y_pred": float(estimate),
        }
        for index, (truth, estimate) in enumerate(zip(y_true, prediction))
    ]


def _fit_pwl_condition(
    data: SimulationData,
    train: np.ndarray,
    validation: np.ndarray,
    config: dict[str, Any],
    *,
    experiment: str,
    model_name: str,
    repeat: int,
    seed: int,
    n_labeled: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, params, validation_mse = tune_pwl(
        data, train, validation, config, seed=seed
    )
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    # [ENGINEERING] Stage-1 internal diagnostics (additive only).  They
    # observe distillation quality, component magnitudes, and selection gaps
    # without changing the fitted model or the tuning protocol.
    features = model.features_
    weak_physics_prediction = model.y_mean_ + model.y_scale_ * (
        features.transform_h(data.weak_x_ph, model.theta_) @ model.g_
    )
    weak_distillation_r2 = _r_squared(data.weak_y, weak_physics_prediction)
    train_prediction = model.predict(data.x_ph[train], data.x_pr[train])
    labeled_prediction_r2 = _r_squared(data.y[train], train_prediction)
    h_test = features.transform_h(data.test_x_ph, model.theta_)
    b_test = features.transform_b(data.test_x_ph, data.test_x_pr)
    physics_component_norm = float(
        np.linalg.norm(model.y_scale_ * (h_test @ model.g_))
    )
    process_component_norm = float(
        np.linalg.norm(model.y_scale_ * (b_test @ model.d_))
    )
    theta_boundary_hit = bool(
        np.any(np.abs(model.theta_) < 1e-3)
        or np.any(np.abs(model.theta_ - 1.0) < 1e-3)
    )
    g_group_norms = {
        name: float(np.linalg.norm(model.g_[indices]))
        for name, indices in zip(features.group_names, features.groups)
    }
    tuning = getattr(model, "tuning_diagnostics_", {}) or {}
    minimum_validation_mse = tuning.get("minimum_validation_mse")
    minimum_vs_selected = (
        None
        if minimum_validation_mse is None
        else float(validation_mse - minimum_validation_mse)
    )
    row = _metric_row(
        experiment=experiment,
        model=model_name,
        source_model="PWL",
        data=data,
        prediction=prediction,
        repeat=repeat,
        seed=seed,
        n_labeled=n_labeled,
        n_train=len(train),
        n_validation=len(validation),
        validation_mse=validation_mse,
        hyperparameters=params,
        details={
            "theta_estimate": model.theta_.tolist(),
            "iterations": model.n_iter_,
            "converged": model.converged_,
            "training_objective": model.training_objective_,
            "final_relative_change": model.history_[-1].relative_change,
            "final_parameter_relative_change": (
                model.history_[-1].parameter_relative_change
            ),
            "g_admm_converged": model.history_[-1].g_admm_converged,
            "theta_admm_converged": model.history_[-1].theta_admm_converged,
            "g_primal_residual": model.history_[-1].g_primal_residual,
            "g_dual_residual": model.history_[-1].g_dual_residual,
            "theta_primal_residual": model.history_[-1].theta_primal_residual,
            "theta_dual_residual": model.history_[-1].theta_dual_residual,
            "g_admm_rho": model.history_[-1].g_admm_rho,
            "theta_admm_rho": model.history_[-1].theta_admm_rho,
            "g_l2_norm": float(np.linalg.norm(model.g_)),
            "d_l2_norm": float(np.linalg.norm(model.d_)),
            "d_max_abs": float(np.max(np.abs(model.d_))),
            "tuning": model.tuning_diagnostics_,
            "active_h_features": list(model.active_features_),
            "weak_distillation_r2": weak_distillation_r2,
            "labeled_prediction_r2": labeled_prediction_r2,
            "physics_component_norm": physics_component_norm,
            "process_component_norm": process_component_norm,
            "theta_boundary_hit": theta_boundary_hit,
            "g_group_norms": g_group_norms,
            "d_coefficients": model.d_.tolist(),
            "minimum_vs_selected_validation_mse": minimum_vs_selected,
        },
        extra=extra,
    )
    return row, _prediction_rows(row, data.test_y, prediction)


def _fit_baseline_condition(
    name: str,
    data: SimulationData,
    train: np.ndarray,
    validation: np.ndarray,
    config: dict[str, Any],
    *,
    experiment: str,
    repeat: int,
    seed: int,
    n_labeled: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    refit = bool(config.get("refit_on_train_validation", False))
    x = np.column_stack((data.x_ph, data.x_pr))
    test_x = np.column_stack((data.test_x_ph, data.test_x_pr))
    if name == "Physics":
        fit_index = _scenario_indices(train, validation) if refit else train
        model = PhysicsGPRegressor(data.physics_model, random_state=seed).fit(
            data.x_ph[fit_index], data.x_pr[fit_index], data.y[fit_index]
        )
        prediction = model.predict(data.test_x_ph, data.test_x_pr)
        params: dict[str, Any] = {}
        validation_mse = None
        details = {"theta_estimate": model.theta_.tolist()}
    else:
        model, params, validation_mse = tune_baseline(
            name,
            x[train],
            data.y[train],
            x[validation],
            data.y[validation],
            seed=seed,
            refit=refit,
        )
        prediction = model.predict(test_x)
        details = {}
    row = _metric_row(
        experiment=experiment,
        model=name,
        source_model=name,
        data=data,
        prediction=prediction,
        repeat=repeat,
        seed=seed,
        n_labeled=n_labeled,
        n_train=len(train),
        n_validation=len(validation),
        validation_mse=validation_mse,
        hyperparameters=params,
        details=details,
        extra=extra,
    )
    return row, _prediction_rows(row, data.test_y, prediction)


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
    for size in sizes:
        subset = data[data["n_labeled"] == size]
        pwl_values.append(subset[subset["model"] == "PWL"]["mse"].to_numpy())
        means = subset[subset["model"] != "PWL"].groupby("model")["mse"].mean()
        best = str(means.idxmin())
        baseline_values.append(subset[subset["model"] == best]["mse"].to_numpy())
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].boxplot(pwl_values, tick_labels=sizes, showfliers=False)
    axes[0].set_ylabel("PWL MSE")
    axes[1].boxplot(baseline_values, tick_labels=sizes, showfliers=False)
    axes[1].set(xlabel="Number of labeled samples", ylabel="Best baseline MSE")
    for axis in axes:
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
