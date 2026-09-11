"""PWL candidate search, model reconstruction, and condition-level fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from joblib import Parallel, delayed, parallel_config

from ..baselines import PhysicsGPRegressor, tune_baseline
from ..core.model import PWLRegressor
from .types import (
    ScenarioData,
    _labeled_dataset_id,
    _r_squared,
    _scenario_indices,
    regression_metrics,
)


@dataclass(frozen=True)
class _PWLCandidateResult:
    """Small, picklable result returned by one hyperparameter worker."""

    params: dict[str, float]
    start_index: int
    theta_init: np.ndarray | None
    mse: float
    squared_errors: np.ndarray

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
        feature_spec=str(model_config.get("feature_spec", "simulation")),
        theta_lower=tuple(
            np.asarray(model_config.get("theta_lower", (0.0, 0.0)), dtype=float)
        ),
        theta_upper=tuple(
            np.asarray(model_config.get("theta_upper", (1.0, 1.0)), dtype=float)
        ),
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
        freeze_theta=bool(model_config.get("freeze_theta", False)),
        refresh_weak_labels=bool(model_config.get("refresh_weak_labels", False)),
        random_state=seed,
    )

def _evaluate_pwl_candidate(
    data: ScenarioData,
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
    data: ScenarioData,
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
    # [ENGINEERING] Random theta starts respect the scenario's box bounds.
    theta_lower = np.asarray(model_config.get("theta_lower", (0.0, 0.0)), dtype=float)
    theta_upper = np.asarray(model_config.get("theta_upper", (1.0, 1.0)), dtype=float)
    # [ENGINEERING] Frozen-theta mode (heat W2): every candidate starts from
    # the configured fixed value instead of calibration or random starts.
    theta_frozen_value = model_config.get("theta_frozen_value")
    if theta_frozen_value is not None:
        theta_frozen_value = np.asarray(theta_frozen_value, dtype=float)
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
            if theta_frozen_value is not None:
                theta_init = theta_frozen_value.copy()
            elif start_index == 0:
                theta_init = None
            else:
                theta_init = rng.uniform(theta_lower, theta_upper)
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
    data: ScenarioData,
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
    data: ScenarioData,
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
    data: ScenarioData,
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
        np.any(np.abs(model.theta_ - model.theta_lower_) < 1e-3)
        or np.any(np.abs(model.theta_ - model.theta_upper_) < 1e-3)
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
            # [ENGINEERING] Case-B authentic mode audit (M1): drift of the
            # refreshed weak labels at the final BCD sweep (0 in fixed mode).
            "weak_refresh_final_drift": (
                float(model.weak_refresh_drift_[-1])
                if getattr(model, "weak_refresh_drift_", ())
                else 0.0
            ),
            "g_l2_norm": float(np.linalg.norm(model.g_)),
            "d_l2_norm": float(np.linalg.norm(model.d_)),
            "d_max_abs": float(np.max(np.abs(model.d_))),
            "tuning": model.tuning_diagnostics_,
            "active_h_features": list(model.active_features_),
            # [ENGINEERING] Audit trail for the orthogonalization drop rule:
            # a non-empty list here means B columns were discarded at fit
            # time (silently emptying B was audit report Bug 1).
            "dropped_b_features": [
                name
                for name, dropped_flag in zip(
                    features.b_names, getattr(features, "b_dropped_", ())
                )
                if dropped_flag
            ],
            "weak_distillation_r2": weak_distillation_r2,
            "labeled_prediction_r2": labeled_prediction_r2,
            "physics_component_norm": physics_component_norm,
            "process_component_norm": process_component_norm,
            "theta_boundary_hit": theta_boundary_hit,
            # [ENGINEERING] Output-mapping audit (alignment arms A1-A3,
            # migration/reports/迁移后续工作/PWL物理模型输出对齐与适用性说明.md
            # §6.3): phi is fitted by OLS on the train fold only, at the
            # initial theta, and stays fixed during the BCD sweeps.
            "phi_intercept": float(model.phi_intercept_),
            "phi_slope": float(model.phi_slope_),
            "g_group_norms": g_group_norms,
            "d_coefficients": model.d_.tolist(),
            "minimum_vs_selected_validation_mse": minimum_vs_selected,
        },
        extra=extra,
    )
    return row, _prediction_rows(row, data.test_y, prediction)

def _fit_baseline_condition(
    name: str,
    data: ScenarioData,
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
        model_cfg = config.get("model", {})
        model = PhysicsGPRegressor(
            data.physics_model,
            theta_lower=tuple(
                np.asarray(model_cfg.get("theta_lower", (0.0, 0.0)), dtype=float)
            ),
            theta_upper=tuple(
                np.asarray(model_cfg.get("theta_upper", (1.0, 1.0)), dtype=float)
            ),
            random_state=seed,
        ).fit(
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
