"""Supervised and two-step physics baselines used in the paper.

[PAPER] identifies the comparison-model families.  [INFERRED] The exact
scikit-learn estimators, compact validation grids, and unspecified defaults
complete details not published by the authors.  [STABILITY] Scaling pipelines
improve conditioning.  [ENGINEERING] Estimator-level n_jobs only affects time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .core.model import PhysicsModel, calibrate_theta

Array = NDArray[np.float64]


def baseline_candidates(seed: int) -> dict[str, tuple[BaseEstimator, dict[str, list[object]]]]:
    """Return [PAPER]-aligned model families with [INFERRED] search grids."""

    return {
        "Ridge": (
            Pipeline((("scale", StandardScaler()), ("model", Ridge()))),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "SVR": (
            Pipeline(
                (
                    ("scale", StandardScaler()),
                    ("model", SVR(kernel="poly")),
                )
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__degree": [2, 3],
                "model__epsilon": [0.01, 0.1],
            },
        ),
        "DT": (
            DecisionTreeRegressor(random_state=seed),
            {"max_depth": [None, 2, 3, 5], "min_samples_leaf": [1, 3]},
        ),
        "RF": (
            RandomForestRegressor(
                n_estimators=200,
                random_state=seed,
                n_jobs=-1,
            ),
            {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2]},
        ),
        "GBDT": (
            GradientBoostingRegressor(random_state=seed),
            {
                "n_estimators": [30, 100, 160],
                "learning_rate": [0.1, 0.15, 0.2],
                "max_depth": [2, 3, 5],
            },
        ),
        "GP": (
            Pipeline(
                (
                    ("scale", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel(1.0) * RBF(1.0)
                            + WhiteKernel(0.1),
                            normalize_y=True,
                            random_state=seed,
                            n_restarts_optimizer=0,
                        ),
                    ),
                )
            ),
            {},
        ),
    }


def tune_baseline(
    name: str,
    x_train: Array,
    y_train: Array,
    x_validation: Array,
    y_validation: Array,
    *,
    seed: int,
    refit: bool = True,
) -> tuple[BaseEstimator, dict[str, object], float]:
    """Select one baseline configuration by explicit validation MSE."""

    candidates = baseline_candidates(seed)
    if name not in candidates:
        raise KeyError(f"Unknown baseline {name!r}. Options: {sorted(candidates)}")
    estimator, grid = candidates[name]
    best_model: BaseEstimator | None = None
    best_params: dict[str, object] = {}
    best_mse = float("inf")
    for params in ParameterGrid(grid or {}):
        model = clone(estimator).set_params(**params)
        model.fit(x_train, y_train)
        mse = float(np.mean((y_validation - model.predict(x_validation)) ** 2))
        if mse < best_mse:
            best_model, best_params, best_mse = model, dict(params), mse
    assert best_model is not None
    if not refit:
        return best_model, best_params, best_mse
    final = clone(estimator).set_params(**best_params)
    final.fit(
        np.vstack((x_train, x_validation)),
        np.concatenate((y_train, y_validation)),
    )
    return final, best_params, best_mse


@dataclass
class PhysicsGPRegressor:
    """Equation (12) calibration followed by a GP discrepancy model."""

    physics_model: PhysicsModel
    theta_lower: tuple[float, ...] = (0.0, 0.0)
    theta_upper: tuple[float, ...] = (1.0, 1.0)
    random_state: int = 0

    def fit(self, x_ph: Array, x_pr: Array, y: Array) -> "PhysicsGPRegressor":
        lower = np.asarray(self.theta_lower, dtype=float)
        upper = np.asarray(self.theta_upper, dtype=float)
        self.theta_ = calibrate_theta(
            x_ph,
            y,
            self.physics_model,
            lower,
            upper,
            seed=self.random_state,
        )
        physics = self.physics_model(x_ph, self.theta_)
        residual = np.asarray(y) - physics
        self.scaler_ = StandardScaler().fit(np.column_stack((x_ph, x_pr)))
        scaled_x = self.scaler_.transform(np.column_stack((x_ph, x_pr)))
        kernel = ConstantKernel(1.0) * RBF(np.ones(scaled_x.shape[1])) + WhiteKernel(0.1)
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=self.random_state,
            n_restarts_optimizer=0,
        ).fit(scaled_x, residual)
        return self

    def predict(self, x_ph: Array, x_pr: Array) -> Array:
        physics = self.physics_model(x_ph, self.theta_)
        scaled_x = self.scaler_.transform(np.column_stack((x_ph, x_pr)))
        return physics + self.gp_.predict(scaled_x)
