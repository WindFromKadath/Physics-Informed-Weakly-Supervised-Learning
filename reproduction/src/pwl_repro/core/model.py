"""PWL estimator: BCD outer loop with consensus ADMM block updates.

[PAPER] covers the Hg+Bd model, the four core objective terms, BCD order, and
the g/theta consensus decompositions.  [INFERRED] Mapping, initialization, and
concrete feature-library choices complete unpublished details.  [STABILITY]
Output scaling, optional d ridge, strict convergence checks, and residual
diagnostics make small-sample runs numerically usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .features import FeatureSpec, PWLFeatureLibrary, get_feature_spec
from .optimization import (
    ConsensusResult,
    consensus_admm,
    group_threshold,
    soft_threshold,
)

Array = NDArray[np.float64]
PhysicsModel = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    objective: float
    relative_change: float
    parameter_relative_change: float
    theta: tuple[float, ...]
    g_admm_iterations: int
    theta_admm_iterations: int
    g_admm_converged: bool
    theta_admm_converged: bool
    g_primal_residual: float
    g_dual_residual: float
    theta_primal_residual: float
    theta_dual_residual: float
    g_admm_rho: float
    theta_admm_rho: float


def calibrate_theta(
    x_ph: Array,
    y: Array,
    physics_model: PhysicsModel,
    lower: Array,
    upper: Array,
    *,
    seed: int = 0,
    starts: int = 4,
) -> Array:
    """[PAPER] Equation (12), with [INFERRED] multi-start L-BFGS-B details."""

    rng = np.random.default_rng(seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def loss(theta: Array) -> float:
        residual = np.asarray(y) - np.asarray(physics_model(x_ph, theta))
        if not np.all(np.isfinite(residual)):
            return 1e30
        return float(np.mean(residual**2))

    candidates = [(lower + upper) / 2.0]
    candidates.extend(rng.uniform(lower, upper) for _ in range(max(0, starts - 1)))
    results = [
        minimize(loss, start, method="L-BFGS-B", bounds=list(zip(lower, upper)))
        for start in candidates
    ]
    best = min(results, key=lambda item: float(item.fun))
    return np.clip(np.asarray(best.x, dtype=float), lower, upper)


class PWLRegressor:
    """Physics-informed weakly-supervised regressor.

    [PAPER] The fitted predictor and four unpenalized-in-d objective terms
    follow equations (4)--(5).  [STABILITY] ``d_ridge > 0`` adds a fifth term
    that is not in the paper; set it to zero to restore the published d block.

    The weak labels are fixed observations from a biased physics model by
    default.  If a callable physics model is supplied, it is used for theta
    initialization and optional linear phi fitting.  With
    ``refresh_weak_labels=True`` (the paper's Case-B protocol, see
    references/docs/09), the weak labels are instead re-evaluated at the
    current theta before every BCD sweep; the lagged refresh keeps each
    block's closed-form proximal update.  Either way the BCD objective
    remains the convex block formulation except for an explicitly nonzero
    d ridge.
    """

    def __init__(
        self,
        *,
        lambda_physics: float = 1.0,
        lambda_l1: float = 0.01,
        lambda_group: float = 0.01,
        theta_lower: tuple[float, ...] = (0.0, 0.0),
        theta_upper: tuple[float, ...] = (1.0, 1.0),
        mapping: str = "identity",
        standardize: bool = True,
        b_profile: str = "expanded",
        feature_spec: str = "simulation",
        admm_rho: float = 1.0,
        admm_adaptive_rho: bool = True,
        admm_tolerance: float = 1e-5,
        admm_max_iter: int = 1000,
        bcd_tolerance: float = 1e-4,
        bcd_parameter_tolerance: float | None = None,
        bcd_max_iter: int = 50,
        d_ridge: float = 0.0,
        freeze_theta: bool = False,
        refresh_weak_labels: bool = False,
        random_state: int = 0,
    ) -> None:
        self.lambda_physics = lambda_physics
        self.lambda_l1 = lambda_l1
        self.lambda_group = lambda_group
        self.theta_lower = theta_lower
        self.theta_upper = theta_upper
        self.mapping = mapping
        self.standardize = standardize
        self.b_profile = b_profile
        # [ENGINEERING] The spec carries input/theta dimensions for another
        # scenario's feature library; bounds must match its theta_dim.
        self.spec: FeatureSpec = get_feature_spec(feature_spec, b_profile)
        self.feature_spec = feature_spec
        _lower = np.asarray(theta_lower, dtype=float)
        _upper = np.asarray(theta_upper, dtype=float)
        expected = (self.spec.theta_dim,)
        if _lower.shape != expected or _upper.shape != expected:
            raise ValueError(
                f"theta bounds must have shape {expected} for feature_spec "
                f"{feature_spec!r}, got {_lower.shape} and {_upper.shape}."
            )
        self.admm_rho = admm_rho
        self.admm_adaptive_rho = admm_adaptive_rho
        self.admm_tolerance = admm_tolerance
        self.admm_max_iter = admm_max_iter
        self.bcd_tolerance = bcd_tolerance
        self.bcd_parameter_tolerance = (
            float(np.sqrt(bcd_tolerance))
            if bcd_parameter_tolerance is None
            else bcd_parameter_tolerance
        )
        self.bcd_max_iter = bcd_max_iter
        if d_ridge < 0.0:
            raise ValueError("d_ridge must be non-negative.")
        self.d_ridge = d_ridge
        # [ENGINEERING] Frozen-theta mode (heat W2): skip the theta block
        # update for scenarios where theta is structurally non-identifiable;
        # theta stays at its init (theta_init, or calibration/midpoint).
        self.freeze_theta = freeze_theta
        # [ENGINEERING] Case-B authentic mode (09 doc layer 3): refresh weak
        # labels at the current theta every BCD sweep.  Mutually exclusive
        # with freeze_theta (a frozen theta makes the refresh a no-op).
        if refresh_weak_labels and freeze_theta:
            raise ValueError(
                "refresh_weak_labels and freeze_theta are mutually exclusive."
            )
        self.refresh_weak_labels = refresh_weak_labels
        self.random_state = random_state

    def fit(
        self,
        x_ph: Array,
        x_pr: Array,
        y: Array,
        weak_x_ph: Array,
        weak_y: Array,
        *,
        theta_init: Array | None = None,
        physics_model: PhysicsModel | None = None,
    ) -> "PWLRegressor":
        x_ph = np.asarray(x_ph, dtype=float)
        x_pr = np.asarray(x_pr, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        weak_x_ph = np.asarray(weak_x_ph, dtype=float)
        weak_y = np.asarray(weak_y, dtype=float).reshape(-1)
        self._validate_inputs(x_ph, x_pr, y, weak_x_ph, weak_y)
        if self.refresh_weak_labels and physics_model is None:
            raise ValueError("refresh_weak_labels requires physics_model.")

        lower = np.asarray(self.theta_lower, dtype=float)
        upper = np.asarray(self.theta_upper, dtype=float)
        self.theta_lower_ = lower
        self.theta_upper_ = upper
        if theta_init is None and physics_model is not None:
            theta = calibrate_theta(
                x_ph,
                y,
                physics_model,
                lower,
                upper,
                seed=self.random_state,
            )
        elif theta_init is None:
            theta = (lower + upper) / 2.0
        else:
            theta = np.clip(np.asarray(theta_init, dtype=float), lower, upper)

        self.phi_intercept_, self.phi_slope_ = self._fit_mapping(
            x_ph, y, theta, physics_model
        )
        mapped_weak_y = self.phi_intercept_ + self.phi_slope_ * weak_y
        self.y_mean_ = float(np.mean(y))
        self.y_scale_ = float(np.std(y))
        if self.y_scale_ < 1e-12:
            self.y_scale_ = 1.0
        y_scaled = (y - self.y_mean_) / self.y_scale_
        weak_scaled = (mapped_weak_y - self.y_mean_) / self.y_scale_

        self.features_ = PWLFeatureLibrary(
            standardize=self.standardize,
            b_profile=self.b_profile,
            spec=self.spec,
        ).fit(
            np.vstack((x_ph, weak_x_ph)),
            x_ph,
            x_pr,
            theta,
        )
        b = self.features_.transform_b(x_ph, x_pr)
        # [INFERRED] The paper does not disclose complete initialization.
        g = np.zeros(len(self.features_.h_names), dtype=float)
        d = np.zeros(len(self.features_.b_names), dtype=float)
        history: list[IterationRecord] = []
        weak_refresh_drift: list[float] = []
        previous_objective = float("inf")

        for outer_iteration in range(1, self.bcd_max_iter + 1):
            if self.refresh_weak_labels:
                # [ENGINEERING] Case-B authentic mode (09 doc layer 3): weak
                # labels are the surrogate evaluated at the CURRENT theta.
                # The lagged refresh keeps each block's closed-form prox;
                # once theta settles the targets stop moving.
                weak_raw = np.asarray(
                    physics_model(weak_x_ph, theta), dtype=float
                )
                mapped_weak_y = self.phi_intercept_ + self.phi_slope_ * weak_raw
                refreshed = (mapped_weak_y - self.y_mean_) / self.y_scale_
                weak_refresh_drift.append(
                    float(np.max(np.abs(refreshed - weak_scaled)))
                )
                weak_scaled = refreshed
            previous_parameters = np.concatenate((g, d, theta))
            h_labeled = self.features_.transform_h(x_ph, theta)
            h_weak = self.features_.transform_h(weak_x_ph, theta)
            g_result = self._update_g(
                h_labeled, b, y_scaled, h_weak, weak_scaled, d, g
            )
            g = g_result.value

            target_d = y_scaled - h_labeled @ g
            if self.d_ridge == 0.0:
                # [PAPER] Unregularized least-squares d update.  A
                # minimum-norm solve remains well-defined when p_B > n.
                d = np.linalg.lstsq(b, target_d, rcond=None)[0]
            else:
                # [STABILITY; OBJECTIVE CHANGE] This ridge term is not in
                # equation (5).  It stabilizes d when the labeled design is
                # underdetermined and is also included in _objective below.
                gram = b.T @ b + self.d_ridge * np.eye(b.shape[1])
                d = np.linalg.solve(gram, b.T @ target_d)

            if self.freeze_theta:
                # [ENGINEERING] A converged dummy result keeps the history
                # schema and the combined convergence check intact while
                # theta stays at its initial value.
                theta_result = ConsensusResult(
                    value=theta,
                    iterations=0,
                    converged=True,
                    primal_residual=0.0,
                    dual_residual=0.0,
                    rho=self.admm_rho,
                    history=(),
                )
            else:
                theta_result = self._update_theta(
                    x_ph,
                    weak_x_ph,
                    y_scaled,
                    weak_scaled,
                    b,
                    d,
                    g,
                    theta,
                    lower,
                    upper,
                )
            theta = np.clip(theta_result.value, lower, upper)
            objective = self._objective(
                x_ph, weak_x_ph, y_scaled, weak_scaled, b, g, d, theta
            )
            relative_change = (
                float("inf")
                if not np.isfinite(previous_objective)
                else abs(previous_objective - objective)
                / max(1.0, abs(previous_objective))
            )
            current_parameters = np.concatenate((g, d, theta))
            parameter_relative_change = float(
                np.linalg.norm(current_parameters - previous_parameters)
                / max(1.0, np.linalg.norm(previous_parameters))
            )
            history.append(
                IterationRecord(
                    iteration=outer_iteration,
                    objective=objective,
                    relative_change=relative_change,
                    parameter_relative_change=parameter_relative_change,
                    theta=tuple(float(value) for value in theta),
                    g_admm_iterations=g_result.iterations,
                    theta_admm_iterations=theta_result.iterations,
                    g_admm_converged=g_result.converged,
                    theta_admm_converged=theta_result.converged,
                    g_primal_residual=g_result.primal_residual,
                    g_dual_residual=g_result.dual_residual,
                    theta_primal_residual=theta_result.primal_residual,
                    theta_dual_residual=theta_result.dual_residual,
                    g_admm_rho=g_result.rho,
                    theta_admm_rho=theta_result.rho,
                )
            )
            if (
                # [STABILITY] The paper does not publish this combined finite-
                # iteration rule.  Requiring objective change, parameter
                # change, and both inner solves prevents false convergence.
                relative_change < self.bcd_tolerance
                and parameter_relative_change < self.bcd_parameter_tolerance
                and g_result.converged
                and theta_result.converged
            ):
                break
            previous_objective = objective

        self.g_ = g
        self.d_ = d
        self.theta_ = theta
        self.history_ = tuple(history)
        self.weak_refresh_drift_ = tuple(weak_refresh_drift)
        self.n_iter_ = len(history)
        self.converged_ = bool(
            history
            and np.isfinite(history[-1].objective)
            and history[-1].relative_change < self.bcd_tolerance
            and history[-1].parameter_relative_change
            < self.bcd_parameter_tolerance
            and history[-1].g_admm_converged
            and history[-1].theta_admm_converged
        )
        self.training_objective_ = history[-1].objective
        self.active_features_ = tuple(
            name
            for name, coefficient in zip(self.features_.h_names, self.g_)
            if abs(coefficient) > 1e-8
        )
        return self

    def predict(self, x_ph: Array, x_pr: Array) -> Array:
        self._check_fitted()
        h = self.features_.transform_h(np.asarray(x_ph, dtype=float), self.theta_)
        b = self.features_.transform_b(
            np.asarray(x_ph, dtype=float), np.asarray(x_pr, dtype=float)
        )
        scaled = h @ self.g_ + b @ self.d_
        return self.y_mean_ + self.y_scale_ * scaled

    def _fit_mapping(
        self,
        x_ph: Array,
        y: Array,
        theta: Array,
        physics_model: PhysicsModel | None,
    ) -> tuple[float, float]:
        if self.mapping == "identity":
            # [INFERRED] Paper defines phi; identity is the simulation default,
            # not a disclosed author implementation.
            return 0.0, 1.0
        if self.mapping != "linear":
            raise ValueError("mapping must be 'identity' or 'linear'.")
        if physics_model is None:
            raise ValueError("linear mapping requires physics_model.")
        physics = np.asarray(physics_model(x_ph, theta), dtype=float)
        design = np.column_stack((np.ones(len(physics)), physics))
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(intercept), float(slope)

    def _update_g(
        self,
        h_labeled: Array,
        b: Array,
        y: Array,
        h_weak: Array,
        weak_y: Array,
        d: Array,
        initial: Array,
    ) -> ConsensusResult:
        # [PAPER] Proposition 1: labeled quadratic, physics quadratic, L1, and
        # group-L2 proximal nodes for the g block.
        identity = np.eye(h_labeled.shape[1])
        labeled_target = y - b @ d

        def prox_labeled(value: Array, step: float) -> Array:
            return np.linalg.solve(
                identity + step * (h_labeled.T @ h_labeled),
                value + step * h_labeled.T @ labeled_target,
            )

        def prox_physics(value: Array, step: float) -> Array:
            weight = step * self.lambda_physics
            return np.linalg.solve(
                identity + weight * (h_weak.T @ h_weak),
                value + weight * h_weak.T @ weak_y,
            )

        def prox_l1(value: Array, step: float) -> Array:
            return soft_threshold(value, self.lambda_l1 * step)

        def prox_group(value: Array, step: float) -> Array:
            return group_threshold(
                value, self.lambda_group * step, self.features_.groups
            )

        return consensus_admm(
            (prox_labeled, prox_physics, prox_l1, prox_group),
            initial,
            rho=self.admm_rho,
            max_iter=self.admm_max_iter,
            tolerance=self.admm_tolerance,
            adaptive_rho=self.admm_adaptive_rho,
        )

    def _update_theta(
        self,
        x_ph: Array,
        weak_x_ph: Array,
        y: Array,
        weak_y: Array,
        b: Array,
        d: Array,
        g: Array,
        initial: Array,
        lower: Array,
        upper: Array,
    ) -> ConsensusResult:
        # [PAPER] Proposition 2: labeled quadratic, physics quadratic, and box
        # projection nodes.  [INFERRED] features.py constructs the explicit
        # n-by-theta_dim affine designs to remove the supplement's dimension
        # ambiguity.
        base_labeled, design_labeled = self.features_.affine_prediction_parts(x_ph, g)
        base_weak, design_weak = self.features_.affine_prediction_parts(weak_x_ph, g)
        target_labeled = y - b @ d - base_labeled
        target_weak = weak_y - base_weak
        identity = np.eye(len(initial))

        def prox_labeled(value: Array, step: float) -> Array:
            return np.linalg.solve(
                identity + step * design_labeled.T @ design_labeled,
                value + step * design_labeled.T @ target_labeled,
            )

        def prox_physics(value: Array, step: float) -> Array:
            weight = step * self.lambda_physics
            return np.linalg.solve(
                identity + weight * design_weak.T @ design_weak,
                value + weight * design_weak.T @ target_weak,
            )

        def prox_box(value: Array, step: float) -> Array:
            del step
            return np.clip(value, lower, upper)

        return consensus_admm(
            (prox_labeled, prox_physics, prox_box),
            initial,
            rho=self.admm_rho,
            max_iter=self.admm_max_iter,
            tolerance=self.admm_tolerance,
            adaptive_rho=self.admm_adaptive_rho,
        )

    def _objective(
        self,
        x_ph: Array,
        weak_x_ph: Array,
        y: Array,
        weak_y: Array,
        b: Array,
        g: Array,
        d: Array,
        theta: Array,
    ) -> float:
        h_labeled = self.features_.transform_h(x_ph, theta)
        h_weak = self.features_.transform_h(weak_x_ph, theta)
        labeled_loss = 0.5 * np.sum((y - h_labeled @ g - b @ d) ** 2)
        physics_loss = (
            0.5
            * self.lambda_physics
            * np.sum((weak_y - h_weak @ g) ** 2)
        )
        l1 = self.lambda_l1 * np.sum(np.abs(g))
        group = self.lambda_group * sum(
            np.linalg.norm(g[index]) for index in self.features_.groups
        )
        # [STABILITY; OBJECTIVE CHANGE] Zero recovers paper equation (5).
        d_penalty = 0.5 * self.d_ridge * float(d @ d)
        return float(labeled_loss + physics_loss + l1 + group + d_penalty)

    def _validate_inputs(
        self,
        x_ph: Array, x_pr: Array, y: Array, weak_x_ph: Array, weak_y: Array
    ) -> None:
        x_ph_dim = self.spec.x_ph_dim
        x_pr_dim = self.spec.x_pr_dim
        if x_ph.ndim != 2 or x_ph.shape[1] != x_ph_dim:
            raise ValueError(f"x_ph must have shape (n, {x_ph_dim}).")
        if x_pr.ndim != 2 or x_pr.shape != (len(x_ph), x_pr_dim):
            raise ValueError(f"x_pr must have shape (n, {x_pr_dim}).")
        if y.shape != (len(x_ph),):
            raise ValueError("y must have shape (n,).")
        if weak_x_ph.ndim != 2 or weak_x_ph.shape[1] != x_ph_dim:
            raise ValueError(f"weak_x_ph must have shape (N, {x_ph_dim}).")
        if weak_y.shape != (len(weak_x_ph),):
            raise ValueError("weak_y must have shape (N,).")
        arrays = (x_ph, x_pr, y, weak_x_ph, weak_y)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Inputs must be finite.")

    def _check_fitted(self) -> None:
        if not hasattr(self, "g_"):
            raise RuntimeError("PWLRegressor is not fitted.")
