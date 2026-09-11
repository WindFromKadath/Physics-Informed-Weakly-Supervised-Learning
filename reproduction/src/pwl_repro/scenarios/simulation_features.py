"""Feature-library definition for the Section IV simulation scenario."""

from __future__ import annotations

import numpy as np

from ..core.features import Array, FeatureSpec, register_feature_spec
from .simulation import saturation

SIMULATION_H_NAMES: tuple[str, ...] = (
    "1",
    "x1",
    "x1^2",
    "x1^3",
    "x2",
    "x2^2",
    "x2^3",
    "x3",
    "x3^2",
    "x3^3",
    "S(x3)",
    "S(x3)*x1",
    "S(x3)*x2",
    "S(x3)*x1^3",
    "S(x3)*x2^3",
    "theta1",
    "theta1*x1",
    "theta1*x1^3",
    "theta1*S(x3)",
    "theta1*x1^3*S(x3)",
    "theta2",
    "theta2*x2",
    "theta2*x2^3",
    "theta2*S(x3)",
    "theta2*x2^3*S(x3)",
)
SIMULATION_EXPANDED_B_NAMES: tuple[str, ...] = (
    # [INFERRED] Broader discrepancy library from the design study.  It can
    # overlap H and is retained for sensitivity analysis, not as paper fact.
    "1",
    "x4",
    "x5",
    "x4^2",
    "x5^2",
    "x4*x5",
    "x4/x5",
    "x5/x4",
    "x1^2",
    "x2^2",
    "x3^3",
    "x1*x2",
)
SIMULATION_COMPACT_B_NAMES: tuple[str, ...] = (
    # [INFERRED] Exactly spans paper equation (11) with three columns and
    # reduces underdetermination when only seven training labels exist.
    "1",
    "x4/x5",
    "x5/x4",
)
SIMULATION_GROUPS: tuple[Array, ...] = (
    # [PAPER] The group penalty has q1+q3=5 groups.  [INFERRED] These exact
    # column memberships follow the reproduction's concrete H library.
    np.arange(0, 4),
    np.arange(4, 7),
    np.arange(7, 15),
    np.arange(15, 20),
    np.arange(20, 25),
)
SIMULATION_GROUP_NAMES: tuple[str, ...] = (
    "x1",
    "x2",
    "x3",
    "theta1",
    "theta2",
)


def _simulation_h_raw(x_ph: Array, theta: Array) -> Array:
    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    theta = np.asarray(theta, dtype=float)
    s = saturation(x3)
    return np.column_stack(
        (
            np.ones_like(x1),
            x1,
            x1**2,
            x1**3,
            x2,
            x2**2,
            x2**3,
            x3,
            x3**2,
            x3**3,
            s,
            s * x1,
            s * x2,
            s * x1**3,
            s * x2**3,
            theta[0] * np.ones_like(x1),
            theta[0] * x1,
            theta[0] * x1**3,
            theta[0] * s,
            theta[0] * x1**3 * s,
            theta[1] * np.ones_like(x1),
            theta[1] * x2,
            theta[1] * x2**3,
            theta[1] * s,
            theta[1] * x2**3 * s,
        )
    )


def _simulation_safe_ratio_inputs(x_pr: Array) -> tuple[Array, Array]:
    # [STABILITY] Equation (11)'s ratio features are undefined at zero.
    x4, x5 = np.asarray(x_pr, dtype=float).T
    x4_safe = np.where(np.abs(x4) < 1e-8, np.where(x4 < 0, -1e-8, 1e-8), x4)
    x5_safe = np.where(np.abs(x5) < 1e-8, np.where(x5 < 0, -1e-8, 1e-8), x5)
    return x4_safe, x5_safe


def _simulation_b_raw_compact(x_ph: Array, x_pr: Array) -> Array:
    x1 = np.asarray(x_ph, dtype=float)[:, 0]
    x4_safe, x5_safe = _simulation_safe_ratio_inputs(x_pr)
    return np.column_stack(
        (
            np.ones_like(x1),
            np.asarray(x_pr, dtype=float)[:, 0] / x5_safe,
            np.asarray(x_pr, dtype=float)[:, 1] / x4_safe,
        )
    )


def _simulation_b_raw_expanded(x_ph: Array, x_pr: Array) -> Array:
    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    x4, x5 = np.asarray(x_pr, dtype=float).T
    x4_safe, x5_safe = _simulation_safe_ratio_inputs(x_pr)
    return np.column_stack(
        (
            np.ones_like(x1),
            x4,
            x5,
            x4**2,
            x5**2,
            x4 * x5,
            x4 / x5_safe,
            x5 / x4_safe,
            x1**2,
            x2**2,
            x3**3,
            x1 * x2,
        )
    )


def simulation_feature_spec(b_profile: str = "expanded") -> FeatureSpec:
    if b_profile == "compact":
        b_names = SIMULATION_COMPACT_B_NAMES
        raw_b = _simulation_b_raw_compact
    elif b_profile == "expanded":
        b_names = SIMULATION_EXPANDED_B_NAMES
        raw_b = _simulation_b_raw_expanded
    else:
        raise ValueError("b_profile must be 'compact' or 'expanded'.")
    return FeatureSpec(
        name="simulation",
        x_ph_dim=3,
        x_pr_dim=2,
        theta_dim=2,
        h_names=SIMULATION_H_NAMES,
        b_names=b_names,
        groups=SIMULATION_GROUPS,
        group_names=SIMULATION_GROUP_NAMES,
        raw_h=_simulation_h_raw,
        raw_b=raw_b,
    )


register_feature_spec("simulation", simulation_feature_spec)
