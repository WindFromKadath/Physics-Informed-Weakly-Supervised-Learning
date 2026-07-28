"""Auditable H and B feature libraries proposed in references/docs/08.

[PAPER] requires H(x_ph, theta), B(x_pr, x_ph), group sparsity, and a theta
subproblem compatible with the published proximal update.  [INFERRED] The
paper does not publish the actual columns of H or B; every concrete feature in
this module is therefore a reproduction design.  [STABILITY] Scaling and
ratio-denominator guards are implementation additions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .simulation import saturation

Array = NDArray[np.float64]


@dataclass
class _Scaler:
    mean: Array
    scale: Array

    @classmethod
    def fit(cls, matrix: Array, *, intercept_index: int | None = None) -> "_Scaler":
        # [STABILITY] Scaling is not a disclosed paper setting.  It keeps L1,
        # group, and ridge penalties comparable across heterogeneous columns.
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        if intercept_index is not None:
            mean[intercept_index] = 0.0
            scale[intercept_index] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, matrix: Array) -> Array:
        return (matrix - self.mean) / self.scale


class PWLFeatureLibrary:
    """[INFERRED] 25-column affine H and configurable 3/12-column B.

    H is affine in the two calibration parameters by construction, matching
    the structure required by the paper's theta proximal update.  The exact
    25 columns, five group memberships, compact B, and expanded B are not
    published by the authors.  Scaling statistics are frozen at fit time, so
    the [STABILITY] standardization preserves theta affinity.
    """

    h_names = (
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
    expanded_b_names = (
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
    compact_b_names = (
        # [INFERRED] Exactly spans paper equation (11) with three columns and
        # reduces underdetermination when only seven training labels exist.
        "1",
        "x4/x5",
        "x5/x4",
    )
    groups = (
        # [PAPER] The group penalty has q1+q3=5 groups.  [INFERRED] These exact
        # column memberships follow the reproduction's concrete H library.
        np.arange(0, 4),
        np.arange(4, 7),
        np.arange(7, 15),
        np.arange(15, 20),
        np.arange(20, 25),
    )
    group_names = ("x1", "x2", "x3", "theta1", "theta2")

    def __init__(
        self,
        standardize: bool = True,
        b_profile: str = "expanded",
    ) -> None:
        self.standardize = standardize
        if b_profile not in {"compact", "expanded"}:
            raise ValueError("b_profile must be 'compact' or 'expanded'.")
        self.b_profile = b_profile
        self.b_names = (
            self.compact_b_names
            if b_profile == "compact"
            else self.expanded_b_names
        )

    @staticmethod
    def _h_raw(x_ph: Array, theta: Array) -> Array:
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

    def _b_raw(self, x_ph: Array, x_pr: Array) -> Array:
        x1, x2, x3 = np.asarray(x_ph, dtype=float).T
        x4, x5 = np.asarray(x_pr, dtype=float).T
        # [STABILITY] Equation (11)'s ratio features are undefined at zero.
        x4_safe = np.where(np.abs(x4) < 1e-8, np.where(x4 < 0, -1e-8, 1e-8), x4)
        x5_safe = np.where(np.abs(x5) < 1e-8, np.where(x5 < 0, -1e-8, 1e-8), x5)
        if self.b_profile == "compact":
            return np.column_stack(
                (
                    np.ones_like(x1),
                    x4 / x5_safe,
                    x5 / x4_safe,
                )
            )
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

    def fit(
        self,
        h_x_ph: Array,
        b_x_ph: Array,
        b_x_pr: Array,
        theta_reference: Array,
    ) -> "PWLFeatureLibrary":
        h_raw = self._h_raw(h_x_ph, theta_reference)
        b_raw = self._b_raw(b_x_ph, b_x_pr)
        if self.standardize:
            self.h_scaler_ = _Scaler.fit(h_raw, intercept_index=0)
            self.b_scaler_ = _Scaler.fit(b_raw, intercept_index=0)
        else:
            self.h_scaler_ = _Scaler(np.zeros(h_raw.shape[1]), np.ones(h_raw.shape[1]))
            self.b_scaler_ = _Scaler(np.zeros(b_raw.shape[1]), np.ones(b_raw.shape[1]))
        return self

    def transform_h(self, x_ph: Array, theta: Array) -> Array:
        self._check_fitted()
        return self.h_scaler_.transform(self._h_raw(x_ph, theta))

    def transform_b(self, x_ph: Array, x_pr: Array) -> Array:
        self._check_fitted()
        return self.b_scaler_.transform(self._b_raw(x_ph, x_pr))

    def affine_prediction_parts(self, x_ph: Array, g: Array) -> tuple[Array, Array]:
        """Return c and G such that H(x, theta) @ g == c + G @ theta.

        [PAPER] The theta subproblem is affine least squares with box
        projection.  [INFERRED] Constructing an explicit n-by-2 G resolves the
        supplement's ambiguous identity-matrix dimension for this library.
        """

        zeros = np.zeros(2)
        base = self.transform_h(x_ph, zeros) @ g
        design = np.empty((len(x_ph), 2), dtype=float)
        for index in range(2):
            unit = np.zeros(2)
            unit[index] = 1.0
            design[:, index] = self.transform_h(x_ph, unit) @ g - base
        return base, design

    def _check_fitted(self) -> None:
        if not hasattr(self, "h_scaler_"):
            raise RuntimeError("Feature library is not fitted.")
