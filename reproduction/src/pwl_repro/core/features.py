"""Auditable H and B feature libraries proposed in references/docs/08.

[PAPER] requires H(x_ph, theta), B(x_pr, x_ph), group sparsity, and a theta
subproblem compatible with the published proximal update.  [INFERRED] The
paper does not publish the actual columns of H or B; every concrete feature in
this module is therefore a reproduction design.  [STABILITY] Scaling and
ratio-denominator guards are implementation additions.  [ENGINEERING] The
``FeatureSpec`` registry lets other scenarios (for example the heat-conduction
migration pilot) inject their own libraries without changing the estimator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
RawFeatureMap = Callable[[Array, Array], Array]


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


@dataclass(frozen=True)
class FeatureSpec:
    """[ENGINEERING] Injectable description of an H/B feature library.

    ``raw_h`` must be affine in ``theta`` so the theta subproblem keeps its
    closed-form least-squares proximal update.  Both raw maps must be
    module-level functions so parallel experiment workers can pickle them.
    """

    name: str
    x_ph_dim: int
    x_pr_dim: int
    theta_dim: int
    h_names: tuple[str, ...]
    b_names: tuple[str, ...]
    groups: tuple[Array, ...]
    group_names: tuple[str, ...]
    raw_h: RawFeatureMap
    raw_b: RawFeatureMap
    # [ENGINEERING] Plumlee residualization: when True, B is projected onto
    # the orthogonal complement of the H shape space at fit time so the g/d
    # decomposition stays identifiable (heat-scenario W1 experiment).
    orthogonalize_b: bool = False


_SPEC_FACTORIES: dict[str, Callable[[str], FeatureSpec]] = {}


def register_feature_spec(name: str, factory: Callable[[str], FeatureSpec]) -> None:
    """Register a feature-library factory keyed by scenario name."""

    _SPEC_FACTORIES[name] = factory


def get_feature_spec(name: str, b_profile: str = "expanded") -> FeatureSpec:
    """Resolve a registered feature spec.

    Scenario packages register their own factories when imported.  Keeping
    that direction explicit prevents the core package from importing a
    scenario package merely to discover features.
    """

    factory = _SPEC_FACTORIES.get(name)
    if factory is None:
        if name == "heat" or name.startswith("heat_"):
            raise ValueError(
                f"Unknown heat feature_spec {name!r}. "
                f"Registered: {sorted(_SPEC_FACTORIES)}."
            )
        if name == "case_b" or name.startswith("case_b_"):
            raise ValueError(
                f"Unknown case_b feature_spec {name!r}. "
                f"Registered: {sorted(_SPEC_FACTORIES)}."
            )
        raise ValueError(
            f"Unknown feature_spec {name!r}. "
            f"Registered: {sorted(_SPEC_FACTORIES)}. "
            "Import the scenario module before constructing its model."
        )
    return factory(b_profile)


class PWLFeatureLibrary:
    """Fit and transform an injected scenario feature specification.

    Scaling statistics are frozen at fit time, so standardization preserves
    theta affinity.  With no explicit ``spec``, the registered ``simulation``
    feature library remains the compatibility default.
    """

    def __init__(
        self,
        standardize: bool = True,
        b_profile: str = "expanded",
        spec: FeatureSpec | None = None,
    ) -> None:
        self.standardize = standardize
        if b_profile not in {"compact", "expanded"}:
            raise ValueError("b_profile must be 'compact' or 'expanded'.")
        self.b_profile = b_profile
        if spec is None:
            spec = get_feature_spec("simulation", b_profile)
        self.spec = spec
        self.h_names = spec.h_names
        self.b_names = spec.b_names
        self.groups = spec.groups
        self.group_names = spec.group_names

    def _h_raw(self, x_ph: Array, theta: Array) -> Array:
        return self.spec.raw_h(x_ph, theta)

    def _b_raw(self, x_ph: Array, x_pr: Array) -> Array:
        return self.spec.raw_b(x_ph, x_pr)

    def fit(
        self,
        h_x_ph: Array,
        b_x_ph: Array,
        b_x_pr: Array,
        theta_reference: Array,
    ) -> "PWLFeatureLibrary":
        h_raw = self._h_raw(h_x_ph, theta_reference)
        b_raw = self._b_raw(b_x_ph, b_x_pr)
        # [STABILITY] Rows whose statistics feed the B scaler.  With active
        # orthogonalization these must be the projection rows: centering
        # B_perp by any nonzero mean reintroduces H/B correlation, and B_perp
        # is zero-mean on the projection rows because H carries an intercept.
        b_scaler_rows = b_raw
        if self.spec.orthogonalize_b:
            # [ENGINEERING] Plumlee residualization: B_perp = B - H_ref @ M
            # with M from least squares on the projection rows.  The H shape
            # space is theta-invariant, so raw B_perp stays orthogonal to
            # H(theta) for every theta.  M and theta_reference are frozen
            # like the scalers, keeping transform_b well-defined on any
            # inputs.
            if self.spec.x_pr_dim == 0:
                # [STABILITY] With x^pr empty (case B) the projection design
                # uses the stacked labeled+weak rows.  Projecting on the
                # n_train labeled rows alone annihilates B whenever
                # n_train <= rank(H): the underdetermined lstsq interpolates
                # every B column exactly and the drop rule below then
                # discards all of B (audit report 案例B实现审查报告 Bug 1).
                h_ref = h_raw
                b_ref = self._b_raw(h_x_ph, np.empty((len(h_x_ph), 0)))
            else:
                # [STABILITY] B depends on x^pr, which is unavailable for the
                # weak rows; fall back to the labeled-row projection.
                h_ref = self._h_raw(b_x_ph, theta_reference)
                b_ref = b_raw
            rank_h = np.linalg.matrix_rank(h_ref)
            if len(h_ref) <= rank_h:
                # [STABILITY] Exact-interpolation guard: with no residual
                # degrees of freedom the projection would drop every B
                # column.  Skip orthogonalization loudly instead of silently
                # disabling the discrepancy term.
                warnings.warn(
                    "orthogonalize_b skipped: "
                    f"{len(h_ref)} projection rows <= rank(H)={rank_h}; "
                    "the least-squares projection would annihilate every "
                    "B column.",
                    stacklevel=2,
                )
                self.b_dropped_ = np.zeros(b_raw.shape[1], dtype=bool)
            else:
                # [STABILITY] Column normalization keeps lstsq's rcond cutoff
                # from discarding small-scale directions: raw Q^2 columns
                # reach 1e10, which otherwise breaks the projection's
                # precision.  The projector is invariant to this rescaling by
                # construction.
                col_norms = np.linalg.norm(h_ref, axis=0)
                col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
                m_normalized = np.linalg.lstsq(
                    h_ref / col_norms, b_ref, rcond=None
                )[0]
                self.b_projection_ = m_normalized / col_norms[:, None]
                self.theta_reference_ = np.asarray(
                    theta_reference, dtype=float
                ).copy()
                b_resid = b_ref - h_ref @ self.b_projection_
                # [STABILITY] Columns fully explained by H leave only
                # numerical noise (~1e-10); the scaler would amplify it to
                # unit-variance garbage.  Zero them here and in transform_b
                # so the design matrices stay consistent between fit and
                # prediction.
                original_norm = np.linalg.norm(b_ref, axis=0)
                residual_norm = np.linalg.norm(b_resid, axis=0)
                self.b_dropped_ = residual_norm <= 1e-8 * np.maximum(
                    original_norm, 1e-12
                )
                b_resid[:, self.b_dropped_] = 0.0
                b_scaler_rows = b_resid
        if self.standardize:
            self.h_scaler_ = _Scaler.fit(h_raw, intercept_index=0)
            self.b_scaler_ = _Scaler.fit(b_scaler_rows, intercept_index=0)
        else:
            self.h_scaler_ = _Scaler(np.zeros(h_raw.shape[1]), np.ones(h_raw.shape[1]))
            self.b_scaler_ = _Scaler(np.zeros(b_raw.shape[1]), np.ones(b_raw.shape[1]))
        return self

    def transform_h(self, x_ph: Array, theta: Array) -> Array:
        self._check_fitted()
        return self.h_scaler_.transform(self._h_raw(x_ph, theta))

    def transform_b(self, x_ph: Array, x_pr: Array) -> Array:
        self._check_fitted()
        b_raw = self._b_raw(x_ph, x_pr)
        if hasattr(self, "b_projection_"):
            b_raw = b_raw - self._h_raw(x_ph, self.theta_reference_) @ (
                self.b_projection_
            )
            b_raw[:, self.b_dropped_] = 0.0
        return self.b_scaler_.transform(b_raw)

    def affine_prediction_parts(self, x_ph: Array, g: Array) -> tuple[Array, Array]:
        """Return c and G such that H(x, theta) @ g == c + G @ theta.

        [PAPER] The theta subproblem is affine least squares with box
        projection.  [INFERRED] Constructing an explicit n-by-theta_dim G
        resolves the supplement's ambiguous identity-matrix dimension for any
        registered library.
        """

        theta_dim = self.spec.theta_dim
        zeros = np.zeros(theta_dim)
        base = self.transform_h(x_ph, zeros) @ g
        design = np.empty((len(x_ph), theta_dim), dtype=float)
        for index in range(theta_dim):
            unit = np.zeros(theta_dim)
            unit[index] = 1.0
            design[:, index] = self.transform_h(x_ph, unit) @ g - base
        return base, design

    def _check_fitted(self) -> None:
        if not hasattr(self, "h_scaler_"):
            raise RuntimeError("Feature library is not fitted.")
