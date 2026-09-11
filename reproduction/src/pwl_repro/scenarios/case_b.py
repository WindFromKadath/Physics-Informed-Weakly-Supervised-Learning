"""Case B (spot-weld nugget diameter) scenario module.

[ENGINEERING] Follows ``references/docs/09_案例B物理代理模型解析.md`` and the
data contract in ``datasets/case_b_spotweld/README.md``: an ordinary-kriging
surrogate trained ONLY on the 35 ANSYS simulations (theta as 4th input),
theta in [0.8, 8.0] (SAVE package prior), x^pr = empty (B is built on x^ph
and therefore needs orthogonalization), and weak labels are surrogate
evaluations refreshed at the current theta (M1 in model.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ..core.features import FeatureSpec, register_feature_spec
from ..core.model import calibrate_theta

Array = NDArray[np.float64]

# [ENGINEERING] Scenario constants (SAVE 1.0 / 09 doc §4).
THETA_LOWER = (0.8,)
THETA_UPPER = (8.0,)
THETA_FALLBACK = np.array([4.0])  # SAVE example bestguess.
LOAD_RANGE = (4.0, 5.3)
CURRENT_RANGE = (21.0, 29.0)
THICKNESS_LEVELS = (1.0, 2.0)
INPUT_COLUMNS = ("load", "current", "thickness")
NOISE_VAR_REFERENCE = 0.2  # Paper-reported replicate variance (≈0.2006 here).

CASE_B_H_NAMES: tuple[str, ...] = (
    "1",
    "x1",
    "x1^2",
    "x2",
    "x2^2",
    "x1*x2",
    "x3",
    "theta",
    "theta*x1",
    "theta*x2",
    "theta*x3",
)
CASE_B_B_NAMES: tuple[str, ...] = (
    "1",
    "x1",
    "x2",
    "x3",
    "x1*x2",
    "x1*x3",
    "x2*x3",
)
CASE_B_GROUPS: tuple[Array, ...] = (
    np.arange(0, 3),
    np.arange(3, 6),
    np.arange(6, 7),
    np.arange(7, 11),
)
CASE_B_GROUP_NAMES: tuple[str, ...] = ("load", "current", "thickness", "theta")

CASE_B_EXT_H_NAMES: tuple[str, ...] = (
    "1",
    "x1",
    "x1^2",
    "x1^3",
    "x2",
    "x2^2",
    "x2^3",
    "x1*x2",
    "x3",
    "x1*x3",
    "x2*x3",
    "x1*x2*x3",
    "theta",
    "theta*x1",
    "theta*x2",
    "theta*x3",
    "theta*x1^2",
    "theta*x2^2",
    "theta*x1*x2",
    "theta*x1*x3",
    "theta*x2*x3",
)
CASE_B_EXT_GROUPS: tuple[Array, ...] = (
    np.arange(0, 4),
    np.arange(4, 8),
    np.arange(8, 12),
    np.arange(12, 21),
)


def case_b_h_raw(x_ph: Array, theta: Array) -> Array:
    """[INFERRED] 11-column affine H (09 doc §4.4 draft): low-order
    polynomials plus first-order theta interactions; thickness (binary) is
    kept linear only, and the four groups follow q1+q3 = 3+1."""

    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    t = np.asarray(theta, dtype=float).reshape(-1)[0]
    return np.column_stack(
        (
            np.ones_like(x1),
            x1,
            x1**2,
            x2,
            x2**2,
            x1 * x2,
            x3,
            t * np.ones_like(x1),
            t * x1,
            t * x2,
            t * x3,
        )
    )


def case_b_b_raw(x_ph: Array, x_pr: Array) -> Array:
    """[INFERRED] 7-column B on x^ph only (x^pr is empty in case B).

    Used together with orthogonalize_b: columns spanned by H are projected
    out and auto-dropped by the library, so the effective discrepancy terms
    are the interaction columns H does not carry.
    """

    del x_pr  # x^pr = empty; B depends on x^ph only.
    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    return np.column_stack(
        (np.ones_like(x1), x1, x2, x3, x1 * x2, x1 * x3, x2 * x3)
    )


CASE_B_EXT_B_NAMES: tuple[str, ...] = CASE_B_B_NAMES + (
    # [ENGINEERING] The ext H spans every column of the base B (audit report
    # 案例B实现审查报告 Bug 2), so without these extra columns the
    # orthogonalization annihilates B at every sample size.  All three are
    # outside the ext H column space (rank-verified).  x3^2 is deliberately
    # absent: thickness is binary, so x3^2 is affine in x3 on the data
    # support and already spanned by H.
    "x1^2*x2",
    "x1*x2^2",
    "x1^2*x3",
)


def case_b_ext_b_raw(x_ph: Array, x_pr: Array) -> Array:
    """[INFERRED] Base B plus H-external interactions for the ext library."""

    base = case_b_b_raw(x_ph, x_pr)
    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    return np.column_stack((base, x1**2 * x2, x1 * x2**2, x1**2 * x3))


def case_b_ext_h_raw(x_ph: Array, theta: Array) -> Array:
    """[INFERRED] 21-column extended affine H (case-B iteration 2).

    The first-cut 11-column H left weak_distillation_r2 ~ 0.28 against the
    kriging surrogate; this extension (cubics + three-way interaction +
    theta x quadratic interactions) reaches R2 0.92-0.98 across the theta
    box on dense surrogate evaluations, while staying affine in theta.
    """

    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    t = np.asarray(theta, dtype=float).reshape(-1)[0]
    return np.column_stack(
        (
            np.ones_like(x1),
            x1,
            x1**2,
            x1**3,
            x2,
            x2**2,
            x2**3,
            x1 * x2,
            x3,
            x1 * x3,
            x2 * x3,
            x1 * x2 * x3,
            t * np.ones_like(x1),
            t * x1,
            t * x2,
            t * x3,
            t * x1**2,
            t * x2**2,
            t * x1 * x2,
            t * x1 * x3,
            t * x2 * x3,
        )
    )


def case_b_feature_spec() -> FeatureSpec:
    """[ENGINEERING] Case-B spec: x_ph=3, x_pr=0, theta=1, 4 groups, and
    mandatory B orthogonalization (B shares H's inputs; 09 doc gap B3)."""

    return FeatureSpec(
        name="case_b",
        x_ph_dim=3,
        x_pr_dim=0,
        theta_dim=1,
        h_names=CASE_B_H_NAMES,
        b_names=CASE_B_B_NAMES,
        groups=CASE_B_GROUPS,
        group_names=CASE_B_GROUP_NAMES,
        raw_h=case_b_h_raw,
        raw_b=case_b_b_raw,
        orthogonalize_b=True,
    )


def get_case_b_feature_spec(name: str = "case_b") -> FeatureSpec:
    if name == "case_b":
        return case_b_feature_spec()
    if name == "case_b_ext":
        return FeatureSpec(
            name="case_b_ext",
            x_ph_dim=3,
            x_pr_dim=0,
            theta_dim=1,
            h_names=CASE_B_EXT_H_NAMES,
            b_names=CASE_B_EXT_B_NAMES,
            groups=CASE_B_EXT_GROUPS,
            group_names=CASE_B_GROUP_NAMES,
            raw_h=case_b_ext_h_raw,
            raw_b=case_b_ext_b_raw,
            orthogonalize_b=True,
        )
    raise ValueError(f"Unknown case_b feature_spec {name!r}.")


def _case_b_spec_factory(name: str):
    """Build a registry factory while retaining the configured spec name."""

    return lambda _b_profile: get_case_b_feature_spec(name)


for _feature_spec_name in ("case_b", "case_b_ext"):
    register_feature_spec(
        _feature_spec_name,
        _case_b_spec_factory(_feature_spec_name),
    )


class CaseBSurrogate:
    """Ordinary-kriging surrogate for the 35 ANSYS simulations.

    [INFERRED] Constant trend + Gaussian correlation with anisotropic
    length-scales estimated by marginal-likelihood maximization, aligned
    with the SAVE/DiceKriging reference implementation (09 doc §4.2).
    Theta (tuning) enters as the 4th input dimension.
    """

    def __init__(self, *, random_state: int = 0) -> None:
        self.random_state = random_state

    def fit(self, model_frame: pd.DataFrame) -> "CaseBSurrogate":
        x = model_frame[["load", "current", "thickness", "tuning"]].to_numpy(
            dtype=float
        )
        y = model_frame["diameter"].to_numpy(dtype=float)
        self.x_mean_ = x.mean(axis=0)
        self.x_scale_ = x.std(axis=0)
        self.x_scale_ = np.where(self.x_scale_ < 1e-12, 1.0, self.x_scale_)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(4),
            length_scale_bounds=(1e-2, 1e3),
        ) + WhiteKernel(1e-10, "fixed")  # Interpolation: simulations are deterministic.
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=self.random_state,
        ).fit((x - self.x_mean_) / self.x_scale_, y)
        self.training_frame_ = model_frame.reset_index(drop=True)
        return self

    def predict(self, x_ph: Array, theta: Array) -> Array:
        self._check_fitted()
        x_ph = np.asarray(x_ph, dtype=float)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        x = np.column_stack(
            (x_ph, np.full(len(x_ph), theta[0], dtype=float))
        )
        return self.gp_.predict((x - self.x_mean_) / self.x_scale_)

    def loo_diagnostics(self) -> dict[str, object]:
        """[ENGINEERING] Leave-one-out quality audit (09 doc gap B5)."""

        self._check_fitted()
        frame = self.training_frame_
        x = frame[["load", "current", "thickness", "tuning"]].to_numpy(
            dtype=float
        )
        y = frame["diameter"].to_numpy(dtype=float)
        predictions = np.empty_like(y)
        for index in range(len(frame)):
            mask = np.arange(len(frame)) != index
            clone = CaseBSurrogate(random_state=self.random_state).fit(
                frame.iloc[mask]
            )
            predictions[index] = clone.predict(
                x[index : index + 1, :3], x[index : index + 1, 3]
            )[0]
        residual = y - predictions
        return {
            "loo_rmse": float(np.sqrt(np.mean(residual**2))),
            "loo_max_abs": float(np.max(np.abs(residual))),
            "kernel": str(self.gp_.kernel_),
        }

    def _check_fitted(self) -> None:
        if not hasattr(self, "gp_"):
            raise RuntimeError("CaseBSurrogate is not fitted.")


@dataclass(frozen=True)
class CaseBScenarioData:
    """[ENGINEERING] One random field-data split behind ScenarioData.

    ``theta_true`` stores the eq.(12) calibration estimate on the labeled
    pool (real case: no ground truth) and is marked in sampling_diagnostics.
    """

    x_ph: Array
    x_pr: Array
    y: Array
    weak_x_ph: Array
    weak_y: Array
    test_x_ph: Array
    test_x_pr: Array
    test_y: Array
    theta_true: Array
    noise_sigma: float
    physics_noise_sigma: float
    labeled_physics_noise: Array
    weak_physics_noise: Array
    test_physics_noise: Array
    discrepancy_scale: float
    covariance: Array
    sampling_diagnostics: Mapping[str, float | int | str]
    surrogate: CaseBSurrogate

    def physics_model(self, x_ph: Array, theta: Array) -> Array:
        """Surrogate evaluation with theta as the 4th input dimension."""

        return self.surrogate.predict(x_ph, theta)

    def labeled_physics_output(self, *, include_noise: bool = True) -> Array:
        output = self.physics_model(self.x_ph, self.theta_true)
        if include_noise:
            output = output + self.labeled_physics_noise
        return output

    def labeled_physics_correlation(self, *, include_noise: bool = True) -> float:
        return float(
            np.corrcoef(
                self.y,
                self.labeled_physics_output(include_noise=include_noise),
            )[0, 1]
        )


def load_case_b_frames(
    data_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (field, model) frames and check the data contract."""

    root = Path(data_root)
    field = pd.read_csv(root / "spotweldfield.csv")
    model = pd.read_csv(root / "spotweldmodel.csv")
    expected_field = list(INPUT_COLUMNS) + ["diameter"]
    expected_model = ["tuning"] + expected_field
    for name, frame, expected in (
        ("spotweldfield", field, expected_field),
        ("spotweldmodel", model, expected_model),
    ):
        missing = [c for c in expected if c not in frame.columns]
        if missing:
            raise ValueError(f"{name}.csv misses columns {missing}.")
    if len(field) != 120 or len(model) != 35:
        raise ValueError(
            f"Expected 120 field and 35 model rows, got {len(field)}, {len(model)}."
        )
    if not all(np.all(np.isfinite(f.to_numpy(dtype=float))) for f in (field, model)):
        raise ValueError("Case-B frames contain non-finite values.")
    return field.reset_index(drop=True), model.reset_index(drop=True)


def pooled_replicate_sigma(field: pd.DataFrame) -> float:
    """Pooled within-setting replicate std (noise floor audit)."""

    variance = field.groupby(list(INPUT_COLUMNS))["diameter"].var(ddof=1)
    return float(np.sqrt(variance.mean()))


def sample_weak_inputs(n_weak: int, seed: int) -> Array:
    """LHS over the FIELD experimental domain (09 doc §4.3); thickness ∈ {1,2}."""

    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n_weak)
    load = LOAD_RANGE[0] + unit[:, 0] * (LOAD_RANGE[1] - LOAD_RANGE[0])
    current = CURRENT_RANGE[0] + unit[:, 1] * (CURRENT_RANGE[1] - CURRENT_RANGE[0])
    thickness = np.where(unit[:, 2] < 0.5, *THICKNESS_LEVELS)
    return np.column_stack((load, current, thickness))


def build_case_b_scenario(
    field: pd.DataFrame,
    surrogate: CaseBSurrogate,
    *,
    n_labeled: int,
    split_seed: int,
    n_weak: int = 120,
    weak_seed: int = 9102026,
    theta_calibration_indices: Array | None = None,
) -> CaseBScenarioData:
    """Random split of the 120 field samples into labeled/test (paper protocol).

    [STABILITY] ``theta_calibration_indices`` restricts the eq.(12) theta_hat
    calibration (and therefore the initial weak labels) to the training rows
    of the labeled pool; without it the validation rows leak into theta_hat
    (audit report 案例B实现审查报告 O1).
    """

    if not 0 < n_labeled < len(field):
        raise ValueError("n_labeled must be within the field sample count.")
    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(len(field))
    labeled_index = np.sort(permutation[:n_labeled])
    test_index = np.sort(permutation[n_labeled:])

    x_all = field[list(INPUT_COLUMNS)].to_numpy(dtype=float)
    y_all = field["diameter"].to_numpy(dtype=float)
    x_ph, y = x_all[labeled_index], y_all[labeled_index]
    test_x_ph, test_y = x_all[test_index], y_all[test_index]

    if theta_calibration_indices is None:
        cal_x_ph, cal_y = x_ph, y
        theta_hat_note = "eq(12) calibration on the labeled pool (no ground truth)"
    else:
        calibration_index = np.asarray(theta_calibration_indices, dtype=int)
        cal_x_ph, cal_y = x_ph[calibration_index], y[calibration_index]
        theta_hat_note = (
            "eq(12) calibration on the training rows only (no ground truth)"
        )
    theta_hat = calibrate_theta(
        cal_x_ph,
        cal_y,
        surrogate.predict,
        np.asarray(THETA_LOWER),
        np.asarray(THETA_UPPER),
        seed=split_seed,
    )
    weak_x_ph = sample_weak_inputs(n_weak, weak_seed)
    weak_y = surrogate.predict(weak_x_ph, theta_hat)

    empty_labeled = np.empty((len(x_ph), 0))
    empty_test = np.empty((len(test_x_ph), 0))
    diagnostics: dict[str, float | int | str] = {
        "scenario": "case_b",
        "n_labeled": n_labeled,
        "split_seed": split_seed,
        "theta_hat_note": theta_hat_note,
        "theta_hat": float(theta_hat[0]),
        "n_weak": n_weak,
        "weak_seed": weak_seed,
        "noise_sigma_pooled": pooled_replicate_sigma(field),
    }
    return CaseBScenarioData(
        x_ph=x_ph,
        x_pr=empty_labeled,
        y=y,
        weak_x_ph=weak_x_ph,
        weak_y=weak_y,
        test_x_ph=test_x_ph,
        test_x_pr=empty_test,
        test_y=test_y,
        theta_true=theta_hat,
        noise_sigma=pooled_replicate_sigma(field),
        physics_noise_sigma=0.0,
        labeled_physics_noise=np.zeros(len(y)),
        weak_physics_noise=np.zeros(n_weak),
        test_physics_noise=np.zeros(len(test_y)),
        discrepancy_scale=1.0,
        covariance=np.empty((0, 0)),
        sampling_diagnostics=diagnostics,
        surrogate=surrogate,
    )


def save_loo_diagnostics(surrogate: CaseBSurrogate, path: str | Path) -> None:
    """Persist the surrogate LOO audit next to the run artifacts."""

    Path(path).write_text(
        json.dumps(surrogate.loo_diagnostics(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
