"""One-dimensional heat-conduction scenario (COMSOL migration pilot).

[ENGINEERING] Scenario constants, the closed-form low-fidelity model, and the
data adapter follow ``migration/reports/legacy/COMSOL场景一维简化参考.md`` (v1)
and the dataset specification ``migration/datasets/README.md`` (v1.3).  The H/B feature
libraries follow ``migration/reports/COMSOL三维热传导场景迁移分析.md`` §6:
theta = 1/k makes the affinity constraint exact, x_pr = P carries the contact
resistance effect the low-fidelity model lacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pwl_repro.core.features import (
    FeatureSpec,
    PWLFeatureLibrary,
    register_feature_spec,
)

Array = NDArray[np.float64]

# [ENGINEERING] Scenario constants from the v1.3 dataset specification.
L = 0.1  # Rod length (m).
T0 = 293.15  # Bottom boundary temperature (K).
R0 = 3e-2  # Contact-resistance reference at 1 MPa (m^2 K/W); high-fidelity only.
K_NOMINAL = 25.0  # Low-fidelity constant conductivity (W/(m K)).
BETA_K = 2e-3  # Temperature coefficient of the high-fidelity k(T).
K_TRUE_RANGE = (15.0, 35.0)  # Batch-level true conductivity range.
THETA_LOWER = (1.0 / 35.0,)  # theta = 1/k box bounds.
THETA_UPPER = (1.0 / 15.0,)
THETA_NOMINAL = np.array([1.0 / K_NOMINAL])

INPUT_COLUMNS = ("Q", "h", "T_inf")
PROCESS_COLUMNS = ("P",)

HEAT_H_NAMES: tuple[str, ...] = (
    "1",
    "Q",
    "Q^2",
    "1/h",
    "Q/h",
    "T_inf",
    "T_inf/h",
    "theta",
    "theta*Q",
    "theta*Q^2",
    "theta*Q/h",
)
HEAT_B_NAMES: tuple[str, ...] = (
    "1",
    "P",
    "1/P",
    "P^-1/2",
    "Q/P",
    "Q*P^-1/2",
    "Q",
    "Q^2",
)
HEAT_GROUPS: tuple[Array, ...] = (
    np.arange(0, 3),
    np.arange(3, 5),
    np.arange(5, 7),
    np.arange(7, 11),
)
HEAT_GROUP_NAMES: tuple[str, ...] = ("Q", "h", "T_inf", "theta")


def low_fidelity_temperature(x_ph: Array, theta: Array) -> Array:
    """Closed-form low-fidelity model (R_c = 0) evaluated at ``k = 1/theta``.

    [ENGINEERING] One-dimensional simplification reference §3.4.  With a
    perfect interface the two interface side temperatures coincide, so this
    also equals the dataset's two-sided average output definition.
    """

    x_ph = np.asarray(x_ph, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    k = 1.0 / theta[0]
    q = x_ph[:, 0]
    h = x_ph[:, 1]
    t_inf = x_ph[:, 2]
    r_lower = L / (2.0 * k)
    r_conv = 1.0 / h
    a = (q * L * (r_lower + r_conv) + (t_inf - T0)) / (L + k * r_conv)
    return T0 - q * L**2 / (8.0 * k) + a * L / 2.0


def heat_h_raw(x_ph: Array, theta: Array) -> Array:
    """[ENGINEERING] 11-column affine H library (migration analysis §6.2)."""

    x_ph = np.asarray(x_ph, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    q, h, t_inf = x_ph.T
    t = theta[0]
    return np.column_stack(
        (
            np.ones_like(q),
            q,
            q**2,
            1.0 / h,
            q / h,
            t_inf,
            t_inf / h,
            t * np.ones_like(q),
            t * q,
            t * q**2,
            t * q / h,
        )
    )


def heat_b_raw(x_ph: Array, x_pr: Array) -> Array:
    """[ENGINEERING] 8-column discrepancy library (migration analysis §6.3).

    P >= 0.5 MPa throughout the dataset, so the negative powers below have no
    singularity; they cover the DeltaT = q * R_c(P) ~ Q * P^(-0.7) power-law
    neighborhood of the contact-resistance mechanism.
    """

    q = np.asarray(x_ph, dtype=float)[:, 0]
    p = np.asarray(x_pr, dtype=float)[:, 0]
    return np.column_stack(
        (
            np.ones_like(q),
            p,
            1.0 / p,
            p**-0.5,
            q / p,
            q * p**-0.5,
            q,
            q**2,
        )
    )


def heat_feature_spec() -> FeatureSpec:
    """[ENGINEERING] Heat-scenario library: x_ph=3, x_pr=1, theta=1, 4 groups."""

    return FeatureSpec(
        name="heat",
        x_ph_dim=3,
        x_pr_dim=1,
        theta_dim=1,
        h_names=HEAT_H_NAMES,
        b_names=HEAT_B_NAMES,
        groups=HEAT_GROUPS,
        group_names=HEAT_GROUP_NAMES,
        raw_h=heat_h_raw,
        raw_b=heat_b_raw,
    )


HEAT_B_NAMES_NO_QQ: tuple[str, ...] = HEAT_B_NAMES[:6]


def heat_b_raw_no_qq(x_ph: Array, x_pr: Array) -> Array:
    """[ENGINEERING] 6-column B without the Q/Q^2 columns (W1 variant v2).

    The first version's theta estimates collapsed to the box boundary because
    B's Q/Q^2 columns span the same space as H's theta group
    (theta*[1, Q, Q^2, Q/h]), letting d absorb the k-calibration error.  The
    remaining columns all carry P and do not overlap H's theta group; column
    order matches HEAT_B_NAMES[:6] so downstream name indexing stays valid.
    """

    q = np.asarray(x_ph, dtype=float)[:, 0]
    p = np.asarray(x_pr, dtype=float)[:, 0]
    return np.column_stack(
        (
            np.ones_like(q),
            p,
            1.0 / p,
            p**-0.5,
            q / p,
            q * p**-0.5,
        )
    )


HEAT_B_NAMES_V3_QINT: tuple[str, ...] = HEAT_B_NAMES_NO_QQ + (
    "q_int*P^-0.7",
    "q_int/(1+hL/k)*P^-0.7",
)

HEAT_B_NAMES_INTERCEPT: tuple[str, ...] = ("1",)


def heat_b_raw_intercept(x_ph: Array, x_pr: Array) -> Array:
    """[ENGINEERING] Intercept-only B (M3 ablation A1, "no B" arm).

    A constant column keeps the Hg + Bd model structure (and the d_ridge
    term) intact while removing every discrepancy-learning direction; the
    d intercept can only absorb a global offset.  Depends on no input, so
    it cannot leak label or high-fidelity information.
    """

    n = len(np.asarray(x_ph, dtype=float))
    return np.ones((n, 1))


def low_fidelity_interface_flux(x_ph: Array, k: float = K_NOMINAL) -> Array:
    """[ENGINEERING] Interface heat flux ``q_int = QL/2 - k*a`` of the
    low-fidelity model (R_c = 0), evaluated pointwise from the closed form.

    ``a`` is the linear profile coefficient of the closed-form solution
    (same expression as in :func:`low_fidelity_temperature`).  Free of any
    high-fidelity solve; used by the v3 B columns (attribution analysis v2
    §8.1).
    """

    x_ph = np.asarray(x_ph, dtype=float)
    q = x_ph[:, 0]
    h = x_ph[:, 1]
    t_inf = x_ph[:, 2]
    r_lower = L / (2.0 * k)
    r_conv = 1.0 / h
    a = (q * L * (r_lower + r_conv) + (t_inf - T0)) / (L + k * r_conv)
    return q * L / 2.0 - k * a


def heat_b_raw_v3_qint(x_ph: Array, x_pr: Array) -> Array:
    """[ENGINEERING] 8-column B: the v2 (no_qq) columns plus two
    mechanism-derived product columns (attribution analysis v2 §8.1).

    ``col1 = q_int * P^(-0.7)`` carries the contact-resistance product
    direction; ``col2 = q_int / (1 + hL/k_nominal) * P^(-0.7)`` carries the
    first-order structure of the top-BC flux-feedback pole.  Both are
    computable from the low-fidelity closed form at k_nominal and depend
    only on (Q, h, T_inf, P).  Verified to cut the noiseless class bias
    from ~7.2 to ~0.44 K^2 (verification report §3).
    """

    base = heat_b_raw_no_qq(x_ph, x_pr)
    x_ph = np.asarray(x_ph, dtype=float)
    p = np.asarray(x_pr, dtype=float)[:, 0]
    q_int = low_fidelity_interface_flux(x_ph, K_NOMINAL)
    s = p**-0.7
    modulation = 1.0 + x_ph[:, 1] * L / K_NOMINAL
    return np.column_stack((base, q_int * s, q_int / modulation * s))


HEAT_B_NAMES_G0_GENERIC: tuple[str, ...] = (
    "h",
    "P",
    "h^2",
    "T_inf^2",
    "P^2",
    "Q*h",
    "Q*T_inf",
    "Q*P",
    "h*T_inf",
    "h*P",
    "T_inf*P",
)


def heat_b_raw_g0_generic(x_ph: Array, x_pr: Array) -> Array:
    """[ENGINEERING] G0 (blind S2 tier): scenario-agnostic quadratic B.

    Full degree-2 polynomial over the four inputs minus the raw columns
    already spanned by H (``1, Q, Q^2, T_inf``), per the framework's own
    constraint-C hygiene (references/docs/08).  No negative powers, no
    q_int, no contact-resistance structure: this is the "mechanism-unknown"
    dictionary.  Any skill here is the PWL framework's, not the designer's.
    """

    x_ph = np.asarray(x_ph, dtype=float)
    q, h, t_inf = x_ph.T
    p = np.asarray(x_pr, dtype=float)[:, 0]
    return np.column_stack(
        (
            h,
            p,
            h**2,
            t_inf**2,
            p**2,
            q * h,
            q * t_inf,
            q * p,
            h * t_inf,
            h * p,
            t_inf * p,
        )
    )


# G1 exponent grid: contact-resistance literature motivates a negative power
# law in P, but the grid deliberately EXCLUDES the true generator exponent
# (0.7); the sparse selector must interpolate from coarse neighbours.
G1_EXPONENT_GRID: tuple[float, ...] = (0.3, 0.5, 1.0)

HEAT_B_NAMES_G1_PRIOR: tuple[str, ...] = HEAT_B_NAMES_NO_QQ + tuple(
    f"{base}*P^-{gamma}"
    for base in ("q_int", "q_int/(1+hL/k)")
    for gamma in G1_EXPONENT_GRID
)


def heat_b_raw_g1_prior(x_ph: Array, x_pr: Array) -> Array:
    """[ENGINEERING] G1 (blind S2 tier): engineering-prior B.

    The no_qq engineering columns plus ``q_int`` (and its top-BC modulated
    variant) times a coarse negative-power grid {0.3, 0.5, 1.0}.  Prices
    "knowing the law family but not the exponent": the oracle exponent 0.7
    is not in the grid.
    """

    base = heat_b_raw_no_qq(x_ph, x_pr)
    x_ph = np.asarray(x_ph, dtype=float)
    p = np.asarray(x_pr, dtype=float)[:, 0]
    q_int = low_fidelity_interface_flux(x_ph, K_NOMINAL)
    modulation = 1.0 + x_ph[:, 1] * L / K_NOMINAL
    extra = [
        column * p**-gamma
        for column in (q_int, q_int / modulation)
        for gamma in G1_EXPONENT_GRID
    ]
    return np.column_stack((base, *extra))


def get_heat_feature_spec(name: str = "heat") -> FeatureSpec:
    """Resolve heat-scenario spec variants (W1 orthogonalization experiment)."""

    if name == "heat":
        return heat_feature_spec()
    if name == "heat_residualized":
        # W1 variant v1: same columns, B projected onto the orthogonal
        # complement of the H shape space inside PWLFeatureLibrary.
        return replace(
            heat_feature_spec(), name="heat_residualized", orthogonalize_b=True
        )
    if name == "heat_no_qq":
        return FeatureSpec(
            name="heat_no_qq",
            x_ph_dim=3,
            x_pr_dim=1,
            theta_dim=1,
            h_names=HEAT_H_NAMES,
            b_names=HEAT_B_NAMES_NO_QQ,
            groups=HEAT_GROUPS,
            group_names=HEAT_GROUP_NAMES,
            raw_h=heat_h_raw,
            raw_b=heat_b_raw_no_qq,
        )
    if name == "heat_v3_qint":
        return FeatureSpec(
            name="heat_v3_qint",
            x_ph_dim=3,
            x_pr_dim=1,
            theta_dim=1,
            h_names=HEAT_H_NAMES,
            b_names=HEAT_B_NAMES_V3_QINT,
            groups=HEAT_GROUPS,
            group_names=HEAT_GROUP_NAMES,
            raw_h=heat_h_raw,
            raw_b=heat_b_raw_v3_qint,
        )
    if name == "heat_no_b":
        return FeatureSpec(
            name="heat_no_b",
            x_ph_dim=3,
            x_pr_dim=1,
            theta_dim=1,
            h_names=HEAT_H_NAMES,
            b_names=HEAT_B_NAMES_INTERCEPT,
            groups=HEAT_GROUPS,
            group_names=HEAT_GROUP_NAMES,
            raw_h=heat_h_raw,
            raw_b=heat_b_raw_intercept,
        )
    if name == "heat_g0_generic":
        return FeatureSpec(
            name="heat_g0_generic",
            x_ph_dim=3,
            x_pr_dim=1,
            theta_dim=1,
            h_names=HEAT_H_NAMES,
            b_names=HEAT_B_NAMES_G0_GENERIC,
            groups=HEAT_GROUPS,
            group_names=HEAT_GROUP_NAMES,
            raw_h=heat_h_raw,
            raw_b=heat_b_raw_g0_generic,
        )
    if name == "heat_g1_prior":
        return FeatureSpec(
            name="heat_g1_prior",
            x_ph_dim=3,
            x_pr_dim=1,
            theta_dim=1,
            h_names=HEAT_H_NAMES,
            b_names=HEAT_B_NAMES_G1_PRIOR,
            groups=HEAT_GROUPS,
            group_names=HEAT_GROUP_NAMES,
            raw_h=heat_h_raw,
            raw_b=heat_b_raw_g1_prior,
        )
    raise ValueError(
        f"Unknown heat feature_spec {name!r}. "
        "Available: 'heat', 'heat_residualized', 'heat_no_qq', 'heat_v3_qint',"
        " 'heat_no_b', 'heat_g0_generic', 'heat_g1_prior'."
    )


def _heat_spec_factory(name: str):
    """Build a registry factory while retaining the configured spec name."""

    return lambda _b_profile: get_heat_feature_spec(name)


for _feature_spec_name in (
    "heat",
    "heat_residualized",
    "heat_no_qq",
    "heat_v3_qint",
    "heat_no_b",
    "heat_g0_generic",
    "heat_g1_prior",
):
    register_feature_spec(
        _feature_spec_name,
        _heat_spec_factory(_feature_spec_name),
    )


@dataclass(frozen=True)
class HeatScenarioData:
    """[ENGINEERING] One dataset batch behind the ScenarioData interface.

    Weak labels are deterministic low-fidelity predictions, so the physics
    noise fields are zero; they exist to mirror ``SimulationData`` and keep
    the shared experiment machinery auditable.
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
    batch_id: str
    k_true: float

    def physics_model(self, x_ph: Array, theta: Array) -> Array:
        """Deterministic low-fidelity model at ``k = 1/theta``."""

        return low_fidelity_temperature(x_ph, theta)

    def labeled_physics_output(self, *, include_noise: bool = True) -> Array:
        """Low-fidelity output at the nominal k paired with the labels."""

        output = self.physics_model(self.x_ph, THETA_NOMINAL)
        if include_noise:
            output = output + self.labeled_physics_noise
        return output

    def labeled_physics_correlation(self, *, include_noise: bool = True) -> float:
        """Pearson correlation between labels and low-fidelity output."""

        return float(
            np.corrcoef(
                self.y,
                self.labeled_physics_output(include_noise=include_noise),
            )[0, 1]
        )


def _read_set_csv(path: Path, expected_y: str) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#")
    expected = list(INPUT_COLUMNS) + list(PROCESS_COLUMNS) + [expected_y]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} misses columns {missing}.")
    return frame


def load_heat_batch(
    datasets_root: str | Path,
    batch_index: int,
) -> HeatScenarioData:
    """Load one batch (reference/engineering/acceptance) as scenario data."""

    batch_dir = Path(datasets_root) / f"batch_{batch_index:02d}"
    metadata = json.loads(
        (batch_dir / "metadata.json").read_text(encoding="utf-8")
    )
    k_true = float(metadata["k_true"])
    if not K_TRUE_RANGE[0] <= k_true <= K_TRUE_RANGE[1]:
        raise ValueError(f"k_true {k_true} outside {K_TRUE_RANGE}.")
    sigma = float(metadata["sigma"])

    reference = _read_set_csv(batch_dir / "reference_measurements.csv", "y_obs")
    engineering = _read_set_csv(
        batch_dir / "engineering_predictions.csv", "y_pred"
    )
    acceptance = _read_set_csv(batch_dir / "acceptance_tests.csv", "y_obs")

    def split(frame: pd.DataFrame, y_column: str) -> tuple[Array, Array, Array]:
        x_ph = frame[list(INPUT_COLUMNS)].to_numpy(dtype=float)
        x_pr = frame[list(PROCESS_COLUMNS)].to_numpy(dtype=float)
        y = frame[y_column].to_numpy(dtype=float)
        return x_ph, x_pr, y

    x_ph, x_pr, y = split(reference, "y_obs")
    weak_x_ph, _, weak_y = split(engineering, "y_pred")
    test_x_ph, test_x_pr, test_y = split(acceptance, "y_obs")
    arrays = (x_ph, x_pr, y, weak_x_ph, weak_y, test_x_ph, test_x_pr, test_y)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"Batch {batch_index:02d} contains non-finite values.")

    sha256 = metadata.get("sha256", {})
    diagnostics: dict[str, float | int | str] = {
        "batch_id": f"batch_{batch_index:02d}",
        "k_true": k_true,
        "sigma": sigma,
        "R0": float(metadata.get("R0", R0)),
        "beta_k": float(metadata.get("beta_k", BETA_K)),
        "k_nominal": float(metadata.get("k_nominal", K_NOMINAL)),
        "sha256_reference": str(sha256.get("reference_measurements.csv", "")),
        "sha256_acceptance": str(sha256.get("acceptance_tests.csv", "")),
        "sha256_engineering": str(sha256.get("engineering_predictions.csv", "")),
    }
    return HeatScenarioData(
        x_ph=x_ph,
        x_pr=x_pr,
        y=y,
        weak_x_ph=weak_x_ph,
        weak_y=weak_y,
        test_x_ph=test_x_ph,
        test_x_pr=test_x_pr,
        test_y=test_y,
        theta_true=np.array([1.0 / k_true]),
        noise_sigma=sigma,
        physics_noise_sigma=0.0,
        labeled_physics_noise=np.zeros(len(y)),
        weak_physics_noise=np.zeros(len(weak_y)),
        test_physics_noise=np.zeros(len(test_y)),
        discrepancy_scale=1.0,
        covariance=np.empty((0, 0)),
        sampling_diagnostics=diagnostics,
        batch_id=f"batch_{batch_index:02d}",
        k_true=k_true,
    )


def subsample_weak_labels(
    data: HeatScenarioData, n_weak: int, *, seed: int
) -> HeatScenarioData:
    """[ENGINEERING] Deterministic nested weak-label subsample (M3 ablation A4).

    One permutation per seed fixes the nesting n_weak=50 ⊂ 100 ⊂ 200, so the
    weak-label-count curve compares nested subsets rather than independent
    draws.  n_weak equal to the pool size returns the data unchanged.
    """

    total = len(data.weak_y)
    if n_weak > total:
        raise ValueError(f"n_weak {n_weak} exceeds the weak pool ({total}).")
    if n_weak == total:
        return data
    index = np.random.default_rng(seed).permutation(total)[:n_weak]
    return replace(
        data,
        weak_x_ph=data.weak_x_ph[index],
        weak_y=data.weak_y[index],
        weak_physics_noise=data.weak_physics_noise[index],
    )


def _r_squared(target: Array, fitted: Array) -> float:
    residual = float(np.sum((target - fitted) ** 2))
    total = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - residual / max(total, 1e-12)


def diagnose_heat_basis(data: HeatScenarioData) -> dict[str, float]:
    """[ENGINEERING] Stage-4 gate: least-squares R² of H (and H+B) on labels.

    Mirrors ``scripts/diagnose_basis.py``: H alone must explain the main
    output trend before full PWL tuning is worthwhile.
    """

    library = PWLFeatureLibrary(standardize=True, spec=heat_feature_spec())
    library.fit(
        np.vstack((data.x_ph, data.weak_x_ph)),
        data.x_ph,
        data.x_pr,
        data.theta_true,
    )
    h_labeled = library.transform_h(data.x_ph, data.theta_true)
    g_hat = np.linalg.lstsq(h_labeled, data.y, rcond=None)[0]
    h_weak = library.transform_h(data.weak_x_ph, THETA_NOMINAL)
    g_weak = np.linalg.lstsq(h_weak, data.weak_y, rcond=None)[0]
    b_labeled = library.transform_b(data.x_ph, data.x_pr)
    joint = np.column_stack((h_labeled, b_labeled))
    joint_hat = np.linalg.lstsq(joint, data.y, rcond=None)[0]
    return {
        "h_labeled_r2": _r_squared(data.y, h_labeled @ g_hat),
        "h_weak_r2": _r_squared(data.weak_y, h_weak @ g_weak),
        "hb_labeled_r2": _r_squared(data.y, joint @ joint_hat),
    }
