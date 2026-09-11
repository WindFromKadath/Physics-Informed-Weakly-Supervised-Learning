"""Heat-conduction scenario tests: physics, adapter, library, end-to-end."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_bvp

from pwl_migration.heat import (
    HEAT_H_NAMES,
    K_NOMINAL,
    L,
    T0,
    THETA_NOMINAL,
    load_heat_batch,
    low_fidelity_interface_flux,
    low_fidelity_temperature,
    subsample_weak_labels,
)
from pwl_migration.heat_experiments import _heat_anchor_checks
from pwl_repro.experiment_api import ExperimentArtifacts
from pwl_repro.features import PWLFeatureLibrary, get_feature_spec
from pwl_repro.model import PWLRegressor

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "migration" / "datasets"


def _x_ph(q: float, h: float, t_inf: float) -> np.ndarray:
    return np.array([[q, h, t_inf]], dtype=float)


def test_closed_form_zero_source_limit():
    # With Q = 0 the profile is linear; the interface value follows directly
    # from the bottom clamp and the top convection balance.
    h, t_inf, k = 12.0, 310.0, 20.0
    y = low_fidelity_temperature(_x_ph(0.0, h, t_inf), [1.0 / k])[0]
    expected = T0 + (t_inf - T0) * (L / 2.0) / (L + k / h)
    assert abs(y - expected) < 1e-10


def test_closed_form_matches_bvp_solution():
    # Independent check: solve k*T'' + Q = 0 with the same boundary
    # conditions via scipy instead of re-deriving the closed form.
    q, h, t_inf, k = 5e4, 12.0, 300.0, 22.0

    def ode(z, state):
        return np.vstack((state[1], -q / k * np.ones_like(z)))

    def bc(ya, yb):
        return np.array([ya[0] - T0, -k * yb[1] - h * (yb[0] - t_inf)])

    z = np.linspace(0.0, L, 200)
    solution = solve_bvp(ode, bc, z, np.zeros((2, z.size)), tol=1e-10)
    y_bvp = float(solution.sol(L / 2.0)[0])
    y_closed = low_fidelity_temperature(_x_ph(q, h, t_inf), [1.0 / k])[0]
    assert abs(y_bvp - y_closed) < 1e-6


def test_closed_form_monotonicity():
    h, t_inf = 12.0, 300.0
    y_q_low = low_fidelity_temperature(_x_ph(1e4, h, t_inf), [1.0 / 22.0])[0]
    y_q_high = low_fidelity_temperature(_x_ph(9e4, h, t_inf), [1.0 / 22.0])[0]
    assert y_q_high > y_q_low
    y_k30 = low_fidelity_temperature(_x_ph(5e4, h, t_inf), [1.0 / 30.0])[0]
    y_k18 = low_fidelity_temperature(_x_ph(5e4, h, t_inf), [1.0 / 18.0])[0]
    assert y_k30 < y_k18  # Higher conductivity carries heat to the clamp.
    y_tinf_low = low_fidelity_temperature(_x_ph(5e4, h, 280.0), [1.0 / 22.0])[0]
    y_tinf_high = low_fidelity_temperature(_x_ph(5e4, h, 320.0), [1.0 / 22.0])[0]
    assert y_tinf_high > y_tinf_low


@pytest.mark.parametrize("batch", range(1, 11))
def test_weak_labels_match_closed_form(batch: int):
    # Mutual validation of CSV semantics and the closed-form implementation:
    # engineering predictions must equal the low-fidelity model at k_nominal.
    data = load_heat_batch(DATASETS_ROOT, batch)
    prediction = data.physics_model(data.weak_x_ph, THETA_NOMINAL)
    assert np.max(np.abs(prediction - data.weak_y)) < 1e-6


def test_load_batch_contract():
    data = load_heat_batch(DATASETS_ROOT, 1)
    assert data.x_ph.shape == (120, 3)
    assert data.x_pr.shape == (120, 1)
    assert data.y.shape == (120,)
    assert data.weak_x_ph.shape == (200, 3)
    assert data.weak_y.shape == (200,)
    assert data.test_x_ph.shape == (200, 3)
    assert data.test_x_pr.shape == (200, 1)
    assert data.theta_true.shape == (1,)
    assert abs(data.theta_true[0] - 1.0 / data.k_true) < 1e-15
    assert data.physics_noise_sigma == 0.0
    arrays = (
        data.x_ph, data.x_pr, data.y,
        data.weak_x_ph, data.weak_y,
        data.test_x_ph, data.test_x_pr, data.test_y,
    )
    for array in arrays:
        assert np.all(np.isfinite(array))
    correlation = data.labeled_physics_correlation()
    assert 0.6 < correlation < 0.95


def test_heat_spec_dims_and_groups():
    spec = get_feature_spec("heat")
    assert (spec.x_ph_dim, spec.x_pr_dim, spec.theta_dim) == (3, 1, 1)
    assert len(spec.h_names) == 11
    assert len(spec.b_names) == 8
    assert len(spec.groups) == 4
    covered = np.concatenate(spec.groups)
    assert sorted(covered.tolist()) == list(range(len(spec.h_names)))


def test_simulation_spec_unchanged():
    compact = get_feature_spec("simulation", "compact")
    assert (compact.x_ph_dim, compact.x_pr_dim, compact.theta_dim) == (3, 2, 2)
    assert len(compact.h_names) == 25
    assert len(compact.b_names) == 3
    expanded = get_feature_spec("simulation", "expanded")
    assert len(expanded.b_names) == 12
    assert len(expanded.groups) == 5


def test_heat_h_is_affine_in_theta():
    data = load_heat_batch(DATASETS_ROOT, 1)
    library = PWLFeatureLibrary(
        standardize=True, spec=get_feature_spec("heat")
    ).fit(
        np.vstack((data.x_ph, data.weak_x_ph)),
        data.x_ph,
        data.x_pr,
        data.theta_true,
    )
    rng = np.random.default_rng(0)
    g = rng.normal(size=len(HEAT_H_NAMES))
    theta = np.array([0.05])
    base, design = library.affine_prediction_parts(data.x_ph, g)
    discrepancy = library.transform_h(data.x_ph, theta) @ g - (
        base + design @ theta
    )
    assert np.max(np.abs(discrepancy)) < 1e-8


def test_theta_bounds_must_match_spec_dim():
    with pytest.raises(ValueError, match="theta bounds"):
        PWLRegressor(feature_spec="heat")


def test_residualized_b_is_orthogonal_to_h():
    # W1 variant v1: after Plumlee residualization on the fit rows, the
    # standardized B block must be orthogonal to H(theta) for any theta,
    # because the H shape space is theta-invariant and H has an intercept.
    data = load_heat_batch(DATASETS_ROOT, 1)
    library = PWLFeatureLibrary(
        standardize=True, spec=get_feature_spec("heat_residualized")
    ).fit(
        np.vstack((data.x_ph, data.weak_x_ph)),
        data.x_ph,
        data.x_pr,
        data.theta_true,
    )
    b_fit = library.transform_b(data.x_ph, data.x_pr)
    for theta in (0.03, 0.04, 0.05, 1.0 / 15.0):
        h_fit = library.transform_h(data.x_ph, np.array([theta]))
        cross = b_fit.T @ h_fit
        assert np.max(np.abs(cross)) < 1e-8


def test_v3_qint_spec_variant():
    # W3: no_qq columns plus two low-fidelity-free q_int product columns.
    spec = get_feature_spec("heat_v3_qint")
    assert (spec.x_ph_dim, spec.x_pr_dim, spec.theta_dim) == (3, 1, 1)
    assert spec.b_names == (
        "1",
        "P",
        "1/P",
        "P^-1/2",
        "Q/P",
        "Q*P^-1/2",
        "q_int*P^-0.7",
        "q_int/(1+hL/k)*P^-0.7",
    )
    assert not spec.orthogonalize_b
    x_ph = np.array([[5e4, 12.0, 300.0], [8e4, 30.0, 290.0]])
    x_pr = np.array([[2.0], [7.5]])
    raw = spec.raw_b(x_ph, x_pr)
    assert raw.shape == (2, 8)
    # Column values against an independent q_int evaluation: q_int = QL/2 - k*a
    # with a from the top-BC balance of the R_c = 0 closed form.
    q, h, t_inf = x_ph.T
    k = K_NOMINAL
    a = (q * L * (L / (2.0 * k) + 1.0 / h) + (t_inf - T0)) / (L + k / h)
    q_int = q * L / 2.0 - k * a
    s = x_pr[:, 0] ** -0.7
    np.testing.assert_allclose(raw[:, 6], q_int * s, rtol=1e-12)
    np.testing.assert_allclose(raw[:, 7], q_int / (1.0 + h * L / k) * s, rtol=1e-12)
    # The helper must agree with the same closed-form coefficient.
    np.testing.assert_allclose(
        low_fidelity_interface_flux(x_ph, k), q_int, rtol=1e-12
    )
    assert np.all(np.isfinite(raw))


def test_end_to_end_heat_v3_qint_fit():
    data = load_heat_batch(DATASETS_ROOT, 1)
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat_v3_qint",
        d_ridge=10.0,
        random_state=0,
    ).fit(
        data.x_ph[:21],
        data.x_pr[:21],
        data.y[:21],
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert np.all(np.isfinite(prediction))
    assert len(model.d_) == 8


def test_no_qq_spec_variant():
    spec = get_feature_spec("heat_no_qq")
    assert (spec.x_ph_dim, spec.x_pr_dim, spec.theta_dim) == (3, 1, 1)
    assert spec.b_names == ("1", "P", "1/P", "P^-1/2", "Q/P", "Q*P^-1/2")
    assert not spec.orthogonalize_b
    x_ph = np.array([[5e4, 12.0, 300.0]])
    x_pr = np.array([[2.0]])
    assert spec.raw_b(x_ph, x_pr).shape == (1, 6)
    residualized = get_feature_spec("heat_residualized")
    assert residualized.orthogonalize_b
    assert residualized.b_names == spec.b_names[:6] + ("Q", "Q^2")
    with pytest.raises(ValueError, match="Unknown heat feature_spec"):
        get_feature_spec("heat_bogus")


def test_end_to_end_heat_fit():
    data = load_heat_batch(DATASETS_ROOT, 1)
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat",
        d_ridge=10.0,
        random_state=0,
    ).fit(
        data.x_ph[:21],
        data.x_pr[:21],
        data.y[:21],
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert np.all(np.isfinite(prediction))
    assert 1.0 / 35.0 - 1e-12 <= model.theta_[0] <= 1.0 / 15.0 + 1e-12
    assert model.history_


def test_freeze_theta_keeps_initial_value():
    # W2: with freeze_theta the block update is skipped entirely, so theta
    # must stay exactly at theta_init while g/d still train.
    data = load_heat_batch(DATASETS_ROOT, 1)
    frozen_value = np.array([0.04])
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat_no_qq",
        d_ridge=10.0,
        freeze_theta=True,
        random_state=0,
    ).fit(
        data.x_ph[:21],
        data.x_pr[:21],
        data.y[:21],
        data.weak_x_ph,
        data.weak_y,
        theta_init=frozen_value,
        physics_model=data.physics_model,
    )
    assert model.theta_[0] == pytest.approx(0.04, abs=1e-15)
    assert all(
        record.theta[0] == pytest.approx(0.04, abs=1e-15)
        for record in model.history_
    )
    assert all(record.theta_admm_iterations == 0 for record in model.history_)
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert np.all(np.isfinite(prediction))
    assert np.linalg.norm(model.g_) > 0.0  # g/d still trained.


def test_refresh_weak_labels_tracks_theta():
    # M1 (Case-B authentic mode): weak labels follow the current theta, so
    # the drift between successive refreshes must shrink as theta settles.
    data = load_heat_batch(DATASETS_ROOT, 1)
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat_no_qq",
        d_ridge=10.0,
        refresh_weak_labels=True,
        random_state=0,
    ).fit(
        data.x_ph[:21],
        data.x_pr[:21],
        data.y[:21],
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    drift = model.weak_refresh_drift_
    assert len(drift) == model.n_iter_
    assert drift[-1] <= drift[0] + 1e-12
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert np.all(np.isfinite(prediction))


def test_refresh_weak_labels_requires_physics_model():
    data = load_heat_batch(DATASETS_ROOT, 1)
    model = PWLRegressor(
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat_no_qq",
        refresh_weak_labels=True,
    )
    with pytest.raises(ValueError, match="physics_model"):
        model.fit(
            data.x_ph[:21],
            data.x_pr[:21],
            data.y[:21],
            data.weak_x_ph,
            data.weak_y,
        )


def test_refresh_and_freeze_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PWLRegressor(
            theta_lower=(1.0 / 35.0,),
            theta_upper=(1.0 / 15.0,),
            feature_spec="heat_no_qq",
            freeze_theta=True,
            refresh_weak_labels=True,
        )


def _synthetic_anchor_artifacts() -> ExperimentArtifacts:
    """Minimal two-repeat, two-size metrics frame for anchor-check tests."""

    details = json.dumps(
        {
            "theta_estimate": [0.0667],
            "theta_boundary_hit": True,
            "d_coefficients": [0.1] * 8,
        }
    )
    rows = []
    for repeat in (0, 1):
        for n_labeled, pwl_rmse in ((10, 8.0), (120, 5.4)):
            rows.append(
                {
                    "experiment": "sample_size",
                    "model": "PWL",
                    "repeat": repeat,
                    "n_labeled": n_labeled,
                    "rmse": pwl_rmse,
                    "mse": pwl_rmse**2,
                    "physics_correlation": 0.8,
                    "theta_true": "[0.05]",
                    "details": details,
                }
            )
            for model, rmse in (
                ("Physics", 5.5),
                ("PhysicsDirect", 15.7),
                ("GP", 5.7 if n_labeled == 120 else 10.9),
                ("Ridge", 6.4 if n_labeled == 120 else 10.1),
            ):
                rows.append(
                    {
                        "experiment": "sample_size",
                        "model": model,
                        "repeat": repeat,
                        "n_labeled": n_labeled,
                        "rmse": rmse,
                        "mse": rmse**2,
                        "physics_correlation": 0.8,
                        "theta_true": "[0.05]",
                        "details": details,
                    }
                )
    metrics = pd.DataFrame(rows)
    conditions = pd.DataFrame(
        {"condition_type": ["dataset"], "noise_sigma": [4.494]}
    )
    return ExperimentArtifacts(
        metrics=metrics,
        predictions=pd.DataFrame(),
        conditions=conditions,
        statistical_tests=pd.DataFrame(),
    )


def test_anchor_checks_theta_m2_revision():
    # M2: theta criteria are info-level nuisance-parameter indicators,
    # never pass/fail gates on "closer to 1/k_true".
    artifacts = _synthetic_anchor_artifacts()
    config = {
        "heat": {"datasets_root": str(DATASETS_ROOT), "batches": [1]},
        "model": {"feature_spec": "heat_v3_qint"},
    }
    checks = _heat_anchor_checks(artifacts, config).set_index("check")
    assert checks.loc["heat_theta_learning_fraction", "status"] == "info"
    assert checks.loc["heat_theta_boundary_hit_rate", "status"] == "info"
    assert checks.loc["heat_theta_batch_spread", "status"] == "info"
    assert "not a success standard" in checks.loc[
        "heat_theta_learning_fraction", "threshold"
    ]
    sensitivity = checks.loc["heat_theta_prediction_sensitivity"]
    assert sensitivity["status"] == "info"
    low, high = (float(v) for v in sensitivity["value"].strip("[]").split(", "))
    assert 1.0 < low <= high < 3.0


def test_no_b_spec_is_constant_and_leak_free():
    # M3 ablation A1: the intercept-only B carries no input dependence at
    # all, so it cannot leak label or high-fidelity information.
    spec = get_feature_spec("heat_no_b")
    assert spec.b_names == ("1",)
    x_ph = np.array([[1e4, 3.0, 273.0], [1e5, 50.0, 323.0]])
    x_pr = np.array([[0.5], [10.0]])
    b = spec.raw_b(x_ph, x_pr)
    assert b.shape == (2, 1)
    assert np.all(b == 1.0)
    # H block stays the 11-column affine library.
    assert len(spec.h_names) == 11
    theta = np.array([0.05])
    h = spec.raw_h(x_ph, theta)
    g = np.arange(1.0, 12.0)
    base, design = np.zeros(2), np.zeros((2, 1))
    base = spec.raw_h(x_ph, np.zeros(1)) @ g
    design = (spec.raw_h(x_ph, np.ones(1)) - spec.raw_h(x_ph, np.zeros(1))) @ g
    assert np.allclose(h @ g, base + design * theta[0])


def test_no_b_end_to_end_fit():
    # M3 ablation A1: the intercept-only B trains end to end and the
    # discrepancy term reduces to a global offset.
    data = load_heat_batch(DATASETS_ROOT, 1)
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat_no_b",
        d_ridge=10.0,
        refresh_weak_labels=True,
        random_state=0,
    ).fit(
        data.x_ph[:21],
        data.x_pr[:21],
        data.y[:21],
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    assert model.d_.shape == (1,)
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert np.all(np.isfinite(prediction))


def test_subsample_weak_labels_is_nested_and_deterministic():
    # M3 ablation A4: subsamples must be nested (50 ⊂ 100 ⊂ 200) and
    # reproducible under the same seed.
    data = load_heat_batch(DATASETS_ROOT, 1)
    s50 = subsample_weak_labels(data, 50, seed=2026)
    s100 = subsample_weak_labels(data, 100, seed=2026)
    s50_again = subsample_weak_labels(data, 50, seed=2026)
    assert len(s50.weak_y) == 50
    assert len(s100.weak_y) == 100
    assert np.array_equal(s50.weak_y, s50_again.weak_y)
    assert np.array_equal(s50.weak_y, s100.weak_y[:50])
    assert s50.weak_x_ph.shape == (50, 3)
    # Full pool returns the data unchanged.
    assert subsample_weak_labels(data, 200, seed=2026) is data
    with pytest.raises(ValueError, match="exceeds the weak pool"):
        subsample_weak_labels(data, 201, seed=2026)


def test_mechanism_aligned_condition_matches_ols():
    # Alignment arm A4: the scenario baseline must reduce to a plain OLS on
    # the train fold with the temperature-rise affine + two q_int columns.
    from pwl_migration.heat_experiments import _mechanism_aligned_condition

    data = load_heat_batch(DATASETS_ROOT, 1)
    train = np.arange(21)
    validation = np.arange(21, 30)
    row, preds = _mechanism_aligned_condition(
        data,
        train,
        validation,
        experiment="sample_size",
        repeat=0,
        seed=2026,
        n_labeled=30,
        extra={"k_true": data.k_true},
    )

    def design(x_ph, x_pr):
        y_lf = data.physics_model(x_ph, THETA_NOMINAL)
        s = x_pr[:, 0] ** -0.7
        q_int = low_fidelity_interface_flux(x_ph, K_NOMINAL)
        modulation = 1.0 + x_ph[:, 1] * L / K_NOMINAL
        return np.column_stack(
            (np.ones(len(y_lf)), y_lf - T0, q_int * s, q_int / modulation * s)
        )

    coef, *_ = np.linalg.lstsq(
        design(data.x_ph[train], data.x_pr[train]),
        data.y[train] - T0,
        rcond=None,
    )
    expected = T0 + design(data.test_x_ph, data.test_x_pr) @ coef

    details = json.loads(row["details"])
    assert row["model"] == "MechanismAligned"
    assert row["n_train"] == len(train)
    assert row["n_validation"] == len(validation)
    assert row["validation_mse"] is None
    np.testing.assert_allclose(
        details["q_int_coefficients"], coef[2:], rtol=1e-12
    )
    assert details["phi_slope"] == pytest.approx(coef[1], rel=1e-12)
    assert details["phi_rise_intercept"] == pytest.approx(coef[0], rel=1e-12)
    # Absolute-temperature equivalence of the rise parametrization (§6.2).
    assert details["phi_intercept"] == pytest.approx(
        T0 + coef[0] - coef[1] * T0, rel=1e-9
    )
    assert row["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((data.test_y - expected) ** 2))), rel=1e-12
    )
    assert len(preds) == len(data.test_y)


def test_mechanism_aligned_fit_ignores_eval_data():
    # Alignment protocol §6.1: the A4 fit must depend on the train fold only;
    # poisoning validation/test labels must leave the coefficients unchanged.
    from dataclasses import replace

    from pwl_migration.heat_experiments import _mechanism_aligned_condition

    data = load_heat_batch(DATASETS_ROOT, 1)
    train = np.arange(21)
    validation = np.arange(21, 30)
    kwargs = dict(
        experiment="sample_size", repeat=0, seed=2026, n_labeled=30
    )
    row_clean, _ = _mechanism_aligned_condition(
        data, train, validation, **kwargs
    )
    poisoned_y = data.y.copy()
    poisoned_y[validation] += 1000.0
    poisoned = replace(data, y=poisoned_y, test_y=data.test_y + 100.0)
    row_poisoned, _ = _mechanism_aligned_condition(
        poisoned, train, validation, **kwargs
    )
    assert json.loads(row_clean["details"]) == json.loads(
        row_poisoned["details"]
    )


def test_linear_mapping_phi_uses_train_fold_only():
    # Alignment arms A1-A3: phi must be the train-fold OLS of y on the
    # low-fidelity output at the initial (calibrated) theta.
    from pwl_repro.core.model import calibrate_theta

    data = load_heat_batch(DATASETS_ROOT, 1)
    lower = np.array([1.0 / 35.0])
    upper = np.array([1.0 / 15.0])
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=(1.0 / 35.0,),
        theta_upper=(1.0 / 15.0,),
        feature_spec="heat_v3_qint",
        mapping="linear",
        d_ridge=10.0,
        refresh_weak_labels=True,
        random_state=0,
    ).fit(
        data.x_ph[:21],
        data.x_pr[:21],
        data.y[:21],
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    theta_init = calibrate_theta(
        data.x_ph[:21],
        data.y[:21],
        data.physics_model,
        lower,
        upper,
        seed=0,
    )
    physics = data.physics_model(data.x_ph[:21], theta_init)
    design = np.column_stack((np.ones(21), physics))
    intercept, slope = np.linalg.lstsq(design, data.y[:21], rcond=None)[0]
    assert model.phi_intercept_ == pytest.approx(intercept, rel=1e-10)
    assert model.phi_slope_ == pytest.approx(slope, rel=1e-10)
    # The linear path must actually engage (identity would give slope 1).
    assert 1.5 < model.phi_slope_ < 4.0


def test_affine_aligned_condition_matches_ols():
    # Blind-test S1 arm: pure affine alignment must reduce to a 2-column OLS
    # on the train fold (diagnostic M1 formalized).
    from pwl_migration.heat_experiments import _affine_aligned_condition

    data = load_heat_batch(DATASETS_ROOT, 1)
    train = np.arange(21)
    validation = np.arange(21, 30)
    row, preds = _affine_aligned_condition(
        data,
        train,
        validation,
        experiment="sample_size",
        repeat=0,
        seed=2026,
        n_labeled=30,
        extra={"k_true": data.k_true},
    )
    y_lf = data.physics_model(data.x_ph[train], THETA_NOMINAL)
    design = np.column_stack((np.ones(len(train)), y_lf - T0))
    coef, *_ = np.linalg.lstsq(design, data.y[train] - T0, rcond=None)
    y_lf_test = data.physics_model(data.test_x_ph, THETA_NOMINAL)
    expected = T0 + np.column_stack(
        (np.ones(len(y_lf_test)), y_lf_test - T0)
    ) @ coef

    details = json.loads(row["details"])
    assert row["model"] == "AffineAligned"
    assert row["validation_mse"] is None
    assert details["phi_slope"] == pytest.approx(coef[1], rel=1e-12)
    assert details["phi_rise_intercept"] == pytest.approx(coef[0], rel=1e-12)
    assert details["phi_intercept"] == pytest.approx(
        T0 + coef[0] - coef[1] * T0, rel=1e-9
    )
    assert row["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((data.test_y - expected) ** 2))), rel=1e-12
    )
    assert len(preds) == len(data.test_y)


def test_affine_aligned_fit_ignores_eval_data():
    # Same §6.1 leak guard as A4: poisoning validation/test labels must not
    # change the fitted coefficients.
    from dataclasses import replace

    from pwl_migration.heat_experiments import _affine_aligned_condition

    data = load_heat_batch(DATASETS_ROOT, 1)
    train = np.arange(21)
    validation = np.arange(21, 30)
    kwargs = dict(experiment="sample_size", repeat=0, seed=2026, n_labeled=30)
    row_clean, _ = _affine_aligned_condition(data, train, validation, **kwargs)
    poisoned_y = data.y.copy()
    poisoned_y[validation] += 1000.0
    poisoned = replace(data, y=poisoned_y, test_y=data.test_y + 100.0)
    row_poisoned, _ = _affine_aligned_condition(
        poisoned, train, validation, **kwargs
    )
    assert json.loads(row_clean["details"]) == json.loads(
        row_poisoned["details"]
    )


def test_g0_generic_spec_is_mechanism_free():
    # Blind S2 tier G0: generic quadratic dictionary.  No negative powers,
    # no q_int, no contact-resistance structure, and no duplication of the
    # raw H columns (1, Q, Q^2, T_inf).
    spec = get_feature_spec("heat_g0_generic")
    assert (spec.x_ph_dim, spec.x_pr_dim, spec.theta_dim) == (3, 1, 1)
    assert len(spec.b_names) == 11
    joined = " ".join(spec.b_names)
    assert "q_int" not in joined
    assert "^-" not in joined and "/" not in joined
    assert not ({"1", "Q", "Q^2", "T_inf"} & set(spec.b_names))
    x_ph = np.array([[5e4, 12.0, 300.0], [8e4, 30.0, 290.0]])
    x_pr = np.array([[2.0], [7.5]])
    raw = spec.raw_b(x_ph, x_pr)
    assert raw.shape == (2, 11)
    q, h, t_inf = x_ph.T
    p = x_pr[:, 0]
    np.testing.assert_allclose(raw[:, 0], h, rtol=1e-12)
    np.testing.assert_allclose(raw[:, 7], q * p, rtol=1e-12)
    np.testing.assert_allclose(raw[:, 10], t_inf * p, rtol=1e-12)
    assert np.all(np.isfinite(raw))


def test_g1_prior_spec_grid_excludes_oracle_exponent():
    # Blind S2 tier G1: no_qq columns plus q_int times a coarse negative
    # power grid {0.3, 0.5, 1.0} that must NOT contain the true 0.7.
    spec = get_feature_spec("heat_g1_prior")
    assert len(spec.b_names) == 12
    assert "q_int*P^-0.7" not in spec.b_names
    assert "q_int/(1+hL/k)*P^-0.7" not in spec.b_names
    assert spec.b_names[:6] == ("1", "P", "1/P", "P^-1/2", "Q/P", "Q*P^-1/2")
    x_ph = np.array([[5e4, 12.0, 300.0], [8e4, 30.0, 290.0]])
    x_pr = np.array([[2.0], [7.5]])
    raw = spec.raw_b(x_ph, x_pr)
    assert raw.shape == (2, 12)
    q_int = low_fidelity_interface_flux(x_ph, K_NOMINAL)
    modulation = 1.0 + x_ph[:, 1] * L / K_NOMINAL
    p = x_pr[:, 0]
    np.testing.assert_allclose(raw[:, 6], q_int * p**-0.3, rtol=1e-12)
    np.testing.assert_allclose(raw[:, 8], q_int * p**-1.0, rtol=1e-12)
    np.testing.assert_allclose(
        raw[:, 9], q_int / modulation * p**-0.3, rtol=1e-12
    )
    assert np.all(np.isfinite(raw))


def test_end_to_end_g0_g1_fit():
    # Both S2 tiers train end to end on a small fold.
    data = load_heat_batch(DATASETS_ROOT, 1)
    for spec_name, n_b in (("heat_g0_generic", 11), ("heat_g1_prior", 12)):
        model = PWLRegressor(
            lambda_physics=1.0,
            lambda_l1=0.01,
            lambda_group=0.01,
            theta_lower=(1.0 / 35.0,),
            theta_upper=(1.0 / 15.0,),
            feature_spec=spec_name,
            d_ridge=10.0,
            random_state=0,
        ).fit(
            data.x_ph[:21],
            data.x_pr[:21],
            data.y[:21],
            data.weak_x_ph,
            data.weak_y,
            physics_model=data.physics_model,
        )
        prediction = model.predict(data.test_x_ph, data.test_x_pr)
        assert np.all(np.isfinite(prediction))
        assert len(model.d_) == n_b
