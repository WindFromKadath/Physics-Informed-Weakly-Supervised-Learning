"""Case-B (spot weld) scenario tests: data contract, surrogate, library,
x_pr=0 plumbing, weak-label refresh, protocol, and end-to-end fit."""

from pathlib import Path

import numpy as np
import pytest

from pwl_repro.case_b import (
    CASE_B_H_NAMES,
    THETA_LOWER,
    THETA_UPPER,
    CaseBSurrogate,
    build_case_b_scenario,
    case_b_feature_spec,
    load_case_b_frames,
    pooled_replicate_sigma,
    sample_weak_inputs,
)
from pwl_repro.features import PWLFeatureLibrary, get_feature_spec
from pwl_repro.model import PWLRegressor

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "case_b_spotweld"


@pytest.fixture(scope="module")
def frames():
    return load_case_b_frames(DATA_ROOT)


@pytest.fixture(scope="module")
def surrogate(frames):
    return CaseBSurrogate().fit(frames[1])


def test_data_contract(frames):
    field, model = frames
    assert len(field) == 120 and len(model) == 35
    settings = field.groupby(["load", "current", "thickness"]).size()
    assert len(settings) == 12 and (settings == 10).all()
    sigma = pooled_replicate_sigma(field)
    assert 0.3 < sigma < 0.6  # pooled variance ≈ 0.2006
    assert field["tuning" if "tuning" in field else "diameter"].notna().all()


def test_surrogate_interpolates_and_uses_theta(surrogate, frames):
    model = frames[1]
    x = model[["load", "current", "thickness"]].to_numpy(dtype=float)
    theta = model["tuning"].to_numpy(dtype=float)
    y = model["diameter"].to_numpy(dtype=float)
    prediction = np.array(
        [surrogate.predict(x[i : i + 1], [theta[i]])[0] for i in range(len(model))]
    )
    # Interpolating surrogate: training points are reproduced closely.
    assert np.max(np.abs(prediction - y)) < 0.2
    # Theta enters as the 4th input: outputs must vary with theta somewhere.
    varied = surrogate.predict(x[:5], [0.8]) - surrogate.predict(x[:5], [8.0])
    assert np.max(np.abs(varied)) > 1e-6
    diagnostics = surrogate.loo_diagnostics()
    assert diagnostics["loo_rmse"] < 0.5


def test_case_b_spec_dims_and_groups():
    spec = get_feature_spec("case_b")
    assert (spec.x_ph_dim, spec.x_pr_dim, spec.theta_dim) == (3, 0, 1)
    assert len(spec.h_names) == 11
    assert len(spec.b_names) == 7
    assert len(spec.groups) == 4
    assert spec.orthogonalize_b
    covered = np.concatenate(spec.groups)
    assert sorted(covered.tolist()) == list(range(len(spec.h_names)))
    ext = get_feature_spec("case_b_ext")
    assert (ext.x_ph_dim, ext.x_pr_dim, ext.theta_dim) == (3, 0, 1)
    assert len(ext.h_names) == 21
    # ext B = base 7 + 3 H-external interactions (audit fix Bug 2).
    assert len(ext.b_names) == 10
    assert len(ext.groups) == 4
    assert ext.orthogonalize_b
    covered = np.concatenate(ext.groups)
    assert sorted(covered.tolist()) == list(range(len(ext.h_names)))


def test_case_b_h_is_affine_in_theta(surrogate, frames):
    field = frames[0]
    data = build_case_b_scenario(
        field, surrogate, n_labeled=100, split_seed=2026
    )
    library = PWLFeatureLibrary(
        standardize=True, spec=case_b_feature_spec()
    ).fit(
        np.vstack((data.x_ph, data.weak_x_ph)),
        data.x_ph,
        data.x_pr,
        data.theta_true,
    )
    rng = np.random.default_rng(0)
    g = rng.normal(size=len(CASE_B_H_NAMES))
    theta = np.array([3.0])
    base, design = library.affine_prediction_parts(data.x_ph, g)
    discrepancy = library.transform_h(data.x_ph, theta) @ g - (
        base + design @ theta
    )
    assert np.max(np.abs(discrepancy)) < 1e-8
    # Residualized B is orthogonal to H on the projection rows, which are
    # the stacked labeled+weak rows when x^pr is empty (audit fix Bug 1).
    stacked = np.vstack((data.x_ph, data.weak_x_ph))
    b_fit = library.transform_b(stacked, np.empty((len(stacked), 0)))
    cross = b_fit.T @ library.transform_h(stacked, np.array([4.0]))
    assert np.max(np.abs(cross)) < 1e-6


def test_weak_inputs_stay_in_field_domain():
    weak = sample_weak_inputs(50, 0)
    assert weak[:, 0].min() >= 4.0 and weak[:, 0].max() <= 5.3
    assert weak[:, 1].min() >= 21.0 and weak[:, 1].max() <= 29.0
    assert set(np.unique(weak[:, 2]).tolist()) <= {1.0, 2.0}


def test_scenario_split_is_leakage_free(surrogate, frames):
    field = frames[0]
    data = build_case_b_scenario(
        field, surrogate, n_labeled=10, split_seed=2026
    )
    assert len(data.y) == 10 and len(data.test_y) == 110
    assert data.x_pr.shape == (10, 0) and data.test_x_pr.shape == (110, 0)
    assert THETA_LOWER[0] <= data.theta_true[0] <= THETA_UPPER[0]
    assert data.physics_noise_sigma == 0.0
    # Reproducible: the same seed yields the identical split.
    again = build_case_b_scenario(
        field, surrogate, n_labeled=10, split_seed=2026
    )
    assert np.array_equal(data.x_ph, again.x_ph)
    assert np.array_equal(data.test_x_ph, again.test_x_ph)
    # Row-level partition of the 120 rows (no row duplicated or dropped).
    # Note: field data are 12 settings x 10 replicates, so input settings
    # legitimately appear on both sides of a sample-level split (paper
    # protocol); setting-level sensitivity is a separate follow-up.
    combined = np.vstack((data.x_ph, data.test_x_ph))
    field_x = field[["load", "current", "thickness"]].to_numpy(dtype=float)
    assert sorted(map(tuple, combined)) == sorted(map(tuple, field_x))


def test_end_to_end_case_b_fit(surrogate, frames):
    field = frames[0]
    data = build_case_b_scenario(
        field, surrogate, n_labeled=100, split_seed=2026
    )
    model = PWLRegressor(
        lambda_physics=1.0,
        lambda_l1=0.01,
        lambda_group=0.01,
        theta_lower=THETA_LOWER,
        theta_upper=THETA_UPPER,
        feature_spec="case_b",
        refresh_weak_labels=True,
        d_ridge=10.0,
        random_state=0,
    ).fit(
        data.x_ph[:70],
        data.x_pr[:70],
        data.y[:70],
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert np.all(np.isfinite(prediction))
    assert THETA_LOWER[0] - 1e-9 <= model.theta_[0] <= THETA_UPPER[0] + 1e-9
    assert len(model.weak_refresh_drift_) == model.n_iter_


def test_small_sample_b_not_annihilated(surrogate, frames):
    """Audit fix Bug 1: stacked projection keeps B alive at n_train=7.

    With the old labeled-row projection the underdetermined lstsq (7 rows vs
    rank-7 H) interpolated every B column and the drop rule discarded all
    seven; only x1*x3 / x2*x3 lie outside the default H column space.
    """

    field = frames[0]
    data = build_case_b_scenario(field, surrogate, n_labeled=10, split_seed=2026)
    library = PWLFeatureLibrary(
        standardize=True, spec=case_b_feature_spec()
    ).fit(
        np.vstack((data.x_ph[:7], data.weak_x_ph)),
        data.x_ph[:7],
        data.x_pr[:7],
        data.theta_true,
    )
    surviving = [
        name for name, dropped in zip(library.b_names, library.b_dropped_)
        if not dropped
    ]
    assert surviving == ["x1*x3", "x2*x3"]


def test_case_b_ext_b_survives_orthogonalization(surrogate, frames):
    """Audit fix Bug 2: only the H-external interactions survive in ext."""

    field = frames[0]
    data = build_case_b_scenario(field, surrogate, n_labeled=10, split_seed=2026)
    library = PWLFeatureLibrary(
        standardize=True, spec=get_feature_spec("case_b_ext")
    ).fit(
        np.vstack((data.x_ph[:7], data.weak_x_ph)),
        data.x_ph[:7],
        data.x_pr[:7],
        data.theta_true,
    )
    surviving = [
        name for name, dropped in zip(library.b_names, library.b_dropped_)
        if not dropped
    ]
    assert surviving == ["x1^2*x2", "x1*x2^2", "x1^2*x3"]


def test_orthogonalize_guard_skips_when_underdetermined():
    """Rank guard: rows <= rank(H) must skip orthogonalization loudly."""

    from pwl_repro.features import FeatureSpec

    def raw_h(x_ph, theta):
        x = np.asarray(x_ph, dtype=float)
        t = np.asarray(theta, dtype=float).reshape(-1)[0]
        return np.column_stack(
            (
                np.ones(len(x)),
                x[:, 0],
                x[:, 0] ** 2,
                x[:, 1],
                t * np.ones(len(x)),
                t * x[:, 0],
            )
        )

    def raw_b(x_ph, x_pr):
        return np.column_stack(
            (np.ones(len(x_ph)), np.asarray(x_pr, dtype=float)[:, 0])
        )

    spec = FeatureSpec(
        name="guard_test",
        x_ph_dim=2,
        x_pr_dim=1,  # B needs x^pr, so the projection uses labeled rows only.
        theta_dim=1,
        h_names=("1", "x1", "x1^2", "x2", "t", "t*x1"),
        b_names=("1", "p"),
        groups=(np.arange(0, 4), np.arange(4, 6)),
        group_names=("x", "theta"),
        raw_h=raw_h,
        raw_b=raw_b,
        orthogonalize_b=True,
    )
    rng = np.random.default_rng(0)
    x_labeled = rng.normal(size=(3, 2))  # 3 rows vs rank-3 H: exact fit.
    p_labeled = rng.normal(size=(3, 1))
    x_weak = rng.normal(size=(20, 2))
    with pytest.warns(UserWarning, match="orthogonalize_b skipped"):
        library = PWLFeatureLibrary(standardize=True, spec=spec).fit(
            np.vstack((x_labeled, x_weak)),
            x_labeled,
            p_labeled,
            np.array([0.5]),
        )
    assert not hasattr(library, "b_projection_")
    assert not library.b_dropped_.any()
    transformed = library.transform_b(x_labeled, p_labeled)
    assert np.all(np.isfinite(transformed))
