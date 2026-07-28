import numpy as np

from pwl_repro.simulation import discrepancy, generate_simulation


def test_simulation_shapes_and_finiteness():
    data = generate_simulation(n_labeled=12, n_weak=15, n_test=18, seed=7)
    assert data.x_ph.shape == (12, 3)
    assert data.x_pr.shape == (12, 2)
    assert data.weak_x_ph.shape == (15, 3)
    assert data.test_x_ph.shape == (18, 3)
    assert data.theta_true.shape == (2,)
    for value in (
        data.y,
        data.weak_y,
        data.test_y,
        data.physics_model(data.x_ph, data.theta_true),
    ):
        assert np.all(np.isfinite(value))


def test_simulation_is_seed_reproducible():
    first = generate_simulation(10, 11, 12, seed=99)
    second = generate_simulation(10, 11, 12, seed=99)
    np.testing.assert_allclose(first.y, second.y)
    np.testing.assert_allclose(first.weak_y, second.weak_y)
    np.testing.assert_allclose(first.theta_true, second.theta_true)


def test_discrepancy_scale_preserves_samples_and_noise():
    base = generate_simulation(20, 25, 30, seed=123, discrepancy_scale=0.0)
    changed = base.with_discrepancy_scale(3.0)
    np.testing.assert_allclose(base.x_ph, changed.x_ph)
    np.testing.assert_allclose(base.y, changed.y)
    np.testing.assert_allclose(base.test_y, changed.test_y)
    np.testing.assert_allclose(
        changed.weak_y - base.weak_y,
        3.0 * discrepancy(base.weak_x_ph),
    )
    assert base.sampling_diagnostics["noise_reference_size"] == 5000


def test_noise_scale_uses_independent_reference_population():
    small = generate_simulation(10, 12, 14, seed=8, noise_reference_size=500)
    large = generate_simulation(40, 12, 60, seed=8, noise_reference_size=500)
    assert small.noise_sigma == large.noise_sigma


def test_generated_inputs_are_stable_for_entire_theta_box():
    data = generate_simulation(30, 30, 30, seed=19, noise_reference_size=500)
    all_x2 = np.concatenate(
        (data.x_ph[:, 1], data.weak_x_ph[:, 1], data.test_x_ph[:, 1])
    )
    denominators = np.column_stack(
        (
            4.0 * all_x2 + 20.0,
            9.0 * all_x2**3 + 4.0 * all_x2 + 20.0,
        )
    )
    assert np.all(np.min(np.abs(denominators), axis=1) >= 1.0)
    assert np.all(np.prod(denominators, axis=1) > 0.0)
