import numpy as np

from pwl_repro.model import PWLRegressor
from pwl_repro.simulation import generate_simulation


def test_pwl_end_to_end_returns_finite_predictions():
    data = generate_simulation(24, 35, 20, seed=123)
    model = PWLRegressor(
        lambda_physics=0.3,
        lambda_l1=0.01,
        lambda_group=0.01,
        admm_tolerance=1e-4,
        admm_max_iter=250,
        bcd_tolerance=1e-3,
        bcd_max_iter=10,
        random_state=123,
    ).fit(
        data.x_ph,
        data.x_pr,
        data.y,
        data.weak_x_ph,
        data.weak_y,
        physics_model=data.physics_model,
    )
    prediction = model.predict(data.test_x_ph, data.test_x_pr)
    assert prediction.shape == (20,)
    assert np.all(np.isfinite(prediction))
    assert np.all(model.theta_ >= 0.0)
    assert np.all(model.theta_ <= 1.0)
    assert model.n_iter_ >= 1
    assert np.isfinite(model.training_objective_)
    final = model.history_[-1]
    assert np.isfinite(final.g_primal_residual)
    assert np.isfinite(final.g_dual_residual)
    assert np.isfinite(final.theta_primal_residual)
    assert np.isfinite(final.theta_dual_residual)
