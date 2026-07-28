import numpy as np

from pwl_repro.optimization import consensus_admm


def test_consensus_admm_matches_quadratic_solution():
    # min 0.5*(x-2)^2 + 0.5*(x+1)^2 has x=0.5
    def prox_first(value, step):
        return (value + 2.0 * step) / (1.0 + step)

    def prox_second(value, step):
        return (value - step) / (1.0 + step)

    result = consensus_admm(
        (prox_first, prox_second),
        np.array([0.0]),
        max_iter=500,
        tolerance=1e-8,
        relative_tolerance=1e-8,
    )
    assert result.converged
    np.testing.assert_allclose(result.value, np.array([0.5]), atol=1e-6)

