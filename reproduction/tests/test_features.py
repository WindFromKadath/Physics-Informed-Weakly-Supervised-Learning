import numpy as np

from pwl_repro.features import PWLFeatureLibrary
from pwl_repro.simulation import generate_simulation


def test_feature_shapes_groups_and_affinity():
    data = generate_simulation(20, 25, 10, seed=3)
    library = PWLFeatureLibrary().fit(
        np.vstack((data.x_ph, data.weak_x_ph)),
        data.x_ph,
        data.x_pr,
        np.array([0.4, 0.6]),
    )
    assert library.transform_h(data.x_ph, np.array([0.2, 0.8])).shape == (20, 25)
    assert library.transform_b(data.x_ph, data.x_pr).shape == (20, 12)
    assert len(library.groups) == 5
    assert sum(map(len, library.groups)) == 25

    rng = np.random.default_rng(4)
    g = rng.normal(size=25)
    theta = np.array([0.27, 0.83])
    base, design = library.affine_prediction_parts(data.x_ph, g)
    expected = library.transform_h(data.x_ph, theta) @ g
    np.testing.assert_allclose(base + design @ theta, expected, atol=1e-10)


def test_compact_b_profile_exactly_spans_process_function():
    data = generate_simulation(30, 20, 10, seed=17)
    library = PWLFeatureLibrary(standardize=False, b_profile="compact").fit(
        np.vstack((data.x_ph, data.weak_x_ph)),
        data.x_ph,
        data.x_pr,
        np.array([0.4, 0.6]),
    )
    design = library.transform_b(data.x_ph, data.x_pr)
    coefficients = np.array([0.0, 0.8, 1.6])
    from pwl_repro.simulation import process_contribution

    np.testing.assert_allclose(
        design @ coefficients,
        process_contribution(data.x_pr),
        atol=1e-10,
    )
