import json
from copy import deepcopy

import numpy as np

from pwl_repro.experiments import (
    derive_label_savings_experiment,
    nested_split_plan,
    run_physics_accuracy_experiment,
    run_sample_size_experiment,
)


def _tiny_config():
    return {
        "experiment": {
            "seed": 11,
            "sample_sizes": [10, 20, 30],
            "repeats": 2,
            "physics_accuracy_targets": [0.8, 0.6, 0.4],
            "physics_accuracy_n_labeled": 20,
            "physics_accuracy_baselines": ["Ridge"],
            "accuracy_calibration_size": 120,
            "accuracy_calibration_seed": 991,
            "accuracy_scale_log10": [-2, 2, 41],
            "accuracy_require_tolerance": False,
            "label_savings_pwl_n_labeled": 30,
            "label_savings_sizes": [30],
            "label_savings_baselines": ["Ridge"],
            "equivalence_margin_fraction": 0.1,
        },
        "simulation": {
            "n_weak": 12,
            "n_test": 15,
            "snr": 5.0,
            "correlation": 0.5,
            "discrepancy_scale": 1.0,
            "physics_noise_fraction": 0.1,
            "noise_reference_size": 500,
        },
        "train_fraction": 0.7,
        "refit_on_train_validation": False,
        "lambda_grid": {
            "physics": [0.3],
            "l1": [0.01],
            "group": [0.01],
        },
        "model": {
            "mapping": "identity",
            "standardize": True,
            "d_ridge": 10.0,
            "selection_rule": "physics_one_standard_error",
            "admm_rho": 1.0,
            "admm_tolerance": 1e-3,
            "admm_max_iter": 40,
            "bcd_tolerance": 1e-2,
            "bcd_max_iter": 3,
            "require_convergence": False,
        },
        "baselines": ["Ridge"],
    }


def test_nested_split_plan_preserves_roles():
    split = nested_split_plan(
        30, [10, 20, 30], train_fraction=0.7, seed=4
    )
    for size, (train, validation) in split.items():
        assert len(train) + len(validation) == size
        assert set(train).isdisjoint(validation)
    assert set(split[10][0]).issubset(split[20][0])
    assert set(split[20][0]).issubset(split[30][0])
    assert set(split[10][1]).issubset(split[20][1])
    assert set(split[20][1]).issubset(split[30][1])


def test_section_iv_protocol_shares_data_and_fixes_pwl_labels():
    config = _tiny_config()
    sample = run_sample_size_experiment(config)
    pwl_details = sample.metrics[sample.metrics["model"] == "PWL"][
        "details"
    ].map(json.loads)
    assert all(
        details["tuning"]["selection_rule"]
        == "physics_one_standard_error"
        for details in pwl_details
    )
    for _, rows in sample.metrics.groupby("repeat"):
        assert rows["dataset_id"].nunique() == 1
        assert sorted(rows["n_labeled"].unique()) == [10, 20, 30]

    savings = derive_label_savings_experiment(sample, config)
    fixed = savings.metrics[savings.metrics["model"] == "PWL-fixed"]
    assert set(fixed["n_labeled"]) == {30}
    assert set(fixed["curve_n_labeled"]) == {30}

    accuracy = run_physics_accuracy_experiment(config)
    for _, rows in accuracy.metrics.groupby("repeat"):
        assert rows["dataset_id"].nunique() == 1
        assert set(rows["model"]) == {"Ridge", "PWL-H", "PWL-M", "PWL-L"}
    calibrated = accuracy.conditions.dropna(subset=["calibration_correlation"])
    assert len(calibrated) == 6
    assert set(calibrated["repeat"]) == {0, 1}
    assert np.all(np.isfinite(calibrated["discrepancy_scale"]))
    assert np.all(np.isfinite(calibrated["correlation_error"]))


def test_parallel_pwl_candidates_match_serial_selection():
    serial_config = _tiny_config()
    serial_config["experiment"]["sample_sizes"] = [10]
    serial_config["experiment"]["repeats"] = 1
    serial_config["lambda_grid"]["physics"] = [0.3, 0.6]
    serial_config["baselines"] = []
    serial_config["model"]["n_jobs"] = 1

    parallel_config = deepcopy(serial_config)
    parallel_config["model"]["n_jobs"] = 2

    serial = run_sample_size_experiment(serial_config)
    parallel = run_sample_size_experiment(parallel_config)
    serial_row = serial.metrics.iloc[0]
    parallel_row = parallel.metrics.iloc[0]

    assert serial_row["hyperparameters"] == parallel_row["hyperparameters"]
    np.testing.assert_allclose(
        serial_row[["validation_mse", "mse", "rmse", "mae"]].to_numpy(
            dtype=float
        ),
        parallel_row[["validation_mse", "mse", "rmse", "mae"]].to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    )
    details = json.loads(parallel_row["details"])
    assert details["tuning"]["n_jobs"] == 2
