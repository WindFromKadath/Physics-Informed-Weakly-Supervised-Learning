import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from pwl_repro.experiments import (
    run_sample_size_experiment,
    validate_experiment_config,
)

REPRODUCTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION / "scripts"))

from compare_to_paper import PAPER_REFERENCE, compare_results
from diagnose_basis import diagnose_basis

SENSITIVITY_DIR = REPRODUCTION / "configs" / "sensitivity"
BASELINE_RESULTS = REPRODUCTION / "results" / "section_iv_single_full_parallel"

# 阶段 2：每个变体相对 base.yaml 的递归差异键集合（s12 三个键）。
# s04/s12 额外包含 model.admm_max_iter=2000：[ENGINEERING] refit 合并数据后
# g-ADMM 偶发触及 1000 上限导致假性不收敛，属工程修正而非研究因子。
EXPECTED_DIFF_KEYS = {
    "base": set(),
    "s01_minimum_selection": {"model.selection_rule"},
    "s02_no_d_ridge": {"model.d_ridge"},
    "s03_expanded_b": {"model.b_profile"},
    "s04_refit": {"refit_on_train_validation", "model.admm_max_iter"},
    "s05_corr_05": {"simulation.correlation"},
    "s06_corr_07": {"simulation.correlation"},
    "s07_physics_noise_00": {"simulation.physics_noise_fraction"},
    "s08_physics_noise_01": {"simulation.physics_noise_fraction"},
    "s09_physics_noise_05": {"simulation.physics_noise_fraction"},
    "s10_linear_mapping": {"model.mapping"},
    "s11_paper_raw": {"simulation.singularity_policy"},
    "s12_paper_text_combo": {
        "model.selection_rule",
        "model.d_ridge",
        "refit_on_train_validation",
        "model.admm_max_iter",
    },
}


def _flatten(mapping, prefix=""):
    items = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            items.update(_flatten(value, dotted + "."))
        else:
            items[dotted] = value
    return items


def _load(name):
    return yaml.safe_load((SENSITIVITY_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def test_sensitivity_configs_validate_and_diff_from_base():
    assert set(EXPECTED_DIFF_KEYS) == {
        path.stem for path in SENSITIVITY_DIR.glob("*.yaml")
    }
    base = _flatten(_load("base"))
    for name, expected_keys in EXPECTED_DIFF_KEYS.items():
        config = _load(name)
        validate_experiment_config(config, "all")
        flat = _flatten(config)
        assert set(flat) == set(base)
        diff = {key for key in flat if flat[key] != base[key]}
        assert diff == expected_keys, f"{name}: {diff} != {expected_keys}"


def test_compare_to_paper_recovers_baseline_gap_without_writes():
    if not BASELINE_RESULTS.exists():
        pytest.skip("基线结果目录不存在。")
    before = {
        path.name: path.stat().st_size for path in BASELINE_RESULTS.iterdir()
    }
    report = compare_results(BASELINE_RESULTS)
    after = {
        path.name: path.stat().st_size for path in BASELINE_RESULTS.iterdir()
    }
    assert before == after, "compare_results 不得新增或修改结果目录文件"

    keys = report["key_metrics"]
    assert keys["iv_a_pwl_rmse"][120] == pytest.approx(2.185, rel=1e-2)
    assert PAPER_REFERENCE["iv_a_pwl_rmse"][120] == pytest.approx(1.378)
    assert 0.9 <= keys["iv_a_mse_over_noise_variance_at_120"] <= 1.2


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


def test_pwl_details_contain_stage1_diagnostics():
    sample = run_sample_size_experiment(_tiny_config())
    pwl_details = sample.metrics[sample.metrics["model"] == "PWL"][
        "details"
    ].map(json.loads)
    assert len(pwl_details) > 0
    numeric_fields = (
        "weak_distillation_r2",
        "labeled_prediction_r2",
        "physics_component_norm",
        "process_component_norm",
        "minimum_vs_selected_validation_mse",
    )
    for details in pwl_details:
        for field in numeric_fields:
            assert field in details
            assert np.isfinite(details[field]), field
        assert isinstance(details["theta_boundary_hit"], bool)
        assert set(details["g_group_norms"]) == {
            "x1",
            "x2",
            "x3",
            "theta1",
            "theta2",
        }
        assert all(
            np.isfinite(value) for value in details["g_group_norms"].values()
        )
        assert all(np.isfinite(value) for value in details["d_coefficients"])
        tuning = details["tuning"]
        assert tuning["selected_rank_by_validation_mse"] >= 1
        assert tuning["n_valid_candidates"] >= 1


def test_diagnose_basis_compact_b_spans_process_contribution():
    result = diagnose_basis(_tiny_config(), seed=0, coverage_size=1500)
    assert result["b_compact_r2_process_contribution"] > 0.999
    assert np.isfinite(result["h_r2_eta_true"])
    assert np.isfinite(result["h_r2_eta_true_plus_discrepancy"])
    assert 0.0 <= result["max_canonical_correlation_h_b_compact"] <= 1.0
