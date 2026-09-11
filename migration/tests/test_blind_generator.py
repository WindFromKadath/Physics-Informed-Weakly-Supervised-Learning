"""Blind-batch generator tests: determinism, contract, physics, isolation."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "reproduction" / "src")
)

from generate_blind_batches import (  # noqa: E402
    SIGMA,
    batch_seeds,
    generate_batch,
)
from pwl_migration.heat import (  # noqa: E402
    K_NOMINAL,
    load_heat_batch,
    low_fidelity_temperature,
)
from verify_attribution import hf_solve  # noqa: E402


def test_seed_protocol_matches_v13_spec():
    # v1.3: batch 1 starts at 20260831 (= 20260731 + 100); offsets fixed.
    seeds = batch_seeds(1)
    assert seeds["k_true"] == 20260831
    assert seeds["reference"] == 20260832
    assert seeds["noise_cal"] == 20260838
    blind = batch_seeds(21)
    assert blind["k_true"] == 20260731 + 2100
    assert blind["noise_acc"] == blind["k_true"] + 6


def test_generation_is_deterministic(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    generate_batch(21, root_a)
    generate_batch(21, root_b)
    for name in (
        "reference_measurements.csv",
        "acceptance_tests.csv",
        "engineering_predictions.csv",
    ):
        assert (root_a / "batch_21" / name).read_bytes() == (
            root_b / "batch_21" / name
        ).read_bytes()


def test_generated_batch_loads_via_heat_adapter(tmp_path):
    generate_batch(21, tmp_path)
    data = load_heat_batch(tmp_path, 21)
    assert data.x_ph.shape == (120, 3)
    assert data.x_pr.shape == (120, 1)
    assert data.weak_x_ph.shape == (200, 3)
    assert data.test_x_ph.shape == (200, 3)
    assert 15.0 <= data.k_true <= 35.0
    assert data.noise_sigma == pytest.approx(SIGMA)
    for array in (data.y, data.weak_y, data.test_y):
        assert np.all(np.isfinite(array))


def test_generated_batch_physics_and_noise(tmp_path):
    root = generate_batch(21, tmp_path)
    meta = json.loads(
        (tmp_path / "batch_21" / "metadata.json").read_text(encoding="utf-8")
    )
    k_true = float(meta["k_true"])
    data = load_heat_batch(tmp_path, 21)
    x_full = np.column_stack((data.x_ph, data.x_pr))
    sol = hf_solve(x_full[:, 0], x_full[:, 1], x_full[:, 2], x_full[:, 3], k_true)
    resid = data.y - sol["y"]
    assert abs(float(np.mean(resid))) < 0.8
    assert 3.5 < float(np.std(resid)) < 5.5  # n=120 sampling window
    # Engineering predictions equal the LF closed form at k_nominal.
    y_lf = low_fidelity_temperature(data.weak_x_ph, np.array([1.0 / K_NOMINAL]))
    assert np.max(np.abs(y_lf - data.weak_y)) < 1e-9
    # Reference and acceptance inputs are disjoint (isolated seeds).
    test_full = np.column_stack((data.test_x_ph, data.test_x_pr))
    ref_keys = set(map(tuple, x_full.tolist()))
    acc_keys = set(map(tuple, test_full.tolist()))
    assert not (ref_keys & acc_keys)


def test_generator_refuses_to_overwrite(tmp_path):
    generate_batch(21, tmp_path)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        # CLI-level guard: re-entering main with the same range must stop.
        sys.argv = ["generate_blind_batches.py", "--start", "21", "--end", "21",
                    "--root", str(tmp_path)]
        from generate_blind_batches import main

        main()
