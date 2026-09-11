"""M2: theta reinterpretation analysis for the heat scenario (2026-08-03).

Question: what does theta = 1/k actually identify in this scenario?  The
contact resistance R_c(P) is in series with the material conduction
resistance, so temperature data can only constrain the TOTAL resistance.
This script quantifies, per batch:

  1. theta_true = 1/k_true                    (material parameter, not identifiable)
  2. theta_eff  = argmin_theta ||y_det - LF(theta)||^2   on noise-free HF outputs
       (the effective-resistance parameter the data physics actually defines)
  3. theta_cal  = argmin_theta ||y_obs - LF(theta)||^2   on the noisy 120 labels
       (equation-(12) calibration: what is empirically recoverable)
  4. theta_hat  = v3 PWL estimate at n=120 (from the frozen baseline results)
  5. prediction sensitivity of the low-fidelity model across the theta box
       [1/35, 1/15], as RMS change on acceptance inputs relative to sigma

Verdict criteria (M2 revised anchor 4):
  * theta_eff far outside the box  ->  "close to 1/k_true" is physically
    unreachable; boundary hits are the correct optimizer behaviour.
  * theta_cal direction consistent with theta_eff (> theta_true)  ->
    calibration follows the effective-resistance mechanism.
  * box prediction sensitivity << sigma  ->  predictions do not hinge on the
    exact theta value; theta acts as a nuisance/calibration parameter.

Run:  uv run python migration/verify_theta_effective.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))

from pwl_migration.heat import (  # noqa: E402
    K_NOMINAL,
    THETA_LOWER,
    THETA_UPPER,
    low_fidelity_temperature,
)

from verify_attribution import hf_solve  # noqa: E402

DATASETS = ROOT / "migration" / "datasets"
RESULTS = ROOT / "migration" / "results" / "heat_v3_qint_min"

# Wide bracket that surely contains any effective-resistance minimum.
CAL_BRACKET = (0.005, 0.6)


def calibrate(x_ph: np.ndarray, y: np.ndarray) -> float:
    """1-D least-squares calibration of theta for the low-fidelity model."""

    def mse(theta: float) -> float:
        return float(np.mean((low_fidelity_temperature(x_ph, np.array([theta])) - y) ** 2))

    result = minimize_scalar(mse, bounds=CAL_BRACKET, method="bounded",
                             options={"xatol": 1e-12})
    return float(result.x)


def main() -> None:
    # theta_hat from the frozen v3 baseline (n=120 rows).
    results = pd.read_csv(RESULTS / "results.csv")
    pwl = results[(results["model"] == "PWL") & (results["n_labeled"] == 120)]
    theta_hat = {}
    for _, row in pwl.iterrows():
        details = json.loads(row["details"])
        theta_hat[int(row["repeat"]) + 1] = (
            float(details["theta_estimate"][0]),
            bool(details["theta_boundary_hit"]),
        )

    print("== M2 theta analysis: effective resistance vs material parameter ==")
    header = (
        f"  {'batch':>5} {'k_true':>7} {'th_true':>8} {'th_eff':>8} {'th_cal':>8}"
        f" {'th_hat':>8} {'hit':>4} {'sens/sigma':>10}"
    )
    print(header)
    rows = []
    for batch in range(1, 11):
        batch_dir = DATASETS / f"batch_{batch:02d}"
        meta = json.loads((batch_dir / "metadata.json").read_text(encoding="utf-8"))
        k_true = float(meta["k_true"])
        sigma = float(meta["sigma"])
        ref = pd.read_csv(batch_dir / "reference_measurements.csv", comment="#")
        acc = pd.read_csv(batch_dir / "acceptance_tests.csv", comment="#")
        x_ph = ref[["Q", "h", "T_inf"]].to_numpy(dtype=float)
        p = ref["P"].to_numpy(dtype=float)
        y_obs = ref["y_obs"].to_numpy(dtype=float)
        # Noise-free HF output on the same inputs -> effective theta.
        y_det = hf_solve(x_ph[:, 0], x_ph[:, 1], x_ph[:, 2], p, k_true)["y"]
        th_true = 1.0 / k_true
        th_eff = calibrate(x_ph, y_det)
        th_cal = calibrate(x_ph, y_obs)
        hat, hit = theta_hat.get(batch, (float("nan"), False))
        # Prediction sensitivity across the theta box on acceptance inputs.
        xa = acc[["Q", "h", "T_inf"]].to_numpy(dtype=float)
        spread = low_fidelity_temperature(xa, np.array(THETA_UPPER)) - low_fidelity_temperature(
            xa, np.array(THETA_LOWER)
        )
        sens = float(np.sqrt(np.mean(spread**2))) / sigma
        rows.append((batch, k_true, th_true, th_eff, th_cal, hat, hit, sens))
        print(
            f"  {batch:>5d} {k_true:>7.2f} {th_true:>8.5f} {th_eff:>8.5f}"
            f" {th_cal:>8.5f} {hat:>8.5f} {str(hit):>4} {sens:>10.3f}"
        )

    arr = pd.DataFrame(
        rows,
        columns=["batch", "k_true", "theta_true", "theta_eff", "theta_cal",
                 "theta_hat", "boundary_hit", "sens_over_sigma"],
    )
    upper = THETA_UPPER[0]
    print()
    print("== summary ==")
    print(f"  theta box: [{THETA_LOWER[0]:.5f}, {upper:.5f}]  (k in [15, 35])")
    print(
        f"  theta_eff range: [{arr.theta_eff.min():.4f}, {arr.theta_eff.max():.4f}]"
        f"  -> batches with theta_eff above box upper:"
        f" {int((arr.theta_eff > upper).sum())}/10"
        f"  (equivalent k_eff = 1/theta_eff in"
        f" [{1.0/arr.theta_eff.max():.2f}, {1.0/arr.theta_eff.min():.2f}] W/(m K))"
    )
    print(
        f"  theta_cal > theta_true (effective-resistance direction):"
        f" {int((arr.theta_cal > arr.theta_true).sum())}/10 batches"
    )
    print(
        f"  |theta_cal - theta_eff| median:"
        f" {float((arr.theta_cal - arr.theta_eff).abs().median()):.5f}"
        f"  (noise floor of calibration on n=120)"
    )
    print(
        f"  box prediction sensitivity / sigma:"
        f" [{arr.sens_over_sigma.min():.3f}, {arr.sens_over_sigma.max():.3f}]"
    )
    hits = arr[arr.theta_hat.notna()]
    print(
        f"  v3 theta_hat boundary hits: {int(hits.boundary_hit.sum())}/{len(hits)};"
        f" theta_hat spread (std): {float(hits.theta_hat.std()):.5f}"
    )
    print()
    print("== interpretation ==")
    print(
        "  R_c(P) is in series with the material resistance and dominates it\n"
        "  (R_c ~ 0.015-0.06 m^2 K/W vs L/(2k) ~ 0.002): the temperature data\n"
        "  constrain total resistance only. theta_eff therefore sits far above\n"
        "  the box built around k_true, boundary hits are the correct optimizer\n"
        "  behaviour, and theta must be read as an effective-resistance /\n"
        "  nuisance parameter, not as recoverable 1/k."
    )


if __name__ == "__main__":
    main()
