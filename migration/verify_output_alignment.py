"""M4: read-only verification of the output-alignment diagnostic (2026-08-05).

Reproduces the M0-M3 diagnostic table of
``migration/reports/迁移后续工作/PWL物理模型输出对齐与适用性说明.md`` §3
against the frozen 20-batch datasets, without touching any frozen config or
result directory.

Protocol (report §3.1):
  * per batch, the first 84 rows of ``reference_measurements.csv`` form the
    fit set (``n_labeled = 120`` x ``train_fraction = 0.7``);
  * the 200 rows of ``acceptance_tests.csv`` form the independent eval set;
    acceptance labels are never used to fit anything;
  * the low-fidelity output is the closed-form model at ``k_nominal = 25``;
  * alignment coefficients are refit independently per batch.

Diagnostic arms (report §3.2), all plain OLS on the fit set:
  M0: y_hat = y_LF
  M1: y_hat = a + b*y_LF
  M2: y_hat = a + b*y_LF + c*P^(-0.7)
  M3: y_hat = a + b*y_LF + c1*q_int*P^(-0.7)
                           + c2*q_int/(1 + h*L/k_nom)*P^(-0.7)

Anchors checked (report §0/§3.3, plus the frozen v3 baseline card):
  mean RMSE M0..M3 = 15.276 / 8.026 / 6.214 / 4.763 K,
  mean affine map a = -447.9, b = 2.526 (i.e. dT_HF ~ -0.6 + 2.53*dT_LF),
  noise floor sigma = 4.494 K, formal PWL (n=120) RMSE = 5.351 K.

Run:  uv run python migration/verify_output_alignment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))

from pwl_migration.heat import (  # noqa: E402
    K_NOMINAL,
    L,
    T0,
    low_fidelity_interface_flux,
    low_fidelity_temperature,
)

DATASETS = ROOT / "migration" / "datasets"
# The report's "formal PWL (20 batches)" number lives in the b20 results; the
# heat_v3_qint_min baseline card only covers 10 batches.
RESULTS = ROOT / "migration" / "results" / "heat_v3_qint_min_b20"

N_FIT = 84  # 120 labeled x 0.7 train fraction (report §3.1)
THETA_NOM = np.array([1.0 / K_NOMINAL])

# Anchors from the report (§0 summary table, §3.3 affine map, §4 noise floor).
ANCHORS = {
    "M0": 15.276,
    "M1": 8.026,
    "M2": 6.214,
    "M3": 4.763,
    "affine_a": -447.9,
    "affine_b": 2.526,
    "sigma": 4.494,
    "pwl_formal": 5.351,
}
TOL = 0.01  # K (or coefficient units); report values carry 3 decimals


def design_matrix(arm: str, y_lf: np.ndarray, x_ph: np.ndarray,
                  x_pr: np.ndarray) -> np.ndarray:
    """Regressor block shared by fit and eval for arms M1-M3."""

    if arm == "M1":
        return np.column_stack((np.ones_like(y_lf), y_lf))
    if arm == "M2":
        return np.column_stack((np.ones_like(y_lf), y_lf, x_pr[:, 0] ** -0.7))
    if arm == "M3":
        s = x_pr[:, 0] ** -0.7
        q_int = low_fidelity_interface_flux(x_ph, K_NOMINAL)
        modulation = 1.0 + x_ph[:, 1] * L / K_NOMINAL
        return np.column_stack(
            (np.ones_like(y_lf), y_lf, q_int * s, q_int / modulation * s)
        )
    raise ValueError(arm)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def main() -> None:
    rows: list[dict[str, float]] = []
    for batch_dir in sorted(DATASETS.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        reference = pd.read_csv(batch_dir / "reference_measurements.csv",
                                comment="#")
        acceptance = pd.read_csv(batch_dir / "acceptance_tests.csv",
                                 comment="#")
        fit = reference.iloc[:N_FIT]

        x_fit = fit[["Q", "h", "T_inf"]].to_numpy()
        p_fit = fit[["P"]].to_numpy()
        y_fit = fit["y_obs"].to_numpy()
        x_acc = acceptance[["Q", "h", "T_inf"]].to_numpy()
        p_acc = acceptance[["P"]].to_numpy()
        y_acc = acceptance["y_obs"].to_numpy()

        y_lf_fit = low_fidelity_temperature(x_fit, THETA_NOM)
        y_lf_acc = low_fidelity_temperature(x_acc, THETA_NOM)

        row: dict[str, float] = {"batch": batch_dir.name}
        row["M0"] = rmse(y_lf_acc, y_acc)
        for arm in ("M1", "M2", "M3"):
            coef, *_ = np.linalg.lstsq(
                design_matrix(arm, y_lf_fit, x_fit, p_fit), y_fit, rcond=None
            )
            pred = design_matrix(arm, y_lf_acc, x_acc, p_acc) @ coef
            row[arm] = rmse(pred, y_acc)
            if arm == "M1":
                row["affine_a"], row["affine_b"] = float(coef[0]), float(coef[1])
        rows.append(row)

    table = pd.DataFrame(rows)
    mean = table.drop(columns="batch").mean()

    # Formal PWL anchor: frozen baseline results, n_labeled = 120, 20 repeats.
    results = pd.read_csv(RESULTS / "results.csv")
    pwl = results[(results["model"] == "PWL") & (results["n_labeled"] == 120)]
    pwl_rmse = float(pwl["rmse"].mean())

    # Noise floor from the batch metadata (identical across batches, v1.3 spec).
    sigmas = {
        json.loads((d / "metadata.json").read_text(encoding="utf-8"))["sigma"]
        for d in sorted(DATASETS.glob("batch_*"))
        if d.is_dir()
    }
    sigma = float(np.mean(sorted(sigmas)))

    pd.set_option("display.float_format", lambda v: f"{v:9.3f}")
    print("Per-batch acceptance RMSE (K) and M1 affine coefficients:")
    print(table.to_string(index=False))
    print()

    checks = [
        ("M0 identity", mean["M0"], ANCHORS["M0"]),
        ("M1 affine", mean["M1"], ANCHORS["M1"]),
        ("M2 affine + P^-0.7", mean["M2"], ANCHORS["M2"]),
        ("M3 affine + q_int terms", mean["M3"], ANCHORS["M3"]),
        ("M1 intercept a", mean["affine_a"], ANCHORS["affine_a"]),
        ("M1 slope b", mean["affine_b"], ANCHORS["affine_b"]),
        ("noise floor sigma", sigma, ANCHORS["sigma"]),
        ("PWL formal (n=120)", pwl_rmse, ANCHORS["pwl_formal"]),
    ]
    print(f"{'check':<28}{'reproduced':>12}{'report':>10}{'diff':>9}  verdict")
    ok_all = True
    for name, actual, expected in checks:
        ok = abs(actual - expected) <= TOL
        ok_all &= ok
        print(f"{name:<28}{actual:>12.3f}{expected:>10.3f}"
              f"{actual - expected:>+9.3f}  {'PASS' if ok else 'FAIL'}")

    # Temperature-rise reparametrization of the mean affine map (report §3.3).
    a, b = mean["affine_a"], mean["affine_b"]
    alpha = a + (b - 1.0) * T0
    print()
    print(f"Temperature-rise form: dT_HF = {alpha:+.3f} + {b:.3f} * dT_LF "
          f"(report: -0.6 + 2.53 * dT_LF)")
    print(f"PWL formal repeats used: {len(pwl)} rows "
          f"(expect 20 batches x 1 repeat at n=120)")
    print()
    print("OVERALL:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
