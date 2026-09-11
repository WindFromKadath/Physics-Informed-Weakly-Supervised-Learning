"""Blind-batch data acceptance (S1): QA gate for batches 21-40.

Two layers:

  * Section 0 (physics regression): pool y_obs - hf_solve(inputs, k_true)
    over the EXISTING batches 1-20 (reference + acceptance sets).  If the
    in-repo solver reproduces the external pipeline's physics, the pooled
    residual has mean ~ 0 and std ~ sigma.  This uses only long-frozen data.
  * Section 1 (new batches): the v1.3 §5.3/§8 checks — finiteness, sha256,
    shapes, k_true range, Newton residuals, noise statistics, dT_int and
    engineering-set consistency, dual-fidelity correlation, set isolation,
    coverage, SNR.

Hard gates: sections 1.1-1.8, 1.11.  Correlation and SNR are data attributes
(reported, soft-gated like the original pipeline's later anchor history).

Run:  uv run python migration/accept_blind_batches.py --start 21 --end 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))
sys.path.insert(0, str(ROOT / "migration"))

from pwl_migration.heat import (  # noqa: E402
    K_NOMINAL,
    low_fidelity_temperature,
)
from verify_attribution import hf_solve  # noqa: E402
from generate_blind_batches import SIGMA, batch_seeds  # noqa: E402

DATASETS = ROOT / "migration" / "datasets"
EXPECTED_SEED_OFFSETS = {
    "k_true": 0, "reference": 1, "acceptance": 2, "engineering": 3,
    "calibration": 4, "noise_ref": 5, "noise_acc": 6, "noise_cal": 7,
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def physics_regression(batches: range) -> dict[str, float]:
    """Section 0: pooled residual of hf_solve against frozen observations."""

    residuals: list[np.ndarray] = []
    for batch in batches:
        batch_dir = DATASETS / f"batch_{batch:02d}"
        meta = json.loads(
            (batch_dir / "metadata.json").read_text(encoding="utf-8")
        )
        k_true = float(meta["k_true"])
        for name in ("reference_measurements.csv", "acceptance_tests.csv"):
            frame = _read(batch_dir / name)
            sol = hf_solve(
                frame["Q"].to_numpy(),
                frame["h"].to_numpy(),
                frame["T_inf"].to_numpy(),
                frame["P"].to_numpy(),
                k_true,
            )
            residuals.append(sol["y"] - frame["y_obs"].to_numpy())
    pooled = np.concatenate(residuals)
    return {
        "n": int(pooled.size),
        "resid_mean": float(np.mean(pooled)),
        "resid_std": float(np.std(pooled)),
        "mse": float(np.mean(pooled**2)),
        "sigma": SIGMA,
    }


def check_batch(batch_index: int) -> dict:
    batch_dir = DATASETS / f"batch_{batch_index:02d}"
    meta = json.loads((batch_dir / "metadata.json").read_text(encoding="utf-8"))
    reference = _read(batch_dir / "reference_measurements.csv")
    acceptance = _read(batch_dir / "acceptance_tests.csv")
    engineering = _read(batch_dir / "engineering_predictions.csv")
    k_true = float(meta["k_true"])

    checks: dict[str, bool] = {}
    details: dict[str, float | str] = {}

    # 1.1 files finite, shapes and columns
    shapes_ok = (
        reference.shape == (120, 6)
        and acceptance.shape == (200, 6)
        and engineering.shape == (200, 5)
    )
    finite_ok = all(
        np.all(np.isfinite(frame.to_numpy(dtype=float)))
        for frame in (reference, acceptance, engineering)
    )
    checks["shapes_and_finite"] = bool(shapes_ok and finite_ok)

    # 1.2 sha256 fingerprints match metadata
    stored = meta.get("sha256", {})
    checks["sha256"] = all(
        hashlib.sha256((batch_dir / name).read_bytes()).hexdigest()
        == stored.get(name)
        for name in (
            "reference_measurements.csv",
            "acceptance_tests.csv",
            "engineering_predictions.csv",
        )
    )

    # 1.3 k_true in range; seeds follow the v1.3 protocol
    checks["k_true_range"] = 15.0 <= k_true <= 35.0
    seeds = meta["seeds"]
    expected = batch_seeds(batch_index)
    checks["seed_protocol"] = all(
        int(seeds[key]) == expected[key] for key in EXPECTED_SEED_OFFSETS
    )

    # 1.4 Newton solver diagnostics recorded at generation time
    diag = meta.get("solver_diagnostics", {})
    checks["newton_residuals"] = all(
        float(diag.get(f"{name}_max_abs_f1_K", 1.0)) < 1e-8
        for name in ("reference", "acceptance")
    )

    # 1.5 noise statistics: y_obs - deterministic HF solve
    resid_stats = {}
    for name, frame in (("reference", reference), ("acceptance", acceptance)):
        sol = hf_solve(
            frame["Q"].to_numpy(), frame["h"].to_numpy(),
            frame["T_inf"].to_numpy(), frame["P"].to_numpy(), k_true,
        )
        resid = frame["y_obs"].to_numpy() - sol["y"]
        resid_stats[name] = (float(np.mean(resid)), float(np.std(resid)))
        # dT_int column must equal T_lower - T_upper (sign convention).
        jump = sol["t_lower"] - sol["t_upper"]
        checks[f"dT_int_{name}"] = bool(
            np.max(np.abs(jump - frame["dT_int"].to_numpy())) < 1e-9
        )
    details["noise_ref_mean"], details["noise_ref_std"] = resid_stats["reference"]
    details["noise_acc_mean"], details["noise_acc_std"] = resid_stats["acceptance"]
    checks["noise_calibration"] = all(
        abs(mean) < 1.0 and 3.5 < std < 5.5
        for mean, std in resid_stats.values()
    )

    # 1.6 engineering predictions equal the LF closed form at k_nominal
    y_lf = low_fidelity_temperature(
        engineering[["Q", "h", "T_inf"]].to_numpy(), np.array([1.0 / K_NOMINAL])
    )
    checks["engineering_lf_match"] = bool(
        np.max(np.abs(y_lf - engineering["y_pred"].to_numpy())) < 1e-9
    )

    # 1.7 dual-fidelity correlation (reference set, noisy) — data attribute
    y_lf_ref = low_fidelity_temperature(
        reference[["Q", "h", "T_inf"]].to_numpy(), np.array([1.0 / K_NOMINAL])
    )
    correlation = float(
        np.corrcoef(reference["y_obs"].to_numpy(), y_lf_ref)[0, 1]
    )
    details["dual_fidelity_correlation"] = correlation
    checks["correlation_range"] = 0.6 <= correlation <= 0.95

    # 1.8 isolation: no exact shared input rows across the three sets
    def key_set(frame: pd.DataFrame) -> set[tuple[float, ...]]:
        return set(map(tuple, frame[["Q", "h", "T_inf", "P"]].to_numpy()))

    ref_keys, acc_keys, eng_keys = (
        key_set(reference), key_set(acceptance), key_set(engineering)
    )
    checks["isolation"] = not (
        ref_keys & acc_keys or ref_keys & eng_keys or acc_keys & eng_keys
    )

    # 1.9 coverage gate recorded at generation time
    checks["coverage_pass"] = bool(meta["coverage"]["pass"])
    details["coverage_p95_ref"] = float(
        meta["coverage"]["reference_to_engineering"]["p95"]
    )
    details["coverage_p95_acc"] = float(
        meta["coverage"]["acceptance_to_engineering"]["p95"]
    )

    # 1.10 SNR attribute: Var(y_det)/sigma^2 on the reference set
    sol_ref = hf_solve(
        reference["Q"].to_numpy(), reference["h"].to_numpy(),
        reference["T_inf"].to_numpy(), reference["P"].to_numpy(), k_true,
    )
    details["snr_reference"] = float(np.var(sol_ref["y"]) / SIGMA**2)

    checks["all_hard"] = all(
        checks[key]
        for key in (
            "shapes_and_finite", "sha256", "k_true_range", "seed_protocol",
            "newton_residuals", "dT_int_reference", "dT_int_acceptance",
            "noise_calibration", "engineering_lf_match", "isolation",
            "coverage_pass",
        )
    )
    return {
        "batch": batch_index,
        "k_true": k_true,
        "checks": checks,
        "details": details,
        "pass": checks["all_hard"] and checks["correlation_range"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=21)
    parser.add_argument("--end", type=int, default=40)
    args = parser.parse_args()

    print("== 0. physics regression on existing batches 1-20 ==")
    regression = physics_regression(range(1, 21))
    print(
        f"  n={regression['n']}, resid mean={regression['resid_mean']:+.4f} K,"
        f" std={regression['resid_std']:.4f} K (sigma={regression['sigma']:.4f}),"
        f" MSE={regression['mse']:.4f}"
    )
    regression_ok = (
        abs(regression["resid_mean"]) < 0.2
        and abs(regression["resid_std"] - SIGMA) < 0.1
    )
    print(f"  physics regression: {'PASS' if regression_ok else 'FAIL'}\n")

    print(f"== 1. blind batches {args.start}-{args.end} ==")
    reports = [check_batch(b) for b in range(args.start, args.end + 1)]
    for report in reports:
        status = "PASS" if report["pass"] else "FAIL"
        details = report["details"]
        print(
            f"  batch_{report['batch']:02d} [{status}] "
            f"k_true={report['k_true']:.3f} "
            f"corr={details['dual_fidelity_correlation']:.3f} "
            f"noise_std(ref/acc)={details['noise_ref_std']:.3f}/"
            f"{details['noise_acc_std']:.3f} "
            f"snr={details['snr_reference']:.2f}"
        )
        failed = [k for k, v in report["checks"].items() if not v]
        if failed:
            print(f"    failed checks: {failed}")

    n_pass = sum(report["pass"] for report in reports)
    print(f"\n  batches passed: {n_pass}/{len(reports)}")
    output = {
        "physics_regression_batches_1_20": regression,
        "physics_regression_pass": regression_ok,
        "batches": reports,
        "n_pass": n_pass,
        "n_total": len(reports),
    }
    out_path = DATASETS / "acceptance_report_blind_21_40.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  report written to {out_path}")
    ok = regression_ok and n_pass == len(reports)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
