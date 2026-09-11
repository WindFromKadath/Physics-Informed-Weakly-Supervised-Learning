"""Blind-test batch generator (S1): batches 21-40 under the v1.3 spec.

In-repo regeneration of the external COMSOL_Link 1D pipeline
(``migration/datasets/README.md`` §9): the high-fidelity physics is the
already-validated ``hf_solve`` from ``verify_attribution.py`` (Kirchhoff BVP
with k(T) = k_true*(1+beta*(T-300)) and R_c(P) = R0*P^-0.7); the low-fidelity
engineering predictions use the closed form at k_nominal = 25.  Batches 1-20
are never touched.

Protocol (v1.3 spec):
  * per batch: reference 120 / acceptance 200 / engineering 200 samples,
    4D Latin-hypercube inputs over Q in [1e4, 1e5], h in [3, 50],
    T_inf in [273, 323], P in [0.5, 10];
  * k_true ~ U[15, 35] per batch; noise sigma = 4.493844638867134 (SNR=5);
  * seeds: base = 20260731 + batch*100; offsets k_true +0, reference +1,
    acceptance +2, engineering +3, noise_ref +5, noise_acc +6 (same formula
    as batches 1-20; batch ranges 21-40 use previously unused seeds);
  * y_obs = y_det + eps; engineering y_pred is deterministic;
  * dT_int = T_lower - T_upper at the interface (sign convention pinned by
    the batch_01 column check in verify_attribution.py).

Run:  uv run python migration/generate_blind_batches.py --start 21 --end 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))
sys.path.insert(0, str(ROOT / "migration"))

from pwl_migration.heat import (  # noqa: E402
    BETA_K,
    K_NOMINAL,
    R0,
    low_fidelity_temperature,
)
from verify_attribution import BOUNDS, hf_solve  # noqa: E402

SEED_BASE = 20260731  # v1.3 spec: set seed = SEED_BASE + batch*100 + offset
SIGMA = 4.493844638867134  # v1.3 spec value, identical across batches 1-20
K_TRUE_RANGE = (15.0, 35.0)
SIZES = {"reference": 120, "acceptance": 200, "engineering": 200}
DIAGONAL = 2.0  # sqrt(4) of the unit 4-cube, per the v1.3 coverage metric

REF_COMMENT = (
    "# units: Q[W/m^3], h[W/(m^2*K)], T_inf[K], P[MPa], y[K] "
    "(y=界面两侧平均温度); dT_int[K]=界面温跃(派生,P通道诊断)"
)
ENG_COMMENT = (
    "# units: Q[W/m^3], h[W/(m^2*K)], T_inf[K], P[MPa], y[K] "
    "(y=界面两侧平均温度)"
)
UNITS = "Q[W/m^3], h[W/(m^2*K)], T_inf[K], P[MPa], y[K]; dT_int[K]"


def batch_seeds(batch_index: int) -> dict[str, int]:
    """v1.3 seed protocol: base = SEED_BASE + batch*100, fixed offsets."""

    base = SEED_BASE + batch_index * 100
    return {
        "k_true": base,
        "reference": base + 1,
        "acceptance": base + 2,
        "engineering": base + 3,
        "calibration": base + 4,
        "noise_ref": base + 5,
        "noise_acc": base + 6,
        "noise_cal": base + 7,
    }


def lhs_inputs(n: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    return qmc.scale(sampler.random(n), BOUNDS[0], BOUNDS[1])


def _write_set(path: Path, comment: str, frame: pd.DataFrame) -> str:
    """Write one set CSV with the v1.3 units comment; return its sha256."""

    text = comment + "\n" + frame.to_csv(index=False)
    path.write_text(text, encoding="utf-8", newline="")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage(query: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Nearest-neighbour coverage in the unit 4-cube (v1.3 §coverage)."""

    unit_query = (query - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    unit_reference = (reference - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])
    distances, _ = cKDTree(unit_reference).query(unit_query)
    mean = float(np.mean(distances))
    p95 = float(np.quantile(distances, 0.95))
    return {
        "mean": mean,
        "p95": p95,
        "mean_pct_of_diag": mean / DIAGONAL,
        "p95_pct_of_diag": p95 / DIAGONAL,
        "frac_within_10pct": float(np.mean(distances < 0.10 * DIAGONAL)),
        "frac_within_15pct": float(np.mean(distances < 0.15 * DIAGONAL)),
    }


def generate_batch(batch_index: int, root: Path) -> dict:
    """Generate one blind batch (3 CSVs + metadata.json) under ``root``."""

    seeds = batch_seeds(batch_index)
    rng_k = np.random.default_rng(seeds["k_true"])
    k_true = float(rng_k.uniform(*K_TRUE_RANGE))

    inputs = {
        name: lhs_inputs(n, seeds[name]) for name, n in SIZES.items()
    }

    frames: dict[str, pd.DataFrame] = {}
    solver_diag: dict[str, float] = {}
    for name in ("reference", "acceptance"):
        x = inputs[name]
        q, h, t_inf, p = x.T
        sol = hf_solve(q, h, t_inf, p, k_true)
        noise = np.random.default_rng(seeds[f"noise_{name[:3]}"]).normal(
            0.0, SIGMA, len(x)
        )
        solver_diag[f"{name}_max_abs_f1_K"] = sol["max_abs_f1_K"]
        solver_diag[f"{name}_max_abs_f2_Wm2"] = sol["max_abs_f2_Wm2"]
        solver_diag[f"{name}_newton_iterations"] = sol["iterations"]
        frames[name] = pd.DataFrame(
            {
                "Q": q,
                "h": h,
                "T_inf": t_inf,
                "P": p,
                "y_obs": sol["y"] + noise,
                "dT_int": sol["t_lower"] - sol["t_upper"],
            }
        )

    x_eng = inputs["engineering"]
    y_lf = low_fidelity_temperature(x_eng[:, :3], np.array([1.0 / K_NOMINAL]))
    frames["engineering"] = pd.DataFrame(
        {
            "Q": x_eng[:, 0],
            "h": x_eng[:, 1],
            "T_inf": x_eng[:, 2],
            "P": x_eng[:, 3],
            "y_pred": y_lf,
        }
    )

    batch_dir = root / f"batch_{batch_index:02d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    sha256 = {
        "reference_measurements.csv": _write_set(
            batch_dir / "reference_measurements.csv",
            REF_COMMENT,
            frames["reference"],
        ),
        "acceptance_tests.csv": _write_set(
            batch_dir / "acceptance_tests.csv",
            REF_COMMENT,
            frames["acceptance"],
        ),
        "engineering_predictions.csv": _write_set(
            batch_dir / "engineering_predictions.csv",
            ENG_COMMENT,
            frames["engineering"],
        ),
    }

    coverage = {
        "direction": "参考/验收 → 工程",
        "reference_to_engineering": _coverage(
            inputs["reference"], inputs["engineering"]
        ),
        "acceptance_to_engineering": _coverage(
            inputs["acceptance"], inputs["engineering"]
        ),
        "threshold_p95": 0.3,
    }
    coverage["pass"] = bool(
        coverage["reference_to_engineering"]["p95"] < 0.3
        and coverage["acceptance_to_engineering"]["p95"] < 0.3
    )

    metadata = {
        "spec_version": "v1.3 (1D)",
        "batch_index": batch_index,
        "solver": "1D: 高保真 Kirchhoff BVP + 低保真闭式解 "
        "(in-repo verify_attribution.hf_solve; 与 comsol_platform/solver1d.py 同源)，"
        "无离散误差",
        "output_definition": "y = 界面两侧平均温度 (T1+T2)/2 = T1(L/2) − dT_int/2",
        "k_true": k_true,
        "sigma": SIGMA,
        "sigma_definition": "试验综合重复性误差（SNR=5）",
        "sigma_low_option": 0.5,
        "R0": R0,
        "beta_k": BETA_K,
        "k_nominal": K_NOMINAL,
        "comsol_version": "N/A（一维求解器，未使用 COMSOL）",
        "seeds": seeds,
        "sizes": dict(SIZES),
        "success_rate": {"reference": 1.0, "acceptance": 1.0,
                         "engineering": 1.0},
        "failures": {"reference": [], "acceptance": [], "engineering": []},
        "nested_reference": "前 k 行即嵌套子集，k ∈ {10,20,...,120}（固定顺序，种子唯一确定）",
        "coverage": coverage,
        "derived_columns": "reference/acceptance 含 dT_int（界面温跃，P 通道诊断）",
        "sha256": sha256,
        "units": UNITS,
        "solver_diagnostics": solver_diag,
        "provenance": "blind-test S1 extension: in-repo generator "
        "migration/generate_blind_batches.py; batches 1-20 untouched",
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (batch_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=21)
    parser.add_argument("--end", type=int, default=40)
    parser.add_argument(
        "--root",
        default=str(ROOT / "migration" / "datasets"),
        help="Output root holding batch_XX directories.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    for batch_index in range(args.start, args.end + 1):
        target = root / f"batch_{batch_index:02d}"
        if target.exists():
            raise SystemExit(
                f"{target} already exists; refusing to overwrite a batch."
            )
        metadata = generate_batch(batch_index, root)
        print(
            f"[generate] batch_{batch_index:02d} "
            f"k_true={metadata['k_true']:.6f} "
            f"coverage_pass={metadata['coverage']['pass']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
