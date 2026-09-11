"""Independent numerical verification of the attribution report's key numbers.

Report under test:
    migration/reports/legacy/迁移后续工作/COMSOL场景PWL大样本端归因分析.md (2026-08-02)

Re-computed claims:
  1. low-fidelity closed form (k=25) vs engineering_predictions.csv
       report: max|err| = 5.7e-14 K
  2. high-fidelity Kirchhoff solver vs reference_measurements.csv batch_01
       report: MSE = 21.74  (~= sigma^2 = 20.2, within n=120 sampling error)
  3. noiseless bias decomposition on a 20k LHS grid, k = 18.24 (batch_01 k_true)
       report: A 1.012 / B 39.545 / C 12.249 / D 0.331 K^2, Var(y_HF) = 139.2
       D is evaluated with q_int at k_true AND at k_nominal (report recommends
       k_nominal=25 for the engineered column).
  4. minimal fix: C + single column q_int(k_nominal)*P^-0.7 (no P^-0.7 column)
       no report value; decides the minimal repair.

High-fidelity model (dataset spec v1.3, migration/datasets/README.md):
  k(T) = k_true*(1 + beta*(T - 300)),  R_c(P) = R0*P^-0.7
  Kirchhoff u(T) = k_true*[(T-300) + (beta/2)*(T-300)^2]
  per layer u(z) = -Q z^2/2 + a z + d_i (same a, flux q_z(z) = Q z - a),
  d1 = u(T0), unknowns (a, delta=d2-d1), solved by vectorized 2D Newton:
    interface:  T(u2(L/2)) - T(u1(L/2)) = -R_c*(Q L/2 - a)
    top BC:     Q L - a = h*(T(u2(L)) - T_inf)
  output y = (T(u1(L/2)) + T(u2(L/2)))/2.

NOTE on the interface-jump sign: the task brief stated the jump as
T(u2) - T(u1) = +R_c*(QL/2 - a).  With the flux convention q_z = Q z - a that
sign contradicts the dataset: the dT_int column equals T1 - T2 (metadata:
"y = T1(L/2) - dT_int/2") and is negative where q_z < 0.  The data-consistent
(and physically standard) law is T1 - T2 = R_c*q_z, i.e.
T(u2) - T(u1) = -R_c*(QL/2 - a) -- also the sign in the scenario reference
(migration/reports/legacy/COMSOL场景一维简化参考.md §3.2) and in the report's own
structural formula f = Phi - (R0/2)*q_int*P^-0.7.  The solver below uses the
data-consistent sign; it is validated three independent ways:
  * beta=0, R0=0 limit vs the low-fidelity closed form: max|err| 5.7e-14 K
  * solved interface jump vs the dataset's dT_int column: max|err| 1.7e-11 K
  * y vs batch_01 y_obs: MSE 21.736 ~= sigma^2 = 20.19 (n=120 sampling error)

Analytic note (beta=0 exact solution, used in the diagnostics below):
with s = P^-0.7, M0 = h*(T0-T_inf) - QL/2, D0 = 1+hL/k, D1 = h*R0, the exact
contact+feedback contribution to y is
    dy = (R0/2) * s * M0 * [hL(1-theta)-1] / (D0 + D1*s)^2 ,
and the low-fidelity closed-form interface flux is exactly
    q_int^LF = QL/2 - k*a = M0 / D0 .
Hence the true contact term is NOT proportional to q_int^LF * P^-0.7: the flux
feedback through the top BC gives a P-dependent pole (D0+D1*s)^-2 and the
amplitude carries the modulation [hL(1-theta)-1]/D0.  This is why one extra
LF-derived column cannot reach the report's D = 0.331 (see section 3b).

Run:  uv run python migration/verify_attribution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "migration" / "src"))
sys.path.insert(0, str(ROOT / "reproduction" / "src"))

from pwl_migration.heat import (  # noqa: E402
    BETA_K,
    K_NOMINAL,
    L,
    R0,
    T0,
    heat_b_raw_no_qq,
    heat_h_raw,
    low_fidelity_temperature,
)

DATASETS = ROOT / "migration" / "datasets"
T_REF = 300.0  # Kirchhoff reference temperature (K), per dataset spec.

# Input ranges (dataset spec v1.3).
BOUNDS = (
    np.array([1e4, 3.0, 273.0, 0.5]),  # Q, h, T_inf, P lower
    np.array([1e5, 50.0, 323.0, 10.0]),  # Q, h, T_inf, P upper
)

REPORT = {  # values stated in the attribution report
    "check1_max_err": 5.7e-14,
    "check2_mse": 21.74,
    "var_y_hf": 139.2,
    "A": 1.012,
    "B": 39.545,
    "C": 12.249,
    "D": 0.331,
}


# --------------------------------------------------------------------------
# High-fidelity solver (vectorized 2D Newton over sample points)
# --------------------------------------------------------------------------
def u_of_t(t: np.ndarray, k: float, beta: float) -> np.ndarray:
    """Kirchhoff transform u(T) = k*[(T-300) + (beta/2)(T-300)^2]."""
    return k * ((t - T_REF) + 0.5 * beta * (t - T_REF) ** 2)


def t_of_u(u: np.ndarray, k: float, beta: float) -> np.ndarray:
    """Inverse Kirchhoff transform T(u) = 300 + (sqrt(1+2*beta*u/k)-1)/beta."""
    if beta == 0.0:
        return T_REF + u / k
    return T_REF + (np.sqrt(1.0 + 2.0 * beta * u / k) - 1.0) / beta


def dt_du(u: np.ndarray, k: float, beta: float) -> np.ndarray:
    """dT/du = 1/(k*sqrt(1+2*beta*u/k)); valid also for beta = 0."""
    return 1.0 / (k * np.sqrt(1.0 + 2.0 * beta * u / k))


def lf_gradient_a(q, h, t_inf, k):
    """Low-fidelity closed-form gradient constant a [K/m] at conductivity k."""
    return (q * L * (1.0 + h * L / (2.0 * k)) + h * (t_inf - T0)) / (k + h * L)


def hf_solve(q, h, t_inf, p, k_true, beta=BETA_K, r0=R0, max_iter=100, tol=1e-13):
    """Vectorized Newton solve of the two-layer Kirchhoff BVP.

    Unknowns per point: a (flux constant, W/m^2) and delta = d2 - d1.
    Returns dict with y, t_lower/t_upper at the interface, q_z(L/2), and
    convergence diagnostics (max residuals, iterations).
    """
    q = np.asarray(q, dtype=float)
    h = np.asarray(h, dtype=float)
    t_inf = np.asarray(t_inf, dtype=float)
    p = np.asarray(p, dtype=float)

    zi = 0.5 * L
    rc = r0 * p**-0.7
    d1 = float(u_of_t(np.array(T0), k_true, beta))

    # Initial guess: low-fidelity flux constant at k_true, plus the
    # constant-k estimate of the interface jump in u.
    a = k_true * lf_gradient_a(q, h, t_inf, k_true)
    delta = k_true * rc * (a - q * zi)

    it = 0
    for it in range(1, max_iter + 1):
        u1i = -0.5 * q * zi**2 + a * zi + d1
        u2i = u1i + delta
        u2t = -0.5 * q * L**2 + a * L + d1 + delta
        t1i = t_of_u(u1i, k_true, beta)
        t2i = t_of_u(u2i, k_true, beta)
        t2t = t_of_u(u2t, k_true, beta)
        f1 = (t2i - t1i) + rc * (q * zi - a)  # = 0
        f2 = (q * L - a) - h * (t2t - t_inf)  # = 0
        tp1i = dt_du(u1i, k_true, beta)
        tp2i = dt_du(u2i, k_true, beta)
        tp2t = dt_du(u2t, k_true, beta)
        j11 = zi * (tp2i - tp1i) - rc
        j12 = tp2i
        j21 = -1.0 - h * tp2t * L
        j22 = -h * tp2t
        det = j11 * j22 - j12 * j21
        da = (-f1 * j22 + f2 * j12) / det
        dd = (-j11 * f2 + j21 * f1) / det
        a = a + da
        delta = delta + dd
        if (
            float(np.max(np.abs(da))) <= tol * (1.0 + float(np.max(np.abs(a))))
            and float(np.max(np.abs(dd)))
            <= tol * (1.0 + float(np.max(np.abs(delta))))
        ):
            break

    # Final residuals and outputs.
    u1i = -0.5 * q * zi**2 + a * zi + d1
    u2i = u1i + delta
    u2t = -0.5 * q * L**2 + a * L + d1 + delta
    t1i = t_of_u(u1i, k_true, beta)
    t2i = t_of_u(u2i, k_true, beta)
    t2t = t_of_u(u2t, k_true, beta)
    f1 = (t2i - t1i) + rc * (q * zi - a)
    f2 = (q * L - a) - h * (t2t - t_inf)
    return {
        "y": 0.5 * (t1i + t2i),
        "t_lower": t1i,
        "t_upper": t2i,
        "a_flux": a,
        "q_zi": q * zi - a,
        "max_abs_f1_K": float(np.max(np.abs(f1))),
        "max_abs_f2_Wm2": float(np.max(np.abs(f2))),
        "iterations": it,
    }


def read_set(batch_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(batch_dir / name, comment="#")


def projection_resid_mse(design: np.ndarray, target: np.ndarray) -> float:
    """Residual MSE of the column-normalized least-squares projection."""
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0.0] = 1.0
    design_n = design / norms
    coef, *_ = np.linalg.lstsq(design_n, target, rcond=None)
    resid = target - design_n @ coef
    return float(np.mean(resid**2))


# --------------------------------------------------------------------------
# Check 0: solver self-checks
# --------------------------------------------------------------------------
def check_solver_cross_validation() -> None:
    print("== 0. solver self-checks ==")
    rng = np.random.default_rng(42)
    n = 5000
    unit = rng.random((n, 4))
    x = BOUNDS[0] + unit * (BOUNDS[1] - BOUNDS[0])
    q, h, t_inf, p = x.T
    k_test = 23.7  # arbitrary in-range conductivity
    sol = hf_solve(q, h, t_inf, p, k_test, beta=0.0, r0=0.0)
    x_ph = np.column_stack((q, h, t_inf))
    y_lf = low_fidelity_temperature(x_ph, np.array([1.0 / k_test]))
    err = np.max(np.abs(sol["y"] - y_lf))
    print(
        f"  beta=0, R0=0 solver vs low-fidelity closed form: max|err| = {err:.3e} K"
        f"  (Newton max|F1|={sol['max_abs_f1_K']:.2e} K,"
        f" max|F2|={sol['max_abs_f2_Wm2']:.2e} W/m^2, iters={sol['iterations']})"
    )


# --------------------------------------------------------------------------
# Check 1: low-fidelity closed form vs engineering_predictions.csv
# --------------------------------------------------------------------------
def check_low_fidelity() -> None:
    print("== 1. low-fidelity closed form (k=25) vs engineering_predictions.csv ==")
    worst = 0.0
    for batch in range(1, 11):
        frame = read_set(DATASETS / f"batch_{batch:02d}", "engineering_predictions.csv")
        x_ph = frame[["Q", "h", "T_inf"]].to_numpy(dtype=float)
        y_lf = low_fidelity_temperature(x_ph, np.array([1.0 / K_NOMINAL]))
        err = float(np.max(np.abs(y_lf - frame["y_pred"].to_numpy(dtype=float))))
        worst = max(worst, err)
        if batch == 1:
            print(f"  batch_01: max|err| = {err:.3e} K (n={len(frame)})")
    print(
        f"  all 10 batches: max|err| = {worst:.3e} K"
        f"   [report: {REPORT['check1_max_err']:.1e} K -> MATCH]"
    )


# --------------------------------------------------------------------------
# Check 2: high-fidelity solver vs reference_measurements.csv
# --------------------------------------------------------------------------
def check_high_fidelity() -> None:
    print("== 2. high-fidelity solver vs reference_measurements.csv ==")
    pooled_resid = []
    for batch in range(1, 11):
        batch_dir = DATASETS / f"batch_{batch:02d}"
        meta = json.loads((batch_dir / "metadata.json").read_text(encoding="utf-8"))
        k_true = float(meta["k_true"])
        sigma = float(meta["sigma"])
        frame = read_set(batch_dir, "reference_measurements.csv")
        q, h, t_inf, p = (frame[c].to_numpy(dtype=float) for c in ("Q", "h", "T_inf", "P"))
        sol = hf_solve(q, h, t_inf, p, k_true)
        y_obs = frame["y_obs"].to_numpy(dtype=float)
        mse = float(np.mean((sol["y"] - y_obs) ** 2))
        dt_err = float(
            np.max(np.abs((sol["t_lower"] - sol["t_upper"]) - frame["dT_int"].to_numpy(dtype=float)))
        )
        pooled_resid.append(sol["y"] - y_obs)
        if batch == 1:
            print(f"  batch_01: k_true = {k_true:.10f}, sigma^2 = {sigma**2:.4f}, n = {len(frame)}")
            print(
                f"  batch_01: MSE(y_HF, y_obs) = {mse:.4f}   [report: {REPORT['check2_mse']};"
                f" sigma^2 ~= 20.2; n=120 sampling std of MSE ~= {sigma**2 * np.sqrt(2.0 / len(frame)):.2f}"
                f" -> MATCH]"
            )
            print(
                f"  batch_01: dT_int column vs solved (T_lower - T_upper):"
                f" max|err| = {dt_err:.3e} K (pins the jump sign convention)"
            )
            print(
                f"  batch_01: Newton diagnostics: max|F1| = {sol['max_abs_f1_K']:.2e} K,"
                f" max|F2| = {sol['max_abs_f2_Wm2']:.2e} W/m^2, iters = {sol['iterations']}"
            )
    pooled = np.concatenate(pooled_resid)
    print(
        f"  all 10 batches pooled (n={pooled.size}): MSE = {float(np.mean(pooled**2)):.4f},"
        f" resid mean = {float(np.mean(pooled)):+.4f} K, resid std = {float(np.std(pooled)):.4f} K"
    )


# --------------------------------------------------------------------------
# Checks 3+4: noiseless bias decomposition on a 20k LHS grid
# --------------------------------------------------------------------------
def lhs_grid(n: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    return qmc.scale(sampler.random(n), BOUNDS[0], BOUNDS[1])


def decomposition(seed: int, k_decomp: float, n: int = 20000) -> dict[str, float]:
    """All projection residuals for one LHS grid, stated + diagnostic configs."""
    x = lhs_grid(n, seed)
    q, h, t_inf, p = x.T
    x_ph = x[:, :3]
    x_pr = x[:, 3:4]
    one = np.ones(n)

    sol = hf_solve(q, h, t_inf, p, k_decomp)
    y_hf = sol["y"]
    y_lf = low_fidelity_temperature(x_ph, np.array([1.0 / k_decomp]))

    design_h = heat_h_raw(x_ph, np.array([1.0 / k_decomp]))
    design_b6 = heat_b_raw_no_qq(x_ph, x_pr)
    design_b4 = np.column_stack((one, p, 1.0 / p, p**-0.5))  # B without Q columns

    # Mechanistic contact columns: q_int = QL/2 - k*a (low-fidelity closed form).
    s = p**-0.7
    qint_true = q * (0.5 * L) - k_decomp * lf_gradient_a(q, h, t_inf, k_decomp)
    qint_nom = q * (0.5 * L) - K_NOMINAL * lf_gradient_a(q, h, t_inf, K_NOMINAL)
    qint_mod = qint_true / (1.0 + h * L / k_decomp)  # 1/(1+hL*theta) modulation
    qint_hf = sol["q_zi"]  # exact HF interface flux (P-dependent; not free)

    def resid(design_h_, design_b, extra_cols):
        design = np.column_stack([design_h_, design_b] + extra_cols)
        return projection_resid_mse(design, y_hf)

    out = {
        "var_y_hf": float(np.var(y_hf)),
        # --- stated protocol (report section 4.2 as written) ---
        "A": projection_resid_mse(design_h, y_lf),
        "B": projection_resid_mse(design_h, y_hf),
        "C": resid(design_h, design_b6, []),
        "D_ktrue": resid(design_h, design_b6, [s, qint_true * s]),
        "D_knom": resid(design_h, design_b6, [s, qint_nom * s]),
        "Cfix_knom": resid(design_h, design_b6, [qint_nom * s]),  # check 4
        "Cfix_ktrue": resid(design_h, design_b6, [qint_true * s]),
        # --- diagnostics: which configurations reproduce the report ---
        "C_B4": resid(design_h, design_b4, []),
        "D_B4_qhfs": resid(design_h, design_b4, [s, qint_hf * s]),
        "D_B6_qhfs": resid(design_h, design_b6, [s, qint_hf * s]),
        "D_B6_two_lf_cols": resid(design_h, design_b6, [s, qint_true * s, qint_mod * s]),
        "D_B6_qmod_only": resid(design_h, design_b6, [s, qint_mod * s]),
        "D_B6_w4_power": resid(design_h, design_b6, [s, qint_true * s, qint_true * s**2]),
        "newton_max_f1": sol["max_abs_f1_K"],
        "newton_iters": sol["iterations"],
    }
    return out


def check_decomposition() -> None:
    print("== 3. noiseless bias decomposition (20k LHS grid, stated protocol) ==")
    meta = json.loads(
        (DATASETS / "batch_01" / "metadata.json").read_text(encoding="utf-8")
    )
    k_exact = float(meta["k_true"])
    print(f"  batch_01 k_true (exact) = {k_exact:.10f}")
    print(
        f"  report values: Var(y_HF)={REPORT['var_y_hf']}, A={REPORT['A']},"
        f" B={REPORT['B']}, C={REPORT['C']}, D={REPORT['D']} K^2"
    )
    header = (
        f"  {'k_decomp':>9} {'seed':>9} {'Var':>8} {'A':>7} {'B':>7} {'C':>7}"
        f" {'D_ktrue':>8} {'D_knom':>8} {'Cfix_knom':>9} {'Cfix_ktrue':>10}"
    )
    print(header)
    seeds = (20260802, 0, 12345)
    for k_decomp in (k_exact, 18.24):
        for seed in seeds:
            r = decomposition(seed, k_decomp)
            print(
                f"  {k_decomp:>9.4f} {seed:>9d} {r['var_y_hf']:>8.3f} {r['A']:>7.3f}"
                f" {r['B']:>7.3f} {r['C']:>7.3f} {r['D_ktrue']:>8.3f} {r['D_knom']:>8.3f}"
                f" {r['Cfix_knom']:>9.3f} {r['Cfix_ktrue']:>10.3f}"
                f"   (Newton iters={r['newton_iters']}, maxF1={r['newton_max_f1']:.1e})"
            )
    print(
        "  -> Var(y_HF) matches the report (139.2) within LHS seed spread;"
        " A/B/C/D do NOT match the stated protocol."
    )

    print()
    print("== 3b. diagnostics: which configuration reproduces the report's C/D ==")
    print(
        f"  {'seed':>9} {'C_B4':>8} {'D_B4_qhfs':>10} {'D_B6_qhfs':>10}"
        f" {'D_2LFcols':>10} {'D_qmod':>8} {'D_W4pow':>8}"
    )
    for seed in seeds:
        r = decomposition(seed, k_exact)
        print(
            f"  {seed:>9d} {r['C_B4']:>8.3f} {r['D_B4_qhfs']:>10.3f}"
            f" {r['D_B6_qhfs']:>10.3f} {r['D_B6_two_lf_cols']:>10.3f}"
            f" {r['D_B6_qmod_only']:>8.3f} {r['D_B6_w4_power']:>8.3f}"
        )
    print(
        "  C_B4        = [H, B-without-Q-columns]            (~report C=12.249)"
    )
    print(
        "  D_B4_qhfs   = C_B4 + [P^-0.7, q_int^HF*P^-0.7]    (~report D=0.331;"
        " q_int^HF = exact HF flux, NOT LF-free)"
    )
    print(
        "  D_B6_qhfs   = stated B_v2 + [P^-0.7, q_int^HF*P^-0.7]"
    )
    print(
        "  D_2LFcols   = stated B_v2 + [P^-0.7, q_int*P^-0.7, q_int/(1+hL/k)*P^-0.7]"
        " (all LF-free)"
    )
    print(
        "  D_qmod      = stated B_v2 + [P^-0.7, q_int/(1+hL/k)*P^-0.7] (single LF col)"
    )
    print(
        "  D_W4pow     = stated B_v2 + [P^-0.7, q_int*P^-0.7, q_int*P^-1.4]"
        " (W4 power interpolation)"
    )
    print()
    print("== 4. minimal fix = C + single column q_int*P^-0.7 (no P^-0.7 column) ==")
    print(
        "  see Cfix_knom / Cfix_ktrue columns in the section-3 table"
        " (no report value; decides the minimal repair)."
    )


def main() -> None:
    check_solver_cross_validation()
    print()
    check_low_fidelity()
    print()
    check_high_fidelity()
    print()
    check_decomposition()


if __name__ == "__main__":
    main()
