"""Independent numerical verification of the simulation-line implementation.

Cross-checks the reproduction against the mathematics it claims to implement
(paper equations (4)-(5), (9)-(12), supplement Propositions 1-2), without
reusing the implementation's own formulas:

* equations (9)-(11) vs hand-computed values;
* realized SNR by Monte Carlo;
* reproducibility under fixed seeds;
* affine decomposition H(x, theta) g == c + G theta;
* all 7 proximal operators via KKT / subgradient optimality conditions
  (captured from the model's own update routines);
* smooth-case g subproblem vs the exact ridge closed form;
* d update vs the analytic ridge solve;
* BCD objective monotonicity and final-theta projected-gradient optimality;
* nested split plan and IV-B correlation calibration.

Run from the repository root:  uv run python reproduction/scripts/verify_sim_correctness.py
"""

from __future__ import annotations

import sys

import numpy as np

import pwl_repro.model as model_module
from pwl_repro.experiments import _calibrate_accuracy_scales, nested_split_plan
from pwl_repro.features import PWLFeatureLibrary
from pwl_repro.model import PWLRegressor
from pwl_repro.simulation import (
    _sample_inputs,
    discrepancy,
    eta_true,
    generate_simulation,
    process_contribution,
    saturation,
    toeplitz_covariance,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------- formulas
x = np.array([[1.0, 1.0, 1.0]])
theta = np.array([0.5, 0.5])
expected_eta = (1.0 - np.exp(-0.5)) * (10 * 0.5 * 1 + 19 + 60) / (9 * 0.5 * 1 + 4 + 20)
check(
    "eq(9) eta_true hand value",
    abs(float(eta_true(x, theta)[0]) - expected_eta) < 1e-12,
    f"code={float(eta_true(x, theta)[0]):.12f} hand={expected_eta:.12f}",
)
expected_disc = (10 + 4 + 2) / (50 + 10)
check(
    "eq(10) discrepancy hand value",
    abs(float(discrepancy(x)[0]) - expected_disc) < 1e-12,
)
expected_h = (4 * 4 + 8 * 1) / (5 * 2 * 1)
check(
    "eq(11) process hand value",
    abs(float(process_contribution(np.array([[2.0, 1.0]]))[0]) - expected_h) < 1e-12,
)
check(
    "S(x3) both signs",
    abs(float(saturation(np.array([1.0]))[0]) - (1 - np.exp(-0.5))) < 1e-12
    and abs(float(saturation(np.array([-1.0]))[0]) - (1 - np.exp(0.5))) < 1e-9,
)

# theta-2 denominator of eq (9) must be handled as a pole over the box
check(
    "eq(9) denominator sign change detected",
    (9 * 0.0 * (-3.0) ** 3 + 4 * (-3.0) + 20) * (9 * 1.0 * (-3.0) ** 3 + 4 * (-3.0) + 20)
    < 0,
    "x2=-3 => denominator crosses zero inside theta2 in [0,1]",
)

# ---------------------------------------------------------------------- SNR
data = generate_simulation(
    n_labeled=10,
    n_weak=10,
    n_test=10,
    seed=0,
    correlation=0.9,
    noise_reference_size=20000,
    noise_reference_seed=123,
)
mc_x, _ = _sample_inputs(
    np.random.default_rng(999),
    200_000,
    data.covariance,
    data.theta_true,
    singularity_policy="reject_near_pole",
    min_abs_x3=0.15,
    min_abs_x4=0.15,
    min_abs_x5=0.15,
    min_abs_eta_denominator=1.0,
    min_abs_discrepancy_denominator=0.75,
    max_abs_component=25.0,
)
signal = eta_true(mc_x[:, :3], data.theta_true) + process_contribution(mc_x[:, 3:])
realized_snr = float(np.var(signal, ddof=1) / data.noise_sigma**2)
check("realized SNR ~= 5", abs(realized_snr - 5.0) < 0.15, f"SNR={realized_snr:.3f}")
check(
    "physics noise scale = 0.25 * output noise",
    abs(data.physics_noise_sigma - 0.25 * data.noise_sigma) < 1e-15,
)

# --------------------------------------------------------- reproducibility
d1 = generate_simulation(12, 20, 30, seed=7, correlation=0.9)
d2 = generate_simulation(12, 20, 30, seed=7, correlation=0.9)
d3 = generate_simulation(12, 20, 30, seed=8, correlation=0.9)
check(
    "same seed => identical arrays",
    all(
        np.array_equal(a, b)
        for a, b in ((d1.x_ph, d2.x_ph), (d1.y, d2.y), (d1.weak_y, d2.weak_y), (d1.test_y, d2.test_y))
    ),
)
check("different seed => different data", not np.array_equal(d1.y, d3.y))

# ------------------------------------------------------- affine decomposition
lib = PWLFeatureLibrary(standardize=True, b_profile="compact").fit(
    d1.x_ph, d1.x_ph, d1.x_pr, np.array([0.6, 0.4])
)
rng = np.random.default_rng(0)
g = rng.normal(size=len(lib.h_names))
max_err = 0.0
for _ in range(20):
    t = rng.uniform(0.0, 1.0, size=2)
    c, G = lib.affine_prediction_parts(d1.x_ph, g)
    err = float(np.max(np.abs(lib.transform_h(d1.x_ph, t) @ g - (c + G @ t))))
    max_err = max(max_err, err)
check("H(x,theta) g == c + G theta", max_err < 1e-9, f"max err={max_err:.2e}")

# ------------------------------------------------- prox KKT (g block, 4 ops)
model = PWLRegressor(
    lambda_physics=2.0,
    lambda_l1=0.3,
    lambda_group=0.7,
    b_profile="compact",
    d_ridge=0.0,
)
model.features_ = lib
captured: dict[str, object] = {}
real_admm = model_module.consensus_admm


def capturing_admm(proxes, initial, **kwargs):
    captured["proxes"] = proxes
    captured["initial"] = np.asarray(initial, dtype=float)
    return real_admm(proxes, initial, **kwargs)


model_module.consensus_admm = capturing_admm
try:
    n, p = 12, len(lib.h_names)
    H1 = lib.transform_h(d1.x_ph[:n], np.array([0.6, 0.4]))
    B1 = lib.transform_b(d1.x_ph[:n], d1.x_pr[:n])
    yv = rng.normal(size=n)
    d_v = rng.normal(size=B1.shape[1])
    weak = d1.weak_x_ph
    H2 = lib.transform_h(weak, np.array([0.6, 0.4]))
    wy = rng.normal(size=len(weak))
    model._update_g(H1, B1, yv, H2, wy, d_v, np.zeros(p))
finally:
    model_module.consensus_admm = real_admm

prox_labeled, prox_physics, prox_l1, prox_group = captured["proxes"]
step = 0.37
v = rng.normal(size=p)
c_target = yv - B1 @ d_v

x_star = prox_labeled(v, step)
kkt = x_star - v + step * H1.T @ (H1 @ x_star - c_target)
check("prox g labeled KKT", np.max(np.abs(kkt)) < 1e-9, f"max={np.max(np.abs(kkt)):.2e}")

x_star = prox_physics(v, step)
kkt = x_star - v + step * model.lambda_physics * H2.T @ (H2 @ x_star - wy)
check("prox g physics KKT", np.max(np.abs(kkt)) < 1e-9, f"max={np.max(np.abs(kkt)):.2e}")

x_star = prox_l1(v, step)
viol = 0.0
for xi, vi in zip(x_star, v):
    if abs(xi) > 1e-12:
        viol = max(viol, abs((xi - vi) / step + model.lambda_l1 * np.sign(xi)))
    else:
        viol = max(viol, max(0.0, abs(vi) / step - model.lambda_l1))
check("prox g L1 subgradient", viol < 1e-9, f"viol={viol:.2e}")

x_star = prox_group(v, step)
viol = 0.0
for idx in lib.groups:
    xs, vs = x_star[idx], v[idx]
    nrm = float(np.linalg.norm(xs))
    if nrm > 1e-12:
        viol = max(viol, float(np.max(np.abs((xs - vs) / step + model.lambda_group * xs / nrm))))
    else:
        viol = max(viol, max(0.0, float(np.linalg.norm(vs)) / step - model.lambda_group))
check("prox g group-L2 subgradient", viol < 1e-9, f"viol={viol:.2e}")

# --------------------------------------------- smooth-case g subproblem exact
# Well-conditioned random design: the structured H at theta == theta_ref has
# identically-zero standardized theta columns (exact singularity), which makes
# any coefficient-level reference ill-posed.  A generic design gives a
# meaningful solver comparison.
model_smooth = PWLRegressor(lambda_physics=2.0, lambda_l1=0.0, lambda_group=0.0, b_profile="compact")
from types import SimpleNamespace

pp_r, n1_r, n2_r = 15, 40, 60
H1r = rng.normal(size=(n1_r, pp_r))
H2r = rng.normal(size=(n2_r, pp_r))
cr = rng.normal(size=n1_r)
wr = rng.normal(size=n2_r)
model_smooth.features_ = SimpleNamespace(groups=(np.arange(pp_r),))
res = model_smooth._update_g(H1r, np.zeros((n1_r, 1)), cr, H2r, wr, np.zeros(1), np.zeros(pp_r))
lam = model_smooth.lambda_physics
augmented_H = np.vstack((H1r, np.sqrt(lam) * H2r))
augmented_c = np.concatenate((cr, np.sqrt(lam) * wr))
exact = np.linalg.lstsq(augmented_H, augmented_c, rcond=None)[0]
rel = float(np.linalg.norm(res.value - exact) / max(1.0, np.linalg.norm(exact)))
check(
    "ADMM g (smooth) == exact ridge",
    rel < 1e-3,
    f"rel err={rel:.2e}, converged={res.converged} (tolerance-limited)",
)

# Composite (L1 + single-group L2) vs an independent high-precision ISTA solve.
l1_r, lg_r = 0.3, 0.4


def _smooth_grad(g):
    return H1r.T @ (H1r @ g - cr) + lam * (H2r.T @ (H2r @ g - wr))


def _objective_g(g):
    return (
        0.5 * np.sum((H1r @ g - cr) ** 2)
        + 0.5 * lam * np.sum((H2r @ g - wr) ** 2)
        + l1_r * np.sum(np.abs(g))
        + lg_r * np.linalg.norm(g)
    )


def _prox_sgl(v, t):
    s = np.sign(v) * np.maximum(np.abs(v) - t * l1_r, 0.0)
    nrm = float(np.linalg.norm(s))
    return s * max(1.0 - t * lg_r / nrm, 0.0) if nrm > 0 else s


lipschitz = float(np.linalg.eigvalsh(H1r.T @ H1r + lam * H2r.T @ H2r).max())
g_ref = np.zeros(pp_r)
for _ in range(200000):
    g_next = _prox_sgl(g_ref - (1.0 / lipschitz) * _smooth_grad(g_ref), 1.0 / lipschitz)
    if np.max(np.abs(g_next - g_ref)) < 1e-13:
        g_ref = g_next
        break
    g_ref = g_next

model_comp = PWLRegressor(
    lambda_physics=lam, lambda_l1=l1_r, lambda_group=lg_r, b_profile="compact"
)
model_comp.features_ = SimpleNamespace(groups=(np.arange(pp_r),))
res_comp = model_comp._update_g(H1r, np.zeros((n1_r, 1)), cr, H2r, wr, np.zeros(1), np.zeros(pp_r))
rel_comp = float(np.linalg.norm(res_comp.value - g_ref) / max(1.0, np.linalg.norm(g_ref)))
gap_comp = float(_objective_g(res_comp.value) - _objective_g(g_ref))
check(
    "ADMM g (L1+group) == ISTA reference",
    rel_comp < 1e-3 and gap_comp < 1e-4,
    f"rel err={rel_comp:.2e}, objective gap={gap_comp:.2e}, converged={res_comp.converged}",
)

# --------------------------------------------- prox KKT (theta block, 3 ops)
model_module.consensus_admm = capturing_admm
try:
    theta0 = np.array([0.5, 0.5])
    model._update_theta(d1.x_ph[:n], weak, yv, wy, B1, d_v, g, theta0, np.zeros(2), np.ones(2))
finally:
    model_module.consensus_admm = real_admm

prox_t_labeled, prox_t_physics, prox_t_box = captured["proxes"]
c1, G1 = lib.affine_prediction_parts(d1.x_ph[:n], g)
c2, G2 = lib.affine_prediction_parts(weak, g)
t1 = yv - B1 @ d_v - c1
t2 = wy - c2

v2 = rng.uniform(-0.5, 1.5, size=2)
x_star = prox_t_labeled(v2, step)
kkt = x_star - v2 + step * G1.T @ (G1 @ x_star - t1)
check("prox theta labeled KKT", np.max(np.abs(kkt)) < 1e-9, f"max={np.max(np.abs(kkt)):.2e}")

x_star = prox_t_physics(v2, step)
kkt = x_star - v2 + step * model.lambda_physics * G2.T @ (G2 @ x_star - t2)
check("prox theta physics KKT", np.max(np.abs(kkt)) < 1e-9, f"max={np.max(np.abs(kkt)):.2e}")

x_star = prox_t_box(v2, step)
check(
    "prox theta box == clip",
    np.array_equal(x_star, np.clip(v2, 0.0, 1.0)),
)

# -------------------------------------------------- end-to-end fit on sim data
sim = generate_simulation(40, 100, 200, seed=11, correlation=0.9)
fit_model = PWLRegressor(
    lambda_physics=1.0,
    lambda_l1=0.01,
    lambda_group=0.01,
    b_profile="compact",
    d_ridge=10.0,
)
fit_model.fit(sim.x_ph, sim.x_pr, sim.y, sim.weak_x_ph, sim.weak_y, physics_model=sim.physics_model)
pred = fit_model.predict(sim.test_x_ph, sim.test_x_pr)
check("fit: finite predictions", bool(np.all(np.isfinite(pred))))
check(
    "fit: theta inside box",
    bool(np.all(fit_model.theta_ >= 0.0) and np.all(fit_model.theta_ <= 1.0)),
    f"theta={fit_model.theta_}",
)

objectives = np.array([record.objective for record in fit_model.history_])
increases = np.diff(objectives)
max_increase = float(np.max(increases)) if len(increases) else 0.0
check(
    "BCD objective (near-)monotone",
    max_increase <= 1e-6 * max(1.0, abs(objectives[0])),
    f"max increase={max_increase:.3e} over {len(objectives)} sweeps, converged={fit_model.converged_}",
)

# d block consistency with the analytic ridge solve at the final g.  Inside
# one BCD sweep the d update uses the theta of the PREVIOUS sweep (theta is
# updated after d), so the reference must use history[-2].theta, not the
# final theta recorded after the last theta block.
y_scaled = (sim.y - fit_model.y_mean_) / fit_model.y_scale_
B_full = fit_model.features_.transform_b(sim.x_ph, sim.x_pr)
theta_at_d_update = (
    np.asarray(fit_model.history_[-2].theta)
    if len(fit_model.history_) >= 2
    else fit_model.theta_
)
H_at_d_update = fit_model.features_.transform_h(sim.x_ph, theta_at_d_update)
target_d = y_scaled - H_at_d_update @ fit_model.g_
gram = B_full.T @ B_full + fit_model.d_ridge * np.eye(B_full.shape[1])
expected_d = np.linalg.solve(gram, B_full.T @ target_d)
check(
    "d update == analytic ridge at final g",
    float(np.max(np.abs(expected_d - fit_model.d_))) < 1e-6,
    f"max diff={float(np.max(np.abs(expected_d - fit_model.d_))):.2e}",
)

# theta optimality: projected gradient of the theta subproblem at convergence
c1f, G1f = fit_model.features_.affine_prediction_parts(sim.x_ph, fit_model.g_)
c2f, G2f = fit_model.features_.affine_prediction_parts(sim.weak_x_ph, fit_model.g_)
weak_scaled = (sim.weak_y - fit_model.y_mean_) / fit_model.y_scale_
grad = -G1f.T @ ((y_scaled - B_full @ fit_model.d_ - c1f) - G1f @ fit_model.theta_) - (
    fit_model.lambda_physics * G2f.T @ ((weak_scaled - c2f) - G2f @ fit_model.theta_)
)
scale = max(1.0, float(np.linalg.norm(grad)))
viol = 0.0
for gi, ti in zip(grad, fit_model.theta_):
    if 1e-6 < ti < 1.0 - 1e-6:
        viol = max(viol, abs(gi) / scale)
    elif ti <= 1e-6:
        viol = max(viol, -gi / scale)
    else:
        viol = max(viol, gi / scale)
check("theta subproblem projected gradient", viol < 2e-2, f"scaled viol={viol:.2e}")

# objective self-consistency
manual_obj = fit_model._objective(
    sim.x_ph, sim.weak_x_ph, y_scaled, weak_scaled, B_full, fit_model.g_, fit_model.d_, fit_model.theta_
)
check(
    "training objective self-consistent",
    abs(manual_obj - fit_model.training_objective_) < 1e-8 * max(1.0, abs(manual_obj)),
)

# ---------------------------------------------------------- nested split plan
plan = nested_split_plan(120, [10, 30, 120], train_fraction=0.7, seed=5)
t10, v10 = plan[10]
t30, v30 = plan[30]
t120, v120 = plan[120]
check(
    "nested splits are nested",
    set(t10) <= set(t30) <= set(t120) and set(v10) <= set(v30) <= set(v120),
)
check(
    "nested split sizes 70/30 per block",
    (len(t10), len(v10)) == (7, 3)
    and (len(t30) - len(t10), len(v30) - len(v10)) == (14, 6)
    and (len(t120) - len(t30), len(v120) - len(v30)) == (63, 27),
    f"counts: {len(t10)}/{len(v10)} {len(t30)}/{len(v30)} {len(t120)}/{len(v120)}",
)
check(
    "nested split covers pool without overlap",
    len(set(t120) & set(v120)) == 0 and len(set(t120) | set(v120)) == 120,
)

# ------------------------------------------------- IV-B calibration accuracy
config = {
    "experiment": {
        "physics_accuracy_targets": [0.85, 0.7, 0.5],
        "accuracy_calibration_size": 8000,
        "accuracy_calibration_seed": 9102026,
        "accuracy_scale_log10": [-3, 3, 801],
        "accuracy_correlation_tolerance": 0.02,
        "accuracy_require_tolerance": True,
    },
    "simulation": {
        "n_weak": 5,
        "n_test": 5,
        "snr": 5.0,
        "correlation": 0.9,
        "physics_noise_fraction": 0.25,
        "singularity_policy": "reject_near_pole",
        "noise_reference_size": 8000,
        "stability": {},
    },
}
base = generate_simulation(40, 5, 5, seed=21, correlation=0.9, noise_reference_size=8000)
cal = _calibrate_accuracy_scales(config, base, repeat=0, seed=21)
check(
    "IV-B calibration hits targets",
    bool((cal["correlation_error"] <= 0.02).all()),
    f"errors={cal['correlation_error'].round(4).tolist()}",
)
scales = cal["discrepancy_scale"].to_numpy()
check(
    "IV-B scales decrease with target correlation",
    bool(scales[0] < scales[1] < scales[2]),
    f"scales={np.round(scales, 4).tolist()}",
)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("All simulation-line verification checks passed.")
