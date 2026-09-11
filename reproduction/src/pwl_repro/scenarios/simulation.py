"""Synthetic experiment from equations (9)--(11) of the PWL paper.

Audit tags used in the reproduction:

* [PAPER] is stated in the paper or supplement.
* [INFERRED] completes a detail that the paper does not disclose.
* [STABILITY] prevents numerical failure and is not part of the paper DGP.
* [ENGINEERING] records provenance without changing the fitted objective.

[PAPER] The equations, Gaussian input family, theta range, and SNR target come
from Section IV.  [INFERRED] The paper does not disclose ``Sigma_x``, the
physics-noise scale, or the finite-sample SNR calibration rule.  [STABILITY]
It also does not explain how samples close to the rational-function poles are
handled.  These choices are explicit so that results cannot silently depend
on undocumented assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationData:
    """One labeled/weak/test split and its data-generating metadata."""

    x_ph: Array
    x_pr: Array
    y: Array
    weak_x_ph: Array
    weak_y: Array
    test_x_ph: Array
    test_x_pr: Array
    test_y: Array
    theta_true: Array
    noise_sigma: float
    physics_noise_sigma: float
    labeled_physics_noise: Array
    weak_physics_noise: Array
    test_physics_noise: Array
    discrepancy_scale: float
    covariance: Array
    sampling_diagnostics: Mapping[str, float | int | str]

    def labeled_subset(self, indices: Array) -> "SimulationData":
        """Return a labeled view while preserving weak data and the test set."""

        indices = np.asarray(indices, dtype=int)
        return SimulationData(
            x_ph=self.x_ph[indices],
            x_pr=self.x_pr[indices],
            y=self.y[indices],
            weak_x_ph=self.weak_x_ph,
            weak_y=self.weak_y,
            test_x_ph=self.test_x_ph,
            test_x_pr=self.test_x_pr,
            test_y=self.test_y,
            theta_true=self.theta_true,
            noise_sigma=self.noise_sigma,
            physics_noise_sigma=self.physics_noise_sigma,
            labeled_physics_noise=self.labeled_physics_noise[indices],
            weak_physics_noise=self.weak_physics_noise,
            test_physics_noise=self.test_physics_noise,
            discrepancy_scale=self.discrepancy_scale,
            covariance=self.covariance,
            sampling_diagnostics=self.sampling_diagnostics,
        )

    def physics_model(self, x_ph: Array, theta: Array) -> Array:
        """Deterministic, biased physics model available to the learner."""

        return eta_true(x_ph, theta) + self.discrepancy_scale * discrepancy(x_ph)

    def labeled_physics_output(self, *, include_noise: bool = True) -> Array:
        """Return the physics output paired with the labeled observations."""

        output = self.physics_model(self.x_ph, self.theta_true)
        if include_noise:
            output = output + self.labeled_physics_noise
        return output

    def labeled_physics_correlation(self, *, include_noise: bool = True) -> float:
        """Pearson correlation used to audit the Section IV-B condition."""

        return float(
            np.corrcoef(
                self.y,
                self.labeled_physics_output(include_noise=include_noise),
            )[0, 1]
        )

    def with_discrepancy_scale(self, scale: float) -> "SimulationData":
        """Change only the discrepancy magnitude while preserving all samples/noise."""

        scale = float(scale)
        weak_y = (
            eta_true(self.weak_x_ph, self.theta_true)
            + scale * discrepancy(self.weak_x_ph)
            + self.weak_physics_noise
        )
        return replace(self, weak_y=weak_y, discrepancy_scale=scale)


def toeplitz_covariance(dim: int = 5, correlation: float = 0.5) -> Array:
    """Return the [INFERRED] covariance Sigma[i,j] = rho ** |i-j|.

    [PAPER] specifies a zero-mean multivariate Gaussian input but does not
    publish Sigma_x.  The configured rho is therefore a reproduction
    assumption, not an author-provided parameter.
    """

    indices = np.arange(dim)
    return correlation ** np.abs(indices[:, None] - indices[None, :])


def _signed_floor(value: Array, floor: float) -> Array:
    # [STABILITY] The published rational functions have genuine poles.
    # This guard only prevents floating division by zero; it is not in paper.
    sign = np.where(value < 0.0, -1.0, 1.0)
    return np.where(np.abs(value) < floor, sign * floor, value)


def saturation(x3: Array) -> Array:
    """[PAPER] S(x3), with [STABILITY] floor and overflow clipping."""

    x3_safe = _signed_floor(np.asarray(x3, dtype=float), 1e-8)
    exponent = np.clip(-1.0 / (2.0 * x3_safe), -60.0, 60.0)
    return 1.0 - np.exp(exponent)


def eta_true(x_ph: Array, theta: Array) -> Array:
    """[PAPER] Equation (9), with [STABILITY] singularity guards."""

    x = np.asarray(x_ph, dtype=float)
    theta = np.asarray(theta, dtype=float)
    x1, x2, x3 = x.T
    numerator = 10.0 * theta[0] * x1**3 + 19.0 * x1 + 60.0
    denominator = 9.0 * theta[1] * x2**3 + 4.0 * x2 + 20.0
    return saturation(x3) * numerator / _signed_floor(denominator, 1e-8)


def discrepancy(x_ph: Array) -> Array:
    """[PAPER] Equation (10), with a [STABILITY] denominator guard."""

    x1, x2, x3 = np.asarray(x_ph, dtype=float).T
    numerator = 10.0 * x1**2 + 4.0 * x2**2 + 2.0 * x3**3
    denominator = 50.0 * x1 * x2 + 10.0
    return numerator / _signed_floor(denominator, 1e-8)


def process_contribution(x_pr: Array) -> Array:
    """[PAPER] Equation (11), with a [STABILITY] denominator guard."""

    x4, x5 = np.asarray(x_pr, dtype=float).T
    numerator = 4.0 * x4**2 + 8.0 * x5**2
    denominator = 5.0 * x4 * x5
    return numerator / _signed_floor(denominator, 1e-8)


def _stability_masks(
    x: Array,
    theta: Array,
    *,
    min_abs_x3: float,
    min_abs_x4: float,
    min_abs_x5: float,
    min_abs_eta_denominator: float,
    min_abs_discrepancy_denominator: float,
    max_abs_component: float,
) -> dict[str, NDArray[np.bool_]]:
    """Return masks for the [STABILITY] near-pole rejection policy.

    Rejection is not described in the paper and changes the Gaussian input
    into a conditional distribution.  Every threshold is therefore exposed
    in YAML and every rejection reason is written to conditions.csv.
    """

    x1, x2, x3, x4, x5 = x.T
    eta_den_at_zero = 4.0 * x2 + 20.0
    eta_den_at_one = 9.0 * x2**3 + 4.0 * x2 + 20.0
    eta_crosses_zero = eta_den_at_zero * eta_den_at_one <= 0.0
    minimum_eta_denominator = np.where(
        eta_crosses_zero,
        0.0,
        np.minimum(np.abs(eta_den_at_zero), np.abs(eta_den_at_one)),
    )
    delta_den = 50.0 * x1 * x2 + 10.0
    values = np.column_stack(
        (
            eta_true(x[:, :3], theta),
            discrepancy(x[:, :3]),
            process_contribution(x[:, 3:]),
        )
    )
    masks = {
        "x3": np.abs(x3) >= min_abs_x3,
        "x4": np.abs(x4) >= min_abs_x4,
        "x5": np.abs(x5) >= min_abs_x5,
        # Calibration explores theta_2 in [0, 1].  Requiring stability only at
        # theta_true lets a calibrated baseline encounter a new denominator pole.
        "eta_denominator": minimum_eta_denominator >= min_abs_eta_denominator,
        "discrepancy_denominator": (
            np.abs(delta_den) >= min_abs_discrepancy_denominator
        ),
        "finite": np.all(np.isfinite(values), axis=1),
        "component_magnitude": np.all(np.abs(values) <= max_abs_component, axis=1),
    }
    accepted = np.ones(len(x), dtype=bool)
    for mask in masks.values():
        accepted &= mask
    masks["accepted"] = accepted
    return masks


def _sample_inputs(
    rng: np.random.Generator,
    n_samples: int,
    covariance: Array,
    theta: Array,
    *,
    singularity_policy: str,
    min_abs_x3: float,
    min_abs_x4: float,
    min_abs_x5: float,
    min_abs_eta_denominator: float,
    min_abs_discrepancy_denominator: float,
    max_abs_component: float,
) -> tuple[Array, dict[str, float | int | str]]:
    if singularity_policy == "paper_raw":
        # [PAPER-like sensitivity mode] Preserve the published Gaussian draw.
        # Formula evaluation still retains 1e-8/overflow machine guards, so
        # this is not evidence of the authors' unknown pole-handling policy.
        sampled = rng.multivariate_normal(np.zeros(5), covariance, size=n_samples)
        return sampled, {
            "singularity_policy": singularity_policy,
            "requested_samples": n_samples,
            "drawn_samples": n_samples,
            "accepted_samples": n_samples,
            "acceptance_rate": 1.0,
        }
    if singularity_policy != "reject_near_pole":
        raise ValueError(
            "singularity_policy must be 'paper_raw' or 'reject_near_pole'."
        )

    accepted: list[Array] = []
    count = 0
    attempts = 0
    drawn = 0
    rejection_counts = {
        "x3": 0,
        "x4": 0,
        "x5": 0,
        "eta_denominator": 0,
        "discrepancy_denominator": 0,
        "finite": 0,
        "component_magnitude": 0,
    }
    while count < n_samples:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Could not sample enough numerically stable inputs.")
        batch_size = max(4 * (n_samples - count), 256)
        batch = rng.multivariate_normal(np.zeros(5), covariance, size=batch_size)
        drawn += len(batch)
        masks = _stability_masks(
            batch,
            theta,
            min_abs_x3=min_abs_x3,
            min_abs_x4=min_abs_x4,
            min_abs_x5=min_abs_x5,
            min_abs_eta_denominator=min_abs_eta_denominator,
            min_abs_discrepancy_denominator=min_abs_discrepancy_denominator,
            max_abs_component=max_abs_component,
        )
        for name in rejection_counts:
            rejection_counts[name] += int(np.count_nonzero(~masks[name]))
        stable = batch[masks["accepted"]]
        if stable.size:
            accepted.append(stable)
            count += len(stable)
    sampled = np.vstack(accepted)[:n_samples]
    diagnostics: dict[str, float | int | str] = {
        "singularity_policy": singularity_policy,
        "requested_samples": n_samples,
        "drawn_samples": drawn,
        "accepted_samples": count,
        "acceptance_rate": float(count / drawn),
        "min_abs_x3": min_abs_x3,
        "min_abs_x4": min_abs_x4,
        "min_abs_x5": min_abs_x5,
        "min_abs_eta_denominator": min_abs_eta_denominator,
        "min_abs_discrepancy_denominator": min_abs_discrepancy_denominator,
        "max_abs_component": max_abs_component,
    }
    for name, rejected in rejection_counts.items():
        diagnostics[f"rejection_rate_{name}"] = float(rejected / drawn)
    return sampled, diagnostics


def generate_simulation(
    n_labeled: int = 40,
    n_weak: int = 100,
    n_test: int = 200,
    *,
    seed: int = 42,
    snr: float = 5.0,
    correlation: float = 0.5,
    covariance: Array | None = None,
    discrepancy_scale: float = 1.0,
    physics_noise_fraction: float = 0.25,
    theta_true: Array | None = None,
    singularity_policy: str = "reject_near_pole",
    min_abs_x3: float = 0.15,
    min_abs_x4: float = 0.15,
    min_abs_x5: float = 0.15,
    min_abs_eta_denominator: float = 1.0,
    min_abs_discrepancy_denominator: float = 0.75,
    max_abs_component: float = 25.0,
    noise_reference_size: int = 5000,
    noise_reference_seed: int | None = None,
) -> SimulationData:
    """Generate the synthetic PWL experiment with audited assumptions.

    [PAPER] SNR follows Var(signal) / Var(noise), theta is system-level, and
    equations (9)--(11) define the signals.  [INFERRED] Physics-noise magnitude
    and the independent reference population are reproduction choices.
    [STABILITY] Inputs near the published formulas' poles are rejected by
    default; this explicitly completes an omitted numerical detail.
    """

    if min(n_labeled, n_weak, n_test) <= 0:
        raise ValueError("All sample counts must be positive.")
    if snr <= 0:
        raise ValueError("snr must be positive.")
    if physics_noise_fraction < 0:
        raise ValueError("physics_noise_fraction must be nonnegative.")
    if noise_reference_size < 100:
        raise ValueError("noise_reference_size must be at least 100.")
    rng = np.random.default_rng(seed)
    covariance_matrix = (
        # [INFERRED] The paper does not publish Sigma_x.
        toeplitz_covariance(5, correlation)
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    if covariance_matrix.shape != (5, 5):
        raise ValueError("covariance must have shape (5, 5).")
    if not np.allclose(covariance_matrix, covariance_matrix.T):
        raise ValueError("covariance must be symmetric.")
    if np.min(np.linalg.eigvalsh(covariance_matrix)) <= 0:
        raise ValueError("covariance must be positive definite.")
    theta = (
        rng.uniform(np.nextafter(0.0, 1.0), 1.0, size=2)
        if theta_true is None
        else np.asarray(theta_true, dtype=float).copy()
    )
    if theta.shape != (2,):
        raise ValueError("theta_true must have shape (2,).")

    total = n_labeled + n_weak + n_test
    sampling_options = {
        "singularity_policy": singularity_policy,
        "min_abs_x3": min_abs_x3,
        "min_abs_x4": min_abs_x4,
        "min_abs_x5": min_abs_x5,
        "min_abs_eta_denominator": min_abs_eta_denominator,
        "min_abs_discrepancy_denominator": min_abs_discrepancy_denominator,
        "max_abs_component": max_abs_component,
    }
    x, sampling_diagnostics = _sample_inputs(
        rng,
        total,
        covariance_matrix,
        theta,
        **sampling_options,
    )
    labeled = x[:n_labeled]
    weak = x[n_labeled : n_labeled + n_weak]
    test = x[n_labeled + n_weak :]

    def signal(rows: Array) -> Array:
        return eta_true(rows[:, :3], theta) + process_contribution(rows[:, 3:])

    reference_rng = np.random.default_rng(
        seed + 1_000_003 if noise_reference_seed is None else noise_reference_seed
    )
    # [INFERRED] Estimate one scenario-independent noise scale on a separate
    # population.  This prevents label-count changes from changing the SNR.
    reference_x, reference_diagnostics = _sample_inputs(
        reference_rng,
        noise_reference_size,
        covariance_matrix,
        theta,
        **sampling_options,
    )
    reference_signal = signal(reference_x)
    noise_sigma = float(np.sqrt(np.var(reference_signal, ddof=1) / snr))
    # [INFERRED] The paper includes physics error/noise but does not disclose
    # this relative standard deviation.
    physics_noise_sigma = float(physics_noise_fraction * noise_sigma)

    y = signal(labeled) + rng.normal(0.0, noise_sigma, n_labeled)
    test_y = signal(test) + rng.normal(0.0, noise_sigma, n_test)
    labeled_physics_noise = rng.normal(0.0, physics_noise_sigma, n_labeled)
    weak_physics_noise = rng.normal(0.0, physics_noise_sigma, n_weak)
    test_physics_noise = rng.normal(0.0, physics_noise_sigma, n_test)
    weak_y = (
        eta_true(weak[:, :3], theta)
        + discrepancy_scale * discrepancy(weak[:, :3])
        + weak_physics_noise
    )
    combined_y = np.concatenate((y, test_y))
    sampling_diagnostics = {
        **sampling_diagnostics,
        "noise_reference_size": noise_reference_size,
        "noise_reference_acceptance_rate": reference_diagnostics["acceptance_rate"],
        "reference_signal_mean": float(np.mean(reference_signal)),
        "reference_signal_std": float(np.std(reference_signal, ddof=1)),
        "target_mean": float(np.mean(combined_y)),
        "target_std": float(np.std(combined_y, ddof=1)),
        "target_abs_q99": float(np.quantile(np.abs(combined_y), 0.99)),
        "target_abs_max": float(np.max(np.abs(combined_y))),
    }

    return SimulationData(
        x_ph=labeled[:, :3],
        x_pr=labeled[:, 3:],
        y=y,
        weak_x_ph=weak[:, :3],
        weak_y=weak_y,
        test_x_ph=test[:, :3],
        test_x_pr=test[:, 3:],
        test_y=test_y,
        theta_true=theta,
        noise_sigma=noise_sigma,
        physics_noise_sigma=physics_noise_sigma,
        labeled_physics_noise=labeled_physics_noise,
        weak_physics_noise=weak_physics_noise,
        test_physics_noise=test_physics_noise,
        discrepancy_scale=float(discrepancy_scale),
        covariance=covariance_matrix,
        sampling_diagnostics=sampling_diagnostics,
    )


PhysicsModel = Callable[[Array, Array], Array]
