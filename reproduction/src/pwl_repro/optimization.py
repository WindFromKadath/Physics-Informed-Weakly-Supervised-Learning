"""Consensus ADMM implementation used by both PWL block updates.

[PAPER] The supplement uses consensus ADMM and supplies the component proximal
operators.  [STABILITY] The absolute/relative residual stopping test and
adaptive-rho policy below are explicit reproduction additions because their
numerical settings are not published.  [ENGINEERING] Residual histories are
retained for result auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Prox = Callable[[Array, float], Array]


@dataclass(frozen=True)
class ConsensusResult:
    value: Array
    iterations: int
    converged: bool
    primal_residual: float
    dual_residual: float
    rho: float
    history: tuple[tuple[float, float], ...]


def soft_threshold(value: Array, threshold: float) -> Array:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def group_threshold(
    value: Array,
    threshold: float,
    groups: Sequence[NDArray[np.int64]],
) -> Array:
    result = np.asarray(value, dtype=float).copy()
    for group in groups:
        norm = float(np.linalg.norm(result[group]))
        if norm <= threshold:
            result[group] = 0.0
        else:
            result[group] *= 1.0 - threshold / norm
    return result


def consensus_admm(
    prox_operators: Sequence[Prox],
    initial: Array,
    *,
    rho: float = 1.0,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
    adaptive_rho: bool = True,
    residual_balance: float = 10.0,
    rho_scale: float = 2.0,
) -> ConsensusResult:
    """Minimize sum_i f_i(x) using scaled consensus ADMM.

    Each callback computes prox_{step*f_i}(v), where step = 1/rho.

    [PAPER] supplies the consensus decomposition.  [STABILITY] Convergence is
    accepted only when both primal and dual residuals meet scaled thresholds;
    adaptive rho balances those residuals during finite computation.
    """

    if not prox_operators:
        raise ValueError("At least one proximal operator is required.")
    if rho <= 0:
        raise ValueError("rho must be positive.")
    z = np.asarray(initial, dtype=float).copy()
    local = np.repeat(z[None, :], len(prox_operators), axis=0)
    dual = np.zeros_like(local)
    history: list[tuple[float, float]] = []
    converged = False
    primal = dual_residual = float("inf")
    current_rho = float(rho)

    for iteration in range(1, max_iter + 1):
        step = 1.0 / current_rho
        for index, prox in enumerate(prox_operators):
            local[index] = prox(z - dual[index], step)
        previous_z = z.copy()
        z = np.mean(local + dual, axis=0)
        dual += local - z

        primal = float(np.linalg.norm(local - z))
        dual_residual = float(
            np.sqrt(len(prox_operators))
            * current_rho
            * np.linalg.norm(z - previous_z)
        )
        history.append((primal, dual_residual))
        primal_limit = np.sqrt(local.size) * tolerance + relative_tolerance * max(
            float(np.linalg.norm(local)),
            np.sqrt(len(prox_operators)) * float(np.linalg.norm(z)),
        )
        dual_limit = np.sqrt(local.size) * tolerance + relative_tolerance * float(
            current_rho * np.linalg.norm(dual)
        )
        if primal <= primal_limit and dual_residual <= dual_limit:
            converged = True
            break
        # [STABILITY] Residual balancing is not a disclosed paper setting.
        # Rescale the scaled dual variables when rho changes.
        if adaptive_rho and primal > residual_balance * dual_residual:
            current_rho *= rho_scale
            dual /= rho_scale
        elif adaptive_rho and dual_residual > residual_balance * primal:
            current_rho /= rho_scale
            dual *= rho_scale

    return ConsensusResult(
        value=z,
        iterations=iteration,
        converged=converged,
        primal_residual=primal,
        dual_residual=dual_residual,
        rho=current_rho,
        history=tuple(history),
    )
