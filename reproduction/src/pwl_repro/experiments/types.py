"""Shared scenario contracts, artifact containers, metrics, and split helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd


class ScenarioData(Protocol):
    """[ENGINEERING] Structural interface the experiment machinery consumes.

    ``SimulationData`` satisfies it for the paper's synthetic DGP; scenario
    adapters (for example the heat-conduction migration pilot) implement the
    same members so tuning, metrics, and artifacts stay shared.
    """

    x_ph: np.ndarray
    x_pr: np.ndarray
    y: np.ndarray
    weak_x_ph: np.ndarray
    weak_y: np.ndarray
    test_x_ph: np.ndarray
    test_x_pr: np.ndarray
    test_y: np.ndarray
    theta_true: np.ndarray
    noise_sigma: float
    physics_noise_sigma: float
    discrepancy_scale: float
    sampling_diagnostics: Mapping[str, Any]

    def physics_model(self, x_ph: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Deterministic, biased physics model available to the learner."""
        ...

    def labeled_physics_correlation(self, *, include_noise: bool = True) -> float:
        """Pearson correlation between labels and the biased physics output."""
        ...

@dataclass(frozen=True)
class ExperimentArtifacts:
    """Raw metrics, predictions, conditions, and statistical tests."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    conditions: pd.DataFrame
    statistical_tests: pd.DataFrame

    @classmethod
    def combine(cls, parts: Iterable["ExperimentArtifacts"]) -> "ExperimentArtifacts":
        parts = list(parts)

        def combine_field(name: str) -> pd.DataFrame:
            frames = [getattr(part, name) for part in parts if not getattr(part, name).empty]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return cls(
            metrics=combine_field("metrics"),
            predictions=combine_field("predictions"),
            conditions=combine_field("conditions"),
            statistical_tests=combine_field("statistical_tests"),
        )

def _empty_artifacts() -> ExperimentArtifacts:
    return ExperimentArtifacts(
        metrics=pd.DataFrame(),
        predictions=pd.DataFrame(),
        conditions=pd.DataFrame(),
        statistical_tests=pd.DataFrame(),
    )

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_true) - np.asarray(y_pred)
    mse = float(np.mean(error**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
    }

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """[ENGINEERING] Diagnostic R^2; NaN when the target has zero variance."""

    truth = np.asarray(y_true, dtype=float)
    estimate = np.asarray(y_pred, dtype=float)
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    if total <= 0.0:
        return float("nan")
    residual = float(np.sum((truth - estimate) ** 2))
    return 1.0 - residual / total

def nested_split_plan(
    max_samples: int,
    sample_sizes: Iterable[int],
    *,
    train_fraction: float,
    seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Create [INFERRED] nested train/validation sets at the paper's ratio.

    Each newly added block is independently split, so earlier training and
    validation observations never change roles as sample size grows.  The
    paper reports a 70/30 split but does not disclose this nesting policy.
    """

    sizes = sorted(set(int(size) for size in sample_sizes))
    if not sizes or sizes[-1] > max_samples:
        raise ValueError("sample_sizes must be nonempty and <= max_samples.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(max_samples)
    train: list[int] = []
    validation: list[int] = []
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    previous = 0
    for size in sizes:
        block = order[previous:size]
        n_train = int(round(train_fraction * len(block)))
        n_train = min(max(1, n_train), len(block) - 1)
        train.extend(int(value) for value in block[:n_train])
        validation.extend(int(value) for value in block[n_train:])
        result[size] = (
            np.asarray(train, dtype=int).copy(),
            np.asarray(validation, dtype=int).copy(),
        )
        previous = size
    return result

def _labeled_dataset_id(data: ScenarioData) -> str:
    digest = hashlib.sha256()
    for array in (
        data.x_ph,
        data.x_pr,
        data.y,
        data.test_x_ph,
        data.test_x_pr,
        data.test_y,
        data.theta_true,
    ):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()[:16]

def _scenario_indices(train: np.ndarray, validation: np.ndarray) -> np.ndarray:
    return np.concatenate((train, validation))
