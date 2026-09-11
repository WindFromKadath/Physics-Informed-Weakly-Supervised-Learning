"""Stable experiment-engine API shared by scenario packages.

The synthetic driver historically exposed several underscore-prefixed
helpers that were also used by the case-B and heat scenarios.  This module
provides a small public boundary without changing the validated experiment
implementation in :mod:`pwl_repro.experiments`.
"""

from .experiments import (
    ExperimentArtifacts,
    _fit_baseline_condition as fit_baseline_condition,
    _fit_pwl_condition as fit_pwl_condition,
    _labeled_dataset_id as labeled_dataset_id,
    _metric_row as metric_row,
    _prediction_rows as prediction_rows,
    _sample_size_tests as sample_size_tests,
    nested_split_plan,
    save_artifacts,
)

__all__ = [
    "ExperimentArtifacts",
    "fit_baseline_condition",
    "fit_pwl_condition",
    "labeled_dataset_id",
    "metric_row",
    "nested_split_plan",
    "prediction_rows",
    "sample_size_tests",
    "save_artifacts",
]
