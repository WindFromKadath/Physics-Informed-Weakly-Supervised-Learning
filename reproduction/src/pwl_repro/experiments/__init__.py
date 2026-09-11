"""Experiment engine split by responsibility.

This package preserves the historical ``pwl_repro.experiments`` import path.
"""

from .protocols import (
    _calibrate_accuracy_scales,
    _generate_pool,
    derive_label_savings_experiment,
    run_experiments,
    run_physics_accuracy_experiment,
    run_sample_size_experiment,
    validate_experiment_config,
)
from .reporting import (
    _plot_accuracy,
    _plot_label_savings,
    _plot_sample_boxplots,
    _plot_sample_size,
    _quality_checks,
    _summary,
    save_artifacts,
)
from .statistics import (
    _add_holm_adjustment,
    _label_savings_tests,
    _paired_frame,
    _physics_accuracy_tests,
    _sample_size_tests,
)
from .tuning import (
    _evaluate_pwl_candidate,
    _fit_baseline_condition,
    _fit_pwl_condition,
    _make_pwl,
    _metric_row,
    _prediction_rows,
    _reconstruct_selected,
    _refit_objective_stagnated,
    tune_pwl,
)
from .types import (
    ExperimentArtifacts,
    ScenarioData,
    _empty_artifacts,
    _labeled_dataset_id,
    _r_squared,
    _scenario_indices,
    nested_split_plan,
    regression_metrics,
)

__all__ = [
    "ExperimentArtifacts",
    "ScenarioData",
    "derive_label_savings_experiment",
    "nested_split_plan",
    "regression_metrics",
    "run_experiments",
    "run_physics_accuracy_experiment",
    "run_sample_size_experiment",
    "save_artifacts",
    "tune_pwl",
    "validate_experiment_config",
]
