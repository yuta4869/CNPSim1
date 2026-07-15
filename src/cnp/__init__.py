"""Movement-agnostic Conditional Neural Process components."""

from .losses import gaussian_nll
from .model import CNP, ConditionalNeuralProcess
from .sampling import CNPBatch, sample_context_target
from .synthetic import (
    SinusoidParameterRange,
    SyntheticEvaluationTask,
    SyntheticTrainingBatch,
    deterministic_regression_metrics,
    evaluate_sinusoids,
    gaussian_regression_metrics,
    interpolation_baselines,
    make_synthetic_evaluation_task,
    sample_sinusoid_parameters,
    sample_synthetic_training_batch,
    sinusoid_regression_baseline,
)

__all__ = [
    "CNP",
    "CNPBatch",
    "ConditionalNeuralProcess",
    "SinusoidParameterRange",
    "SyntheticEvaluationTask",
    "SyntheticTrainingBatch",
    "deterministic_regression_metrics",
    "evaluate_sinusoids",
    "gaussian_nll",
    "gaussian_regression_metrics",
    "interpolation_baselines",
    "make_synthetic_evaluation_task",
    "sample_context_target",
    "sample_sinusoid_parameters",
    "sample_synthetic_training_batch",
    "sinusoid_regression_baseline",
]
