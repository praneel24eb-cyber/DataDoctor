"""Evaluation metrics, review-budget curve, and bootstrap hypothesis testing."""
from datadoctor.evaluation.metrics import (
    evaluate_detection_metrics,
    compute_spearman_correlations,
)
from datadoctor.evaluation.bootstrap import (
    compute_bootstrap_ci,
    paired_wilcoxon_test,
)
from datadoctor.evaluation.budget_curve import (
    evaluate_review_budget_curve,
)

__all__ = [
    "evaluate_detection_metrics",
    "compute_spearman_correlations",
    "compute_bootstrap_ci",
    "paired_wilcoxon_test",
    "evaluate_review_budget_curve",
]

