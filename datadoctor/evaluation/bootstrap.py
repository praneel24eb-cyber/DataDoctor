from typing import Tuple, Callable, Any
import numpy as np
from scipy.stats import wilcoxon


def compute_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes non-parametric percentile 95% bootstrap confidence intervals.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_estimate = float(metric_fn(y_true, y_pred))

    bootstrap_estimates = np.zeros(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        boot_idx = rng.integers(0, n, size=n)
        bootstrap_estimates[b] = metric_fn(y_true[boot_idx], y_pred[boot_idx])

    ci_lower = float(np.percentile(bootstrap_estimates, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1.0 - alpha / 2.0)))

    return point_estimate, ci_lower, ci_upper


def paired_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Computes paired Wilcoxon signed-rank test between two sets of paired metrics.
    
    Returns:
        (statistic, p_value)
    """
    diff = scores_a - scores_b
    if np.all(diff == 0):
        return 0.0, 1.0
    res = wilcoxon(scores_a, scores_b)
    return float(res.statistic), float(res.pvalue)

