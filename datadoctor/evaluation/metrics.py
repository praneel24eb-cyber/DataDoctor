from typing import Dict, Optional, Any
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def evaluate_detection_metrics(
    scores: np.ndarray,
    is_noisy: np.ndarray,
    estimated_noise_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Computes comprehensive detection ranking metrics over the training split.
    
    Metrics:
    - auroc: Area Under ROC Curve
    - auprc: Area Under Precision-Recall Curve
    - p_at_k: Precision at top K samples (1%, 2%, 5%, 10%, 20%)
    - r_at_k: Recall at top K samples
    - p_at_num_errors: Precision at K = total true errors
    - f1_at_estimated_noise: F1 score at top round(eta_hat * N) threshold
    """
    n = len(scores)
    num_errors = int(np.sum(is_noisy))
    metrics: Dict[str, float] = {}

    if num_errors == 0 or num_errors == n:
        # Edge cases: no noise or all noise
        metrics["auroc"] = 1.0 if num_errors == n else 0.5
        metrics["auprc"] = 1.0 if num_errors == n else 0.0
        metrics["precision_at_num_errors"] = 1.0 if num_errors == n else 0.0
        return metrics

    # AUROC & AUPRC
    metrics["auroc"] = float(roc_auc_score(is_noisy, scores))
    metrics["auprc"] = float(average_precision_score(is_noisy, scores))

    # Sorted order: descending by score (most suspicious first)
    sort_order = np.argsort(-scores)
    sorted_is_noisy = is_noisy[sort_order]

    # Precision & Recall at K percent
    k_percents = [0.01, 0.02, 0.05, 0.10, 0.20]
    for p in k_percents:
        k = max(1, int(round(p * n)))
        tp = np.sum(sorted_is_noisy[:k])
        metrics[f"p_at_{int(p*100)}pct"] = float(tp / k)
        metrics[f"r_at_{int(p*100)}pct"] = float(tp / num_errors)

    # Precision at K = number of true errors
    k_errors = max(1, num_errors)
    tp_errors = np.sum(sorted_is_noisy[:k_errors])
    metrics["precision_at_num_errors"] = float(tp_errors / k_errors)

    # F1 at estimated noise rate
    eta = estimated_noise_rate if estimated_noise_rate is not None else (num_errors / n)
    k_eta = max(1, int(round(eta * n)))
    preds = np.zeros(n, dtype=bool)
    preds[sort_order[:k_eta]] = True
    metrics["f1_at_noise_rate"] = float(f1_score(is_noisy, preds))

    return metrics


def compute_spearman_correlations(
    view_scores: Dict[str, np.ndarray],
    is_noisy: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Computes pairwise Spearman rank correlations between detector views
    separately on clean and noisy subsets to verify independence.
    """
    clean_mask = ~is_noisy
    noisy_mask = is_noisy

    pairs = [
        ("v1", "v2"),
        ("v1", "v3"),
        ("v2", "v3"),
        ("v1", "v2_B"),
    ]

    clean_corrs = {}
    noisy_corrs = {}
    all_corrs = {}

    for k1, k2 in pairs:
        if k1 in view_scores and k2 in view_scores:
            s1 = view_scores[k1]
            s2 = view_scores[k2]
            key = f"{k1}_{k2}"

            # Overall
            rho_all, _ = spearmanr(s1, s2)
            all_corrs[key] = float(rho_all) if not np.isnan(rho_all) else 0.0

            # Clean
            if np.sum(clean_mask) > 1:
                rho_clean, _ = spearmanr(s1[clean_mask], s2[clean_mask])
                clean_corrs[key] = float(rho_clean) if not np.isnan(rho_clean) else 0.0

            # Noisy
            if np.sum(noisy_mask) > 1:
                rho_noisy, _ = spearmanr(s1[noisy_mask], s2[noisy_mask])
                noisy_corrs[key] = float(rho_noisy) if not np.isnan(rho_noisy) else 0.0

    return {
        "all": all_corrs,
        "clean": clean_corrs,
        "noisy": noisy_corrs,
    }

