from typing import Tuple, Optional
import numpy as np


def estimate_confident_joint_and_noise_rate(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """MIT-licensed implementation of Confident Joint estimation (Northcutt et al., 2021).
    
    Computes the joint distribution matrix C_{s, y*} and estimates noise rate:
    \\hat{\\eta} = 1.0 - \\frac{trace(C)}{N}.
    
    Does not require cleanlab to be installed.
    """
    n_samples, num_classes = pred_probs.shape
    
    # 1. Compute per-class expected self-confidence thresholds t_j = E[p(y=j | x) | given_label = j]
    thresholds = np.zeros(num_classes, dtype=np.float64)
    for j in range(num_classes):
        mask = (labels == j)
        if np.any(mask):
            thresholds[j] = np.mean(pred_probs[mask, j])
        else:
            thresholds[j] = 1.0 / num_classes

    # 2. Count confident joint entries
    C = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(n_samples):
        s = labels[i]
        probs_i = pred_probs[i]
        # Candidate classes exceeding class threshold
        above_thresh = np.where(probs_i >= thresholds)[0]
        if len(above_thresh) > 0:
            # Pick argmax among those exceeding threshold
            best_c = above_thresh[np.argmax(probs_i[above_thresh])]
        else:
            best_c = np.argmax(probs_i)
        C[s, best_c] += 1.0

    # 3. Estimated noise rate is fraction of off-diagonal elements in C
    diag_sum = np.trace(C)
    total_samples = np.sum(C)
    estimated_noise_rate = float(1.0 - (diag_sum / max(total_samples, 1.0)))
    estimated_noise_rate = float(np.clip(estimated_noise_rate, 0.01, 0.90))

    return C, estimated_noise_rate


def compute_cleanlab_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Computes Confident Learning label issue suspicion scores ranked by Normalized Margin.
    
    Flagged samples (off-diagonals) receive scores in [1.0, 2.0], ranked by normalized margin.
    Unflagged samples receive scores in [0.0, 1.0), ranked by normalized margin.
    Higher score = MORE suspicious.
    
    Uses cleanlab.filter.find_label_issues if cleanlab is installed;
    otherwise falls back to native MIT implementation.
    """
    n_samples, num_classes = pred_probs.shape
    scores = np.zeros(n_samples, dtype=np.float32)

    # Compute normalized margin: margin_i = p_{i, y_i} - max_{c != y_i} p_{i, c}
    margins = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        y_i = labels[i]
        p_given = pred_probs[i, y_i]
        other_probs = [pred_probs[i, c] for c in range(num_classes) if c != y_i]
        max_other = max(other_probs) if other_probs else 0.0
        margins[i] = p_given - max_other

    ranked_indices = None
    try:
        import cleanlab
        from cleanlab.filter import find_label_issues
        # Rank by normalized_margin as specified in PRD
        ranked_indices = find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by="normalized_margin",
        )
    except (ImportError, Exception):
        # Fallback to native MIT Confident Joint filtering
        C, _ = estimate_confident_joint_and_noise_rate(labels, pred_probs)
        # Class thresholds
        thresholds = np.zeros(num_classes, dtype=np.float64)
        for j in range(num_classes):
            mask = (labels == j)
            thresholds[j] = np.mean(pred_probs[mask, j]) if np.any(mask) else 1.0 / num_classes

        flagged = []
        for i in range(n_samples):
            s = labels[i]
            probs_i = pred_probs[i]
            above = np.where(probs_i >= thresholds)[0]
            if len(above) > 0:
                best_c = above[np.argmax(probs_i[above])]
            else:
                best_c = np.argmax(probs_i)
            # Flagged if assigned to a class other than given label s
            if best_c != s:
                flagged.append(i)

        if len(flagged) > 0:
            flagged = np.array(flagged, dtype=np.int64)
            # Sort flagged by margin ascending (most negative margin first = most suspicious)
            flagged_order = np.argsort(margins[flagged])
            ranked_indices = flagged[flagged_order]
        else:
            ranked_indices = np.array([], dtype=np.int64)

    # Assign scores
    flagged_set = set(ranked_indices.tolist())
    n_flagged = len(ranked_indices)

    # Flagged items get scores in [1.0, 2.0]
    for pos, idx in enumerate(ranked_indices):
        scores[idx] = float(2.0 - (pos / max(n_flagged, 1)))

    # Unflagged items get scores in [0.0, 1.0) based on margin
    # Smaller margin = more suspicious -> higher score
    for i in range(n_samples):
        if i not in flagged_set:
            # margins in [-1.0, 1.0], map to [0.0, 1.0)
            scores[i] = float(np.clip(1.0 - ((margins[i] + 1.0) / 2.0), 0.0, 0.9999))

    return scores

