from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np

from datadoctor.schema import DatasetArtifact, HygieneReport


def calibrate_neardup_threshold(
    embeddings: np.ndarray,
    ground_truth_pairs: Set[Tuple[int, int]],
    threshold_candidates: Optional[np.ndarray] = None,
) -> float:
    """Calibrates tau_dup by maximizing F1 score against known ground-truth duplicate pairs."""
    if threshold_candidates is None:
        threshold_candidates = np.linspace(0.90, 0.99, 19)

    best_f1 = -1.0
    best_thresh = 0.965

    # Compute pairwise similarities for candidate pairs
    gt_normalized = {tuple(sorted((p[0], p[1]))) for p in ground_truth_pairs}
    if not gt_normalized:
        return 0.965

    for thresh in threshold_candidates:
        # Evaluate threshold
        detected_pairs = set()
        # Fast evaluation across pairs
        for (i, j) in gt_normalized:
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= thresh:
                detected_pairs.add((i, j))

        tp = len(detected_pairs.intersection(gt_normalized))
        fp = len(detected_pairs - gt_normalized)
        fn = len(gt_normalized - detected_pairs)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return best_thresh


def scan_dataset_hygiene(
    artifact: DatasetArtifact,
    tau_dup: float = 0.965,
    k_mutual: int = 15,
    backbone_key: str = "A",
) -> HygieneReport:
    """Three-Pass Dataset Hygiene Scanner operating over the entire dataset (train + test).
    
    Pass 1: Exact Duplicates via SHA-1 hashing (zero cosine drift).
    Pass 2: Near-Duplicates via Mutual k-NN graph on Backbone A embeddings with cosine sim >= tau_dup.
    Pass 3: Cross-Split Train/Test Leakage and In-Split Label Conflicts.
    """
    n_total = len(artifact.sample_ids)
    splits = artifact.split
    labels = artifact.given_labels
    hashes = artifact.sha1_hashes

    # --- PASS 1: Exact Duplicates via SHA-1 ---
    hash_groups = defaultdict(list)
    for idx, h in enumerate(hashes):
        hash_groups[h].append(idx)

    exact_dup_groups = [group for group in hash_groups.values() if len(group) > 1]
    exact_dup_pairs: Set[Tuple[int, int]] = set()
    for grp in exact_dup_groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                exact_dup_pairs.add((min(grp[i], grp[j]), max(grp[i], grp[j])))

    # --- PASS 2: Near-Duplicates via Mutual k-NN on Backbone A ---
    near_dup_pairs: List[Tuple[int, int, float]] = []
    if backbone_key in artifact.embeddings:
        emb_all = artifact.embeddings[backbone_key]  # shape: (N_total, d)
        
        # Build top-k nearest neighbors on all samples
        # Chunked dot product
        batch_size = 1000
        topk_indices = np.zeros((n_total, k_mutual), dtype=np.int32)
        topk_sims = np.zeros((n_total, k_mutual), dtype=np.float32)

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            chunk = emb_all[start:end]
            sims = np.dot(chunk, emb_all.T)
            
            for l_idx in range(end - start):
                g_idx = start + l_idx
                # Zero out self
                sims[l_idx, g_idx] = -1.0
                top_idx = np.argpartition(-sims[l_idx], k_mutual - 1)[:k_mutual]
                sorted_top = top_idx[np.argsort(-sims[l_idx, top_idx])]
                topk_indices[g_idx] = sorted_top
                topk_sims[g_idx] = sims[l_idx, sorted_top]

        # Check Mutual k-NN and sim >= tau_dup
        seen_near_pairs: Set[Tuple[int, int]] = set()
        for i in range(n_total):
            for rank_pos in range(k_mutual):
                j = topk_indices[i, rank_pos]
                sim_ij = topk_sims[i, rank_pos]
                if sim_ij >= tau_dup:
                    # Check if i is also in j's top-k neighbors (Mutual k-NN)
                    if i in topk_indices[j]:
                        pair = (min(i, j), max(i, j))
                        # Do not count if already an exact SHA-1 duplicate
                        if pair not in exact_dup_pairs and pair not in seen_near_pairs:
                            seen_near_pairs.add(pair)
                            near_dup_pairs.append((pair[0], pair[1], float(sim_ij)))

    # --- PASS 3: Cross-Split Leakage & Label Conflicts ---
    cross_split_leakage: List[Tuple[int, int, str]] = []
    label_conflicts: List[Tuple[int, int]] = []

    # Check exact duplicates
    for i, j in exact_dup_pairs:
        if splits[i] != splits[j]:
            cross_split_leakage.append((i, j, "exact_sha1"))
        elif labels[i] != labels[j]:
            label_conflicts.append((i, j))

    # Check near duplicates
    for i, j, sim in near_dup_pairs:
        if splits[i] != splits[j]:
            cross_split_leakage.append((i, j, f"near_dup_{sim:.4f}"))
        elif labels[i] != labels[j]:
            label_conflicts.append((i, j))

    summary = {
        "n_exact_duplicate_groups": len(exact_dup_groups),
        "n_exact_duplicate_pairs": len(exact_dup_pairs),
        "n_near_duplicate_pairs": len(near_dup_pairs),
        "n_cross_split_leakage_pairs": len(cross_split_leakage),
        "n_label_conflict_pairs": len(label_conflicts),
        "tau_dup": tau_dup,
        "k_mutual": k_mutual,
    }

    return HygieneReport(
        exact_duplicate_groups=exact_dup_groups,
        near_duplicate_pairs=near_dup_pairs,
        cross_split_leakage_pairs=cross_split_leakage,
        label_conflict_pairs=label_conflicts,
        summary=summary,
    )
