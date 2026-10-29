from typing import Dict, Tuple, Optional
import numpy as np
from scipy.stats import rankdata

from datadoctor.schema import DatasetArtifact


def scores_to_ranks_and_percentiles(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts continuous suspicion scores to 1-indexed average ranks and percentiles.
    
    Higher score = MORE suspicious.
    Rank 1 = MOST suspicious.
    Percentile u = 1.0 = MOST suspicious.
    
    u_m(i) = (N - rank_m(i)) / (N - 1) in [0, 1].
    """
    n = len(scores)
    if n <= 1:
        return np.ones(n, dtype=np.float64), np.ones(n, dtype=np.float64)

    # Higher score gets smaller rank number (rank 1 is highest score)
    ranks = rankdata(-scores, method="average")
    percentiles = (n - ranks) / (n - 1)
    percentiles = np.clip(percentiles, 0.0, 1.0)
    return ranks, percentiles


def compute_rank_fusion(
    artifact: DatasetArtifact,
    kappa: int = 60,
) -> DatasetArtifact:
    """Executes Reciprocal Rank Fusion (RRF) and required ablation variants.
    
    Operates strictly on the training split (N_train).
    
    Default 3-View RRF (rrf_v1_v2_v3):
    S_RRF(i) = \\sum_{m \\in {V1, V2, V3}} \\frac{1}{\\kappa + rank_m(i)} + 10^{-9} \\cdot S_{V1}(i)
    
    Populates:
    - artifact.fused_score
    - artifact.fused_rank (strictly 1 .. N_train permutation)
    - artifact.u_rrf in [0, 1]
    - artifact.meta['ablations']: 'rrf_v1_v3', 'rrf_v1_v2B_v3', 'mean_percentile'
    """
    n_train = artifact.n_train
    if n_train == 0:
        return artifact

    # Compute ranks and percentiles for each present view
    for view_key, scores in artifact.view_scores.items():
        ranks, percentiles = scores_to_ranks_and_percentiles(scores)
        artifact.view_ranks[view_key] = ranks
        artifact.view_percentiles[view_key] = percentiles

    # Ensure core views V1, V2, V3 exist
    v1_rank = artifact.view_ranks.get("v1")
    v2_rank = artifact.view_ranks.get("v2")
    v3_rank = artifact.view_ranks.get("v3")
    v1_score = artifact.view_scores.get("v1", np.zeros(n_train, dtype=np.float32))

    if v1_rank is None or v2_rank is None or v3_rank is None:
        raise KeyError("Views v1, v2, and v3 must be present in artifact.view_scores for RRF fusion.")

    # 1. Main 3-View RRF: RRF(V1, V2_A, V3)
    s_rrf = (
        1.0 / (kappa + v1_rank) +
        1.0 / (kappa + v2_rank) +
        1.0 / (kappa + v3_rank) +
        1e-9 * v1_score
    )
    artifact.fused_score = s_rrf.astype(np.float64)

    # 1-indexed strict permutation rank: np.argsort(np.argsort(-S_RRF)) + 1
    fused_rank = np.argsort(np.argsort(-s_rrf)) + 1
    artifact.fused_rank = fused_rank.astype(np.int64)

    if n_train > 1:
        artifact.u_rrf = ((n_train - fused_rank) / (n_train - 1)).astype(np.float64)
    else:
        artifact.u_rrf = np.ones(n_train, dtype=np.float64)

    # 2. Ablation Variants
    ablations: Dict[str, np.ndarray] = {}

    # Ablation 1: 2-View cross-backbone model fusion without k-NN (V1 + V3)
    s_v1_v3 = (
        1.0 / (kappa + v1_rank) +
        1.0 / (kappa + v3_rank) +
        1e-9 * v1_score
    )
    ablations["rrf_v1_v3"] = s_v1_v3.astype(np.float64)

    # Ablation 2: 3-View fusion where V2 runs on Backbone B (v2_B)
    if "v2_B" in artifact.view_ranks:
        v2B_rank = artifact.view_ranks["v2_B"]
        s_v1_v2B_v3 = (
            1.0 / (kappa + v1_rank) +
            1.0 / (kappa + v2B_rank) +
            1.0 / (kappa + v3_rank) +
            1e-9 * v1_score
        )
        ablations["rrf_v1_v2B_v3"] = s_v1_v2B_v3.astype(np.float64)

    # Ablation 3: Mean Percentile (1/3 * (u_V1 + u_V2 + u_V3))
    u1 = artifact.view_percentiles["v1"]
    u2 = artifact.view_percentiles["v2"]
    u3 = artifact.view_percentiles["v3"]
    ablations["mean_percentile"] = ((u1 + u2 + u3) / 3.0).astype(np.float64)

    artifact.meta["ablations"] = ablations
    return artifact

