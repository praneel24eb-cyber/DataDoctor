from typing import Optional
import numpy as np

from datadoctor.schema import DatasetArtifact


def assign_route_buckets(
    artifact: DatasetArtifact,
    budget_cap_fraction: float = 0.02,
    delta_gap: float = 0.35,
) -> DatasetArtifact:
    """Noise-Adaptive Disjoint Percentile Router.
    
    Assigns every training sample i in {0, ..., N_train-1} into strictly disjoint buckets:
    - 'auto_flag': All 3 views agree the sample is highly suspicious.
    - 'auto_clear': All 3 views agree suspicion is low (never overwritten!).
    - 'llm_judge': Parametric (V1) and Geometric (V2) views strongly disagree,
                   capped at B = floor(budget_cap_fraction * N_train) by highest S_RRF.
    - 'human_queue': Ambiguous samples, plus leftover disagreement candidates beyond budget cap B.
    """
    n_train = artifact.n_train
    if n_train == 0:
        return artifact

    # Estimated noise rate clipped to [0.05, 0.45]
    eta_hat = float(np.clip(artifact.estimated_noise_rate, 0.05, 0.45))
    tau_high = 1.0 - 0.5 * eta_hat
    tau_low = max(0.50, 1.0 - 2.5 * eta_hat)

    # Per-view percentiles
    u_v1 = artifact.view_percentiles.get("v1")
    u_v2 = artifact.view_percentiles.get("v2")
    u_v3 = artifact.view_percentiles.get("v3")

    if u_v1 is None or u_v2 is None or u_v3 is None:
        raise KeyError("Views v1, v2, v3 percentiles must be populated before routing.")

    fused_score = artifact.fused_score
    if fused_score is None:
        raise KeyError("artifact.fused_score must be populated before routing.")

    buckets = np.empty(n_train, dtype=object)
    disagreement_candidates = []

    # Strictly disjoint cascade
    for i in range(n_train):
        min_u = min(u_v1[i], u_v2[i], u_v3[i])
        max_u = max(u_v1[i], u_v2[i], u_v3[i])
        max_v1_v2 = max(u_v1[i], u_v2[i])
        gap = abs(u_v1[i] - u_v2[i])

        if min_u >= tau_high:
            buckets[i] = "auto_flag"
        elif max_u <= tau_low:
            buckets[i] = "auto_clear"
        elif max_v1_v2 >= tau_high and gap >= delta_gap:
            buckets[i] = "candidate"
            disagreement_candidates.append(i)
        else:
            buckets[i] = "human_queue"

    # Enforce hard LLM budget cap B = floor(budget_cap_fraction * N_train)
    B = int(np.floor(budget_cap_fraction * n_train))
    if len(disagreement_candidates) > 0:
        disagreement_candidates = np.array(disagreement_candidates, dtype=np.int64)
        # Sort candidates descending by fused_score
        cand_scores = fused_score[disagreement_candidates]
        sorted_cand_idx = disagreement_candidates[np.argsort(-cand_scores)]

        routed_to_llm = sorted_cand_idx[:B]
        routed_to_human = sorted_cand_idx[B:]

        for idx in routed_to_llm:
            buckets[idx] = "llm_judge"
        for idx in routed_to_human:
            buckets[idx] = "human_queue"

    artifact.route_bucket = buckets.astype(str)
    return artifact

