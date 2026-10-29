import numpy as np
from datadoctor.schema import DatasetArtifact


def compute_simifeat_scores(
    artifact: DatasetArtifact,
    backbone_key: str = "A",
    k: int = 15,
) -> np.ndarray:
    """Zhu et al. (ICML 2022) SimiFeat-R local consensus baseline.
    
    Measures local representation disagreement:
    S_simifeat(i) = 1.0 - \\frac{\\sum_{j \\in \\mathcal{N}_k(i), \\tilde{y}_j = \\tilde{y}_i} s_{ij}}{\\sum_{j \\in \\mathcal{N}_k(i)} s_{ij}}
    Higher score = MORE suspicious.
    """
    t_mask = artifact.train_mask
    train_labels = artifact.given_labels[t_mask]
    n_train = artifact.n_train

    knn_data = artifact.knn_cache.get(backbone_key)
    if knn_data is None:
        from datadoctor.detectors.v2_knn import build_knn_index
        emb_train = artifact.embeddings[backbone_key][t_mask]
        hashes_train = artifact.sha1_hashes[t_mask]
        knn_data = build_knn_index(emb_train, hashes_train, k_max=max(k, 50))
        artifact.knn_cache[backbone_key] = knn_data

    indices = knn_data.indices[:, :k]
    sims = np.maximum(knn_data.similarities[:, :k], 1e-6)

    neighbor_labels = train_labels[indices]
    agreements = (neighbor_labels == train_labels[:, np.newaxis]).astype(np.float32)

    agreement_score = np.sum(sims * agreements, axis=1) / np.sum(sims, axis=1)
    scores = 1.0 - agreement_score
    return scores.astype(np.float32)

