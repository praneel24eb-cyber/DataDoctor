import numpy as np
from datadoctor.schema import DatasetArtifact


def compute_deep_knn_scores(
    artifact: DatasetArtifact,
    backbone_key: str = "A",
    k: int = 15,
) -> np.ndarray:
    """Bahri et al. (ICML 2020) unweighted k-NN label disagreement baseline.
    
    Produces discrete steps in {0/k, 1/k, ..., k/k}.
    Demonstrates the value of continuous similarity-weighting in View 2.
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
    neighbor_labels = train_labels[indices]
    disagreements = (neighbor_labels != train_labels[:, np.newaxis]).astype(np.float32)

    # Unweighted fraction of disagreeing neighbors
    scores = np.mean(disagreements, axis=1)
    return scores.astype(np.float32)

