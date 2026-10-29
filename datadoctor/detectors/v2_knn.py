from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

from datadoctor.schema import DatasetArtifact, KNNIndexData


def build_knn_index(
    embeddings_train: np.ndarray,
    sha1_hashes_train: np.ndarray,
    k_max: int = 50,
    batch_size: int = 2000,
) -> KNNIndexData:
    """Builds and caches the top k_max nearest neighbours on training embeddings.
    
    CRITICAL INVARIANT: Exact-Duplicate Exclusion Rule.
    When building the neighbour list for training sample i, exclude index j if:
    j == i OR sha1_hashes[i] == sha1_hashes[j].
    (Never rely on a cosine threshold to filter exact duplicates).
    """
    n_samples = len(embeddings_train)
    actual_k = min(k_max, n_samples - 1)
    
    indices_out = np.zeros((n_samples, actual_k), dtype=np.int32)
    sims_out = np.zeros((n_samples, actual_k), dtype=np.float32)

    # Fast chunked dot product
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        chunk = embeddings_train[start:end]
        sims_chunk = np.dot(chunk, embeddings_train.T) # Shape: (B, N)

        for local_idx in range(end - start):
            global_idx = start + local_idx
            i_hash = sha1_hashes_train[global_idx]
            sims_i = sims_chunk[local_idx]

            # Find candidates: sort descending by similarity
            # Take top actual_k + buffer to allow for duplicates
            buffer_size = min(n_samples, actual_k + 100)
            candidate_indices = np.argpartition(-sims_i, buffer_size - 1)[:buffer_size]
            candidate_indices = candidate_indices[np.argsort(-sims_i[candidate_indices])]

            valid_j = []
            valid_sims = []
            for j in candidate_indices:
                if j == global_idx:
                    continue
                if sha1_hashes_train[j] == i_hash:
                    continue
                valid_j.append(j)
                valid_sims.append(sims_i[j])
                if len(valid_j) == actual_k:
                    break

            # If fewer than actual_k valid non-duplicate neighbours exist
            while len(valid_j) < actual_k:
                valid_j.append(valid_j[-1] if valid_j else global_idx)
                valid_sims.append(0.0)

            indices_out[global_idx] = valid_j
            sims_out[global_idx] = valid_sims

    return KNNIndexData(indices=indices_out, similarities=sims_out)


def compute_v2_scores(
    artifact: DatasetArtifact,
    backbone_key: str = "A",
    k: int = 15,
) -> np.ndarray:
    """View 2: Similarity-Weighted k-NN Disagreement on Backbone embeddings.
    
    Formula:
    S_V2(i) = \\frac{\\sum_{j \\in \\mathcal{N}_k(i)} w_{ij} \\mathbb{I}(\\tilde{y}_j \\neq \\tilde{y}_i)}{\\sum_{j \\in \\mathcal{N}_k(i)} w_{ij}}
    where w_{ij} = \\max(s_{ij}, 10^{-6}).
    """
    t_mask = artifact.train_mask
    train_labels = artifact.given_labels[t_mask]
    n_train = artifact.n_train

    if backbone_key not in artifact.knn_cache:
        if backbone_key not in artifact.embeddings:
            raise KeyError(f"Embeddings for backbone {backbone_key} not found in artifact.")
        emb_train = artifact.embeddings[backbone_key][t_mask]
        hashes_train = artifact.sha1_hashes[t_mask]
        artifact.knn_cache[backbone_key] = build_knn_index(emb_train, hashes_train, k_max=max(k, 50))

    knn_data = artifact.knn_cache[backbone_key]
    indices = knn_data.indices[:, :k]
    sims = knn_data.similarities[:, :k]

    # Clamp weights
    weights = np.maximum(sims, 1e-6)

    # Check neighbor label disagreement
    neighbor_labels = train_labels[indices]
    disagreements = (neighbor_labels != train_labels[:, np.newaxis]).astype(np.float32)

    v2_scores = np.sum(weights * disagreements, axis=1) / np.sum(weights, axis=1)
    return v2_scores.astype(np.float32)

