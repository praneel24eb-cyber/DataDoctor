import numpy as np
from datadoctor.schema import DatasetArtifact


def compute_v1_scores(artifact: DatasetArtifact) -> np.ndarray:
    """View 1: Out-Of-Fold Self-Confidence on Backbone A.
    
    Formula: S_V1(i) = 1.0 - p_hat^(A)_{i, y_tilde_i}
    Higher score = MORE suspicious.
    
    Operates strictly on the training split (N_train).
    """
    if "A" not in artifact.oof_probs:
        raise KeyError("OOF probabilities for Backbone A not found in artifact.")
    
    oof_A = artifact.oof_probs["A"]
    train_labels = artifact.given_labels[artifact.train_mask]
    
    n_train = artifact.n_train
    v1_scores = np.zeros(n_train, dtype=np.float32)
    for i in range(n_train):
        v1_scores[i] = 1.0 - oof_A[i, train_labels[i]]
        
    return v1_scores

