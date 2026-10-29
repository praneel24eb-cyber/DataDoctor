import numpy as np
from datadoctor.schema import DatasetArtifact


def compute_v3_scores(artifact: DatasetArtifact) -> np.ndarray:
    """View 3: Out-Of-Fold Self-Confidence on independent Backbone B.
    
    Formula: S_V3(i) = 1.0 - p_hat^(B)_{i, y_tilde_i}
    Provides true feature independence from Backbone A.
    Higher score = MORE suspicious.
    """
    if "B" not in artifact.oof_probs:
        raise KeyError("OOF probabilities for Backbone B not found in artifact.")
    
    oof_B = artifact.oof_probs["B"]
    train_labels = artifact.given_labels[artifact.train_mask]
    
    n_train = artifact.n_train
    v3_scores = np.zeros(n_train, dtype=np.float32)
    for i in range(n_train):
        v3_scores[i] = 1.0 - oof_B[i, train_labels[i]]
        
    return v3_scores

