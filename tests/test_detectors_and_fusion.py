import numpy as np
import pytest

from datadoctor.schema import DatasetArtifact
from datadoctor.detectors.oof_probe import compute_oof_probabilities
from datadoctor.detectors.v1_self_conf import compute_v1_scores
from datadoctor.detectors.v2_knn import compute_v2_scores, build_knn_index
from datadoctor.detectors.v3_cross_probe import compute_v3_scores
from datadoctor.baselines.cleanlab_wrap import compute_cleanlab_scores


def test_cleanlab_vs_loss_self_conf_distinct():
    """Gate 2 Assertion: Assert cleanlab_cl and loss_self_conf produce different
    rankings on a synthetic dataset with class-imbalanced confidence.
    """
    num_classes = 3
    # Two classes with different confidence baselines:
    # Class 0: low confidence baseline (avg p ~ 0.55) -> threshold t_0 ~ 0.55
    # Class 1: high confidence baseline (avg p ~ 0.95) -> threshold t_1 ~ 0.95
    labels = np.array([0] * 30 + [1] * 30)
    pred_probs = np.zeros((60, num_classes), dtype=np.float32)
    
    # Class 0 background
    pred_probs[:30, 0] = 0.55
    pred_probs[:30, 1] = 0.45
    
    # Class 1 background
    pred_probs[30:, 1] = 0.95
    pred_probs[30:, 0] = 0.05
    
    # Sample 0 (label 0): has lower self-confidence (0.42), but no other class exceeds its threshold
    pred_probs[0, 0] = 0.42
    pred_probs[0, 1] = 0.29
    pred_probs[0, 2] = 0.29
    # Self-conf loss = 1 - 0.42 = 0.58
    
    # Sample 30 (label 1): has higher self-confidence (0.44), but class 0 prob is 0.56 >= t_0 (0.55)!
    # This sample gets FLAGGED in off-diagonal C_{1, 0}
    pred_probs[30, 1] = 0.44
    pred_probs[30, 0] = 0.56
    # Self-conf loss = 1 - 0.44 = 0.56
    
    # Raw self-conf score: Sample 0 (0.58) is MORE suspicious than Sample 30 (0.56)
    loss_self_conf = 1.0 - np.array([pred_probs[i, labels[i]] for i in range(60)])
    cleanlab_scores = compute_cleanlab_scores(labels, pred_probs)
    
    # Under loss_self_conf: sample 0 has higher score than sample 30
    assert loss_self_conf[0] > loss_self_conf[30]
    
    # Under cleanlab_cl: sample 30 is flagged (score >= 1.0) while sample 0 is unflagged (score < 1.0)
    assert cleanlab_scores[30] > cleanlab_scores[0]
    
    rank_self_conf = np.argsort(-loss_self_conf)
    rank_cleanlab = np.argsort(-cleanlab_scores)
    
    assert not np.array_equal(rank_self_conf, rank_cleanlab), (
        "cleanlab_cl and loss_self_conf must produce distinct rankings under class imbalance"
    )


def test_v2_knn_excludes_exact_duplicates_via_sha1():
    """Gate 2 Assertion: Assert v2_knn excludes exact duplicates strictly via
    sha1_hashes[i] == sha1_hashes[j] (never cosine threshold).
    """
    n_samples = 10
    d = 16
    rng = np.random.default_rng(42)
    
    # Create distinct embeddings
    emb = rng.standard_normal((n_samples, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    
    # Make sample 1 an exact duplicate of sample 0 with identical hash
    emb[1] = emb[0]
    
    hashes = np.array([f"unique_hash_{i}" for i in range(n_samples)], dtype="<U40")
    hashes[1] = hashes[0] # Exact duplicate hash!
    
    knn_data = build_knn_index(emb, hashes, k_max=3)
    
    # For sample 0, index 1 MUST be excluded despite having cosine similarity = 1.0
    assert 1 not in knn_data.indices[0], "Exact duplicate index 1 must be excluded from sample 0's neighbors"
    assert 0 not in knn_data.indices[1], "Exact duplicate index 0 must be excluded from sample 1's neighbors"


def test_fast_oof_probe_convergence_and_shapes():
    """Tests that PyTorch L-BFGS OOF probe runs and outputs valid probability distributions."""
    n_samples = 120
    d = 32
    num_classes = 3
    rng = np.random.default_rng(101)
    
    emb = rng.standard_normal((n_samples, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.repeat(np.arange(num_classes), n_samples // num_classes)
    
    oof_probs = compute_oof_probabilities(emb, labels, num_classes=num_classes, n_folds=3, seed=42)
    
    assert oof_probs.shape == (n_samples, num_classes)
    assert not np.isnan(oof_probs).any()
    # Check that each row sums to 1.0 within float precision
    row_sums = np.sum(oof_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    assert np.all(oof_probs >= 0.0)
