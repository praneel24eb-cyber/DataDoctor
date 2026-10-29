import numpy as np
import pytest

from datadoctor.schema import DatasetArtifact
from datadoctor.fusion.rank_fusion import compute_rank_fusion
from datadoctor.evaluation.metrics import evaluate_detection_metrics, compute_spearman_correlations
from datadoctor.evaluation.budget_curve import evaluate_review_budget_curve
from datadoctor.evaluation.bootstrap import compute_bootstrap_ci, paired_wilcoxon_test


def test_rrf_fusion_and_ablations():
    """Gate 3 Assertion: Assert rank_fusion.py populates fused_score, fused_rank (1...N_train),
    and u_rrf (in [0, 1]), plus ablation keys rrf_v1_v3 and rrf_v1_v2B_v3.
    """
    n_train = 50
    rng = np.random.default_rng(42)

    v1 = rng.uniform(0.1, 0.9, n_train).astype(np.float32)
    v2 = rng.uniform(0.1, 0.9, n_train).astype(np.float32)
    v3 = rng.uniform(0.1, 0.9, n_train).astype(np.float32)
    v2_B = rng.uniform(0.1, 0.9, n_train).astype(np.float32)

    artifact = DatasetArtifact(
        dataset_name="test_rrf",
        modality="text",
        sample_ids=np.arange(n_train),
        split=np.array(["train"] * n_train),
        raw_references=[f"ref_{i}" for i in range(n_train)],
        sha1_hashes=np.array([f"h_{i}" for i in range(n_train)], dtype="<U40"),
        class_names=["c0", "c1"],
        given_labels=np.zeros(n_train, dtype=np.int64),
        view_scores={"v1": v1, "v2": v2, "v3": v3, "v2_B": v2_B},
    )

    compute_rank_fusion(artifact, kappa=60)

    assert artifact.fused_score is not None
    assert len(artifact.fused_score) == n_train
    assert artifact.fused_rank is not None
    # Strictly 1 .. N_train permutation
    assert set(artifact.fused_rank.tolist()) == set(range(1, n_train + 1))

    assert artifact.u_rrf is not None
    assert len(artifact.u_rrf) == n_train
    assert np.all(artifact.u_rrf >= 0.0) and np.all(artifact.u_rrf <= 1.0)
    assert np.isclose(np.max(artifact.u_rrf), 1.0)
    assert np.isclose(np.min(artifact.u_rrf), 0.0)

    # Check ablation keys
    assert "ablations" in artifact.meta
    ablations = artifact.meta["ablations"]
    assert "rrf_v1_v3" in ablations
    assert "rrf_v1_v2B_v3" in ablations
    assert "mean_percentile" in ablations


def test_budget_curve_train_isolation_and_modes():
    """Gate 3 Assertion: Assert budget_curve.py runs only on train_mask and evaluates on clean
    test_mask, comparing oracle_relabel, remove, and ensemble_argmax_relabel.
    """
    n_train = 80
    n_test = 40
    n_total = n_train + n_test
    d = 16
    rng = np.random.default_rng(99)

    # Embeddings for Backbone A
    emb_A = rng.standard_normal((n_total, d)).astype(np.float32)
    emb_A /= np.linalg.norm(emb_A, axis=1, keepdims=True)

    true_labels = np.array([0] * (n_total // 2) + [1] * (n_total - n_total // 2))
    given_labels = true_labels.copy()
    # Inject noise into 10 training samples
    given_labels[:10] = 1 - given_labels[:10]

    splits = np.array(["train"] * n_train + ["test"] * n_test)

    # Fake OOF predictions
    oof_A = np.full((n_train, 2), 0.5, dtype=np.float32)
    oof_A[10:, :] = 0.1
    for i in range(10, n_train):
        oof_A[i, given_labels[i]] = 0.9

    artifact = DatasetArtifact(
        dataset_name="test_budget",
        modality="text",
        sample_ids=np.arange(n_total),
        split=splits,
        raw_references=[f"ref_{i}" for i in range(n_total)],
        sha1_hashes=np.array([f"h_{i}" for i in range(n_total)], dtype="<U40"),
        class_names=["c0", "c1"],
        given_labels=given_labels,
        true_labels=true_labels,
        is_noisy=(given_labels[:n_train] != true_labels[:n_train]),
        embeddings={"A": emb_A},
        oof_probs={"A": oof_A},
    )

    # Compute V1 score and fused score
    artifact.view_scores["v1"] = 1.0 - np.array([oof_A[i, given_labels[i]] for i in range(n_train)])
    artifact.view_scores["v2"] = artifact.view_scores["v1"].copy()
    artifact.view_scores["v3"] = artifact.view_scores["v1"].copy()
    compute_rank_fusion(artifact)

    results = evaluate_review_budget_curve(
        artifact,
        budgets=[0.05, 0.10],
        modes=["oracle_relabel", "remove", "ensemble_argmax_relabel"],
        backbone_key="A",
    )

    assert "oracle_relabel" in results
    assert "remove" in results
    assert "ensemble_argmax_relabel" in results

    for m in ["oracle_relabel", "remove", "ensemble_argmax_relabel"]:
        assert 0.0 in results[m]
        assert 0.05 in results[m]
        assert 0.10 in results[m]
        assert 0.0 <= results[m][0.10] <= 1.0


def test_metrics_and_bootstrap():
    """Verifies AUROC, AUPRC, Spearman correlation, and bootstrap confidence intervals."""
    rng = np.random.default_rng(7)
    n = 100
    is_noisy = np.zeros(n, dtype=bool)
    is_noisy[:20] = True
    scores = np.zeros(n)
    scores[:20] = rng.uniform(0.7, 1.0, 20)
    scores[20:] = rng.uniform(0.0, 0.5, 80)

    metrics = evaluate_detection_metrics(scores, is_noisy)
    assert metrics["auroc"] > 0.95
    assert metrics["auprc"] > 0.90
    assert metrics["p_at_20pct"] > 0.85

    # Bootstrap CI
    pt, lo, hi = compute_bootstrap_ci(
        lambda y, s: float(np.mean((s > 0.6) == y)),
        is_noisy,
        scores,
        n_bootstraps=50,
    )
    assert lo <= pt <= hi

    # Spearman
    view_scores = {"v1": scores, "v2": scores + rng.normal(0, 0.1, n)}
    corrs = compute_spearman_correlations(view_scores, is_noisy)
    assert "v1_v2" in corrs["all"]

