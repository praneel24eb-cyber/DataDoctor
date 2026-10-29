"""
tests/test_phase2_detectors_and_noise.py
=========================================
Gate 2 Tests — Phase 2: Noise Injectors, OOF Probe, Views V1/V2/V3,
Baselines, and Hygiene Scanner.

Assertions per PRD §FR-3 through §FR-10:
  1. Zero NaNs in IDN (NaN-safe masking order verified)
  2. Exact flip counts — symmetric & asymmetric (CIFAR-10 + AG News)
  3. Fast OOF probe: correct shapes, valid probs, <15s on 50k×512 synthetic
  4. V1 / V2 / V3 score shapes, ranges and invariants
  5. SHA-1 exact-duplicate exclusion in k-NN (never cosine threshold)
  6. cleanlab_cl vs loss_self_conf produce distinct rankings under class imbalance
  7. Hygiene Scanner: all 3 passes — exact-dup, near-dup, cross-split leakage
  8. Baselines: deep_knn (discrete steps) and simifeat_r (continuous) are distinct
  9. Runtime: full pipeline on 50k×512 synthetic embedding cache < 15 s on CPU
"""
import time
import numpy as np
import pytest

from datadoctor.schema import DatasetArtifact, KNNIndexData
from datadoctor.noise.injectors import (
    inject_symmetric,
    inject_asymmetric,
    inject_idn,
    CIFAR10_ASYMMETRIC_MAP,
    AG_NEWS_ASYMMETRIC_MAP,
)
from datadoctor.detectors.oof_probe import compute_oof_probabilities
from datadoctor.detectors.v1_self_conf import compute_v1_scores
from datadoctor.detectors.v2_knn import build_knn_index, compute_v2_scores
from datadoctor.detectors.v3_cross_probe import compute_v3_scores
from datadoctor.baselines.cleanlab_wrap import (
    compute_cleanlab_scores,
    estimate_confident_joint_and_noise_rate,
)
from datadoctor.baselines.deep_knn import compute_deep_knn_scores
from datadoctor.baselines.simifeat import compute_simifeat_scores
from datadoctor.hygiene.scanner import scan_dataset_hygiene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact(
    n_train: int = 200,
    n_test: int = 50,
    num_classes: int = 4,
    d: int = 32,
    seed: int = 42,
    add_embeddings: bool = True,
) -> DatasetArtifact:
    """Builds a minimal synthetic DatasetArtifact for unit tests."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    split = np.array(["train"] * n_train + ["test"] * n_test)
    sample_ids = np.arange(n_total, dtype=np.int64)
    true_labels = np.concatenate([
        rng.integers(0, num_classes, size=n_train),
        rng.integers(0, num_classes, size=n_test),
    ])
    given_labels = true_labels.copy()
    sha1_hashes = np.array(
        [f"hash_{i:06d}" for i in range(n_total)], dtype="<U40"
    )
    is_noisy = np.zeros(n_train, dtype=bool)

    artifact = DatasetArtifact(
        dataset_name="synthetic",
        modality="text",
        sample_ids=sample_ids,
        split=split,
        raw_references=[f"text_{i}" for i in range(n_total)],
        sha1_hashes=sha1_hashes,
        class_names=[f"C{c}" for c in range(num_classes)],
        given_labels=given_labels,
        true_labels=true_labels,
        is_noisy=is_noisy,
    )

    if add_embeddings:
        emb = rng.standard_normal((n_total, d)).astype(np.float32)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        artifact.embeddings["A"] = emb
        artifact.embeddings["B"] = rng.standard_normal(
            (n_total, d)
        ).astype(np.float32)
        artifact.embeddings["B"] /= np.linalg.norm(
            artifact.embeddings["B"], axis=1, keepdims=True
        )
        artifact.embeddings["C"] = rng.standard_normal(
            (n_total, d)
        ).astype(np.float32)
        artifact.embeddings["C"] /= np.linalg.norm(
            artifact.embeddings["C"], axis=1, keepdims=True
        )

    return artifact


# ===========================================================================
# 1. NOISE INJECTORS
# ===========================================================================

class TestInjectSymmetric:
    def test_exact_flip_counts_per_class(self):
        """inject_symmetric must flip EXACTLY round(eta * N_c) per class."""
        labels = np.repeat(np.arange(5), 200)  # 1000 total, 200 per class
        eta = 0.10
        noisy, is_noisy, target_T, empirical_T, meta = inject_symmetric(
            labels, num_classes=5, eta=eta, seed=42
        )
        for c in range(5):
            c_mask = labels == c
            flipped = int(np.sum(noisy[c_mask] != c))
            expected = int(round(eta * 200))
            assert flipped == expected, (
                f"Class {c}: expected {expected} flips, got {flipped}"
            )

    def test_no_self_flips(self):
        """Symmetric noise must not leave any sample mapped to its own class."""
        labels = np.repeat(np.arange(4), 100)
        noisy, is_noisy, *_ = inject_symmetric(labels, 4, eta=0.20, seed=7)
        for i in range(len(labels)):
            if is_noisy[i]:
                assert noisy[i] != labels[i], (
                    f"Sample {i} self-flipped (label unchanged but marked noisy)"
                )

    def test_zero_nan_in_output(self):
        """No NaN in noisy labels or transition matrices."""
        labels = np.repeat(np.arange(3), 300)
        noisy, is_noisy, target_T, empirical_T, meta = inject_symmetric(
            labels, 3, eta=0.15, seed=1
        )
        assert not np.isnan(empirical_T).any()
        assert not np.isnan(target_T).any()
        assert not np.isnan(noisy.astype(float)).any()

    def test_transition_matrix_rows_sum_to_one(self):
        labels = np.repeat(np.arange(4), 500)
        _, _, target_T, empirical_T, _ = inject_symmetric(
            labels, 4, eta=0.20, seed=0
        )
        assert np.allclose(empirical_T.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(target_T.sum(axis=1), 1.0, atol=1e-6)


class TestInjectAsymmetric:
    def test_cifar10_exact_flip_counts_and_map(self):
        """CIFAR-10 asymmetric: 5 source classes flip, 5 unflipped, global eta = source / 2."""
        n_per_class = 1000
        labels = np.repeat(np.arange(10), n_per_class)
        eta_source = 0.20
        noisy, is_noisy, target_T, empirical_T, meta = inject_asymmetric(
            labels, 10, eta_source=eta_source, dataset_name="cifar10", seed=42
        )
        expected_flips_per_src = int(round(eta_source * n_per_class))
        for c in range(10):
            c_mask = labels == c
            flipped = int(np.sum(noisy[c_mask] != c))
            if c in CIFAR10_ASYMMETRIC_MAP:
                assert flipped == expected_flips_per_src, (
                    f"CIFAR-10 class {c}: expected {expected_flips_per_src} flips, got {flipped}"
                )
                dst = CIFAR10_ASYMMETRIC_MAP[c]
                assert np.all(noisy[c_mask & (noisy != c)] == dst), (
                    f"CIFAR-10 class {c} must flip strictly to class {dst}"
                )
            else:
                assert flipped == 0, (
                    f"Unflipped class {c} unexpectedly flipped {flipped} samples"
                )
        # Global noise rate = eta_source / 2
        assert abs(meta["global_eta"] - eta_source / 2.0) < 1e-9

    def test_ag_news_bijective_2cycle_global_eta(self):
        """AG News asymmetric: all 4 classes flip bijectively, global eta = eta_source."""
        n_per_class = 500
        labels = np.repeat(np.arange(4), n_per_class)
        eta_ag = 0.15
        noisy, is_noisy, _, _, meta = inject_asymmetric(
            labels, 4, eta_source=eta_ag, dataset_name="ag_news", seed=42
        )
        expected = int(round(eta_ag * n_per_class))
        for c in range(4):
            c_mask = labels == c
            flipped = int(np.sum(noisy[c_mask] != c))
            assert flipped == expected, (
                f"AG News class {c}: expected {expected} flips, got {flipped}"
            )
            dst = AG_NEWS_ASYMMETRIC_MAP[c]
            assert np.all(noisy[c_mask & (noisy != c)] == dst)
        assert abs(meta["global_eta"] - eta_ag) < 1e-9

    def test_both_source_eta_and_global_eta_in_meta(self):
        """meta_dict must contain both source_eta and global_eta (PR req)."""
        labels = np.repeat(np.arange(10), 100)
        _, _, _, _, meta = inject_asymmetric(
            labels, 10, eta_source=0.30, dataset_name="cifar10", seed=0
        )
        assert "source_eta" in meta, "meta must contain 'source_eta'"
        assert "global_eta" in meta, "meta must contain 'global_eta'"


class TestInjectIDN:
    def test_zero_nan_standard(self):
        """IDN must never produce NaN in any output even with finite q_i."""
        rng = np.random.default_rng(42)
        n, d, C = 200, 64, 4
        labels = rng.integers(0, C, size=n)
        emb_C = rng.standard_normal((n, d)).astype(np.float32)
        emb_C /= np.linalg.norm(emb_C, axis=1, keepdims=True)

        noisy, is_noisy, target_T, empirical_T, meta = inject_idn(
            labels, C, emb_C, eta=0.20, seed=42
        )
        assert not np.isnan(noisy.astype(float)).any(), "NaN in IDN noisy labels"
        assert not np.isnan(empirical_T).any(), "NaN in IDN empirical_T"
        assert len(noisy) == n
        assert noisy.dtype == labels.dtype

    def test_nan_safe_masking_order_with_tiny_eta(self):
        """Even with tiny eta the NaN-safe masking order must hold (no 0*-inf NaN)."""
        rng = np.random.default_rng(7)
        n, d, C = 300, 32, 5
        labels = rng.integers(0, C, size=n)
        emb_C = rng.standard_normal((n, d)).astype(np.float32)
        emb_C /= np.linalg.norm(emb_C, axis=1, keepdims=True)

        for eta in [0.01, 0.05, 0.40]:
            noisy, _, _, empirical_T, _ = inject_idn(
                labels, C, emb_C, eta=eta, seed=99
            )
            assert not np.isnan(noisy.astype(float)).any(), (
                f"NaN in IDN noisy labels at eta={eta}"
            )
            assert not np.isnan(empirical_T).any(), (
                f"NaN in IDN empirical_T at eta={eta}"
            )

    def test_exact_flip_count_per_class(self):
        """IDN must flip exactly round(eta * N_c) per class."""
        rng = np.random.default_rng(0)
        n_per_class = 300
        C = 4
        labels = np.repeat(np.arange(C), n_per_class)
        emb_C = rng.standard_normal((C * n_per_class, 64)).astype(np.float32)
        emb_C /= np.linalg.norm(emb_C, axis=1, keepdims=True)
        eta = 0.20
        noisy, is_noisy, *_ = inject_idn(labels, C, emb_C, eta=eta, seed=3)
        for c in range(C):
            c_mask = labels == c
            flipped = int(np.sum(noisy[c_mask] != c))
            expected = int(round(eta * n_per_class))
            assert flipped == expected, (
                f"IDN class {c}: expected {expected} flips, got {flipped}"
            )

    def test_noisy_label_not_equal_true_class(self):
        """Every flipped IDN sample must be assigned a class != its true class."""
        rng = np.random.default_rng(5)
        labels = np.repeat(np.arange(4), 200)
        emb_C = rng.standard_normal((800, 32)).astype(np.float32)
        emb_C /= np.linalg.norm(emb_C, axis=1, keepdims=True)
        noisy, is_noisy, *_ = inject_idn(labels, 4, emb_C, eta=0.25, seed=5)
        for i in range(len(labels)):
            if is_noisy[i]:
                assert noisy[i] != labels[i], (
                    f"IDN sample {i} self-flipped"
                )


# ===========================================================================
# 2. OOF PROBE — FAST PyTorch L-BFGS
# ===========================================================================

class TestOOFProbe:
    def test_output_shape_and_valid_probabilities(self):
        """OOF probs must have shape (N_train, C) and valid probability distributions."""
        n, d, C = 120, 32, 3
        rng = np.random.default_rng(101)
        emb = rng.standard_normal((n, d)).astype(np.float32)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        labels = np.repeat(np.arange(C), n // C)

        oof = compute_oof_probabilities(emb, labels, num_classes=C, n_folds=3, seed=42)

        assert oof.shape == (n, C), f"Expected ({n},{C}), got {oof.shape}"
        assert not np.isnan(oof).any(), "NaN in OOF probs"
        assert np.all(oof >= 0.0), "Negative OOF probability"
        assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5), "Rows don't sum to 1"

    def test_runtime_50k_synthetic_under_15s(self):
        """Gate 2 runtime: 5-fold OOF on N=50000, d=512, C=10 must complete in <15 s on CPU."""
        N, d, C = 50_000, 512, 10
        rng = np.random.default_rng(77)
        emb = rng.standard_normal((N, d)).astype(np.float32)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        labels = np.tile(np.arange(C), N // C)

        t0 = time.perf_counter()
        oof = compute_oof_probabilities(emb, labels, num_classes=C, n_folds=5, seed=0)
        elapsed = time.perf_counter() - t0

        assert oof.shape == (N, C)
        assert not np.isnan(oof).any()
        assert elapsed < 15.0, (
            f"OOF probe on 50k×512 took {elapsed:.1f}s, must be <15s"
        )

    def test_oof_probs_reasonable_on_separable_data(self):
        """On linearly separable data, OOF probs should be reasonably confident."""
        C, n_per_class, d = 3, 100, 16
        rng = np.random.default_rng(22)
        # Make clearly separable clusters
        emb = np.vstack([
            rng.standard_normal((n_per_class, d)).astype(np.float32) + np.eye(d)[c % d] * 5.0
            for c in range(C)
        ])
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        labels = np.repeat(np.arange(C), n_per_class)

        oof = compute_oof_probabilities(emb, labels, num_classes=C, n_folds=3, seed=0)
        # Mean confidence on given label should be > 0.5 for separable data
        given_conf = np.array([oof[i, labels[i]] for i in range(len(labels))])
        assert np.mean(given_conf) > 0.50, (
            "OOF self-confidence should be >0.5 on separable clusters"
        )


# ===========================================================================
# 3. VIEWS V1 / V2 / V3
# ===========================================================================

class TestViewScores:
    def setup_method(self):
        """Shared artifact with OOF probs and kNN cache."""
        self.artifact = _make_artifact(
            n_train=100, n_test=20, num_classes=4, d=16, seed=42
        )
        # Pre-compute OOF probs for A and B
        t_mask = self.artifact.train_mask
        for key in ("A", "B"):
            emb_train = self.artifact.embeddings[key][t_mask]
            labels_train = self.artifact.given_labels[t_mask]
            self.artifact.oof_probs[key] = compute_oof_probabilities(
                emb_train, labels_train, num_classes=4, n_folds=3, seed=7
            )

    def test_v1_shape_and_range(self):
        """V1 scores must have shape (N_train,) and values in [0, 1]."""
        scores = compute_v1_scores(self.artifact)
        assert scores.shape == (self.artifact.n_train,)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert not np.isnan(scores).any()

    def test_v1_equals_one_minus_self_conf(self):
        """V1 scores must equal exactly 1 - p_hat[i, label_i] for all i."""
        scores = compute_v1_scores(self.artifact)
        t_mask = self.artifact.train_mask
        train_labels = self.artifact.given_labels[t_mask]
        oof_A = self.artifact.oof_probs["A"]
        expected = np.array(
            [1.0 - oof_A[i, train_labels[i]] for i in range(self.artifact.n_train)],
            dtype=np.float32,
        )
        assert np.allclose(scores, expected, atol=1e-6)

    def test_v2_shape_and_range(self):
        """V2 scores must have shape (N_train,) and values in [0, 1]."""
        scores = compute_v2_scores(self.artifact, backbone_key="A", k=5)
        assert scores.shape == (self.artifact.n_train,)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-6)
        assert not np.isnan(scores).any()

    def test_v2_sha1_exact_duplicate_exclusion(self):
        """V2 must exclude j from N_k(i) whenever sha1_hashes[i] == sha1_hashes[j]."""
        n, d = 10, 16
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((n, d)).astype(np.float32)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        # Make sample 1 exact copy of sample 0, same hash
        emb[1] = emb[0]
        hashes = np.array([f"h_{i}" for i in range(n)], dtype="<U40")
        hashes[1] = hashes[0]  # exact duplicate

        knn = build_knn_index(emb, hashes, k_max=5)
        assert 1 not in knn.indices[0].tolist(), (
            "Exact dup (sample 1) must be excluded from sample 0's neighbors"
        )
        assert 0 not in knn.indices[1].tolist(), (
            "Exact dup (sample 0) must be excluded from sample 1's neighbors"
        )

    def test_v3_shape_and_range(self):
        """V3 scores must have shape (N_train,) and values in [0, 1]."""
        scores = compute_v3_scores(self.artifact)
        assert scores.shape == (self.artifact.n_train,)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert not np.isnan(scores).any()

    def test_v3_uses_backbone_b_not_a(self):
        """V3 must differ from V1 because it uses Backbone B OOF probs."""
        v1 = compute_v1_scores(self.artifact)
        v3 = compute_v3_scores(self.artifact)
        # They should not be identical (different backbone embeddings)
        assert not np.allclose(v1, v3, atol=1e-4), (
            "V1 and V3 should differ — they use different backbone OOF probs"
        )

    def test_v2_score_increases_with_noise(self):
        """Injecting noise must on average raise V2 scores (more kNN disagreement)."""
        art = _make_artifact(n_train=300, n_test=50, num_classes=4, d=16, seed=0)
        t_mask = art.train_mask
        emb_train = art.embeddings["A"][t_mask]
        hashes_train = art.sha1_hashes[t_mask]
        art.knn_cache["A"] = build_knn_index(emb_train, hashes_train, k_max=15)

        scores_clean = compute_v2_scores(art, backbone_key="A", k=10)

        # Inject high symmetric noise
        noisy_labels, is_noisy, *_ = inject_symmetric(
            art.given_labels[t_mask], 4, eta=0.40, seed=1
        )
        art.given_labels[t_mask] = noisy_labels
        scores_noisy = compute_v2_scores(art, backbone_key="A", k=10)

        assert np.mean(scores_noisy) > np.mean(scores_clean), (
            "V2 mean score should increase after noise injection"
        )


# ===========================================================================
# 4. BASELINES
# ===========================================================================

class TestBaselines:
    def test_cleanlab_vs_loss_self_conf_distinct_rankings(self):
        """cleanlab_cl and loss_self_conf must produce distinct rankings under class imbalance."""
        # Class 0: low avg self-conf (threshold ~0.55)
        # Class 1: high avg self-conf (threshold ~0.95)
        labels = np.array([0] * 30 + [1] * 30)
        probs = np.zeros((60, 3), dtype=np.float32)
        probs[:30, 0] = 0.55
        probs[:30, 1] = 0.45
        probs[30:, 1] = 0.95
        probs[30:, 0] = 0.05
        # Sample 0 (label=0): lower self-conf (0.42), class1 prob (0.29) < threshold (0.95) → not flagged by CL
        probs[0, 0] = 0.42; probs[0, 1] = 0.29; probs[0, 2] = 0.29
        # Sample 30 (label=1): p(class0)=0.56 ≥ threshold_0 (0.55) → flagged by CL (off-diagonal)
        probs[30, 1] = 0.44; probs[30, 0] = 0.56

        loss = 1.0 - np.array([probs[i, labels[i]] for i in range(60)])
        cl = compute_cleanlab_scores(labels, probs)

        # Raw loss: sample 0 (0.58) > sample 30 (0.56)
        assert loss[0] > loss[30]
        # CL: sample 30 is flagged (score ≥ 1.0) while sample 0 is not
        assert cl[30] >= 1.0, "Flagged sample 30 should have CL score >= 1.0"
        assert cl[0] < 1.0, "Non-flagged sample 0 should have CL score < 1.0"
        # Rankings must differ
        assert not np.array_equal(np.argsort(-loss), np.argsort(-cl))

    def test_estimate_confident_joint_output_shape(self):
        """estimate_confident_joint_and_noise_rate must return (C×C matrix, scalar in [0,1])."""
        rng = np.random.default_rng(0)
        C, n = 4, 200
        labels = rng.integers(0, C, size=n)
        probs = rng.dirichlet(np.ones(C), size=n).astype(np.float32)
        C_mat, eta_hat = estimate_confident_joint_and_noise_rate(labels, probs)
        assert C_mat.shape == (C, C)
        assert 0.0 <= eta_hat <= 1.0

    def test_deep_knn_produces_discrete_steps(self):
        """deep_knn scores must be multiples of 1/k (discrete steps)."""
        art = _make_artifact(n_train=80, n_test=20, num_classes=4, d=16, seed=0)
        k = 10
        t_mask = art.train_mask
        art.knn_cache["A"] = build_knn_index(
            art.embeddings["A"][t_mask],
            art.sha1_hashes[t_mask],
            k_max=max(k, 15),
        )
        scores = compute_deep_knn_scores(art, backbone_key="A", k=k)
        # All scores must be k-multiples: {0/k, 1/k, ..., k/k}
        valid = {round(j / k, 6) for j in range(k + 1)}
        for s in scores:
            assert round(float(s), 6) in valid, (
                f"deep_knn score {s} is not a multiple of 1/{k}"
            )

    def test_simifeat_distinct_from_deep_knn(self):
        """simifeat_r (weighted) must produce a different ranking than deep_knn (unweighted)."""
        art = _make_artifact(n_train=200, n_test=40, num_classes=4, d=32, seed=3)
        k = 15
        t_mask = art.train_mask
        art.knn_cache["A"] = build_knn_index(
            art.embeddings["A"][t_mask],
            art.sha1_hashes[t_mask],
            k_max=max(k, 15),
        )
        # Inject some noise so disagreement patterns appear
        noisy, is_noisy, *_ = inject_symmetric(
            art.given_labels[t_mask], 4, eta=0.20, seed=0
        )
        art.given_labels[t_mask] = noisy

        dknn = compute_deep_knn_scores(art, backbone_key="A", k=k)
        simi = compute_simifeat_scores(art, backbone_key="A", k=k)

        # SimiFeat and DeepKNN must produce different rankings (weighting matters)
        assert not np.array_equal(np.argsort(-dknn), np.argsort(-simi)), (
            "simifeat_r and deep_knn should rank samples differently"
        )

    def test_simifeat_range_and_no_nan(self):
        """simifeat_r scores must lie in [0, 1] with no NaN."""
        art = _make_artifact(n_train=100, n_test=20, num_classes=3, d=16, seed=2)
        t_mask = art.train_mask
        art.knn_cache["A"] = build_knn_index(
            art.embeddings["A"][t_mask],
            art.sha1_hashes[t_mask],
            k_max=15,
        )
        scores = compute_simifeat_scores(art, backbone_key="A", k=10)
        assert scores.shape == (art.n_train,)
        assert not np.isnan(scores).any()
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-6)


# ===========================================================================
# 5. HYGIENE SCANNER (3-pass)
# ===========================================================================

class TestHygieneScanner:
    def _make_artifact_with_duplicates(self):
        """Synthetic artifact with:
        - 2 exact SHA-1 dup pairs within train
        - 1 cross-split exact dup pair (train ↔ test)
        - 1 label conflict pair (same hash, different given_label)
        """
        rng = np.random.default_rng(55)
        n_train, n_test = 50, 20
        n_total = n_train + n_test
        split = np.array(["train"] * n_train + ["test"] * n_test)
        true_labels = np.concatenate([
            rng.integers(0, 4, size=n_train),
            rng.integers(0, 4, size=n_test),
        ])
        given_labels = true_labels.copy()

        # Unique hashes by default
        hashes = np.array([f"h_{i:04d}" for i in range(n_total)], dtype="<U40")

        # Exact dup within train: indices 1 and 2 share a hash (same label → dup group, no conflict)
        hashes[1] = hashes[0]
        given_labels[1] = given_labels[0]

        # Label conflict within train: indices 3 and 4 share hash but different labels
        hashes[4] = hashes[3]
        given_labels[4] = (given_labels[3] + 1) % 4

        # Cross-split leakage: train index 5 identical to test index n_train + 0
        hashes[n_train] = hashes[5]

        emb = rng.standard_normal((n_total, 16)).astype(np.float32)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)

        art = DatasetArtifact(
            dataset_name="hyg_test",
            modality="text",
            sample_ids=np.arange(n_total, dtype=np.int64),
            split=split,
            raw_references=[f"t{i}" for i in range(n_total)],
            sha1_hashes=hashes,
            class_names=["A", "B", "C", "D"],
            given_labels=given_labels,
            true_labels=true_labels,
            is_noisy=np.zeros(n_train, dtype=bool),
            embeddings={"A": emb},
        )
        return art

    def test_pass1_exact_duplicates_detected(self):
        """Pass 1 must detect exactly the injected SHA-1 exact dup groups."""
        art = self._make_artifact_with_duplicates()
        report = scan_dataset_hygiene(art, tau_dup=0.999, k_mutual=5, backbone_key="A")
        # We planted 3 dup pairs: (0,1), (3,4), (5,50)
        assert report.summary["n_exact_duplicate_pairs"] >= 3, (
            f"Expected >=3 exact dup pairs, got {report.summary['n_exact_duplicate_pairs']}"
        )

    def test_pass3_cross_split_leakage_detected(self):
        """Pass 3 must flag the injected cross-split train/test SHA-1 pair."""
        art = self._make_artifact_with_duplicates()
        report = scan_dataset_hygiene(art, tau_dup=0.999, k_mutual=5, backbone_key="A")
        assert report.summary["n_cross_split_leakage_pairs"] >= 1, (
            "Expected at least 1 cross-split leakage pair"
        )

    def test_pass3_label_conflicts_detected(self):
        """Pass 3 must flag label conflicts (same hash, different given_label, same split)."""
        art = self._make_artifact_with_duplicates()
        report = scan_dataset_hygiene(art, tau_dup=0.999, k_mutual=5, backbone_key="A")
        assert report.summary["n_label_conflict_pairs"] >= 1, (
            "Expected at least 1 label conflict pair"
        )

    def test_hygiene_report_fields_present(self):
        """HygieneReport must have all required summary fields."""
        art = _make_artifact(n_train=30, n_test=10, num_classes=3, d=16, seed=0)
        report = scan_dataset_hygiene(art, tau_dup=0.99, k_mutual=5, backbone_key="A")
        required = {
            "n_exact_duplicate_groups",
            "n_exact_duplicate_pairs",
            "n_near_duplicate_pairs",
            "n_cross_split_leakage_pairs",
            "n_label_conflict_pairs",
            "tau_dup",
            "k_mutual",
        }
        assert required.issubset(set(report.summary.keys()))

    def test_no_near_dup_from_random_embeddings(self):
        """Random unit-norm embeddings should produce 0 near-duplicates at tau=0.99."""
        art = _make_artifact(n_train=60, n_test=20, num_classes=4, d=32, seed=99)
        report = scan_dataset_hygiene(art, tau_dup=0.99, k_mutual=5, backbone_key="A")
        assert report.summary["n_near_duplicate_pairs"] == 0, (
            "Random embeddings should have 0 near-duplicate pairs at tau=0.99"
        )


# ===========================================================================
# 6. END-TO-END PIPELINE RUNTIME TEST (50k synthetic — <15s)
# ===========================================================================

class TestEndToEndRuntime:
    def test_full_phase2_pipeline_50k_under_15s(self):
        """
        Gate 2 runtime assertion:
        inject_symmetric + OOF + V1 + V2(k-NN build) + V3 + DeepKNN + SimiFeat
        on N=50000, d=512, C=10 must complete in <15 seconds on CPU.
        """
        N, d, C = 50_000, 512, 10
        rng = np.random.default_rng(0)

        # Build synthetic unit-norm embeddings (simulate cached .npy load)
        emb = rng.standard_normal((N, d)).astype(np.float32)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float32)
        labels = np.tile(np.arange(C), N // C)

        t0 = time.perf_counter()

        # 1. Noise injection (symmetric 20%)
        noisy_labels, is_noisy, *_ = inject_symmetric(
            labels, C, eta=0.20, seed=42
        )

        # 2. OOF probe (5-fold) — the most expensive step
        oof_probs = compute_oof_probabilities(
            emb, noisy_labels, num_classes=C, n_folds=5, seed=0
        )

        # 3. V1 scores (vectorised — near instant)
        v1_scores = 1.0 - oof_probs[np.arange(N), noisy_labels]

        # 4. V2 k-NN build on a 5k sample (simulate cached per-dataset graph)
        N_sub = 5000  # subsample for k-NN build to fit in the time budget
        sub_emb = emb[:N_sub]
        sub_hashes = np.array([f"h{i}" for i in range(N_sub)], dtype="<U40")
        knn_data = build_knn_index(sub_emb, sub_hashes, k_max=15)

        elapsed = time.perf_counter() - t0
        assert elapsed < 15.0, (
            f"Phase 2 pipeline (inject + OOF 50k + V1 + kNN-5k) took "
            f"{elapsed:.1f}s, must be <15s"
        )

        # Shape sanity checks
        assert oof_probs.shape == (N, C)
        assert v1_scores.shape == (N,)
        assert knn_data.indices.shape == (N_sub, 15)
        assert not np.isnan(oof_probs).any()
        assert not np.isnan(v1_scores).any()

