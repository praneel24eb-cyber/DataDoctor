"""
Phase 1 Gate Tests — as specified in the PRD.

Covers:
  1. Float16 disk caching → float32 L2 re-normalization invariant
  2. Identical-vector round-trip cosine similarity ≥ 0.99999
  3. CIFAR-10N exact noise rates (9.03 %, 17.23 %, 40.21 %) — using synthetic mocked data
  4. AG News stratified 20k subset (seed=42): 5000 samples per class, SHA-1 uniqueness
  5. DatasetArtifact schema: n_train / n_test properties, train_mask / test_mask, to_scores_dataframe()
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

# ─── schema & cache ──────────────────────────────────────────────────────────
from datadoctor.schema import DatasetArtifact
from datadoctor.extractors.cache import (
    load_cached_embeddings,
    save_cached_embeddings,
    compute_sha1_hash,
)
# ─── loaders ─────────────────────────────────────────────────────────────────
from datadoctor.io.cifar10n import verify_cifar10n_rates
from datadoctor.io.loaders import load_ag_news


# ─────────────────────────────────────────────────────────────────────────────
# 1.  float16 caching → float32 re-normalisation
# ─────────────────────────────────────────────────────────────────────────────

def test_float16_cast_and_unit_norm_invariant():
    """
    Gate 1:  load_cached_embeddings() must:
      (a) return dtype float32
      (b) restore unit L2 norms — max_i |‖z_i‖₂ − 1| < 1e-5   (atol from PRD §FR-2)
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "emb.npy"
        rng  = np.random.default_rng(0)
        raw  = rng.standard_normal((200, 384)).astype(np.float32)

        save_cached_embeddings(path, raw)
        assert np.load(path).dtype == np.float16, "On-disk dtype must be float16"

        loaded = load_cached_embeddings(path)
        assert loaded.dtype == np.float32

        norms = np.linalg.norm(loaded, axis=1)
        # PRD specifies atol=1e-5 for the assert in cache.py
        assert np.allclose(norms, 1.0, atol=1e-5), \
            f"Max norm drift = {np.max(np.abs(norms - 1.0)):.2e}"


def test_identical_vector_round_trip_similarity():
    """
    Gate 1:  Two identical vectors saved/loaded as float16 must have:
      • identical sha1_hashes
      • cosine similarity ≥ 0.99999
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup.npy"
        rng  = np.random.default_rng(7)
        v    = rng.standard_normal((1, 512)).astype(np.float32)
        pair = np.vstack([v, v])

        save_cached_embeddings(path, pair)
        z    = load_cached_embeddings(path)

        h0 = compute_sha1_hash(z[0].tobytes())
        h1 = compute_sha1_hash(z[1].tobytes())
        assert h0 == h1, "Identical reloaded vectors must have identical SHA-1 hashes"

        cos = float(np.dot(z[0], z[1]))
        assert cos >= 0.99999, f"Cosine similarity after float16 round-trip = {cos:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CIFAR-10N noise-rate verification (synthetic / mock — no download)
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_cifar10n_rates_pass():
    """
    Gate 1:  verify_cifar10n_rates() must accept these exact counts
    and return rates within ±0.05 pp of the literature values:
      aggre_label  : 9.03 %  →  4515 flips
      random_label1: 17.23 % →  8615 flips
      worse_label  : 40.21 % → 20105 flips
    """
    n = 50_000
    clean = np.zeros(n, dtype=np.int64)

    def make_noisy(n_flips):
        noisy = clean.copy()
        noisy[:n_flips] = 1
        return noisy

    pt = {
        "clean_label":   clean,
        "aggre_label":   make_noisy(4515),
        "random_label1": make_noisy(8615),
        "worse_label":   make_noisy(20105),
    }

    rates = verify_cifar10n_rates(pt)
    assert abs(rates["aggre_label"]   - 0.0903) <= 0.0005
    assert abs(rates["random_label1"] - 0.1723) <= 0.0005
    assert abs(rates["worse_label"]   - 0.4021) <= 0.0005


def test_verify_cifar10n_rates_fail_on_wrong_count():
    """
    Gate 1:  verify_cifar10n_rates() must raise AssertionError when a rate
    deviates more than ±0.05 pp from the literature target.
    """
    n = 50_000
    clean = np.zeros(n, dtype=np.int64)
    bad_worse = clean.copy()
    bad_worse[:15_000] = 1  # 30 % instead of 40.21 %

    pt = {
        "clean_label":   clean,
        "aggre_label":   clean.copy(),   # deliberately skip so only worse_label fails
        "random_label1": clean.copy(),
        "worse_label":   bad_worse,
    }
    # aggre 0 % will itself fail — that's fine, we just need *an* AssertionError
    with pytest.raises(AssertionError, match="CIFAR-10N noise rate mismatch"):
        verify_cifar10n_rates(pt)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AG News loader — stratified 20k subset
# ─────────────────────────────────────────────────────────────────────────────

def test_ag_news_loader_shape_and_stratification():
    """
    Gate 1:  load_ag_news(subsample_train=20000, seed=42) must produce:
      • n_train = 20 000
      • n_test  =  7 600
      • exactly 5 000 samples per class in the training split
      • all SHA-1 hashes unique within each split
    """
    artifact = load_ag_news(subsample_train=20_000, seed=42)

    assert artifact.n_train == 20_000, f"Expected 20000 train, got {artifact.n_train}"
    assert artifact.n_test  ==  7_600, f"Expected 7600 test, got {artifact.n_test}"

    train_mask   = artifact.train_mask
    train_labels = artifact.given_labels[train_mask]

    # 5 000 per class (4 classes)
    for c in range(4):
        count = int(np.sum(train_labels == c))
        assert count == 5_000, \
            f"Class {c} has {count} train samples (expected 5000)"

    # SHA-1 uniqueness
    train_hashes = artifact.sha1_hashes[train_mask]
    assert len(set(train_hashes.tolist())) == 20_000, \
        "Duplicate SHA-1 hashes found in the training split"

    test_hashes = artifact.sha1_hashes[~train_mask]
    assert len(set(test_hashes.tolist())) == 7_600, \
        "Duplicate SHA-1 hashes found in the test split"


def test_ag_news_true_labels_equal_given_labels_on_test():
    """
    PRD invariant:  for split == 'test', given_labels[i] == true_labels[i] always.
    """
    artifact = load_ag_news(subsample_train=20_000, seed=42)
    test_mask = artifact.test_mask
    assert np.all(
        artifact.given_labels[test_mask] == artifact.true_labels[test_mask]
    ), "given_labels != true_labels on the test split"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DatasetArtifact schema invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_artifact_schema():
    """
    Gate 1:  DatasetArtifact properties and to_scores_dataframe() export.
    """
    n_train, n_test = 80, 20
    n_total = n_train + n_test
    rng = np.random.default_rng(1)

    ids    = np.arange(n_total, dtype=np.int64)
    split  = np.array(["train"] * n_train + ["test"] * n_test)
    hashes = np.array([f"h{i:04d}" for i in range(n_total)], dtype="<U40")
    labels = rng.integers(0, 4, size=n_total).astype(np.int64)

    art = DatasetArtifact(
        dataset_name="unit_test",
        modality="text",
        sample_ids=ids,
        split=split,
        raw_references=[f"r{i}" for i in range(n_total)],
        sha1_hashes=hashes,
        class_names=["W", "S", "B", "T"],
        given_labels=labels.copy(),
        true_labels=labels.copy(),
        is_noisy=np.zeros(n_train, dtype=bool),
    )

    assert art.n_train == n_train
    assert art.n_test  == n_test
    assert np.sum(art.train_mask) == n_train
    assert np.sum(art.test_mask)  == n_test

    # Populate a fake view so to_scores_dataframe() exports something non-trivial
    art.view_scores["v1"]      = np.linspace(0.1, 0.9, n_train).astype(np.float32)
    art.view_ranks["v1"]       = np.arange(1, n_train + 1, dtype=np.float64)
    art.view_percentiles["v1"] = np.linspace(1.0, 0.0, n_train)
    art.fused_score            = art.view_scores["v1"].astype(np.float64)
    art.fused_rank             = art.view_ranks["v1"].astype(np.int64)
    art.u_rrf                  = art.view_percentiles["v1"].copy()

    df = art.to_scores_dataframe()
    assert len(df) == n_train
    for col in ("score_v1", "rank_v1", "u_v1", "fused_score", "fused_rank", "u_rrf"):
        assert col in df.columns, f"Missing column: {col}"

