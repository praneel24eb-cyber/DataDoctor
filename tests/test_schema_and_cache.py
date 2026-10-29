import tempfile
from pathlib import Path
import numpy as np
import pytest

from datadoctor.schema import DatasetArtifact, KNNIndexData
from datadoctor.extractors.cache import (
    load_cached_embeddings,
    save_cached_embeddings,
    compute_sha1_hash,
)
from datadoctor.io.cifar10n import verify_cifar10n_rates


def test_float16_drift_and_l2_renormalization():
    """Gate 1 Assertion: load_cached_embeddings() casts float16 -> float32
    and restores unit L2 norms (max_i | ||z_i||_2 - 1.0 | < 1e-6).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_emb.npy"
        
        # Generate random unnormalized embeddings
        rng = np.random.default_rng(42)
        raw_emb = rng.standard_normal((100, 384)).astype(np.float32)
        
        # Save as float16 via save_cached_embeddings
        save_cached_embeddings(cache_file, raw_emb)
        
        # Raw disk load should be float16
        raw_loaded = np.load(cache_file)
        assert raw_loaded.dtype == np.float16
        
        # Datadoctor load_cached_embeddings must cast to float32 and restore unit norm
        reloaded = load_cached_embeddings(cache_file)
        assert reloaded.dtype == np.float32
        
        norms = np.linalg.norm(reloaded, axis=1)
        max_drift = np.max(np.abs(norms - 1.0))
        assert max_drift < 1e-6, f"Unit norm drift exceeded tolerance: {max_drift}"


def test_identical_vectors_reloaded_similarity():
    """Gate 1 Assertion: Two identical vectors saved as float16 and reloaded
    have identical sha1_hashes and cosine similarity >= 0.99999.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "identical.npy"
        
        rng = np.random.default_rng(123)
        vec = rng.standard_normal((1, 512)).astype(np.float32)
        # Repeat vector twice
        two_vecs = np.repeat(vec, 2, axis=0)
        
        save_cached_embeddings(cache_file, two_vecs)
        reloaded = load_cached_embeddings(cache_file)
        
        hash1 = compute_sha1_hash(reloaded[0].tobytes())
        hash2 = compute_sha1_hash(reloaded[1].tobytes())
        assert hash1 == hash2, "Identical reloaded vectors must have identical SHA-1 hashes"
        
        cos_sim = float(np.dot(reloaded[0], reloaded[1]))
        assert cos_sim >= 0.99999, f"Cosine similarity degraded below 0.99999: {cos_sim}"


def test_verify_cifar10n_rates_exact():
    """Gate 1 Assertion: verify_cifar10n_rates() verifies 9.03%, 17.23%, 40.21%
    with strict assertion errors on deviations.
    """
    n = 50000
    clean = np.zeros(n, dtype=np.int64)
    
    # 9.03% = 4515 flips
    aggre = clean.copy()
    aggre[:4515] = 1
    
    # 17.23% = 8615 flips
    rand1 = clean.copy()
    rand1[:8615] = 1
    
    # 40.21% = 20105 flips
    worse = clean.copy()
    worse[:20105] = 1
    
    valid_dict = {
        "clean_label": clean,
        "aggre_label": aggre,
        "random_label1": rand1,
        "worse_label": worse,
    }
    
    computed_rates = verify_cifar10n_rates(valid_dict)
    assert abs(computed_rates["aggre_label"] - 0.0903) <= 0.0005
    assert abs(computed_rates["random_label1"] - 0.1723) <= 0.0005
    assert abs(computed_rates["worse_label"] - 0.4021) <= 0.0005
    
    # Test invalid rate triggers AssertionError
    invalid_dict = valid_dict.copy()
    invalid_worse = clean.copy()
    invalid_worse[:15000] = 1 # 30% instead of 40.21%
    invalid_dict["worse_label"] = invalid_worse
    
    with pytest.raises(AssertionError, match="CIFAR-10N noise rate mismatch"):
        verify_cifar10n_rates(invalid_dict)


def test_schema_and_dataframe_export():
    """Verifies DatasetArtifact contract, split masking, and to_scores_dataframe."""
    n_train = 80
    n_test = 20
    n_total = n_train + n_test
    
    sample_ids = np.arange(n_total, dtype=np.int64)
    split = np.array(["train"] * n_train + ["test"] * n_test)
    raw_refs = [f"sample_{i}" for i in range(n_total)]
    hashes = np.array([f"hash_{i}" for i in range(n_total)], dtype="<U40")
    labels = np.random.randint(0, 4, size=n_total, dtype=np.int64)
    
    artifact = DatasetArtifact(
        dataset_name="test_schema",
        modality="text",
        sample_ids=sample_ids,
        split=split,
        raw_references=raw_refs,
        sha1_hashes=hashes,
        class_names=["c0", "c1", "c2", "c3"],
        given_labels=labels.copy(),
        true_labels=labels.copy(),
        is_noisy=np.zeros(n_train, dtype=bool),
    )
    
    assert artifact.n_train == n_train
    assert artifact.n_test == n_test
    assert np.all(artifact.train_mask == (split == "train"))
    assert np.all(artifact.test_mask == (split == "test"))
    
    # Populate view scores, ranks, and percentiles on train
    artifact.view_scores["v1"] = np.linspace(0.1, 0.9, n_train)
    artifact.view_ranks["v1"] = np.arange(1, n_train + 1)
    artifact.view_percentiles["v1"] = np.linspace(1.0, 0.0, n_train)
    artifact.fused_score = artifact.view_scores["v1"] * 2.0
    artifact.fused_rank = artifact.view_ranks["v1"]
    artifact.u_rrf = artifact.view_percentiles["v1"]
    
    df = artifact.to_scores_dataframe()
    assert len(df) == n_train
    assert "score_v1" in df.columns
    assert "rank_v1" in df.columns
    assert "u_v1" in df.columns
    assert "fused_score" in df.columns
    assert "fused_rank" in df.columns
    assert "u_rrf" in df.columns

