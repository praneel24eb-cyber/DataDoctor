import numpy as np
import pytest

from datadoctor.schema import DatasetArtifact
from datadoctor.hygiene.scanner import scan_dataset_hygiene


def test_hygiene_scanner_passes():
    """Gate 3 Assertion: Verifies Pass 1 (SHA-1 exact), Pass 2 (Mutual k-NN near-dup),
    and Pass 3 (Cross-split leakage & label conflicts).
    """
    n_train = 60
    n_test = 20
    n_total = n_train + n_test
    d = 16
    rng = np.random.default_rng(42)

    # Base embeddings
    emb = rng.standard_normal((n_total, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    hashes = np.array([f"unique_hash_{i}" for i in range(n_total)], dtype="<U40")
    labels = np.array([0] * (n_total // 2) + [1] * (n_total - n_total // 2))
    splits = np.array(["train"] * n_train + ["test"] * n_test)

    # 1. Exact Duplicate across train and test: Sample 5 (train) and Sample 65 (test)
    hashes[65] = hashes[5]
    emb[65] = emb[5]

    # 2. In-train exact duplicate with conflicting label: Sample 10 and Sample 11
    hashes[11] = hashes[10]
    emb[11] = emb[10]
    labels[10] = 0
    labels[11] = 1 # Conflict!

    # 3. Near-duplicate in train: Sample 20 and Sample 21 with high cosine similarity
    emb[21] = emb[20] * 0.98 + rng.standard_normal(d) * 0.02
    emb[21] /= np.linalg.norm(emb[21])
    # Different hashes so not exact
    hashes[21] = "near_hash_21"

    artifact = DatasetArtifact(
        dataset_name="test_hygiene",
        modality="vision",
        sample_ids=np.arange(n_total),
        split=splits,
        raw_references=[f"ref_{i}" for i in range(n_total)],
        sha1_hashes=hashes,
        class_names=["c0", "c1"],
        given_labels=labels,
        true_labels=labels.copy(),
        embeddings={"A": emb},
    )

    report = scan_dataset_hygiene(artifact, tau_dup=0.95, k_mutual=5)

    # Check Pass 1: exact duplicates found
    assert len(report.exact_duplicate_groups) >= 2

    # Check Pass 2: near duplicates found
    near_pairs = [(p[0], p[1]) for p in report.near_duplicate_pairs]
    assert (20, 21) in near_pairs or (21, 20) in near_pairs

    # Check Pass 3: cross-split leakage detected for pair (5, 65)
    leakage_pairs = [(p[0], p[1]) for p in report.cross_split_leakage_pairs]
    assert (5, 65) in leakage_pairs or (65, 5) in leakage_pairs

    # Check Pass 3: label conflict detected for pair (10, 11)
    conflict_pairs = [(p[0], p[1]) for p in report.label_conflict_pairs]
    assert (10, 11) in conflict_pairs or (11, 10) in conflict_pairs

