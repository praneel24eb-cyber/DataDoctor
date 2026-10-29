import tempfile
from pathlib import Path
import numpy as np
import pytest

from datadoctor.schema import DatasetArtifact
from datadoctor.router.rules import assign_route_buckets
from datadoctor.router.judge import (
    run_llm_judge,
    BLIND_PROMPT_TEMPLATE,
    extract_clean_medoids,
)
from datadoctor.router.cache_db import LLMCacheDB
from datadoctor.fusion.rank_fusion import compute_rank_fusion


def test_route_bucket_disjointness_and_budget_cap():
    """Gate 4 Assertion: Assert route_bucket preserves 'auto_clear' samples,
    scales thresholds with estimated_noise_rate, and strictly respects
    sum(route_bucket == 'llm_judge') <= B.
    """
    n_train = 100
    
    # Direct percentile control to test routing cascade logic cleanly
    u_v1 = np.full(n_train, 0.50, dtype=np.float64)
    u_v2 = np.full(n_train, 0.50, dtype=np.float64)
    u_v3 = np.full(n_train, 0.50, dtype=np.float64)

    # 1. auto_clear: samples 0..9 have all u <= tau_low (0.75 for eta=0.10)
    u_v1[:10] = 0.20
    u_v2[:10] = 0.20
    u_v3[:10] = 0.20

    # 2. auto_flag: samples 10..14 have all u >= tau_high (0.95 for eta=0.10)
    u_v1[10:15] = 0.98
    u_v2[10:15] = 0.98
    u_v3[10:15] = 0.98

    # 3. Disagreement candidates: samples 15..34 have max(u_v1, u_v2) >= 0.95 and |u_v1 - u_v2| >= 0.35
    u_v1[15:35] = 0.96
    u_v2[15:35] = 0.40
    u_v3[15:35] = 0.60

    artifact = DatasetArtifact(
        dataset_name="test_router",
        modality="text",
        sample_ids=np.arange(n_train),
        split=np.array(["train"] * n_train),
        raw_references=[f"text_{i}" for i in range(n_train)],
        sha1_hashes=np.array([f"hash_{i}" for i in range(n_train)], dtype="<U40"),
        class_names=["c0", "c1"],
        given_labels=np.zeros(n_train, dtype=np.int64),
        estimated_noise_rate=0.10,
        view_scores={"v1": u_v1, "v2": u_v2, "v3": u_v3},
    )

    compute_rank_fusion(artifact)
    
    # Override percentiles directly to test router's exact math
    artifact.view_percentiles["v1"] = u_v1
    artifact.view_percentiles["v2"] = u_v2
    artifact.view_percentiles["v3"] = u_v3

    # Cap B = floor(0.05 * 100) = 5 calls
    assign_route_buckets(artifact, budget_cap_fraction=0.05)

    assert artifact.route_bucket is not None
    buckets = artifact.route_bucket

    # 1. auto_clear samples must be preserved and never overwritten
    assert np.all(buckets[:10] == "auto_clear")

    # 2. auto_flag samples preserved
    assert np.all(buckets[10:15] == "auto_flag")

    # 3. LLM calls strictly capped at B = 5
    llm_calls = np.sum(buckets == "llm_judge")
    assert llm_calls == 5, f"Expected exactly 5 LLM calls, got {llm_calls}"

    # 4. Overflow disagreement candidates assigned to human_queue
    human_queue_candidates = np.sum(buckets[15:35] == "human_queue")
    assert human_queue_candidates == 15


def test_prompt_template_and_sqlite_cache_hit():
    """Gate 4 Assertion: Assert BLIND_PROMPT_TEMPLATE.format(...) formats cleanly
    without KeyError on JSON braces and hits the SQLite cache on a second seed with identical sample SHA-1.
    """
    # 1. Format template test
    formatted = BLIND_PROMPT_TEMPLATE.format(
        class_names='"World", "Sports"',
        reference_block='- [World]: "News 1"\n- [Sports]: "News 2"',
        target_text="A thrilling soccer match ended today.",
    )
    assert '{"label": "<exact category name>"' in formatted
    assert "A thrilling soccer match ended today." in formatted

    # 2. SQLite Cache hit across calls
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_llm_cache.db"
        cache_db = LLMCacheDB(db_file)

        n_train = 20
        d = 16
        rng = np.random.default_rng(88)

        emb_A = rng.standard_normal((n_train, d)).astype(np.float32)
        emb_A /= np.linalg.norm(emb_A, axis=1, keepdims=True)

        labels = np.array([0] * 10 + [1] * 10)
        raw_texts = [f"Text sample {i}" for i in range(n_train)]
        hashes = np.array([f"sha1_{i}" for i in range(n_train)], dtype="<U40")

        artifact = DatasetArtifact(
            dataset_name="test_judge",
            modality="text",
            sample_ids=np.arange(n_train),
            split=np.array(["train"] * n_train),
            raw_references=raw_texts,
            sha1_hashes=hashes,
            class_names=["World", "Sports"],
            given_labels=labels,
            embeddings={"A": emb_A},
            view_scores={"v1": np.ones(n_train), "v2": np.ones(n_train), "v3": np.ones(n_train)},
        )
        compute_rank_fusion(artifact)

        # Route 2 samples to llm_judge
        artifact.route_bucket = np.array(["auto_clear"] * n_train)
        artifact.route_bucket[2] = "llm_judge"
        artifact.route_bucket[3] = "llm_judge"

        # First run: populates cache
        run_llm_judge(artifact, provider="mock", cache_db=cache_db)
        summary1 = cache_db.get_token_summary()
        assert summary1["total_queries"] == 2

        # Second run: identical texts/hashes must hit cache with 0 extra queries added
        artifact2 = DatasetArtifact(
            dataset_name="test_judge_seed2",
            modality="text",
            sample_ids=np.arange(n_train),
            split=np.array(["train"] * n_train),
            raw_references=raw_texts,
            sha1_hashes=hashes,
            class_names=["World", "Sports"],
            given_labels=labels,
            embeddings={"A": emb_A},
            view_scores={"v1": np.ones(n_train), "v2": np.ones(n_train), "v3": np.ones(n_train)},
        )
        compute_rank_fusion(artifact2)
        artifact2.route_bucket = artifact.route_bucket.copy()

        run_llm_judge(artifact2, provider="mock", cache_db=cache_db)
        summary2 = cache_db.get_token_summary()
        assert summary2["total_queries"] == 2, "Identical samples should hit cache without adding new queries"
        assert artifact2.llm_suggested_label[2] == artifact.llm_suggested_label[2]


def test_modality_guard_raises_on_vision():
    """Verifies that LLM Judge raises ValueError when invoked on vision datasets."""
    artifact = DatasetArtifact(
        dataset_name="cifar10_toy",
        modality="vision",
        sample_ids=np.arange(10),
        split=np.array(["train"] * 10),
        raw_references=[f"img_{i}" for i in range(10)],
        sha1_hashes=np.array([f"h_{i}" for i in range(10)], dtype="<U40"),
        class_names=["cat", "dog"],
        given_labels=np.zeros(10, dtype=np.int64),
        route_bucket=np.array(["llm_judge"] * 10),
    )

    with pytest.raises(ValueError, match="scoped strictly to text datasets"):
        run_llm_judge(artifact, provider="mock")
