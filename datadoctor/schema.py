from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Literal, Tuple
import numpy as np
import pandas as pd


@dataclass
class KNNIndexData:
    indices: np.ndarray      # Shape: (N_train, k_max), dtype: int32 (excludes self & SHA-1 exact dups)
    similarities: np.ndarray # Shape: (N_train, k_max), dtype: float32 (cosine similarity in [-1, 1])


@dataclass
class HygieneReport:
    exact_duplicate_groups: List[List[int]] = field(default_factory=list)
    near_duplicate_pairs: List[Tuple[int, int, float]] = field(default_factory=list)
    cross_split_leakage_pairs: List[Tuple[int, int, str]] = field(default_factory=list)
    label_conflict_pairs: List[Tuple[int, int]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetArtifact:
    dataset_name: str
    modality: Literal["vision", "text"]
    sample_ids: np.ndarray                     # Shape: (N_total,), dtype: int64
    split: np.ndarray                          # Shape: (N_total,), values in {"train", "test"}
    raw_references: List[str]                  # Length: N_total (filepaths or raw text strings)
    sha1_hashes: np.ndarray                    # Shape: (N_total,), dtype: "<U40" (exact content hash)
    class_names: List[str]                     # Length: C
    
    # Labels: Shape (N_total,), dtype: int64, values in 0..C-1
    # Note: For i where split[i] == "test", given_labels[i] == true_labels[i] always.
    given_labels: np.ndarray                   # Observed/corrupted labels \tilde{y} on train
    true_labels: Optional[np.ndarray] = None   # Latent clean labels y (benchmark mode only)
    is_noisy: Optional[np.ndarray] = None      # Shape: (N_train,), bool: given_labels[train] != true_labels[train]
    
    # Per-Backbone Cached Representations (Keys: "A", "B", "C")
    # Embeddings shape: (N_total, D), float32, strictly L2-normalized upon load
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    # OOF Probs shape: (N_train, C), float32 (computed strictly on train split)
    oof_probs: Dict[str, np.ndarray] = field(default_factory=dict)
    knn_cache: Dict[str, KNNIndexData] = field(default_factory=dict)
    
    # Training-Split Suspicion Scores, Ranks & Percentiles — Shape: (N_train,)
    # Higher score = MORE suspicious; Rank 1 = MOST suspicious; Percentile u = 1.0 = MOST suspicious
    view_scores: Dict[str, np.ndarray] = field(default_factory=dict)      # Keys: "v1", "v2", "v3", "v2_B"
    view_ranks: Dict[str, np.ndarray] = field(default_factory=dict)       # 1-indexed average ranks in [1, N_train]
    view_percentiles: Dict[str, np.ndarray] = field(default_factory=dict) # u_m in [0, 1]
    
    baseline_scores: Dict[str, np.ndarray] = field(default_factory=dict)  # Keys: "loss_self_conf", "cleanlab_cl", "deep_knn", "simifeat_r", "random"
    fused_score: Optional[np.ndarray] = None   # Shape: (N_train,), float64
    fused_rank: Optional[np.ndarray] = None    # Shape: (N_train,), int64 (strictly 1..N_train permutation)
    u_rrf: Optional[np.ndarray] = None         # Shape: (N_train,), float64 in [0, 1]: (N_train - fused_rank) / (N_train - 1)
    
    # Routing & LLM Judge Outputs — Shape: (N_train,)
    estimated_noise_rate: float = 0.10         # Estimated from Confident Joint trace
    route_bucket: Optional[np.ndarray] = None  # Values in {"auto_flag", "llm_judge", "human_queue", "auto_clear"}
    llm_suggested_label: Optional[np.ndarray] = None  # Shape: (N_train,), int64 (-1 if not called)
    llm_confidence: Optional[np.ndarray] = None       # Shape: (N_train,), float32
    llm_rationale: Optional[List[str]] = None         # Length: N_train ("" if not called)
    
    # Metadata (dataset_hash, seed, transition matrices, timings, versions)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def train_mask(self) -> np.ndarray:
        return self.split == "train"

    @property
    def test_mask(self) -> np.ndarray:
        return self.split == "test"

    @property
    def n_train(self) -> int:
        return int(np.sum(self.train_mask))

    @property
    def n_test(self) -> int:
        return int(np.sum(self.test_mask))

    def to_scores_dataframe(self) -> pd.DataFrame:
        """Exports the FR-12 compliant `scores.parquet` table for the training split."""
        t_mask = self.train_mask
        data = {
            "sample_id": self.sample_ids[t_mask],
            "given_label": self.given_labels[t_mask],
        }
        if self.true_labels is not None:
            data["true_label"] = self.true_labels[t_mask]
        if self.is_noisy is not None:
            data["is_noisy"] = self.is_noisy
        for k, v in self.view_scores.items():
            data[f"score_{k}"] = v
        for k, v in self.view_ranks.items():
            data[f"rank_{k}"] = v
        for k, v in self.view_percentiles.items():
            data[f"u_{k}"] = v
        for k, v in self.baseline_scores.items():
            data[f"baseline_{k}"] = v
        if self.fused_score is not None:
            data["fused_score"] = self.fused_score
        if self.fused_rank is not None:
            data["fused_rank"] = self.fused_rank
        if self.u_rrf is not None:
            data["u_rrf"] = self.u_rrf
        if self.route_bucket is not None:
            data["route_bucket"] = self.route_bucket
        if self.llm_suggested_label is not None:
            data["llm_suggested_label"] = self.llm_suggested_label
        if self.llm_confidence is not None:
            data["llm_confidence"] = self.llm_confidence
        if self.llm_rationale is not None:
            data["llm_rationale"] = self.llm_rationale
        return pd.DataFrame(data)

