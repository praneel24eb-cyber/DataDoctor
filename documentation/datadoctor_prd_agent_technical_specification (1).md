# DataDoctor: Agent-Ready Product Requirements Document (PRD) & Technical Specification

**Version:** 2.1.0 (Final — Patched, Zero-Cost & Agent-Executable)

**Date:** October 29, 2026

**Authors:** Prateek Khemka & Nikhil

**License:** MIT (Core Library) | Optional `[baselines]` extra isolated for AGPL dependencies (`cleanlab`)

**Target Environment:** Python $\ge 3.10$, CPU + Free-Tier GPU (Google Colab T4 / Kaggle $2\times$T4), Free-Tier LLM APIs (Groq / Google Gemini)

---

## 0. Instructions for Coding Agents

When implementing this repository, you **MUST** obey the following architectural constraints:

1. **Strict Modularity via `DatasetArtifact`:** No detector, baseline, fusion method, or router may load raw datasets directly. Every downstream component accepts and mutates or reads from the `DatasetArtifact` container (`datadoctor/schema.py`).
2. **Train vs. Test Split Isolation ($N_{\text{train}}$ vs. $N_{\text{test}}$):**
   * All label noise injection, OOF probability estimation, detector scoring, rank fusion, and LLM routing operate **strictly on the training split** ($N_{\text{train}}$ samples, where `split == "train"`).
   * The test split ($N_{\text{test}}$ samples, `split == "test"`) is **never** corrupted with synthetic noise and is only accessed by: (a) the Hygiene Scanner (Pass 3 cross-split train/test leakage check) and (b) the Review-Budget Curve evaluator (measuring clean test-set accuracy after training-set label repair).
   * All ranks $\text{rank}_m(i) \in \{1, \dots, N_{\text{train}}\}$ and percentiles $u_m(i) \in [0, 1]$ are computed over $N_{\text{train}}$.
3. **Zero Paid APIs or Proprietary Locks:**
   * All embedding models (`facebook/dinov2-small`, `openai/clip-vit-base-patch32`, `torchvision/resnet50`, `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `scikit-learn` TF-IDF+SVD) must run locally via HuggingFace `transformers`, `sentence-transformers`, or `torchvision`.
   * The LLM Judge is **Text-Only (AG News)** in the core benchmark and must support **Groq Cloud Free Tier** (`llama-3.3-70b-versatile`), **Google AI Studio Free Tier** (`gemini-2.5-flash`), and an offline **`mock` fallback** so unit tests pass with zero network calls.
4. **AGPL Isolation:** `cleanlab` is licensed under AGPL-3.0. **NEVER** import `cleanlab` inside `datadoctor/detectors/`, `datadoctor/fusion/`, or core modules. Only import `cleanlab` lazily inside `datadoctor/baselines/cleanlab_wrap.py` wrapped in a `try...except ImportError` block.
5. **Disk Caching & Float32 Re-Normalization:**
   * Embeddings are cached to disk as `np.float16` `.npy` files to save space.
   * **CRITICAL BUG PREVENTION:** Immediately upon loading `.npy` embeddings into memory, cast to `np.float32` and **re-apply $L_2$ normalization** $\mathbf{z}_i \leftarrow \mathbf{z}_i / \max(\|\mathbf{z}_i\|_2, 10^{-12})$ so that `float16` quantization drift never degrades unit-norm dot products.
6. **Exact Mathematical Conventions:** Higher detector or fusion scores **always** mean **more suspicious (more likely to be a label error)**. Rank $1$ is the **most** suspicious sample; percentile $u = 1.0$ is the **most** suspicious sample.

---

## 1. Executive Summary & Core Contributions

DataDoctor ranks the training labels most likely to be wrong in a labelled text or image classification dataset, enabling a reviewer with a fixed review budget $B$ to repair as many true errors as possible.

### Defensible Academic Framing (Mandatory Wording for Reports/README)

* **Error Prevalence:** Benchmark test sets average at least $3.3\%$ label errors (Northcutt et al., NeurIPS 2021); LLM-assisted audits of NLP datasets report $6\text{--}21\%$ error rates (Nahum et al., EMNLP 2025); human crowd annotations on CIFAR-10N exhibit $9.03\%$ (`aggre_label`, majority vote of 3 annotators), $17.23\%$ (`random_label1`, single annotator), and $40.21\%$ (`worse_label`, worst-case of 3 annotators) error rates (Wei et al., ICLR 2022).
* **Honest Scope of Novelty:** Individual label-error signals (OOF self-confidence, $k$-NN disagreement, Confident Learning, SimiFeat) are established prior art. DataDoctor's contribution is **the parameter-free multi-view fusion, the adaptive disagreement-routed LLM verification protocol, and a rigorous review-budget benchmark**.
* **Complementary Three-View Fusion:** Combines three views via unsupervised **Reciprocal Rank Fusion (RRF)**:
  * **View 1 (`v1_self_conf`):** Out-of-fold (OOF) multinomial logistic regression self-confidence $1 - \hat{p}^{(A)}_{i, \tilde{y}_i}$ on primary self-supervised/pretrained **Backbone A**. (Strictly named *Self-Confidence*, never *Confident Learning*).
  * **View 2 (`v2_knn`):** Model-free similarity-weighted $k$-NN label disagreement on **Backbone A** embeddings. *(Note: Because V1 and V2 share Backbone A features, their errors are partially correlated; we explicitly measure their Spearman correlation and ablate V2 on Backbone B as well as 2-view V1+V3 fusion).*
  * **View 3 (`v3_cross_probe`):** OOF multinomial logistic regression self-confidence $1 - \hat{p}^{(B)}_{i, \tilde{y}_i}$ on an independent **Backbone B** (providing true feature independence).
* **Adaptive Disagreement Routing (Text MVP):** Scales routing thresholds dynamically using the Confident Joint estimated noise rate $\hat{\eta}$, routing only high-suspicion text samples where the parametric model view ($V1$) and geometric view ($V2$) strongly disagree to an LLM Judge under a hard call cap $B$. We explicitly benchmark `llm_relabel` against a zero-cost `ensemble_argmax_relabel` baseline on the exact same routed indices to isolate the true value added by the LLM.
* **Limitations Disclosed in Report:** (1) Clean AG News already contains natural baseline label noise ($\sim 3\text{--}5\%$), and modern LLMs have likely encountered AG News during pretraining (data contamination), which we quantify via a clean test-set zero-shot check; (2) $32\times 32$ CIFAR-10 images are too low-resolution for reliable zero-shot vision-LLM auditing, so LLM routing is scoped to Text (AG News).

---

## 2. Target Repository Structure & Priority Tiers

To ensure course-project feasibility, modules are split into **Core MVP (Phases 1–4)** and **Stretch Polish (Phase 5 Dashboard)**.

```text
datadoctor/
├── pyproject.toml
├── README.md
├── configs/
│   ├── grid.yaml                  # Explicit 90-run synthetic + 15-run CIFAR-10N matrix
│   ├── backbones.yaml             # Backbone A, B, C identifiers & batch sizes
│   └── llm.yaml                   # Groq / Gemini free-tier rate limits & virtual token pricing
├── datadoctor/
│   ├── __init__.py
│   ├── schema.py                  # DatasetArtifact, KNNIndexData, HygieneReport
│   ├── io/
│   │   ├── __init__.py
│   │   ├── loaders.py             # AG News (20k stratified train + 7.6k test), CSV, ImageFolder
│   │   └── cifar10n.py            # CIFAR-10 & CIFAR-10N loader + exact rate verifier
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── cache.py               # SHA-256 disk cache (.npy float16 -> float32 L2 re-norm)
│   │   ├── vision.py              # DINOv2 (A), CLIP (B), ResNet-50 (C)
│   │   └── text.py                # MiniLM-L6 (A), BGE-small (B), TF-IDF+SVD (C)
│   ├── noise/
│   │   ├── __init__.py
│   │   └── injectors.py           # Exact-count Symmetric, Literature Asymmetric, NaN-safe IDN
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── oof_probe.py           # Fast PyTorch Batched L-BFGS OOF Logistic Regression
│   │   ├── v1_self_conf.py        # View 1: Backbone A Self-Confidence
│   │   ├── v2_knn.py              # View 2: SHA1-filtered Similarity-Weighted k-NN
│   │   └── v3_cross_probe.py      # View 3: Backbone B Self-Confidence
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── rank_fusion.py         # RRF (computes fused_rank & u_rrf), Mean Percentile, 2-View Ablations
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── cleanlab_wrap.py       # Confident Joint + find_label_issues(ranked_by="normalized_margin")
│   │   ├── deep_knn.py            # Bahri et al. (ICML 2020) unweighted k-NN disagreement
│   │   └── simifeat.py            # Zhu et al. (ICML 2022) SimiFeat-R implementation
│   ├── hygiene/
│   │   ├── __init__.py
│   │   └── scanner.py             # Pass 1: SHA-1 exact, Pass 2: Mutual k-NN, Pass 3: Leakage
│   ├── router/
│   │   ├── __init__.py
│   │   ├── rules.py               # Noise-adaptive disjoint bucket assignment + routing baselines
│   │   ├── judge.py               # Escaped JSON Blind & Anchored Text LLM Judge
│   │   └── cache_db.py            # Sample-level cross-seed SQLite cache + token ledger
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # AUROC, AUPRC, P@K, R@K, F1, Clean/Noisy Spearman, LLM Accuracy
│   │   ├── budget_curve.py        # Retraining curve: oracle, remove, ensemble_argmax, llm_relabel
│   │   └── bootstrap.py           # 95% Bootstrap CIs & paired Wilcoxon signed-rank test
│   └── cli.py                     # Typer CLI entrypoints
├── benchmarks/
│   ├── run_grid.py                # Executes cached benchmark grid from configs/grid.yaml
│   ├── run_budget_curve.py        # Generates R(b) accuracy recovery curves
│   └── make_figures.py            # Matplotlib/Seaborn publication figures
├── app/                           # [STRETCH GOAL - Build only after Phases 1-4 pass]
│   └── dashboard.py               # Streamlit Reviewer UI with SQLite persistence
└── tests/
    ├── test_schema_and_cache.py
    ├── test_noise.py
    ├── test_detectors_and_fusion.py
    ├── test_hygiene.py
    ├── test_router_and_judge.py
    └── test_evaluation.py
```

---

## 3. Core Data Contract (`datadoctor/schema.py`)

All modules communicate via `DatasetArtifact`. Notice that `sha1_hashes` is a first-class field so exact duplicates are identified deterministically without floating-point cosine drift, and `u_rrf` is explicitly stored for downstream reference selection.

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Literal
import numpy as np
import pandas as pd

@dataclass
class KNNIndexData:
    indices: np.ndarray      # Shape: (N_train, k_max), dtype: int32 (excludes self & SHA-1 exact dups)
    similarities: np.ndarray # Shape: (N_train, k_max), dtype: float32 (cosine similarity in [-1, 1])

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

    def to_scores_dataframe(self) -> pd.DataFrame:
        """Exports the FR-12 compliant `scores.parquet` table for the training split."""
        t_mask = self.train_mask
        data = {
            "sample_id": self.sample_ids[t_mask],
            "given_label": self.given_labels[t_mask],
        }
        if self.true_labels is not None:
            data["true_label"] = self.true_labels[t_mask]
            data["is_noisy"] = self.is_noisy
        for k, v in self.view_scores.items():
            data[f"score_{k}"] = v
        for k, v in self.view_ranks.items():
            data[f"rank_{k}"] = v
        for k, v in self.view_percentiles.items():
            data[f"u_{k}"] = v
        if self.fused_score is not None:
            data["fused_score"] = self.fused_score
            data["fused_rank"] = self.fused_rank
            data["u_rrf"] = self.u_rrf
        if self.route_bucket is not None:
            data["route_bucket"] = self.route_bucket
        if self.llm_suggested_label is not None:
            data["llm_suggested_label"] = self.llm_suggested_label
            data["llm_confidence"] = self.llm_confidence
            data["llm_rationale"] = self.llm_rationale
        return pd.DataFrame(data)
```

---

## 4. Detailed Module Specifications (Functional Requirements)

### FR-1: Data Loaders & SHA-1 Hashing (`datadoctor/io/`)

Every loader must populate `sha1_hashes` at load time:
* For images: `hashlib.sha1(image_uint8.tobytes()).hexdigest()`.
* For text: `hashlib.sha1(" ".join(text.lower().strip().split()).encode("utf-8")).hexdigest()`.

1. **CIFAR-10 & CIFAR-10N (`cifar10n.py`):**
   * Loads standard CIFAR-10 ($N_{\text{train}} = 50,000$, $N_{\text{test}} = 10,000$, $C = 10$) via `torchvision.datasets.CIFAR10`.
   * Downloads/loads `CIFAR-10_human.pt` (Wei et al., 2022) containing keys: `clean_label`, `aggre_label`, `random_label1`, `random_label2`, `random_label3`, `worse_label`.
   * **Mandatory Verification Helper:** `verify_cifar10n_rates(pt_dict)` must assert that the empirical noise rates on the $50,000$ training images match:
     * `aggre_label`: $9.03\% \pm 0.05\%$
     * `random_label1`: $17.23\% \pm 0.05\%$
     * `worse_label`: $40.21\% \pm 0.05\%$
2. **AG News (`loaders.py`):**
   * Loads `fancyzhx/ag_news` ($C = 4$: `["World", "Sports", "Business", "Sci/Tech"]`).
   * Default benchmark setting: `subsample_train: int = 20000` (stratified $5,000$ per class from the $120,000$ training split using `seed=42`) + full $N_{\text{test}} = 7,600$ test split.

### FR-2: Feature Extractors & Re-Normalized Disk Cache (`datadoctor/extractors/`)

1. **Disk Format:** Extractors run inference in `torch.float16` (on GPU) or `torch.float32` (on CPU) with `torch.no_grad()`, normalize vectors, and write `.cache/datadoctor/{dataset_hash}_{backbone_id}.npy` in `np.float16`.
2. **Mandatory Load Re-Normalization (`cache.py`):**
   ```python
   def load_cached_embeddings(path: Path) -> np.ndarray:
       emb = np.load(path).astype(np.float32)
       norms = np.linalg.norm(emb, axis=1, keepdims=True)
       return emb / np.maximum(norms, 1e-12)
   ```
   This guarantees that even after `float16 -> float32` conversion, every vector has unit $L_2$ norm to within $10^{-7}$ and two identical embeddings have dot product $\ge 1.0 - 10^{-6}$.

| Modality | Role | Backbone ID | HuggingFace / Torchvision Model | Dim ($d$) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision** | **Backbone A (V1, V2)** | `dinov2_vits14` | `facebook/dinov2-small` | $384$ | Self-supervised; zero ImageNet label leakage. Upsample $32\times 32 \to 224\times 224$. |
| **Vision** | **Backbone B (V3, V2_B)** | `clip_vitb32` | `openai/clip-vit-base-patch32` (vision) | $512$ | Contrastive image-text supervision; distinct feature space. |
| **Vision** | **Backbone C (IDN Noise Only)** | `resnet50_tv` | `torchvision.models.resnet50(DEFAULT)` | $2048$ | Used strictly to generate Instance-Dependent Noise. |
| **Text** | **Backbone A (V1, V2)** | `minilm_l6` | `sentence-transformers/all-MiniLM-L6-v2` | $384$ | Fast sentence encoder. |
| **Text** | **Backbone B (V3, V2_B)** | `bge_small` | `BAAI/bge-small-en-v1.5` | $384$ | Distinct training recipe and hard-negative mining. |
| **Text** | **Backbone C (IDN Noise Only)** | `tfidf_svd384` | `TfidfVectorizer(max_features=20000) + TruncatedSVD(384)` | $384$ | Purely lexical representation independent of neural transformers. |

### FR-3: Fast PyTorch Batched OOF Probability Engine (`datadoctor/detectors/oof_probe.py`)

Running `scikit-learn` CPU `LogisticRegression` 900 times on $50,000 \times 512$ arrays takes hours. To ensure the 90-cell grid finishes in **under 15 minutes total**, implement `compute_oof_probabilities` using **PyTorch L-BFGS** (`torch.optim.LBFGS` on CUDA if available, or multi-threaded CPU tensors):
* **Input:** Training embeddings $\mathbf{Z}_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times d}$, noisy training labels $\tilde{\mathbf{y}} \in \{0, \dots, C-1\}^{N_{\text{train}}}$, $M = 5$ folds, regularization $C_{\text{reg}} = 1.0$ (with weight decay $\lambda = \frac{1}{C_{\text{reg}} \cdot N_{\text{fold}}}$), `n_repeats = 1`.
* **Hyperparameter $C_{\text{reg}}$ Selection:** Tuned once on Seed 0 from $\{0.1, 1.0, 10.0\}$ and cached in `artifact.meta[f"best_C_{backbone}"]`.
* **Speed Requirement:** Must complete a 5-fold fit on $N_{\text{train}} = 50,000, d = 512, C = 10$ in $< 2.5\text{ seconds}$ on a T4 GPU or $< 8\text{ seconds}$ on a modern 8-core CPU.

### FR-4, FR-5, FR-6: The Three Views & Backbone-B Ablation (`datadoctor/detectors/`)

All views operate strictly on the $N_{\text{train}}$ training samples:

1. **View 1 — Self-Confidence (`v1_self_conf.py`):**
   $$S_{V1}(i) = 1.0 - \hat{p}^{(A)}_{i, \tilde{y}_i}$$
2. **View 2 — SHA1-Filtered Similarity-Weighted $k$-NN Disagreement (`v2_knn.py`):**
   * **Cached $k$-NN Graph:** Because embeddings $\mathbf{Z}^{(A)}_{\text{train}}$ and `sha1_hashes[train]` do not change across noise seeds, compute the top $k_{\max} = 50$ neighbours **once per dataset/backbone** and cache to `.cache/datadoctor/{dataset_hash}_{backbone_id}_knn50.npz`.
   * **Exact-Duplicate Exclusion Rule:** When building the neighbour list for training sample $i$, exclude index $j$ if $j == i$ **OR** `sha1_hashes[i] == sha1_hashes[j]`. (Never rely on a cosine threshold to filter exact duplicates).
   * For a given $k \in \{5, 10, 15, 25, 50\}$ (default $k = 15$), slice the first $k$ valid neighbours $\mathcal{N}_k(i)$, clamp weights $w_{ij} = \max(s_{ij}, 10^{-6})$, and compute:
     $$S_{V2}(i) = \frac{\sum_{j \in \mathcal{N}_k(i)} w_{ij} \, \mathbb{I}(\tilde{y}_j \neq \tilde{y}_i)}{\sum_{j \in \mathcal{N}_k(i)} w_{ij}}$$
   * **Backbone-B Geometry Ablation (`v2_B`):** Also compute $S_{V2\_B}(i)$ using the cached $k$-NN graph of **Backbone B** so we can compare `RRF(V1, V2_A, V3)` vs. `RRF(V1, V2_B, V3)` in the ablation table.
3. **View 3 — Cross-Backbone Self-Confidence (`v3_cross_probe.py`):**
   $$S_{V3}(i) = 1.0 - \hat{p}^{(B)}_{i, \tilde{y}_i}$$

### FR-7: Rank Conversion & Reciprocal Rank Fusion (`datadoctor/fusion/rank_fusion.py`)

Let $N = N_{\text{train}}$.
1. **Per-View Ranks & Percentiles:**
   * For each view $m$, compute 1-indexed average ranks $\text{rank}_m(i) \in [1, N]$ via `scipy.stats.rankdata(-scores, method="average")`.
   * Compute suspicion percentile $u_m(i) = \frac{N - \text{rank}_m(i)}{N - 1} \in [0, 1]$.
2. **Reciprocal Rank Fusion (Default 3-View `rrf_v1_v2_v3`):**
   With smoothing constant $\kappa = 60$:
   $$S_{\text{RRF}}(i) = \sum_{m \in \{V1, V2, V3\}} \frac{1}{\kappa + \text{rank}_m(i)} + 10^{-9} \cdot S_{V1}(i)$$
   *(The $10^{-9} \cdot S_{V1}(i)$ term breaks exact rank ties deterministically).*
3. **Fused Rank & Fused Percentile (`u_rrf`):**
   * `fused_rank` = 1-indexed strict permutation rank `np.argsort(np.argsort(-S_RRF)) + 1` in $\{1, \dots, N_{\text{train}}\}$.
   * `u_rrf` = $\frac{N_{\text{train}} - \text{fused\_rank}}{N_{\text{train}} - 1} \in [0, 1]$.
4. **Required Ablation Variants in `rank_fusion.py`:**
   * `rrf_v1_v3` (2-view cross-backbone model fusion without $k$-NN).
   * `rrf_v1_v2B_v3` (3-view fusion where V2 runs on Backbone B instead of Backbone A).
   * `mean_percentile` ($\frac{1}{3}(u_{V1} + u_{V2} + u_{V3})$).

### FR-8: Distinct Baselines (`datadoctor/baselines/`)

To ensure no two baselines compute the same ranking, implement these exact definitions on $N_{\text{train}}$:
1. **`loss_self_conf`**: Raw self-confidence $S_{V1}(i) = 1 - \hat{p}^{(A)}_{i, \tilde{y}_i}$.
2. **`cleanlab_cl` (`cleanlab_wrap.py`)**:
   * **Distinct from `loss_self_conf`!** Uses the Confident Joint to filter and rank issues by **Normalized Margin** ($\hat{p}_{i, \tilde{y}_i} - \max_{c \neq \tilde{y}_i} \hat{p}_{i, c}$), prioritizing samples inside the Confident Joint off-diagonals first:
   * If `cleanlab` is installed: call `ranked_indices = cleanlab.filter.find_label_issues(labels=given_labels, pred_probs=oof_probs_A, return_indices_ranked_by="normalized_margin")`. Convert `ranked_indices` into a full length-$N_{\text{train}}$ score array where flagged indices get scores $2.0 - \frac{\text{pos}}{N_{\text{train}}}$ and unflagged indices are ranked below them by normalized margin $1.0 - (\hat{p}_{i, \tilde{y}_i} - \max_{c \neq \tilde{y}_i} \hat{p}_{i, c}) / 2.0$.
   * Also implement `estimate_confident_joint_and_noise_rate(given_labels, oof_probs_A)` from scratch (Northcutt et al., 2021 Eq. 1–3) under the MIT license so `artifact.estimated_noise_rate` ($\hat{\eta} = 1 - \frac{\text{trace}(C_{\tilde{y}, y^*})}{N_{\text{train}}}$) is always available even without `cleanlab` installed, and unit-test that it matches `cleanlab.count.compute_confident_joint` on synthetic arrays.
3. **`deep_knn` (`deep_knn.py`)**: Unweighted fraction of $k=15$ neighbours on Backbone A with $\tilde{y}_j \neq \tilde{y}_i$ (producing discrete steps $\{0/15, 1/15, \dots, 15/15\}$, demonstrating why V2's similarity weighting matters).
4. **`simifeat_r` (`simifeat.py`)**: Zhu et al. (ICML 2022) SimiFeat-R local consensus score on Backbone A.
5. **`random`**: Seeded uniform random float array in $[0, 1]$.

### FR-9: Exact-Count, Literature-Standard & `NaN`-Safe Noise Injectors (`datadoctor/noise/injectors.py`)

Noise is injected **only** into `given_labels[train_mask]`. All injectors flip exact integer counts per source class and return `(noisy_train_labels, is_noisy_mask, target_T, empirical_T, meta_dict)`:

1. **Symmetric Noise (`inject_symmetric`, $\eta \in \{0.10, 0.20, 0.40\}$):**
   * For each class $c \in \{0, \dots, C-1\}$, select exactly $m_c = \text{round}(\eta \cdot N_c)$ indices without replacement and flip uniformly to $\{0, \dots, C-1\} \setminus \{c\}$.
2. **Asymmetric Class-Conditional Noise (`inject_asymmetric`, $\eta_{\text{source}} \in \{0.10, 0.20, 0.40\}$):**
   * **CIFAR-10 Standard Literature Map (Patrini et al., CVPR 2017; Wei et al., ICLR 2022):**
     * `9 (truck) -> 1 (automobile)`
     * `2 (bird) -> 0 (airplane)`
     * `4 (deer) -> 7 (horse)`
     * `3 (cat) -> 5 (dog)`
     * `5 (dog) -> 3 (cat)`
     * *(Classes 0, 1, 6, 7, 8 remain unflipped sources; 5 of 10 classes flip with exact source rate $\eta_{\text{source}} \in \{0.10, 0.20, 0.40\}$, yielding exact overall dataset noise rate $\eta_{\text{global}} = \eta_{\text{source}} / 2 \in \{0.05, 0.10, 0.20\}$. Always log both `source_eta` and `global_eta` in `meta_dict`!)*
   * **AG News Bijective 2-Cycle Map ($\eta \in \{0.10, 0.20, 0.40\}$):**
     * `0 (World) <-> 2 (Business)` and `1 (Sports) <-> 3 (Sci/Tech)`. (Every class is a source class; $\eta_{\text{global}} = \eta_{\text{source}}$).
3. **`NaN`-Safe Instance-Dependent Noise (`inject_idn`, $\eta \in \{0.10, 0.20, 0.40\}$):**
   * **Input:** `embeddings_C[train_mask]` (strictly Backbone C).
   * Draw per-instance flip propensity $q_i \sim \mathcal{N}_{\text{trunc}}(\mu=\eta, \sigma=0.10)$ and **clip strictly to $[10^{-4}, 0.95]$** so $q_i > 0$ everywhere.
   * Draw projection matrix $\mathbf{W} \sim \mathcal{N}(0, 1)$ of shape $(d_C, C)$ and compute raw finite logits $\mathbf{L} = \mathbf{Z}^{(C)}_{\text{train}} \mathbf{W}$.
   * **CRITICAL `NaN` PREVENTION:** First multiply finite logits by $q_i$: $\tilde{\mathbf{L}}_i = q_i \cdot \mathbf{L}_i$. **Only after multiplication**, mask the true class $\tilde{L}_{i, y_i} = -\infty$ so `0.0 * -np.inf` can never occur.
   * Within each true class $c$, sort samples descending by $\max_{k \neq c} \tilde{L}_{i, k}$ and flip the top $m_c = \text{round}(\eta \cdot N_c)$ samples to $\arg\max_{k \neq c} \tilde{L}_{i, k}$.

### FR-10: Three-Pass Hygiene Scanner (`datadoctor/hygiene/scanner.py`)

Operates across the full dataset ($N_{\text{total}} = N_{\text{train}} + N_{\text{test}}$):
1. **Pass 1 (Exact Duplicates via SHA-1):** Group samples sharing identical `sha1_hashes`. Zero reliance on embedding cosine similarity.
2. **Pass 2 (Near-Duplicates via Mutual $k$-NN):**
   * Using re-normalized Backbone A embeddings, find pairs $(i, j)$ that are not SHA-1 identical, are Mutual $k$-NNs ($j \in \mathcal{N}_{15}(i)$ and $i \in \mathcal{N}_{15}(j)$), and satisfy $s_{ij} \ge \tau_{\text{dup}}$.
   * Include `calibrate_neardup_threshold(embeddings, ground_truth_pairs)` to tune $\tau_{\text{dup}}$ by maximizing $F_1$ on known duplicate lists (or default to $\tau_{\text{dup}} = 0.965$).
3. **Pass 3 (Cross-Split Leakage & Label Conflicts):**
   * Flag any exact or near-duplicate pair spanning `train` and `test` (`split[i] != split[j]`).
   * Flag any exact or near-duplicate pair inside `train` with conflicting labels (`given_labels[i] != given_labels[j]`).

### FR-11: Adaptive Disagreement Router & Unconfounded Text LLM Judge (`datadoctor/router/`)

#### 1. Noise-Adaptive Disjoint Percentile Router (`rules.py`)
Fixed percentile thresholds ($0.95$) fail when noise is $40\%$ vs. $5\%$. Let $\hat{\eta} = \text{clip}(\text{artifact.estimated\_noise\_rate}, 0.05, 0.45)$ be the Confident Joint estimated noise rate on the training split. Define adaptive thresholds:
* High-suspicion threshold: $\tau_{\text{high}} = 1.0 - 0.5 \cdot \hat{\eta}$ (e.g., $0.95$ at $\hat{\eta}=10\%$, $0.80$ at $\hat{\eta}=40\%$).
* Clear threshold: $\tau_{\text{low}} = \max(0.50, 1.0 - 2.5 \cdot \hat{\eta})$ (e.g., $0.75$ at $\hat{\eta}=10\%$, $0.50$ at $\hat{\eta}=20\%$).
* Disagreement gap: $\delta_{\text{gap}} = 0.35$.

Assign every training sample $i \in \{0, \dots, N_{\text{train}}-1\}$ using this **strictly disjoint `if / elif / else` cascade**:
1. **`if` $\min(u_{V1}(i), u_{V2}(i), u_{V3}(i)) \ge \tau_{\text{high}}$:**
   * Assign `cand_bucket[i] = "auto_flag"`.
2. **`elif` $\max(u_{V1}(i), u_{V2}(i), u_{V3}(i)) \le \tau_{\text{low}}$:**
   * Assign `cand_bucket[i] = "auto_clear"`.
3. **`elif` $\max(u_{V1}(i), u_{V2}(i)) \ge \tau_{\text{high}}$ `and` $|u_{V1}(i) - u_{V2}(i)| \ge \delta_{\text{gap}}$:**
   * Mark as a `disagreement_candidate`.
4. **`else`:**
   * Assign `cand_bucket[i] = "human_queue"`.

Finally, take all `disagreement_candidate` indices, sort them descending by `fused_score` ($S_{\text{RRF}}$), and assign the top $B = \lfloor b_{\text{llm}} \cdot N_{\text{train}} \rfloor$ (default $b_{\text{llm}} = 0.02$, i.e., $400$ samples on AG News 20k) to `route_bucket = "llm_judge"`. Any leftover `disagreement_candidate` beyond cap $B$ is assigned to `route_bucket = "human_queue"`. **(Note: `auto_clear` samples remain `"auto_clear"` and are never overwritten!)**

#### 2. Text-Only LLM Judge with Escaped JSON Braces (`judge.py`)
* **Modality Guard:** If `artifact.modality != "text"`, `judge.py` raises a clear `ValueError("LLM Judge in MVP is scoped strictly to text datasets (AG News).")` unless `--allow-vision-stretch` is explicitly set.
* **Clean Medoid Context Pool:** Trusted reference pool consists of training samples $j$ where `u_rrf[j] < 0.25` and $\tilde{y}_j$ agrees with $100\%$ of its top-5 Backbone-A neighbours. To maximize **cross-seed SQLite cache hits**, select 1 static clean medoid (the sample in class $c$ closest to the class mean embedding $\bar{\mathbf{z}}^{(A)}_c$) for each of the $C=4$ AG News classes. Because these 4 class medoids depend mainly on the clean cluster cores, cached prompts keyed by `(sample_sha1, medoid_hashes, model_id, prompt_mode)` reuse responses across runs.
* **Escaped Python `.format()` Prompt Template:**
  ```python
  BLIND_PROMPT_TEMPLATE = (
      "You are an expert dataset auditor. Classify the TARGET TEXT into exactly one "
      "of these valid categories: {class_names}.\n\n"
      "Verified reference examples:\n"
      "{reference_block}\n\n"
      "TARGET TEXT: \"{target_text}\"\n\n"
      "Respond ONLY with valid JSON matching this exact schema:\n"
      "{{\"label\": \"<exact category name>\", \"confidence\": 0.95, \"rationale\": \"<one concise sentence>\"}}"
  )
  ```
* **Acceptance Rule for LLM Relabelling:** When the LLM returns `{"label": L, "confidence": conf}`, convert `L` to class integer `c_llm`. In `llm_relabel` mode, replace `given_labels[i]` with `c_llm` if and only if `conf >= 0.60`; otherwise fall back to the ensemble argmax $\arg\max_c (\hat{\mathbf{p}}^{(A)}_i + \hat{\mathbf{p}}^{(B)}_i)$.

### FR-12: Unconfounded Evaluation & Review-Budget Curve (`datadoctor/evaluation/`)

1. **Detection Ranking Metrics (`metrics.py`):**
   * Evaluated over $N_{\text{train}}$ against `is_noisy`: `auroc`, `auprc`, `precision_at_k` and `recall_at_k` for $K \in \{1\%, 2\%, 5\%, 10\%, 20\%\}$ of $N_{\text{train}}$, `precision_at_num_errors` ($K = \sum \text{is\_noisy}$), and boolean `f1` at the top $\lfloor \hat{\eta} \cdot N_{\text{train}} \rfloor$ threshold.
   * **Spearman Independence Table:** Compute pairwise Spearman rank correlations $\rho(V1, V2_A)$, $\rho(V1, V3)$, $\rho(V2_A, V3)$, and $\rho(V1, V2_B)$ separately on clean (`~is_noisy`) and noisy (`is_noisy`) samples.
2. **Unconfounded LLM Judge Evaluation (AG News):**
   * **Direct Routed Accuracy:** On the $B$ samples routed to `llm_judge`, report:
     1. **LLM Zero-Shot Clean Test Accuracy:** Accuracy of the LLM Judge on a stratified $500$-sample slice of the clean AG News test set.
     2. **Routed Error-Detection Precision:** Fraction of the $B$ routed samples that are truly corrupted (`is_noisy[i] == True`), comparing **Disagreement Routing** vs. **Top-$B$ RRF Routing** vs. **Random-$B$ Routing**.
     3. **Routed Relabel Accuracy (LLM vs. Free Classifier Argmax):** On truly corrupted routed samples, compare how often `llm_suggested_label[i] == true_labels[i]` versus how often the zero-cost classifier ensemble $\arg\max_c (\hat{\mathbf{p}}^{(A)}_i + \hat{\mathbf{p}}^{(B)}_i) == \text{true\_labels}[i]$.
3. **Review-Budget Curve (`budget_curve.py`):**
   * Evaluates budgets $b \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$ of $N_{\text{train}}$.
   * **Ordering $\pi$ for `datadoctor_rrf`:** All training samples sorted descending by `fused_score` ($S_{\text{RRF}}$).
   * **Ordering $\pi_{\text{llm}}$ for `datadoctor_llm` (Evaluated at $b \in \{0.01, 0.02\}$ within the LLM cap):** Prioritizes `auto_flag` samples first (ordered by $S_{\text{RRF}}$), followed immediately by `llm_judge` samples (where the LLM disagreed with `given_label` and had `confidence >= 0.60`, ordered by `llm_confidence * S_RRF`), followed by the remaining `fused_rank` order.
   * **Four Repair Modes Compared at Every Budget $b$:**
     1. `oracle_relabel`: Top $K = \lfloor b \cdot N_{\text{train}} \rfloor$ samples under $\pi$ get `true_labels[i]`.
     2. `remove`: Top $K$ samples under $\pi$ are removed from training.
     3. `ensemble_argmax_relabel` *(New Unconfounded Baseline!)*: Top $K$ samples under $\pi$ are automatically relabelled to $\arg\max_{c \neq \tilde{y}_i} (\hat{p}^{(A)}_{i, c} + \hat{p}^{(B)}_{i, c})$ with **zero LLM calls**.
     4. `llm_relabel` *(AG News, $b \le 0.02$)*: Top $K$ samples under $\pi_{\text{llm}}$ use `llm_suggested_label[i]` (if routed and `conf >= 0.60`) and `ensemble_argmax_relabel` for `auto_flag` items. Comparing Mode 4 against Mode 3 isolates whether calling the LLM beats free ensemble relabelling!

---

## 5. Configuration Files (`configs/`)

Agents must create these exact YAML files so `benchmarks/run_grid.py` is 100% config-driven:

### `configs/grid.yaml`
```yaml
project: datadoctor_benchmark
random_seeds: [42, 43, 44, 45, 46]

datasets:
  ag_news:
    modality: text
    subsample_train: 20000
    backbone_A: minilm_l6
    backbone_B: bge_small
    backbone_C: tfidf_svd384
    run_llm_judge: true
  cifar10:
    modality: vision
    subsample_train: null   # Full 50,000 train / 10,000 test
    backbone_A: dinov2_vits14
    backbone_B: clip_vitb32
    backbone_C: resnet50_tv
    run_llm_judge: false

synthetic_grid:
  noise_types: ["symmetric", "asymmetric", "idn"]
  noise_rates: [0.10, 0.20, 0.40]   # For CIFAR-10 asymmetric, source_eta = [0.10, 0.20, 0.40] -> global_eta = [0.05, 0.10, 0.20]
  # Total synthetic cells = 2 datasets * 3 noise_types * 3 noise_rates * 5 seeds = 90 cells

real_noise_grid:
  dataset: cifar10n
  label_sets: ["aggre_label", "random_label1", "worse_label"]
  cv_seeds: [42, 43, 44, 45, 46]
  # Total real-noise cells = 3 label_sets * 5 seeds = 15 cells

detectors:
  knn_k_default: 15
  knn_k_ablations: [5, 10, 15, 25, 50]
  rrf_k_default: 60
  rrf_k_ablations: [1, 10, 60, 600]

budget_curve:
  budgets: [0.01, 0.02, 0.05, 0.10, 0.20]
  repair_modes: ["oracle_relabel", "remove", "ensemble_argmax_relabel", "llm_relabel"]
```

### `configs/llm.yaml`
```yaml
default_provider: groq
budget_cap_fraction: 0.02   # B = 400 calls max on AG News 20k
min_confidence_to_relabel: 0.60

providers:
  groq:
    model_id: "llama-3.3-70b-versatile"
    temperature: 0.0
    max_rpm: 28             # Free-tier safe rate limit (requests per minute)
    virtual_cost_per_1m_input_tokens: 0.59
    virtual_cost_per_1m_output_tokens: 0.79
  gemini:
    model_id: "gemini-2.5-flash"
    temperature: 0.0
    max_rpm: 14
    virtual_cost_per_1m_input_tokens: 0.15
    virtual_cost_per_1m_output_tokens: 0.60
  mock:
    model_id: "deterministic-mock-v1"
    temperature: 0.0
    max_rpm: 100000
    virtual_cost_per_1m_input_tokens: 0.0
    virtual_cost_per_1m_output_tokens: 0.0
```

---

## 6. Phased Implementation & Gate Verification Plan (For Agents)

Coding agents must implement and verify Phases 1–4 first (Core Research & CLI MVP) before building Phase 5 (Figures & Stretch Streamlit UI).

### Phase 1: Schema, I/O Loaders, SHA-1 Hashing, and Re-Normalized Embedding Cache
* **Files to build:** `pyproject.toml`, `configs/*.yaml`, `datadoctor/schema.py`, `datadoctor/io/loaders.py`, `datadoctor/io/cifar10n.py`, `datadoctor/extractors/cache.py`, `datadoctor/extractors/vision.py`, `datadoctor/extractors/text.py`.
* **Gate 1 Test (`tests/test_schema_and_cache.py`):**
  * Assert `load_cached_embeddings()` casts `float16 -> float32` and restores unit $L_2$ norms ($\max_i |\|\mathbf{z}_i\|_2 - 1.0| < 10^{-6}$).
  * Assert two identical vectors saved as `float16` and reloaded have identical `sha1_hashes` and cosine similarity $\ge 0.99999$.
  * Assert `verify_cifar10n_rates()` verifies $9.03\%, 17.23\%, 40.21\%$.

### Phase 2: Exact Noise Injectors, Fast PyTorch OOF Probe, Views V1–V3, and Distinct Baselines
* **Files to build:** `datadoctor/noise/injectors.py`, `datadoctor/detectors/oof_probe.py`, `datadoctor/detectors/v1_self_conf.py`, `datadoctor/detectors/v2_knn.py`, `datadoctor/detectors/v3_cross_probe.py`, `datadoctor/baselines/*`.
* **Gate 2 Test (`tests/test_noise.py`, `tests/test_detectors_and_fusion.py`):**
  * Assert `inject_idn` never produces `NaN` even if $q_i$ is forced to $0.0$.
  * Assert `inject_asymmetric` on CIFAR-10 flips only the 5 literature source classes with exact counts ($\eta_{\text{global}} = \eta_{\text{source}} / 2$) and on AG News uses the bijective 2-cycle map ($\eta_{\text{global}} = \eta_{\text{source}}$).
  * Assert `cleanlab_cl` and `loss_self_conf` produce **different** rankings on a synthetic dataset with class-imbalanced confidence.
  * Assert `v2_knn` excludes exact duplicates strictly via `sha1_hashes[i] == sha1_hashes[j]`.

### Phase 3: RRF Fusion, Hygiene Scanner, and Unconfounded Evaluation Suite
* **Files to build:** `datadoctor/fusion/rank_fusion.py`, `datadoctor/hygiene/scanner.py`, `datadoctor/evaluation/metrics.py`, `datadoctor/evaluation/budget_curve.py`, `datadoctor/evaluation/bootstrap.py`.
* **Gate 3 Test (`tests/test_hygiene.py`, `tests/test_evaluation.py`):**
  * Assert `rank_fusion.py` populates `fused_score`, `fused_rank` ($1\dots N_{\text{train}}$), and `u_rrf` ($\in [0, 1]$), plus ablation keys `rrf_v1_v3` and `rrf_v1_v2B_v3`.
  * Assert `budget_curve.py` runs only on `train_mask` and evaluates on clean `test_mask`, comparing `oracle_relabel`, `remove`, and `ensemble_argmax_relabel`.

### Phase 4: Adaptive Disagreement Router, Text LLM Judge, and CLI
* **Files to build:** `datadoctor/router/rules.py`, `datadoctor/router/cache_db.py`, `datadoctor/router/judge.py`, `datadoctor/cli.py`.
* **Gate 4 Test (`tests/test_router_and_judge.py`):**
  * Assert that `route_bucket` preserves `"auto_clear"` samples (never overwriting them with `"human_queue"`), scales thresholds with `estimated_noise_rate`, and strictly respects $\sum \mathbb{I}(\text{route\_bucket} == \text{"llm\_judge"}) \le B$.
  * Assert `BLIND_PROMPT_TEMPLATE.format(...)` formats cleanly without `KeyError` on JSON braces and hits the SQLite cache on a second seed with identical sample SHA-1.

### Phase 5: Benchmark Runners, Publication Figures, and Stretch Streamlit UI
* **Files to build:** `benchmarks/run_grid.py`, `benchmarks/run_budget_curve.py`, `benchmarks/make_figures.py`, `app/dashboard.py`, `README.md`.
* **Gate 5 Test:** Run `python benchmarks/run_grid.py --smoke-test` to execute an end-to-end miniature grid across all 3 noise types, outputting `reports/grid_results.csv`, Spearman correlation tables, and generating the report plots in `reports/figures/`.