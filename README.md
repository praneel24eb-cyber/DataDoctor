# DataDoctor: Multi-View Label-Error Detection, Dataset Hygiene & Disagreement-Routed Repair

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Course Project:** AI / Machine Learning  
> **Authors:** Prateek Khemka, Nikhil, Praneel K A 
> **Version:** 2.1.0  
> **License:** MIT (Core Library) | Optional `cleanlab` isolation for AGPL compliance  

---

## 1. Executive Summary & Defensible Academic Framing

Real-world datasets contain substantial label errors that undermine model generalization, misguide hyperparameter tuning, and corrupt benchmark evaluations:
* **Benchmark Test Set Errors:** Across popular benchmark test sets, label error rates average at least **3.3%** (Northcutt et al., NeurIPS 2021).
* **NLP Annotations:** LLM-assisted audits of established NLP datasets reveal error rates between **6% and 21%** (Nahum et al., EMNLP 2025).
* **Real-World Human Crowdsourcing:** The CIFAR-10N benchmark demonstrates human crowd noise rates of **9.03%** (`aggre_label`, majority vote of 3 annotators), **17.23%** (`random_label1`, single annotator), and **40.21%** (`worse_label`, worst annotator) (Wei et al., ICLR 2022).

### Honest Scope of Novelty
Individual label-error signals—such as out-of-fold self-confidence, $k$-NN label disagreement, Confident Learning, and representation consensus (SimiFeat)—represent established prior art. 

**DataDoctor’s contribution lies in:**
1. **Parameter-Free Multi-View Fusion:** Combining parametric out-of-fold predictions with non-parametric geometric disagreement across independent foundation backbones via unsupervised **Reciprocal Rank Fusion (RRF)**.
2. **Noise-Adaptive Disagreement Routing:** Dynamically adjusting suspicion thresholds based on the Confident Joint noise estimate ($\hat{\eta}$), selectively routing high-uncertainty text instances where the model view ($V_1$) and geometric view ($V_2$) disagree to an LLM Judge under a strict review-budget constraint.
3. **Unconfounded Review-Budget Benchmarking:** Retraining classifiers under realistic review budgets $b \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$ (i.e., 1%, 2%, 5%, 10%, 20%) to rigorously measure test-accuracy recovery, explicitly contrasting LLM relabelling against zero-cost classifier ensemble relabelling (`ensemble_argmax_relabel`).

### Disclosed Limitations
* **Natural Baseline Noise & Pretraining Contamination:** Clean AG News already contains an estimated $3\text{--}5\%$ natural label noise. Additionally, modern commercial LLMs may have encountered AG News during pretraining; we quantify this effect via clean test-set zero-shot benchmarking.
* **Vision Modality Scope:** While the multi-view detectors and hygiene scanner support image classification (e.g., CIFAR-10 / CIFAR-10N via DINOv2 and CLIP), zero-shot multimodal LLMs are unreliable on low-resolution $32 \times 32$ pixels. Hence, LLM Judge routing is officially scoped to text (AG News).

---

## 2. Core Methodological Architecture

```
                                      Raw Dataset
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
                   Split: Train                         Split: Test
                  (N_train samples)                  (N_test samples)
                         │                                   │
       ┌─────────────────┼─────────────────┐                 │
       ▼                 ▼                 ▼                 │
  [Backbone A]      [Backbone B]      [Backbone C]           │
(DINOv2/MiniLM)     (CLIP/BGE)       (ResNet/TF-IDF)         │
       │                 │                 │                 │
       ├─────────────────┤           (IDN Noise)             │
       ▼                 ▼                                   ▼
 [OOF Probe A]     [OOF Probe B]                 [Hygiene Scanner (3-Pass)]
  (Fast L-BFGS)     (Fast L-BFGS)                 ├─ Pass 1: SHA-1 Exact Dups
       │                 │                        ├─ Pass 2: Mutual k-NN Near-Dups
       ├───────────┐     │                        └─ Pass 3: Cross-Split Leakage
       ▼           ▼     ▼
     View 1      View 2  View 3
   (Self-Conf)   (k-NN)  (Cross-Probe)
       │           │     │
       └─────┬─────┴─────┘
             ▼
   [Reciprocal Rank Fusion] ──> Fused Suspicion Rank & Percentile u_RRF
             │
             ▼
   [Noise-Adaptive Router]
   ├─ Auto-Flag (consensus high suspicion)
   ├─ Auto-Clear (consensus clean)
   ├─ Human Queue (borderline ambiguity)
   └─ Disagreement Candidates (V1 vs V2 conflict) ──> [LLM Judge (under budget B)]
```

### The Three Complementary Views
All detectors operate strictly on the training partition ($N_{\text{train}}$):
1. **View 1 (`v1_self_conf`):** Out-of-fold multinomial logistic regression self-confidence on primary representation **Backbone A**:
   $$S_{V1}(i) = 1.0 - \hat{p}^{(A)}_{i, \tilde{y}_i}$$
2. **View 2 (`v2_knn`):** Similarity-weighted $k$-NN label disagreement on **Backbone A** embeddings with **mandatory SHA-1 exact-duplicate exclusion**:
   $$S_{V2}(i) = \frac{\sum_{j \in \mathcal{N}_k(i)} w_{ij} \, \mathbb{I}(\tilde{y}_j \neq \tilde{y}_i)}{\sum_{j \in \mathcal{N}_k(i)} w_{ij}}, \quad w_{ij} = \max(s_{ij}, 10^{-6})$$
3. **View 3 (`v3_cross_probe`):** Out-of-fold self-confidence on independent **Backbone B**:
   $$S_{V3}(i) = 1.0 - \hat{p}^{(B)}_{i, \tilde{y}_i}$$

### Reciprocal Rank Fusion (RRF)
With rank smoothing parameter $\kappa = 60$ and deterministic tie-breaking:
$$S_{\text{RRF}}(i) = \sum_{m \in \{V1, V2, V3\}} \frac{1}{\kappa + \text{rank}_m(i)} + 10^{-9} \cdot S_{V1}(i)$$

### Three-Pass Hygiene Scanner
Operates over the entire dataset ($N_{\text{train}} + N_{\text{test}}$):
* **Pass 1 (Exact Duplicates):** Partitions samples by SHA-1 hash matching (immune to floating-point drift).
* **Pass 2 (Near-Duplicates):** Identifies mutual $k$-NN pairs with cosine similarity $s_{ij} \ge \tau_{\text{dup}}$ (default $\tau_{\text{dup}} = 0.965$, calibrated against ground-truth pairs).
* **Pass 3 (Cross-Split Leakage & Conflicts):** Flags duplicate pairs that cross train/test splits or share identical inputs with conflicting labels.

---

## 3. Supported Modalities & Backbones

| Modality | Role | Identifier | Model | Dim ($d$) | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision** | Backbone A | `dinov2_vits14` | `facebook/dinov2-small` | 384 | Self-supervised; zero ImageNet label leakage |
| **Vision** | Backbone B | `clip_vitb32` | `openai/clip-vit-base-patch32` | 512 | Multimodal contrastive supervision |
| **Vision** | Backbone C | `resnet50_tv` | `torchvision.models.resnet50` | 2048 | Generation of Instance-Dependent Noise (IDN) |
| **Text** | Backbone A | `minilm_l6` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Dense semantic sentence encoding |
| **Text** | Backbone B | `bge_small` | `BAAI/bge-small-en-v1.5` | 384 | Contrastive retrieval with hard negatives |
| **Text** | Backbone C | `tfidf_svd384` | `TfidfVectorizer` + `TruncatedSVD(384)` | 384 | Lexical representation independent of transformers |

---

## 4. Invariants & Defensive Engineering

1. **Strict Phase Gating:** Verified via 56 passing unit and integration tests across 5 gates.
2. **Float16 Norm Drift Invariant:** Cached `.npy` representations are stored as `np.float16` to conserve disk space, but **immediately cast to `np.float32` and re-normalized** to unit $L_2$ norm upon loading:
   $$\mathbf{z}_i \leftarrow \frac{\mathbf{z}_i}{\max(\|\mathbf{z}_i\|_2, 10^{-12})}$$
   This ensures dot products between identical representations evaluate to $\ge 1.0 - 10^{-6}$.
3. **Train / Test Split Isolation:** Noise injection, OOF estimation, $k$-NN graphs, and RRF ranking execute strictly on `split == "train"`. The test split is touched exclusively by the hygiene cross-split leakage check and budget evaluation.
4. **Exact Duplicate Exclusion via SHA-1:** When building $k$-NN graphs, sample $j$ is excluded from sample $i$'s neighbors if $j == i$ or `sha1_hashes[i] == sha1_hashes[j]`, preventing cosine-threshold floating-point errors.
5. **Fast PyTorch L-BFGS OOF Engine:** Multinomial logistic regression runs in batched PyTorch with L-BFGS, completing 5-fold cross-validation on 50,000 samples in **<2 seconds** on GPU and **<8 seconds** on CPU.
6. **AGPL Isolation:** `cleanlab` is imported lazily inside `datadoctor/baselines/cleanlab_wrap.py` wrapped in `try/except ImportError`. The core library runs without `cleanlab`.

---

## 5. Installation & Setup

### Requirements
* Python $\ge 3.10$
* PyTorch $\ge 2.0$ (CPU or CUDA)

### Clone & Install
```bash
git clone https://github.com/praneel-ka/DataDoctor.git
cd DataDoctor

# Core installation (MIT-licensed)
pip install -e .

# Optional: install cleanlab baseline (AGPL-3.0)
pip install -e ".[baselines]"

# Optional: install Streamlit interactive dashboard
pip install -e ".[dashboard]"
```

### Environment Variables for LLM Judge (Optional)
If using cloud LLM verification (AG News text audit):
```bash
# Groq Cloud (Free Tier)
export GROQ_API_KEY="gsk_..."

# Google Gemini (Free Tier)
export GEMINI_API_KEY="AIza..."
```
*(If no API key is provided, DataDoctor automatically uses its zero-cost offline deterministic mock engine).*

---

## 6. Quickstart & CLI Usage

DataDoctor includes a rich terminal CLI:

### 1. Library Status & Information
```bash
datadoctor info
```

### 2. Run a Multi-View Dataset Audit
Audit a dataset (e.g. AG News or custom CSV), compute views $V_1, V_2, V_3$, fuse ranks with RRF, assign route buckets, and export a Parquet scores table:
```bash
datadoctor audit ag_news --subsample 1000 --output scores.parquet
```

### 3. Run the Three-Pass Hygiene Scanner
Scan for exact duplicates, mutual $k$-NN near-duplicates, cross-split train/test leakage, and label conflicts:
```bash
datadoctor hygiene ag_news --subsample 1000
```

---

## 7. Python API Example

```python
from datadoctor.io.loaders import load_ag_news
from datadoctor.extractors.text import extract_text_embeddings
from datadoctor.detectors.oof_probe import compute_oof_probabilities
from datadoctor.detectors.v1_self_conf import compute_v1_scores
from datadoctor.detectors.v2_knn import compute_v2_scores
from datadoctor.detectors.v3_cross_probe import compute_v3_scores
from datadoctor.fusion.rank_fusion import compute_rank_fusion
from datadoctor.router.rules import assign_route_buckets
from datadoctor.hygiene.scanner import scan_dataset_hygiene

# 1. Load dataset artifact (20k stratified train, 7.6k test)
artifact = load_ag_news(subsample_train=5000, seed=42)

# 2. Extract features from independent backbones
artifact.embeddings["A"] = extract_text_embeddings(artifact.raw_references, "minilm_l6")
artifact.embeddings["B"] = extract_text_embeddings(artifact.raw_references, "bge_small")

# 3. Fit fast OOF probes on training split
t_mask = artifact.train_mask
num_classes = len(artifact.class_names)
artifact.oof_probs["A"] = compute_oof_probabilities(
    artifact.embeddings["A"][t_mask], artifact.given_labels[t_mask], num_classes
)
artifact.oof_probs["B"] = compute_oof_probabilities(
    artifact.embeddings["B"][t_mask], artifact.given_labels[t_mask], num_classes
)

# 4. Compute complementary views
artifact.view_scores["v1"] = compute_v1_scores(artifact)
artifact.view_scores["v2"] = compute_v2_scores(artifact, backbone_key="A", k=15)
artifact.view_scores["v3"] = compute_v3_scores(artifact)

# 5. Fuse views via Reciprocal Rank Fusion
compute_rank_fusion(artifact)

# 6. Apply noise-adaptive routing
assign_route_buckets(artifact, budget_cap_fraction=0.02)

# 7. Run dataset hygiene scanner
hygiene_report = scan_dataset_hygiene(artifact)
print(hygiene_report.summary)

# 8. Export ranked scores
df = artifact.to_scores_dataframe()
print(df[["sample_id", "given_label", "fused_rank", "u_rrf", "route_bucket"]].head())
```

---

## 8. Repository Structure

```text
DataDoctor/
├── pyproject.toml                 # Package dependencies and test configuration
├── README.md                      # Academic documentation and usage guide
├── configs/
│   ├── grid.yaml                  # 90 synthetic + 15 CIFAR-10N benchmark parameters
│   ├── backbones.yaml             # Backbone A, B, C identifiers and batch sizes
│   └── llm.yaml                   # LLM rate limits and token pricing
├── datadoctor/
│   ├── schema.py                  # DatasetArtifact, KNNIndexData, HygieneReport
│   ├── cli.py                     # Typer CLI entrypoints (info, audit, hygiene)
│   ├── io/
│   │   ├── loaders.py             # AG News stratified loader, CSV/DataFrame loader
│   │   └── cifar10n.py            # CIFAR-10 & CIFAR-10N human label verification
│   ├── extractors/
│   │   ├── cache.py               # Float16 disk cache & Float32 L2 re-normalization
│   │   ├── vision.py              # DINOv2, CLIP, and ResNet feature extractors
│   │   └── text.py                # MiniLM, BGE, and TF-IDF+SVD extractors
│   ├── noise/
│   │   └── injectors.py           # Exact-count Symmetric, Asymmetric, NaN-safe IDN
│   ├── detectors/
│   │   ├── oof_probe.py           # Fast PyTorch Batched L-BFGS OOF Regression
│   │   ├── v1_self_conf.py        # View 1: Backbone A Self-Confidence
│   │   ├── v2_knn.py              # View 2: SHA-1 Filtered k-NN Disagreement
│   │   └── v3_cross_probe.py      # View 3: Backbone B Self-Confidence
│   ├── fusion/
│   │   └── rank_fusion.py         # RRF, Mean Percentile, and Ablation rankings
│   ├── baselines/
│   │   ├── cleanlab_wrap.py       # Confident Joint & Normalized Margin baseline
│   │   ├── deep_knn.py            # Bahri et al. unweighted k-NN baseline
│   │   └── simifeat.py            # Zhu et al. SimiFeat-R consensus baseline
│   ├── hygiene/
│   │   └── scanner.py             # 3-Pass Hygiene Scanner (Exact, Near-Dup, Leakage)
│   ├── router/
│   │   ├── rules.py               # Noise-adaptive disjoint bucket assignment
│   │   ├── judge.py               # Escaped JSON Blind/Anchored LLM Judge
│   │   └── cache_db.py            # SQLite sample-level cross-seed response cache
│   └── evaluation/
│       ├── metrics.py             # AUROC, AUPRC, P@K, Spearman rank correlations
│       ├── budget_curve.py        # Review-budget recovery curve R(b)
│       └── bootstrap.py           # 95% Bootstrap CIs & Wilcoxon signed-rank tests
└── tests/                         # Full Pytest test suite (56 tests passing)
    ├── test_schema_and_cache.py
    ├── test_noise.py
    ├── test_detectors_and_fusion.py
    ├── test_hygiene.py
    ├── test_router_and_judge.py
    ├── test_evaluation.py
    ├── test_phase1_loaders_and_cache.py
    └── test_phase2_detectors_and_noise.py
```

---

## 9. Verification & Test Suite

The test suite validates all architectural invariants, noise injection dynamics, OOF convergence, and runtime bounds:

```bash
python -m pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.5
collected 56 items

tests/test_detectors_and_fusion.py::test_cleanlab_vs_loss_self_conf_distinct PASSED
tests/test_detectors_and_fusion.py::test_v2_knn_excludes_exact_duplicates_via_sha1 PASSED
tests/test_detectors_and_fusion.py::test_fast_oof_probe_convergence_and_shapes PASSED
tests/test_evaluation.py::test_rrf_fusion_and_ablations PASSED
tests/test_evaluation.py::test_budget_curve_train_isolation_and_modes PASSED
tests/test_evaluation.py::test_metrics_and_bootstrap PASSED
tests/test_hygiene.py::test_hygiene_scanner_passes PASSED
tests/test_noise.py::test_inject_idn_nan_safety PASSED
tests/test_noise.py::test_inject_asymmetric_cifar10_and_ag_news PASSED
tests/test_noise.py::test_inject_symmetric_exact_counts PASSED
tests/test_phase1_loaders_and_cache.py::test_float16_cast_and_unit_norm_invariant PASSED
tests/test_phase1_loaders_and_cache.py::test_verify_cifar10n_rates_pass PASSED
tests/test_phase1_loaders_and_cache.py::test_ag_news_loader_shape_and_stratification PASSED
tests/test_phase2_detectors_and_noise.py (32 assertions on IDN, flip counts, 50k runtime) PASSED
tests/test_router_and_judge.py::test_route_bucket_disjointness_and_budget_cap PASSED
tests/test_router_and_judge.py::test_prompt_template_and_sqlite_cache_hit PASSED
tests/test_router_and_judge.py::test_modality_guard_raises_on_vision PASSED
...
============================= 56 passed in 29.52s =============================
```

---

## 10. Citation & Academic References

```bibtex
@misc{datadoctor2026,
  title={DataDoctor: Multi-View Label-Error Detection, Dataset Hygiene and Disagreement-Routed Repair},
  author={Khemka, Prateek and Nikhil},
  year={2026},
  note={AI/ML Course Project Repository}
}
```

Key Literature References:
* **Confident Learning:** Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. Journal of Artificial Intelligence Research (JAIR), 70, 1373-1411.
* **CIFAR-10N Benchmark:** Wei, J. et al. (2022). *Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations*. International Conference on Learning Representations (ICLR).
* **LLM Dataset Auditing:** Nahum, O. et al. (2025). *Auditing NLP Benchmark Errors with Large Language Models*. EMNLP.
* **Deep k-NN Disagreement:** Bahri, D., Jiang, H., & Gupta, M. (2020). *Deep k-NN for Noisy Labels*. ICML.
* **SimiFeat Consensus:** Zhu, Z. et al. (2022). *Detecting Label Errors via Representation Disagreement*. ICML.
