from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from datadoctor.extractors.cache import load_cached_embeddings, save_cached_embeddings

TEXT_BACKBONES = {
    "minilm_l6": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    "bge_small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    },
    "tfidf_svd384": {
        "dim": 384,
        "max_features": 20000,
    },
}


def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def extract_text_embeddings(
    texts: List[str],
    backbone_id: str,
    cache_path: Optional[Union[str, Path]] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extracts L2-normalized dense text embeddings with disk cache support.
    
    Backbones:
    - minilm_l6: sentence-transformers/all-MiniLM-L6-v2 (Backbone A)
    - bge_small: BAAI/bge-small-en-v1.5 (Backbone B)
    - tfidf_svd384: TF-IDF + TruncatedSVD (Backbone C, for IDN noise)
    """
    if cache_path is not None and Path(cache_path).exists():
        return load_cached_embeddings(cache_path)

    if backbone_id == "tfidf_svd384":
        vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(texts)
        n_components = min(384, tfidf_mat.shape[1] - 1 if tfidf_mat.shape[1] > 1 else 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        emb = svd.fit_transform(tfidf_mat).astype(np.float32)
        # Pad with zeros if n_components < 384 (e.g. small synthetic test datasets)
        if emb.shape[1] < 384:
            pad = np.zeros((emb.shape[0], 384 - emb.shape[1]), dtype=np.float32)
            emb = np.hstack([emb, pad])
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        if cache_path is not None:
            save_cached_embeddings(cache_path, emb)
        return emb

    if backbone_id not in TEXT_BACKBONES:
        raise ValueError(f"Unknown text backbone_id: {backbone_id}. Choose from {list(TEXT_BACKBONES.keys())}")

    model_name = TEXT_BACKBONES[backbone_id]["model_name"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            out = model(**encoded)
            pooled = _mean_pooling(out, encoded["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            all_embs.append(normalized.cpu().numpy().astype(np.float32))

    emb = np.concatenate(all_embs, axis=0)
    if cache_path is not None:
        save_cached_embeddings(cache_path, emb)
    return emb

