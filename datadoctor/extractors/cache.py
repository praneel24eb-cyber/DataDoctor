import hashlib
from pathlib import Path
from typing import Union
import numpy as np


def compute_sha1_hash(data: Union[str, bytes, np.ndarray]) -> str:
    """Computes SHA-1 hash for string, bytes, or numpy uint8 image buffer.
    
    Conventions:
    - Text: hashlib.sha1(" ".join(text.lower().strip().split()).encode("utf-8")).hexdigest()
    - Image uint8: hashlib.sha1(image_uint8.tobytes()).hexdigest()
    - Raw bytes: hashlib.sha1(data).hexdigest()
    """
    if isinstance(data, str):
        normalized_text = " ".join(data.lower().strip().split()).encode("utf-8")
        return hashlib.sha1(normalized_text).hexdigest()
    elif isinstance(data, np.ndarray):
        return hashlib.sha1(data.tobytes()).hexdigest()
    elif isinstance(data, bytes):
        return hashlib.sha1(data).hexdigest()
    else:
        raise TypeError(f"Unsupported data type for hashing: {type(data)}")


def save_cached_embeddings(path: Union[str, Path], emb: np.ndarray) -> Path:
    """Saves embeddings to disk in float16 precision to minimize footprint.
    
    Embeddings are L2 normalized before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # L2 normalize first
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normed_emb = emb / np.maximum(norms, 1e-12)
    
    # Save as float16
    np.save(path, normed_emb.astype(np.float16))
    return path


def load_cached_embeddings(path: Union[str, Path]) -> np.ndarray:
    """Loads cached embeddings, casts to float32 and restores strict unit L2 norm.
    
    MANDATORY INVARIANT: float16 quantization drift can degrade unit-norm dot products.
    Immediately upon loading, cast to float32 and re-normalize:
    Z <- Z / max(||Z||_2, 1e-12)
    """
    path = Path(path)
    emb = np.load(path).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)

