from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoImageProcessor, AutoModel

from datadoctor.extractors.cache import load_cached_embeddings, save_cached_embeddings

VISION_BACKBONES = {
    "dinov2_vits14": {
        "model_name": "facebook/dinov2-small",
        "dim": 384,
    },
    "clip_vitb32": {
        "model_name": "openai/clip-vit-base-patch32",
        "dim": 512,
    },
    "resnet50_tv": {
        "dim": 2048,
    },
}


def extract_vision_embeddings(
    images_uint8: np.ndarray,
    backbone_id: str,
    cache_path: Optional[Union[str, Path]] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extracts L2-normalized dense vision embeddings with disk cache support.
    
    Args:
        images_uint8: Array of shape (N, H, W, C) in uint8 [0, 255].
        backbone_id: 'dinov2_vits14', 'clip_vitb32', or 'resnet50_tv'.
        cache_path: Optional path to save/load .npy float16 cache.
    """
    if cache_path is not None and Path(cache_path).exists():
        return load_cached_embeddings(cache_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backbone_id == "resnet50_tv":
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        model = resnet50(weights=weights).to(device)
        # Remove classification head to get 2048-d pool5 features
        model.fc = torch.nn.Identity()
        model.eval()

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images_uint8), batch_size):
                batch = images_uint8[i : i + batch_size]
                # Convert (B, H, W, C) to (B, C, H, W) float in [0, 1]
                t_batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
                t_batch = preprocess(t_batch).to(device)
                feats = model(t_batch)
                feats = F.normalize(feats, p=2, dim=1)
                all_embs.append(feats.cpu().numpy().astype(np.float32))

        emb = np.concatenate(all_embs, axis=0)
        if cache_path is not None:
            save_cached_embeddings(cache_path, emb)
        return emb

    if backbone_id not in VISION_BACKBONES:
        raise ValueError(f"Unknown vision backbone_id: {backbone_id}. Choose from {list(VISION_BACKBONES.keys())}")

    model_name = VISION_BACKBONES[backbone_id]["model_name"]
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(images_uint8), batch_size):
            batch = list(images_uint8[i : i + batch_size])
            inputs = processor(images=batch, return_tensors="pt").to(device)
            if hasattr(model, "get_image_features"):
                # CLIP vision model
                feats = model.get_image_features(**inputs)
            else:
                out = model(**inputs)
                # DINOv2 uses pooler_output or CLS token [0]
                feats = out.last_hidden_state[:, 0]
            feats = F.normalize(feats, p=2, dim=1)
            all_embs.append(feats.cpu().numpy().astype(np.float32))

    emb = np.concatenate(all_embs, axis=0)
    if cache_path is not None:
        save_cached_embeddings(cache_path, emb)
    return emb

