import hashlib
import json
import os
import re
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from datadoctor.schema import DatasetArtifact
from datadoctor.router.cache_db import LLMCacheDB

BLIND_PROMPT_TEMPLATE = (
    "You are an expert dataset auditor. Classify the TARGET TEXT into exactly one "
    "of these valid categories: {class_names}.\n\n"
    "Verified reference examples:\n"
    "{reference_block}\n\n"
    "TARGET TEXT: \"{target_text}\"\n\n"
    "Respond ONLY with valid JSON matching this exact schema:\n"
    "{{\"label\": \"<exact category name>\", \"confidence\": 0.95, \"rationale\": \"<one concise sentence>\"}}"
)


def extract_clean_medoids(
    artifact: DatasetArtifact,
    backbone_key: str = "A",
) -> Tuple[List[int], str, str]:
    """Finds 1 static clean medoid per class to provide context references.
    
    Criteria:
    - u_rrf < 0.25 (top 25% cleanest samples)
    - Given label agrees with 100% of top-5 Backbone-A neighbours
    - Closest to the class mean embedding
    
    Returns:
        (medoid_indices, reference_block_text, medoids_combined_hash)
    """
    t_mask = artifact.train_mask
    train_labels = artifact.given_labels[t_mask]
    n_train = artifact.n_train
    emb_A = artifact.embeddings[backbone_key][t_mask]
    u_rrf = artifact.u_rrf if artifact.u_rrf is not None else np.zeros(n_train)
    raw_texts = [artifact.raw_references[i] for i in range(len(artifact.split)) if artifact.split[i] == "train"]

    knn_data = artifact.knn_cache.get(backbone_key)
    if knn_data is None:
        from datadoctor.detectors.v2_knn import build_knn_index
        hashes_train = artifact.sha1_hashes[t_mask]
        knn_data = build_knn_index(emb_A, hashes_train, k_max=15)
        artifact.knn_cache[backbone_key] = knn_data

    top5 = knn_data.indices[:, :5]
    top5_labels = train_labels[top5]
    agrees_100pct = np.all(top5_labels == train_labels[:, np.newaxis], axis=1)

    medoid_indices = []
    ref_lines = []
    medoid_hashes = []

    for c_idx, c_name in enumerate(artifact.class_names):
        c_cands = np.where((train_labels == c_idx) & (u_rrf < 0.25) & agrees_100pct)[0]
        if len(cands := c_cands) == 0:
            # Fallback to any sample of class c
            cands = np.where(train_labels == c_idx)[0]

        c_emb = emb_A[cands]
        class_mean = np.mean(c_emb, axis=0, keepdims=True)
        class_mean /= max(np.linalg.norm(class_mean), 1e-12)
        sims = np.dot(c_emb, class_mean.T).squeeze(-1)
        best_cand_local = np.argmax(sims)
        best_cand = cands[best_cand_local]

        medoid_indices.append(int(best_cand))
        medoid_hashes.append(artifact.sha1_hashes[t_mask][best_cand])
        
        # Format clean reference line (truncate long texts to 120 chars)
        ref_text = raw_texts[best_cand].replace("\n", " ")[:120]
        ref_lines.append(f"- [{c_name}]: \"{ref_text}...\"")

    reference_block = "\n".join(ref_lines)
    medoids_hash = hashlib.sha1("".join(medoid_hashes).encode("utf-8")).hexdigest()
    return medoid_indices, reference_block, medoids_hash


def _call_mock_llm(target_text: str, class_names: List[str]) -> Tuple[str, float, str]:
    """Offline deterministic mock provider for zero-cost testing and CI."""
    text_lower = target_text.lower()
    for name in class_names:
        if name.lower() in text_lower:
            return name, 0.95, f"Matched keyword '{name}' in text."
    # Deterministic hash fallback
    idx = int(hashlib.sha1(target_text.encode("utf-8")).hexdigest(), 16) % len(class_names)
    return class_names[idx], 0.85, "Classified based on contextual text patterns."


def _parse_llm_json_response(raw_text: str, class_names: List[str]) -> Tuple[str, float, str]:
    """Extracts valid JSON object from LLM response text."""
    # Attempt to locate JSON code block or bare braces
    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            label = str(data.get("label", class_names[0])).strip()
            conf = float(data.get("confidence", 0.70))
            rationale = str(data.get("rationale", "")).strip()
            return label, conf, rationale
        except Exception:
            pass

    return class_names[0], 0.50, "Failed to parse JSON response."


def run_llm_judge(
    artifact: DatasetArtifact,
    provider: str = "mock",
    model_id: Optional[str] = None,
    allow_vision_stretch: bool = False,
    cache_db: Optional[LLMCacheDB] = None,
) -> DatasetArtifact:
    """Runs the LLM Judge on samples routed to 'llm_judge' bucket.
    
    HARD INVARIANT: In MVP, scoped strictly to Text modality (AG News).
    Raises ValueError on vision unless allow_vision_stretch is True.
    """
    if artifact.modality != "text" and not allow_vision_stretch:
        raise ValueError("LLM Judge in MVP is scoped strictly to text datasets (AG News).")

    if artifact.route_bucket is None:
        raise KeyError("artifact.route_bucket must be assigned before running LLM Judge.")

    t_mask = artifact.train_mask
    n_train = artifact.n_train
    raw_texts = [artifact.raw_references[i] for i in range(len(artifact.split)) if artifact.split[i] == "train"]
    train_hashes = artifact.sha1_hashes[t_mask]

    if cache_db is None:
        cache_db = LLMCacheDB()

    # Model ID default
    if model_id is None:
        if provider == "groq":
            model_id = "llama-3.3-70b-versatile"
        elif provider == "gemini":
            model_id = "gemini-2.5-flash"
        else:
            model_id = "deterministic-mock-v1"

    # Extract 1 static reference medoid per class
    _, reference_block, medoids_hash = extract_clean_medoids(artifact)
    class_names_str = ", ".join(f'"{c}"' for c in artifact.class_names)
    name_to_idx = {name.lower(): i for i, name in enumerate(artifact.class_names)}

    suggested_labels = np.full(n_train, -1, dtype=np.int64)
    confidences = np.zeros(n_train, dtype=np.float32)
    rationales = ["" for _ in range(n_train)]

    routed_indices = np.where(artifact.route_bucket == "llm_judge")[0]

    for idx in routed_indices:
        target_text = raw_texts[idx]
        sample_sha1 = train_hashes[idx]

        # Check SQLite Cache
        cached = cache_db.get(sample_sha1, medoids_hash, model_id, prompt_mode="blind")
        if cached is not None:
            lbl_str = cached["suggested_label"]
            conf = cached["confidence"]
            rat = cached["rationale"]
        else:
            # Format prompt
            prompt_text = BLIND_PROMPT_TEMPLATE.format(
                class_names=class_names_str,
                reference_block=reference_block,
                target_text=target_text.replace('"', '\\"'),
            )

            if provider == "mock":
                lbl_str, conf, rat = _call_mock_llm(target_text, artifact.class_names)
                raw_resp = json.dumps({"label": lbl_str, "confidence": conf, "rationale": rat})
                cache_db.put(
                    sample_sha1, medoids_hash, model_id, "blind", prompt_text,
                    raw_resp, lbl_str, conf, rat, input_tokens=80, output_tokens=30
                )
            elif provider == "groq":
                import groq
                client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                )
                raw_resp = response.choices[0].message.content
                lbl_str, conf, rat = _parse_llm_json_response(raw_resp, artifact.class_names)
                cache_db.put(
                    sample_sha1, medoids_hash, model_id, "blind", prompt_text,
                    raw_resp, lbl_str, conf, rat,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
            elif provider == "gemini":
                # Google GenAI / Gemini API
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt_text)
                raw_resp = response.text
                lbl_str, conf, rat = _parse_llm_json_response(raw_resp, artifact.class_names)
                cache_db.put(
                    sample_sha1, medoids_hash, model_id, "blind", prompt_text,
                    raw_resp, lbl_str, conf, rat,
                    input_tokens=100, output_tokens=35,
                )
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")

        # Map label string to class index
        mapped_idx = name_to_idx.get(lbl_str.lower(), -1)
        suggested_labels[idx] = mapped_idx
        confidences[idx] = conf
        rationales[idx] = rat

    artifact.llm_suggested_label = suggested_labels
    artifact.llm_confidence = confidences
    artifact.llm_rationale = rationales
    return artifact

