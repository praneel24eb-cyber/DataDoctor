from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

from datadoctor.schema import DatasetArtifact
from datadoctor.extractors.cache import compute_sha1_hash

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


def load_ag_news(
    subsample_train: Optional[int] = 20000,
    seed: int = 42,
    cache_dir: Optional[Union[str, Path]] = None,
) -> DatasetArtifact:
    """Loads the AG News benchmark dataset and constructs a DatasetArtifact.
    
    Standard benchmark setting:
    - 20,000 stratified training samples (5,000 per class from the 120,000 total)
    - 7,600 test samples (untouched)
    - C = 4 classes: World, Sports, Business, Sci/Tech
    """
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", cache_dir=str(cache_dir) if cache_dir else None)
    train_split = ds["train"]
    test_split = ds["test"]

    raw_train_texts = list(train_split["text"])
    raw_train_labels = np.array(train_split["label"], dtype=np.int64)

    raw_test_texts = list(test_split["text"])
    raw_test_labels = np.array(test_split["label"], dtype=np.int64)

    # Stratified subsampling on train split
    # Deduplicate within each class pool by SHA-1 to guarantee unique hashes
    if subsample_train is not None and subsample_train < len(raw_train_labels):
        rng = np.random.default_rng(seed)
        per_class = subsample_train // len(AG_NEWS_CLASSES)
        selected_train_indices = []
        for c in range(len(AG_NEWS_CLASSES)):
            c_indices = np.where(raw_train_labels == c)[0]
            # Remove duplicates: keep only first occurrence of each SHA-1 within this class
            seen_hashes: set = set()
            unique_c_indices = []
            for idx in c_indices:
                h = compute_sha1_hash(raw_train_texts[idx])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique_c_indices.append(idx)
            unique_c_indices = np.array(unique_c_indices, dtype=np.int64)
            chosen = rng.choice(unique_c_indices, size=per_class, replace=False)
            selected_train_indices.extend(chosen)
        selected_train_indices = np.sort(np.array(selected_train_indices, dtype=np.int64))

        train_texts = [raw_train_texts[i] for i in selected_train_indices]
        train_labels = raw_train_labels[selected_train_indices]
    else:
        train_texts = raw_train_texts
        train_labels = raw_train_labels

    n_train = len(train_labels)
    n_test = len(raw_test_labels)
    n_total = n_train + n_test

    all_texts = train_texts + raw_test_texts
    true_labels = np.concatenate([train_labels, raw_test_labels], axis=0)
    given_labels = true_labels.copy()

    split = np.array(["train"] * n_train + ["test"] * n_test)
    sample_ids = np.arange(n_total, dtype=np.int64)
    sha1_hashes = np.array([compute_sha1_hash(t) for t in all_texts], dtype="<U40")
    is_noisy = np.zeros(n_train, dtype=bool)

    return DatasetArtifact(
        dataset_name="ag_news",
        modality="text",
        sample_ids=sample_ids,
        split=split,
        raw_references=all_texts,
        sha1_hashes=sha1_hashes,
        class_names=AG_NEWS_CLASSES,
        given_labels=given_labels,
        true_labels=true_labels,
        is_noisy=is_noisy,
        meta={"subsample_train": subsample_train, "seed": seed},
    )


def load_from_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    split_col: Optional[str] = None,
    dataset_name: str = "custom_text",
    class_names: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetArtifact:
    """Constructs a DatasetArtifact from a pandas DataFrame for text classification."""
    texts = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].values

    if class_names is None:
        class_names = sorted(list(set(labels_raw)))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    
    if np.issubdtype(labels_raw.dtype, np.integer):
        numeric_labels = labels_raw.astype(np.int64)
    else:
        numeric_labels = np.array([label_to_idx[y] for y in labels_raw], dtype=np.int64)

    n_total = len(texts)
    sample_ids = np.arange(n_total, dtype=np.int64)
    sha1_hashes = np.array([compute_sha1_hash(t) for t in texts], dtype="<U40")

    if split_col is not None and split_col in df.columns:
        split = df[split_col].values.astype(str)
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(n_total)
        n_train = int(n_total * train_ratio)
        split = np.array(["test"] * n_total)
        split[shuffled[:n_train]] = "train"

    n_train_actual = int(np.sum(split == "train"))
    is_noisy = np.zeros(n_train_actual, dtype=bool)

    return DatasetArtifact(
        dataset_name=dataset_name,
        modality="text",
        sample_ids=sample_ids,
        split=split,
        raw_references=texts,
        sha1_hashes=sha1_hashes,
        class_names=[str(c) for c in class_names],
        given_labels=numeric_labels.copy(),
        true_labels=numeric_labels.copy(),
        is_noisy=is_noisy,
    )


def load_from_csv(
    filepath: Union[str, Path],
    text_col: str,
    label_col: str,
    split_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
    **kwargs,
) -> DatasetArtifact:
    """Reads a CSV file into a DatasetArtifact."""
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    name = dataset_name or filepath.stem
    return load_from_dataframe(df, text_col=text_col, label_col=label_col, split_col=split_col, dataset_name=name, **kwargs)

