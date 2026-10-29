import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import torch
import torchvision
from torchvision.datasets import CIFAR10

from datadoctor.schema import DatasetArtifact
from datadoctor.extractors.cache import compute_sha1_hash

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10N_URL = (
    "https://github.com/UCSC-REAL/cifar-10-100n-dataset/raw/main/data/CIFAR-10_human.pt"
)


def verify_cifar10n_rates(pt_dict: Dict[str, Any]) -> Dict[str, float]:
    """Verifies that empirical noise rates on the 50,000 CIFAR-10 training images
    match the literature-standard rates reported in Wei et al. (ICLR 2022):
    - aggre_label: 9.03% +/- 0.05%
    - random_label1: 17.23% +/- 0.05%
    - worse_label: 40.21% +/- 0.05%
    
    Raises:
        AssertionError: if any rate is outside its specified tolerance.
        
    Returns:
        Dictionary of computed empirical noise rates.
    """
    clean = np.asarray(pt_dict["clean_label"])
    n_samples = len(clean)
    assert n_samples == 50000, f"Expected 50000 training samples, got {n_samples}"

    rates = {}
    target_specs = {
        "aggre_label": (0.0903, 0.0005),
        "random_label1": (0.1723, 0.0005),
        "worse_label": (0.4021, 0.0005),
    }

    for key, (expected, tol) in target_specs.items():
        if key in pt_dict:
            noisy = np.asarray(pt_dict[key])
            rate = float(np.mean(noisy != clean))
            rates[key] = rate
            assert abs(rate - expected) <= tol, (
                f"CIFAR-10N noise rate mismatch for {key}: "
                f"got {rate * 100:.2f}%, expected {expected * 100:.2f}% +/- {tol * 100:.2f}%"
            )

    return rates


def download_cifar10n(save_path: Union[str, Path]) -> Path:
    """Downloads CIFAR-10_human.pt from the official UCSC-REAL repository."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        import requests
        response = requests.get(CIFAR10N_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return save_path


def load_cifar10(
    root: Union[str, Path] = "./data",
    label_type: str = "clean",
    cifar10n_path: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> DatasetArtifact:
    """Loads CIFAR-10 or CIFAR-10N and builds a DatasetArtifact.
    
    Args:
        root: Data directory for CIFAR-10 images.
        label_type: "clean", "aggre_label", "random_label1", "random_label2",
                    "random_label3", or "worse_label".
        cifar10n_path: Path to CIFAR-10_human.pt. If None and label_type != 'clean',
                       attempts to find or download to {root}/CIFAR-10_human.pt.
        download: Whether to download CIFAR10 if not present.
    """
    root = Path(root)
    train_set = CIFAR10(root=str(root), train=True, download=download)
    test_set = CIFAR10(root=str(root), train=False, download=download)

    train_data = train_set.data       # Shape: (50000, 32, 32, 3), uint8
    test_data = test_set.data         # Shape: (10000, 32, 32, 3), uint8
    train_clean_targets = np.array(train_set.targets, dtype=np.int64)
    test_targets = np.array(test_set.targets, dtype=np.int64)

    n_train = len(train_clean_targets)
    n_test = len(test_targets)
    n_total = n_train + n_test

    # Compute SHA-1 hashes for all images
    all_data = np.concatenate([train_data, test_data], axis=0)
    sha1_hashes = np.array([compute_sha1_hash(all_data[i]) for i in range(n_total)], dtype="<U40")

    split = np.array(["train"] * n_train + ["test"] * n_test)
    sample_ids = np.arange(n_total, dtype=np.int64)
    raw_references = [f"cifar10_{s}_{i}" for s, n in [("train", n_train), ("test", n_test)] for i in range(n)]

    true_labels = np.concatenate([train_clean_targets, test_targets], axis=0)

    # Resolve train labels
    if label_type == "clean":
        given_train_labels = train_clean_targets.copy()
    else:
        if cifar10n_path is None:
            cifar10n_path = root / "CIFAR-10_human.pt"
        if not Path(cifar10n_path).exists():
            download_cifar10n(cifar10n_path)
        pt_dict = torch.load(cifar10n_path, map_location="cpu")
        verify_cifar10n_rates(pt_dict)
        given_train_labels = np.asarray(pt_dict[label_type], dtype=np.int64)

    given_labels = np.concatenate([given_train_labels, test_targets], axis=0)
    is_noisy = (given_train_labels != train_clean_targets)

    return DatasetArtifact(
        dataset_name=f"cifar10_{label_type}",
        modality="vision",
        sample_ids=sample_ids,
        split=split,
        raw_references=raw_references,
        sha1_hashes=sha1_hashes,
        class_names=CIFAR10_CLASSES,
        given_labels=given_labels,
        true_labels=true_labels,
        is_noisy=is_noisy,
        meta={"cifar10n_label_type": label_type},
    )

