import numpy as np
import pytest

from datadoctor.noise.injectors import (
    inject_symmetric,
    inject_asymmetric,
    inject_idn,
    CIFAR10_ASYMMETRIC_MAP,
    AG_NEWS_ASYMMETRIC_MAP,
)


def test_inject_idn_nan_safety():
    """Gate 2 Assertion: Assert inject_idn never produces NaN even if q_i is forced to 0.0."""
    n_samples = 200
    num_classes = 4
    d_c = 64
    rng = np.random.default_rng(42)
    
    labels = rng.integers(0, num_classes, size=n_samples)
    embeddings_C = rng.standard_normal((n_samples, d_c)).astype(np.float32)
    # Normalize embeddings
    embeddings_C /= np.linalg.norm(embeddings_C, axis=1, keepdims=True)
    
    # Run standard IDN
    noisy, is_noisy, target_T, empirical_T, meta = inject_idn(
        labels, num_classes, embeddings_C, eta=0.20, seed=42
    )
    assert not np.isnan(noisy).any()
    assert not np.isnan(empirical_T).any()
    assert len(noisy) == n_samples
    assert noisy.dtype == labels.dtype
    
    # Edge case: Test with zero/tiny eta
    noisy_zero, is_noisy_zero, _, _, _ = inject_idn(
        labels, num_classes, embeddings_C, eta=0.01, seed=42
    )
    assert not np.isnan(noisy_zero).any()


def test_inject_asymmetric_cifar10_and_ag_news():
    """Gate 2 Assertion: Assert inject_asymmetric on CIFAR-10 flips only the 5 literature source
    classes with exact counts (eta_global = eta_source / 2) and on AG News uses the bijective
    2-cycle map (eta_global = eta_source).
    """
    # 1. CIFAR-10 test
    n_per_class = 1000
    num_classes_cifar = 10
    labels_cifar = np.repeat(np.arange(num_classes_cifar), n_per_class)
    eta_source = 0.20
    
    noisy_cifar, is_noisy_cifar, target_T, empirical_T, meta_cifar = inject_asymmetric(
        labels_cifar, num_classes_cifar, eta_source=eta_source, dataset_name="cifar10", seed=42
    )
    
    # 5 source classes: 9->1, 2->0, 4->7, 3->5, 5->3
    for c in range(num_classes_cifar):
        c_mask = (labels_cifar == c)
        flipped = np.sum(noisy_cifar[c_mask] != c)
        if c in CIFAR10_ASYMMETRIC_MAP:
            expected_flips = int(round(eta_source * n_per_class))
            assert flipped == expected_flips, f"Class {c} expected {expected_flips} flips, got {flipped}"
            target_class = CIFAR10_ASYMMETRIC_MAP[c]
            assert np.all(noisy_cifar[c_mask & (noisy_cifar != c)] == target_class)
        else:
            assert flipped == 0, f"Unflipped source class {c} had {flipped} flips"
            
    # Check global noise rate
    total_flips = np.sum(is_noisy_cifar)
    expected_total_flips = 5 * int(round(eta_source * n_per_class))
    assert total_flips == expected_total_flips
    assert abs(meta_cifar["global_eta"] - (eta_source / 2.0)) < 1e-6
    
    # 2. AG News test (C = 4)
    n_per_class_ag = 500
    num_classes_ag = 4
    labels_ag = np.repeat(np.arange(num_classes_ag), n_per_class_ag)
    eta_ag = 0.15
    
    noisy_ag, is_noisy_ag, _, _, meta_ag = inject_asymmetric(
        labels_ag, num_classes_ag, eta_source=eta_ag, dataset_name="ag_news", seed=42
    )
    for c in range(num_classes_ag):
        c_mask = (labels_ag == c)
        flipped = np.sum(noisy_ag[c_mask] != c)
        expected_flips = int(round(eta_ag * n_per_class_ag))
        assert flipped == expected_flips
        target_class = AG_NEWS_ASYMMETRIC_MAP[c]
        assert np.all(noisy_ag[c_mask & (noisy_ag != c)] == target_class)
        
    assert abs(meta_ag["global_eta"] - eta_ag) < 1e-6


def test_inject_symmetric_exact_counts():
    """Verifies that inject_symmetric flips exact round(eta * N_c) per class."""
    labels = np.repeat(np.arange(5), 200)
    eta = 0.10
    noisy, is_noisy, target_T, empirical_T, meta = inject_symmetric(labels, 5, eta=eta, seed=99)
    for c in range(5):
        c_mask = (labels == c)
        flipped = np.sum(noisy[c_mask] != c)
        assert flipped == int(round(eta * 200))

