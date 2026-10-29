from typing import Dict, Tuple, Any, Optional
import numpy as np
from scipy.stats import truncnorm

from datadoctor.schema import DatasetArtifact

# Literature Transition Maps
CIFAR10_ASYMMETRIC_MAP = {
    9: 1,  # truck -> automobile
    2: 0,  # bird -> airplane
    4: 7,  # deer -> horse
    3: 5,  # cat -> dog
    5: 3,  # dog -> cat
}

AG_NEWS_ASYMMETRIC_MAP = {
    0: 2,  # World -> Business
    2: 0,  # Business -> World
    1: 3,  # Sports -> Sci/Tech
    3: 1,  # Sci/Tech -> Sports
}


def _compute_transition_matrix(true_labels: np.ndarray, noisy_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes empirical transition matrix T where T[j, k] = P(noisy = k | true = j)."""
    T = np.zeros((num_classes, num_classes), dtype=np.float64)
    for j in range(num_classes):
        mask = (true_labels == j)
        n_j = np.sum(mask)
        if n_j > 0:
            for k in range(num_classes):
                T[j, k] = np.sum(noisy_labels[mask] == k) / n_j
    return T


def inject_symmetric(
    labels: np.ndarray,
    num_classes: int,
    eta: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Injects exact-count class-conditional symmetric noise.
    
    For each class c, exactly round(eta * N_c) samples are flipped uniformly
    to {0, ..., C-1} \\ {c}.
    """
    rng = np.random.default_rng(seed)
    noisy_labels = labels.copy()
    n_samples = len(labels)

    for c in range(num_classes):
        c_indices = np.where(labels == c)[0]
        n_c = len(c_indices)
        m_c = int(round(eta * n_c))
        if m_c > 0 and n_c > 0:
            chosen = rng.choice(c_indices, size=m_c, replace=False)
            possible_targets = [k for k in range(num_classes) if k != c]
            new_labels = rng.choice(possible_targets, size=m_c, replace=True)
            noisy_labels[chosen] = new_labels

    is_noisy = (noisy_labels != labels)
    empirical_T = _compute_transition_matrix(labels, noisy_labels, num_classes)
    
    # Target theoretical T
    target_T = np.zeros((num_classes, num_classes), dtype=np.float64)
    for c in range(num_classes):
        target_T[c, c] = 1.0 - eta
        for k in range(num_classes):
            if k != c:
                target_T[c, k] = eta / (num_classes - 1)

    meta = {
        "noise_type": "symmetric",
        "eta": eta,
        "seed": seed,
        "flipped_count": int(np.sum(is_noisy)),
        "empirical_noise_rate": float(np.mean(is_noisy)),
    }
    return noisy_labels, is_noisy, target_T, empirical_T, meta


def inject_asymmetric(
    labels: np.ndarray,
    num_classes: int,
    eta_source: float,
    dataset_name: str = "cifar10",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Injects literature-standard class-conditional asymmetric noise.
    
    CIFAR-10 literature map (Patrini et al. 2017 / Wei et al. 2022):
    5 of 10 classes flip to their semantic counterpart with source rate eta_source.
    Global dataset noise rate is exactly eta_source / 2.
    
    AG News bijective 2-cycle map:
    All 4 classes flip (0 <-> 2, 1 <-> 3), so global noise rate equals eta_source.
    """
    rng = np.random.default_rng(seed)
    noisy_labels = labels.copy()

    if "cifar10" in dataset_name.lower():
        trans_map = CIFAR10_ASYMMETRIC_MAP
        global_eta = eta_source / 2.0
    elif "ag_news" in dataset_name.lower() or num_classes == 4:
        trans_map = AG_NEWS_ASYMMETRIC_MAP
        global_eta = eta_source
    else:
        # Fallback 1-step cycle
        trans_map = {c: (c + 1) % num_classes for c in range(num_classes)}
        global_eta = eta_source

    for src_class, dst_class in trans_map.items():
        if src_class >= num_classes or dst_class >= num_classes:
            continue
        c_indices = np.where(labels == src_class)[0]
        n_c = len(c_indices)
        m_c = int(round(eta_source * n_c))
        if m_c > 0 and n_c > 0:
            chosen = rng.choice(c_indices, size=m_c, replace=False)
            noisy_labels[chosen] = dst_class

    is_noisy = (noisy_labels != labels)
    empirical_T = _compute_transition_matrix(labels, noisy_labels, num_classes)
    
    target_T = np.eye(num_classes, dtype=np.float64)
    for src_class, dst_class in trans_map.items():
        if src_class < num_classes and dst_class < num_classes:
            target_T[src_class, src_class] = 1.0 - eta_source
            target_T[src_class, dst_class] = eta_source

    meta = {
        "noise_type": "asymmetric",
        "source_eta": eta_source,
        "global_eta": global_eta,
        "seed": seed,
        "flipped_count": int(np.sum(is_noisy)),
        "empirical_noise_rate": float(np.mean(is_noisy)),
    }
    return noisy_labels, is_noisy, target_T, empirical_T, meta


def inject_idn(
    labels: np.ndarray,
    num_classes: int,
    embeddings_C: np.ndarray,
    eta: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Injects NaN-Safe Instance-Dependent Noise (IDN) via Backbone C features.
    
    Procedure:
    1. Draw instance flip propensity q_i ~ N_trunc(mu=eta, sigma=0.10) clipped to [1e-4, 0.95].
    2. Draw projection matrix W ~ N(0, 1) shape (d_C, C) and compute raw logits L = Z^(C) W.
    3. CRITICAL NaN PREVENTION: First multiply finite logits by q_i: L_tilde_i = q_i * L_i.
       Only after multiplication, mask the true class L_tilde_{i, y_i} = -inf.
    4. Within each true class c, sort descending by max_{k != c} L_tilde_{i, k} and flip
       the top m_c = round(eta * N_c) samples to argmax_{k != c} L_tilde_{i, k}.
    """
    rng = np.random.default_rng(seed)
    n_samples, d_c = embeddings_C.shape
    noisy_labels = labels.copy()

    # 1. Truncated normal draw clipped strictly to [1e-4, 0.95]
    a, b = (1e-4 - eta) / 0.10, (0.95 - eta) / 0.10
    q = truncnorm.rvs(a, b, loc=eta, scale=0.10, size=n_samples, random_state=seed)
    q = np.clip(q, 1e-4, 0.95)

    # 2. Random projection matrix
    W = rng.standard_normal((d_c, num_classes), dtype=np.float32)
    raw_logits = np.dot(embeddings_C, W)

    # 3. Multiply first by q_i to prevent 0.0 * -inf = NaN
    scaled_logits = raw_logits * q[:, np.newaxis]

    # Mask true class strictly after scaling
    for i in range(n_samples):
        scaled_logits[i, labels[i]] = -np.inf

    # 4. Flip top m_c samples per class
    for c in range(num_classes):
        c_indices = np.where(labels == c)[0]
        n_c = len(c_indices)
        m_c = int(round(eta * n_c))
        if m_c > 0 and n_c > 0:
            c_sub_logits = scaled_logits[c_indices]
            max_other_logits = np.max(c_sub_logits, axis=1)
            # Sort descending by max other logit
            sort_order = np.argsort(-max_other_logits)
            chosen = c_indices[sort_order[:m_c]]
            # Assign to argmax other class
            noisy_labels[chosen] = np.argmax(scaled_logits[chosen], axis=1)

    is_noisy = (noisy_labels != labels)
    empirical_T = _compute_transition_matrix(labels, noisy_labels, num_classes)
    target_T = empirical_T.copy() # IDN has sample-specific probabilities

    meta = {
        "noise_type": "idn",
        "eta": eta,
        "seed": seed,
        "flipped_count": int(np.sum(is_noisy)),
        "empirical_noise_rate": float(np.mean(is_noisy)),
    }
    return noisy_labels, is_noisy, target_T, empirical_T, meta


def inject_noise_into_artifact(
    artifact: DatasetArtifact,
    noise_type: str,
    eta: float,
    seed: int = 42,
) -> DatasetArtifact:
    """Injects noise strictly into the training split of the DatasetArtifact."""
    t_mask = artifact.train_mask
    train_labels = artifact.true_labels[t_mask] if artifact.true_labels is not None else artifact.given_labels[t_mask]
    num_classes = len(artifact.class_names)

    if noise_type == "symmetric":
        noisy_train, is_noisy, target_T, empirical_T, meta = inject_symmetric(
            train_labels, num_classes, eta, seed=seed
        )
    elif noise_type == "asymmetric":
        noisy_train, is_noisy, target_T, empirical_T, meta = inject_asymmetric(
            train_labels, num_classes, eta, dataset_name=artifact.dataset_name, seed=seed
        )
    elif noise_type == "idn":
        if "C" not in artifact.embeddings:
            raise KeyError("Backbone C embeddings required for Instance-Dependent Noise (IDN).")
        emb_C_train = artifact.embeddings["C"][t_mask]
        noisy_train, is_noisy, target_T, empirical_T, meta = inject_idn(
            train_labels, num_classes, emb_C_train, eta, seed=seed
        )
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Mutate artifact training given_labels strictly
    new_given = artifact.given_labels.copy()
    new_given[t_mask] = noisy_train
    artifact.given_labels = new_given
    artifact.is_noisy = is_noisy
    artifact.meta["noise_injection"] = meta
    artifact.meta["transition_matrix"] = empirical_T
    return artifact

