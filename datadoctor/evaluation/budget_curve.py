from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from datadoctor.schema import DatasetArtifact


def _fit_and_eval_linear_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    c_reg: float = 1.0,
    device: Optional[str] = None,
) -> float:
    """Trains a fast PyTorch L-BFGS multinomial logistic regression on repaired train set
    and returns test accuracy on clean test set.
    """
    if len(x_train) == 0:
        return 0.0

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d = x_train.shape[1]
    n_tr = len(x_train)
    weight_decay = 1.0 / max(c_reg * n_tr, 1e-6)

    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    xv = torch.from_numpy(x_test).to(device)
    yv = torch.from_numpy(y_test).long().to(device)

    weight = torch.zeros((num_classes, d), dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros(num_classes, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=30,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = torch.matmul(xt, weight.t()) + bias
        loss = F.cross_entropy(logits, yt) + 0.5 * weight_decay * torch.sum(weight ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        test_logits = torch.matmul(xv, weight.t()) + bias
        preds = torch.argmax(test_logits, dim=-1)
        acc = float((preds == yv).float().mean().cpu().item())

    return acc


def evaluate_review_budget_curve(
    artifact: DatasetArtifact,
    budgets: Optional[List[float]] = None,
    modes: Optional[List[str]] = None,
    backbone_key: str = "A",
) -> Dict[str, Dict[float, float]]:
    """Evaluates clean test-set accuracy recovery across human review budgets.
    
    HARD INVARIANT:
    - Label repair and sample removal operate strictly on split == 'train'.
    - Evaluation is performed exclusively on clean split == 'test'.
    
    Modes:
    1. oracle_relabel: top K samples under fused ranking get true_labels.
    2. remove: top K samples under fused ranking are dropped from training.
    3. ensemble_argmax_relabel: top K samples get argmax_{c != y_tilde} (p_hat^A + p_hat^B).
    4. llm_relabel: top K samples use llm_suggested_label (if conf >= 0.60) else ensemble argmax.
    """
    if budgets is None:
        budgets = [0.01, 0.02, 0.05, 0.10, 0.20]
    if modes is None:
        modes = ["oracle_relabel", "remove", "ensemble_argmax_relabel"]

    t_mask = artifact.train_mask
    test_mask = artifact.test_mask
    n_train = artifact.n_train
    num_classes = len(artifact.class_names)

    # Feature representations
    x_train_orig = artifact.embeddings[backbone_key][t_mask]
    y_train_orig = artifact.given_labels[t_mask].copy()
    y_train_clean = artifact.true_labels[t_mask] if artifact.true_labels is not None else y_train_orig

    x_test = artifact.embeddings[backbone_key][test_mask]
    y_test = artifact.given_labels[test_mask] # Test given == clean

    # Determine baseline ranking order pi (descending fused_score or fallback)
    if artifact.fused_score is not None:
        pi = np.argsort(-artifact.fused_score)
    elif "v1" in artifact.view_scores:
        pi = np.argsort(-artifact.view_scores["v1"])
    else:
        pi = np.arange(n_train)

    # Compute unconfounded ensemble argmax labels: argmax_{c != y_tilde} (p_A + p_B)
    oof_A = artifact.oof_probs.get("A", np.zeros((n_train, num_classes)))
    oof_B = artifact.oof_probs.get("B", oof_A)
    ensemble_probs = oof_A + oof_B

    ensemble_argmax_labels = np.zeros(n_train, dtype=np.int64)
    for i in range(n_train):
        y_curr = y_train_orig[i]
        cands = [c for c in range(num_classes) if c != y_curr]
        if len(cands) > 0:
            ensemble_argmax_labels[i] = cands[np.argmax(ensemble_probs[i, cands])]
        else:
            ensemble_argmax_labels[i] = y_curr

    # Determine pi_llm ordering if llm_relabel is requested
    pi_llm = pi
    if "llm_relabel" in modes and artifact.route_bucket is not None:
        # Prioritize auto_flag, then llm_judge with conf >= 0.60, then remaining fused rank
        auto_flag_mask = (artifact.route_bucket == "auto_flag")
        llm_called_mask = (
            (artifact.route_bucket == "llm_judge") &
            (artifact.llm_suggested_label is not None) &
            (artifact.llm_suggested_label >= 0) &
            (artifact.llm_confidence is not None) &
            (artifact.llm_confidence >= 0.60)
        )
        remaining_mask = ~auto_flag_mask & ~llm_called_mask

        auto_flags = pi[auto_flag_mask[pi]]
        llm_calls = pi[llm_called_mask[pi]]
        remaining = pi[remaining_mask[pi]]
        pi_llm = np.concatenate([auto_flags, llm_calls, remaining])

    # Initial zero-repair baseline accuracy
    base_acc = _fit_and_eval_linear_classifier(
        x_train_orig, y_train_orig, x_test, y_test, num_classes
    )

    results: Dict[str, Dict[float, float]] = {m: {0.0: base_acc} for m in modes}

    for b in budgets:
        k = max(1, int(round(b * n_train)))

        for mode in modes:
            order = pi_llm if mode == "llm_relabel" else pi
            target_indices = order[:k]

            if mode == "oracle_relabel":
                y_rep = y_train_orig.copy()
                y_rep[target_indices] = y_train_clean[target_indices]
                acc = _fit_and_eval_linear_classifier(
                    x_train_orig, y_rep, x_test, y_test, num_classes
                )
            elif mode == "remove":
                keep_mask = np.ones(n_train, dtype=bool)
                keep_mask[target_indices] = False
                acc = _fit_and_eval_linear_classifier(
                    x_train_orig[keep_mask], y_train_orig[keep_mask], x_test, y_test, num_classes
                )
            elif mode == "ensemble_argmax_relabel":
                y_rep = y_train_orig.copy()
                y_rep[target_indices] = ensemble_argmax_labels[target_indices]
                acc = _fit_and_eval_linear_classifier(
                    x_train_orig, y_rep, x_test, y_test, num_classes
                )
            elif mode == "llm_relabel":
                y_rep = y_train_orig.copy()
                for idx in target_indices:
                    if (
                        artifact.llm_suggested_label is not None and
                        artifact.llm_suggested_label[idx] >= 0 and
                        artifact.llm_confidence is not None and
                        artifact.llm_confidence[idx] >= 0.60
                    ):
                        y_rep[idx] = artifact.llm_suggested_label[idx]
                    else:
                        y_rep[idx] = ensemble_argmax_labels[idx]
                acc = _fit_and_eval_linear_classifier(
                    x_train_orig, y_rep, x_test, y_test, num_classes
                )
            else:
                raise ValueError(f"Unknown repair mode: {mode}")

            results[mode][float(b)] = acc

    return results

