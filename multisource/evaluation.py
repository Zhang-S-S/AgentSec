from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def select_threshold(labels: list[float] | np.ndarray, probabilities: list[float] | np.ndarray,
                     strategy: str = "max_f1", target_recall: float = 0.80) -> tuple[float, dict]:
    truth = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        warnings.warn("Threshold selection split has a single class; using deterministic threshold 0.5")
        return 0.5, {"strategy": strategy, "warning": "single_class_fallback", "target_recall": target_recall}
    candidates = np.unique(np.concatenate(([0.0], probs, [1.0])))
    records = []
    for threshold in candidates:
        predicted = probs >= threshold
        records.append((float(threshold),
                        float(precision_score(truth, predicted, zero_division=0)),
                        float(recall_score(truth, predicted, zero_division=0)),
                        float(f1_score(truth, predicted, zero_division=0))))
    if strategy == "max_f1":
        chosen = max(records, key=lambda row: (row[3], row[1], row[2], row[0]))
        warning = None
    elif strategy == "max_precision_at_recall":
        valid = [row for row in records if row[2] >= target_recall]
        if valid:
            chosen = max(valid, key=lambda row: (row[1], row[3], row[2], row[0]))
            warning = None
        else:
            chosen = max(records, key=lambda row: (row[2], row[1], row[3], -abs(row[0] - 0.5)))
            warning = "target_recall_unavailable_fallback_to_max_recall"
            warnings.warn("No validation threshold meets target recall; falling back to maximum recall")
    else:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")
    return chosen[0], {"strategy": strategy, "target_recall": target_recall,
                       "validation_precision": chosen[1], "validation_recall": chosen[2],
                       "validation_f1": chosen[3], "warning": warning}


def classification_metrics(labels: list[float] | np.ndarray, probabilities: list[float] | np.ndarray,
                           threshold: float) -> dict:
    truth = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    predicted = (probs >= threshold).astype(np.int64)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    result = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "specificity": float(specificity), "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
    }
    if len(np.unique(truth)) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, probs))
        result["pr_auc"] = float(average_precision_score(truth, probs))
    else:
        warnings.warn("ROC-AUC and PR-AUC are undefined for a single-class split")
        result["roc_auc"] = result["pr_auc"] = float("nan")
    return result


def bootstrap_confidence_intervals(labels: list[float], probabilities: list[float], threshold: float,
                                   seed: int = 42, samples: int = 2000) -> dict:
    truth, probs = np.asarray(labels), np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {key: [] for key in ("accuracy", "precision", "recall", "f1", "mcc")}
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        metrics = classification_metrics(truth[indices], probs[indices], threshold)
        for key in collected:
            collected[key].append(metrics[key])
    return {key: {"mean": float(np.nanmean(values)),
                  "ci95": [float(x) for x in np.nanpercentile(values, [2.5, 97.5])]}
            for key, values in collected.items()}
