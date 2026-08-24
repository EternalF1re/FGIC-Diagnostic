from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def paired_accuracy_statistics(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    bootstrap_samples: int = 100_000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    if not (labels.shape == predictions_a.shape == predictions_b.shape):
        raise ValueError("Paired arrays must have identical shapes")
    correct_a = predictions_a == labels
    correct_b = predictions_b == labels
    differences = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 2_000):
        count = min(2_000, bootstrap_samples - start)
        indices = rng.integers(0, len(labels), size=(count, len(labels)))
        draws[start:start + count] = differences[indices].mean(axis=1)
    discordant_a = int(np.sum(correct_a & ~correct_b))
    discordant_b = int(np.sum(~correct_a & correct_b))
    discordant = discordant_a + discordant_b
    p_value = 1.0 if discordant == 0 else float(binomtest(discordant_a, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "accuracy_difference": float(differences.mean()),
        "ci95_percentile": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "mcnemar_exact_two_sided_p": p_value,
        "discordant_a_correct": discordant_a,
        "discordant_b_correct": discordant_b,
    }

