import numpy as np

from fgic_diagnostic.metrics import classification_metrics, paired_accuracy_statistics


def test_classification_metrics_perfect():
    labels = np.array([0, 1, 2, 0])
    assert classification_metrics(labels, labels) == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
    }


def test_paired_statistics_identical_predictions():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])
    result = paired_accuracy_statistics(labels, predictions, predictions, bootstrap_samples=100, seed=42)
    assert result["accuracy_difference"] == 0.0
    assert result["mcnemar_exact_two_sided_p"] == 1.0
    assert result["ci95_percentile"] == [0.0, 0.0]

