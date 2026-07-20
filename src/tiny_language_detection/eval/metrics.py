"""Evaluation metrics for language detection models."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(
    predictions: list[int],
    labels: list[int],
    languages: list[str],
    duration_groups: list[str],
) -> dict:
    """Compute evaluation metrics for language detection.

    Args:
        predictions:
            List of predicted class labels (0 or 1).
        labels:
            List of true class labels (0 or 1).
        languages:
            List of language codes corresponding to each sample.
        duration_groups:
            List of duration group labels for each sample
            (e.g., "0-2", "2-4").

    Returns:
        Dictionary containing:
            - overall_accuracy: Float between 0 and 1.
            - accuracy_by_duration: Mapping from duration group to accuracy.
            - per_language_accuracy: Mapping from language code to accuracy.
            - confusion_matrix: 2x2 matrix as list of lists.
            - total_samples: Total number of samples evaluated.

    Raises:
        ValueError:
            If input lists have mismatched lengths.
    """
    if len(predictions) != len(labels):
        msg = (
            f"predictions ({len(predictions)}) and labels ({len(labels)}) "
            "must have the same length"
        )
        raise ValueError(msg)
    if len(predictions) != len(languages):
        msg = (
            f"predictions ({len(predictions)}) and languages ({len(languages)}) "
            "must have the same length"
        )
        raise ValueError(msg)
    if len(predictions) != len(duration_groups):
        msg = (
            f"predictions ({len(predictions)}) and duration_groups "
            f"({len(duration_groups)}) must have the same length"
        )
        raise ValueError(msg)

    total_samples = len(predictions)

    # Handle empty input
    if total_samples == 0:
        return {
            "overall_accuracy": 0.0,
            "accuracy_by_duration": {},
            "per_language_accuracy": {},
            "confusion_matrix": [[0, 0], [0, 0]],
            "total_samples": 0,
        }

    predictions_arr = np.array(predictions)
    labels_arr = np.array(labels)

    # Overall accuracy
    correct = (predictions_arr == labels_arr).sum()
    overall_accuracy = float(correct / total_samples)

    # Accuracy by duration group
    duration_correct: defaultdict[str, int] = defaultdict(int)
    duration_total: defaultdict[str, int] = defaultdict(int)

    for pred, label, dur_group in zip(predictions, labels, duration_groups):
        duration_total[dur_group] += 1
        if pred == label:
            duration_correct[dur_group] += 1

    accuracy_by_duration: dict[str, float] = {}
    for group in duration_total:
        accuracy_by_duration[group] = float(
            duration_correct[group] / duration_total[group]
        )

    # Per-language accuracy
    lang_correct: defaultdict[str, int] = defaultdict(int)
    lang_total: defaultdict[str, int] = defaultdict(int)

    for pred, label, lang in zip(predictions, labels, languages):
        lang_total[lang] += 1
        if pred == label:
            lang_correct[lang] += 1

    per_language_accuracy: dict[str, float] = {}
    for lang in lang_total:
        per_language_accuracy[lang] = float(lang_correct[lang] / lang_total[lang])

    # Confusion matrix using scikit-learn
    cm = confusion_matrix(labels_arr, predictions_arr)
    confusion_matrix_list: list[list[int]] = cm.tolist()

    return {
        "overall_accuracy": overall_accuracy,
        "accuracy_by_duration": accuracy_by_duration,
        "per_language_accuracy": per_language_accuracy,
        "confusion_matrix": confusion_matrix_list,
        "total_samples": total_samples,
    }
