from __future__ import annotations
"""
Model evaluation utilities shared across all models.
"""

import numpy as np
import pandas as pd


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Intersection over Union for binary predictions."""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_metrics_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute a full suite of binary classification metrics."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, balanced_accuracy_score,
        average_precision_score,
    )
    metrics = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "iou": round(compute_iou(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["pr_auc"] = round(average_precision_score(y_true, y_proba), 4)
    return metrics
