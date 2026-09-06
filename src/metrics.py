from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def roc_auc_score_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC with a clear failure if only one class is present."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        raise ValueError("ROC-AUC is undefined when only one class is present in y_true.")
    return float(roc_auc_score(y_true, y_score))


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return _safe_divide(tp, tp + fn)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return _safe_divide(tn, tn + fp)


def precision_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return _safe_divide(tp, tp + fp)


def f1_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return _safe_divide(2 * tp, 2 * tp + fp + fn)


def accuracy_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def confusion_matrix_safe(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Return a dictionary of standard binary classification metrics."""
    cm = confusion_matrix_safe(y_true, y_pred)
    result = {
        "confusion_matrix": cm,
        "accuracy": accuracy_score_safe(y_true, y_pred),
        "sensitivity": sensitivity_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "precision": precision_score_safe(y_true, y_pred),
        "f1_score": f1_score_safe(y_true, y_pred),
    }

    if y_score is not None:
        result["roc_auc"] = roc_auc_score_safe(y_true, y_score)
    return result
