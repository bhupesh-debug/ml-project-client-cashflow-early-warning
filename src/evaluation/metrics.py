"""Comprehensive Model Evaluation Metrics Module.

Computes statistical and business-aligned metrics for both regression
(cash flow forecasting) and classification (Daily Cash Target breach).
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    accuracy_score,
)


def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates directional correctness (sign match) between true and predicted delta."""
    true_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    return float(np.mean(true_dir == pred_dir))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates financial forecasting metrics for continuous cash flow prediction."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mda = mean_directional_accuracy(y_true, y_pred)

    # Bounded MAPE avoiding zero division
    denom = np.maximum(np.abs(y_true), 1000.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MDA": round(mda, 4),
        "MAPE": round(mape, 2),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Calculates risk detection metrics for cash target breach probability."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Brier_Score": round(brier, 4),
    }
