"""Unit tests for models and evaluation."""

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.evaluation.metrics import evaluate_classification, evaluate_regression
from src.models.baseline import HeuristicLiquidityRuleClassifier, NaiveRollingRunRateRegressor


def test_naive_regressor():
    X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1000.0]])
    y = np.array([13000.0])
    model = NaiveRollingRunRateRegressor(ncf_roll_idx=6)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds[0] == 13000.0


def test_heuristic_classifier():
    X = np.array([[10.0, 1.5], [10.0, 8.0]])  # Burn coverage 1.5w vs 8.0w
    y = np.array([1, 0])
    model = HeuristicLiquidityRuleClassifier(burn_cov_idx=1, threshold_weeks=4.0)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (2, 2)
    assert probs[0, 1] > probs[1, 1]  # Lower coverage should have higher breach probability


def test_evaluation_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    reg_metrics = evaluate_regression(y_true, y_pred)
    assert "MAE" in reg_metrics
    assert reg_metrics["MAE"] == 10.0
    assert reg_metrics["R2"] > 0.95

    y_clf_true = np.array([1, 0, 1, 0])
    y_clf_pred = np.array([1, 0, 1, 0])
    y_clf_prob = np.array([0.9, 0.1, 0.8, 0.2])
    clf_metrics = evaluate_classification(y_clf_true, y_clf_pred, y_clf_prob)
    assert clf_metrics["Accuracy"] == 1.0
    assert clf_metrics["ROC_AUC"] == 1.0
