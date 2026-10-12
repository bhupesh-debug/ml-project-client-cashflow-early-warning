"""Baseline Models for Benchmark Comparison.

Implements standard heuristic financial baselines:
1. Naive Rolling Run-Rate Regressor (13 * rolling 4-week mean cash flow)
2. Heuristic Liquidity Rule Classifier (Flags breach if burn coverage < 4 weeks or Prior probability)
"""

from __future__ import annotations

import numpy as np


class NaiveRollingRunRateRegressor:
    """Heuristic baseline projecting forward 13 weeks as 13 * recent 4-week average net cash flow."""

    def __init__(self, ncf_roll_idx: int = 6):
        # Index 6 corresponds to ncf_roll_mean_4w in FEATURE_COLUMNS
        self.ncf_roll_idx = ncf_roll_idx
        self.fitted_ = False
        self.historical_mean_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveRollingRunRateRegressor:
        self.historical_mean_ = float(np.mean(y))
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before predict.")
        # Project 13 weeks based on rolling 4w feature column
        projections = X[:, self.ncf_roll_idx] * 13.0
        return projections


class HeuristicLiquidityRuleClassifier:
    """Rule-based heuristic classifier comparing cash coverage against 4 weeks."""

    def __init__(self, burn_cov_idx: int = 1, threshold_weeks: float = 4.0):
        self.burn_cov_idx = burn_cov_idx
        self.threshold_weeks = threshold_weeks
        self.prior_breach_rate_ = 0.30
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> HeuristicLiquidityRuleClassifier:
        self.prior_breach_rate_ = float(np.mean(y))
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before predict_proba.")
        burn_cov = X[:, self.burn_cov_idx]
        # Empirical sigmoid approximation around threshold
        diff = self.threshold_weeks - burn_cov
        p1 = 1.0 / (1.0 + np.exp(-1.5 * diff))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.50).astype(int)
