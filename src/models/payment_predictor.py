"""Stage 2: Invoice Days-to-Pay Quantile Regressor.

Replaces theoretical contract terms (Net 30) with empirical customer payment distributions.
Trains Quantile Gradient Boosted Regressors at alpha = 0.10, 0.50, 0.90 with asymmetric loss
to penalize late payment underestimation, feeding empirical cash inflows into Stage 3.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

INVOICE_FEATURES = [
    "log_amount_usd",
    "terms_days",
    "prior_avg_days_late",
    "open_balance_ratio",
    "is_new_customer",
    "ind_Contractor",
    "ind_Dental",
    "ind_Services",
    "ind_Tech",
]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculates pinball (quantile) loss: L_q(y, y_hat) = max(q(y - y_hat), (q-1)(y - y_hat))."""
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


class InvoiceDaysToPayPredictor:
    """Quantile regression model predicting days-to-pay distributions (P10, P50, P90)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.quantiles = [0.10, 0.50, 0.90]
        self.models: Dict[float, HistGradientBoostingRegressor] = {
            q: HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=120,
                learning_rate=0.05,
                random_state=random_state + int(q * 100),
            )
            for q in self.quantiles
        }
        self.fitted_ = False

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocesses raw invoice rows into model feature matrix."""
        df_feat = df.copy()

        df_feat["log_amount_usd"] = np.log1p(df_feat["amount_usd"].clip(lower=0.0))
        df_feat["prior_avg_days_late"] = df_feat["prior_avg_days_late"].fillna(5.0)
        df_feat["open_balance_ratio"] = df_feat["open_balance_ratio"].fillna(0.3)
        df_feat["is_new_customer"] = df_feat["is_new_customer"].fillna(0).astype(float)

        # Industry one-hot
        df_feat["ind_Contractor"] = (df_feat["industry"] == "Contractor").astype(float)
        df_feat["ind_Dental"] = (df_feat["industry"] == "Dental").astype(float)
        df_feat["ind_Services"] = (df_feat["industry"] == "Services").astype(float)
        df_feat["ind_Tech"] = (df_feat["industry"] == "Tech").astype(float)

        X = df_feat[INVOICE_FEATURES].to_numpy(dtype=float)
        return X, INVOICE_FEATURES

    def fit(self, invoices_df: pd.DataFrame) -> InvoiceDaysToPayPredictor:
        """Fits quantile regressors across all defined quantiles."""
        X, _ = self.prepare_features(invoices_df)
        y = invoices_df["days_to_pay"].to_numpy(dtype=float)

        for q, model in self.models.items():
            model.fit(X, y)

        self.fitted_ = True
        logger.info(f"Fitted quantile models for quantiles: {self.quantiles}")
        return self

    def predict_distribution(self, invoices_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts P10, P50, and P90 days-to-pay for every invoice."""
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before predict.")

        X, _ = self.prepare_features(invoices_df)
        res = invoices_df.copy()

        for q in self.quantiles:
            col_name = f"pred_days_p{int(q*100)}"
            res[col_name] = np.clip(self.models[q].predict(X), 1.0, 150.0).round(1)

        # Expected settlement date based on P50 median
        if "issue_date" in res.columns:
            res["expected_settlement_date"] = pd.to_datetime(res["issue_date"]) + pd.to_timedelta(
                res["pred_days_p50"], unit="D"
            )

        return res

    def evaluate(self, invoices_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates MAE and Pinball loss across each quantile."""
        preds_df = self.predict_distribution(invoices_df)
        y_true = invoices_df["days_to_pay"].to_numpy(dtype=float)

        metrics = {}
        # P50 MAE vs Contractual Terms Baseline MAE
        baseline_pred = invoices_df["terms_days"].to_numpy(dtype=float)
        baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
        p50_mae = float(mean_absolute_error(y_true, preds_df["pred_days_p50"].to_numpy()))

        metrics["baseline_terms_mae_days"] = round(baseline_mae, 2)
        metrics["model_p50_mae_days"] = round(p50_mae, 2)
        metrics["mae_improvement_pct"] = round(((baseline_mae - p50_mae) / baseline_mae) * 100.0, 1)

        for q in self.quantiles:
            pred_col = f"pred_days_p{int(q*100)}"
            metrics[f"pinball_loss_p{int(q*100)}"] = round(
                pinball_loss(y_true, preds_df[pred_col].to_numpy(), q), 3
            )

        return metrics
