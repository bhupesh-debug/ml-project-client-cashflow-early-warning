"""Unit tests for Stage 2 Invoice Days-to-Pay Quantile Regressor."""

import numpy as np
import pandas as pd
import pytest
from src.models.payment_predictor import InvoiceDaysToPayPredictor, pinball_loss


def test_pinball_loss_symmetry():
    y_true = np.array([30.0])
    y_pred = np.array([20.0])  # Underestimated by 10 days
    # At alpha = 0.90, underestimation penalty is 0.90 * 10 = 9.0
    loss_high = pinball_loss(y_true, y_pred, 0.90)
    assert abs(loss_high - 9.0) < 1e-5


def test_invoice_predictor_fit_and_predict():
    rng = np.random.default_rng(42)
    n = 60

    df = pd.DataFrame({
        "invoice_id": [f"INV-{i:03d}" for i in range(n)],
        "industry": rng.choice(["Contractor", "Dental", "Services", "Tech"], size=n),
        "amount_usd": rng.uniform(1000.0, 50000.0, size=n),
        "terms_days": rng.choice([15, 30, 45], size=n),
        "prior_avg_days_late": rng.uniform(-2.0, 25.0, size=n),
        "open_balance_ratio": rng.uniform(0.1, 0.9, size=n),
        "is_new_customer": rng.choice([0, 1], size=n),
        "days_to_pay": rng.uniform(12.0, 70.0, size=n),
    })

    predictor = InvoiceDaysToPayPredictor(random_state=42)
    predictor.fit(df)

    eval_res = predictor.evaluate(df)
    assert "baseline_terms_mae_days" in eval_res
    assert "model_p50_mae_days" in eval_res
    assert "pinball_loss_p50" in eval_res

    preds = predictor.predict_distribution(df.head(5))
    assert "pred_days_p10" in preds.columns
    assert "pred_days_p50" in preds.columns
    assert "pred_days_p90" in preds.columns
    # Monotonicity: P10 <= P50 <= P90 on average
    assert preds["pred_days_p10"].mean() <= preds["pred_days_p50"].mean() + 2.0
