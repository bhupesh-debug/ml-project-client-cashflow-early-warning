"""Unit and integration tests for inference engine."""

import os
import pytest
from src.data.generate_synthetic_data import generate_full_dataset
from src.inference.pipeline import CashFlowGuardInferenceEngine
from src.models.train import train_and_evaluate_all


@pytest.fixture(scope="session")
def setup_models(tmp_path_factory):
    tmp_models_dir = tmp_path_factory.mktemp("models")
    df = generate_full_dataset(num_companies=5, weeks_history=30, seed=42)
    train_and_evaluate_all(df, artifacts_dir=str(tmp_models_dir))
    return str(tmp_models_dir)


def test_inference_prediction_flow(setup_models):
    engine = CashFlowGuardInferenceEngine(setup_models)

    payload = {
        "industry": "Commercial Contracting",
        "ending_cash_balance": 80000.0,
        "monthly_opex": 100000.0,
        "daily_debt_service": 500.0,
        "weekly_billed_revenue": 25000.0,
        "total_weekly_outflows": 23000.0,
        "ar_balance": 180000.0,
        "dso_days": 55.0,
        "wip_contract_value": 300000.0,
        "wip_underbillings": 45000.0,
        "recent_weekly_net_cash_flow": -2000.0,
        "ode_margin": 0.15,
    }

    res = engine.predict(payload)
    assert "predicted_13wk_net_cash_flow" in res
    assert "daily_cash_target_breach_probability" in res
    assert 0.0 <= res["daily_cash_target_breach_probability"] <= 1.0
    assert res["risk_tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert len(res["interventions"]) > 0
