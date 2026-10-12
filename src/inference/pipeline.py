"""Production Inference and Advisory Decision Engine.

Loads trained model artifacts, performs feature transformation on new or real-time
client financial inputs, and outputs predictive forecasts along with actionable
cash-flow intervention playbooks (TAPW Protocol: Target, AR, AP, WIP).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS


class CashFlowGuardInferenceEngine:
    """Inference engine for predicting 13-week cash flow and 45-day cash target breach risk."""

    def __init__(self, models_dir: str = "reports/models"):
        self.models_dir = Path(models_dir)
        self.reg_model_path = self.models_dir / "cashflow_regressor_hgbr.joblib"
        self.clf_model_path = self.models_dir / "breach_classifier_hgbc.joblib"

        if not self.reg_model_path.exists() or not self.clf_model_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. Please run model training first."
            )

        self.regressor = joblib.load(self.reg_model_path)
        self.classifier = joblib.load(self.clf_model_path)

    def extract_features_from_dict(self, payload: Dict[str, Any]) -> np.ndarray:
        """Transforms a client state dictionary into model feature vector."""
        # Baseline calculations
        cash = float(payload.get("ending_cash_balance", 100000.0))
        monthly_opex = float(payload.get("monthly_opex", 75000.0))
        daily_debt = float(payload.get("daily_debt_service", 500.0))
        daily_target = (monthly_opex / 30.0) * 60.0 + (daily_debt * 60.0)

        weekly_outflows = float(payload.get("total_weekly_outflows", monthly_opex / 4.33))
        weekly_rev = float(payload.get("weekly_billed_revenue", monthly_opex / 3.5))
        ar_balance = float(payload.get("ar_balance", weekly_rev * 6.0))
        dso_days = float(payload.get("dso_days", 52.0))

        contract_val = max(1.0, float(payload.get("wip_contract_value", weekly_rev * 10.0)))
        wip_under = float(payload.get("wip_underbillings", 0.0))
        wip_over = float(payload.get("wip_overbillings", 0.0))
        wip_cost = float(payload.get("wip_cost_incurred", contract_val * 0.60))

        recent_ncf = float(payload.get("recent_weekly_net_cash_flow", 2500.0))

        industry = payload.get("industry", "Commercial Contracting")

        features = {
            "cash_to_daily_target_ratio": cash / max(daily_target, 1.0),
            "burn_coverage_weeks": cash / max(weekly_outflows, 1.0),
            "ar_to_monthly_rev_ratio": ar_balance / max(weekly_rev * 4.33, 1.0),
            "dso_days": dso_days,
            "ncf_lag_1": recent_ncf,
            "ncf_lag_2": recent_ncf,
            "ncf_roll_mean_4w": recent_ncf,
            "ncf_roll_std_4w": abs(recent_ncf) * 0.15,
            "ncf_roll_mean_8w": recent_ncf,
            "collections_roll_mean_4w": weekly_outflows + recent_ncf,
            "outflows_roll_mean_4w": weekly_outflows,
            "wip_underbilling_ratio": wip_under / contract_val,
            "wip_overbilling_ratio": wip_over / contract_val,
            "wip_cost_intensity": wip_cost / contract_val,
            "ode_margin_est": float(payload.get("ode_margin", 0.18)),
            "month_sin": 0.5,
            "month_cos": 0.86,
            "is_quarter_end": 0.0,
            "ind_Contracting": 1.0 if industry == "Commercial Contracting" else 0.0,
            "ind_ProfServices": 1.0 if industry == "Professional Services" else 0.0,
            "ind_Healthcare": 1.0 if industry == "Healthcare & Dental" else 0.0,
            "ind_SaaS_Tech": 1.0 if industry == "B2B SaaS & Tech" else 0.0,
        }

        feature_vector = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=float)
        return feature_vector

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference and generates structured advisory output."""
        X = self.extract_features_from_dict(payload)

        pred_13wk_cash = float(self.regressor.predict(X)[0])
        breach_prob = float(self.classifier.predict_proba(X)[0, 1])

        # Risk tiering
        if breach_prob >= 0.65:
            risk_tier = "CRITICAL"
        elif breach_prob >= 0.40:
            risk_tier = "HIGH"
        elif breach_prob >= 0.20:
            risk_tier = "MODERATE"
        else:
            risk_tier = "LOW"

        # Actionable Strategic Recommendations based on risk tier and WIP/DSO metrics
        interventions: List[str] = []
        dso = float(payload.get("dso_days", 45.0))
        wip_under = float(payload.get("wip_underbillings", 0.0))

        if breach_prob >= 0.40:
            interventions.append(
                "Activate TAPW Emergency Cash Protocol: Enforce Daily Cash Target threshold monitoring."
            )
            interventions.append(
                "Halt non-essential discretionary owner draws and defer non-critical capex commitments."
            )

        if dso > 50.0:
            interventions.append(
                f"Accelerate AR Collections: DSO is elevated at {dso:.0f} days. Implement 2/10 Net 30 early payment discount."
            )

        if wip_under > 15000.0:
            interventions.append(
                f"Recover WIP Underbillings: ${wip_under:,.0f} in incurred project work is unbilled. Enforce milestone billing."
            )

        if not interventions:
            interventions.append(
                "Maintain Current Pacing: Client shows strong liquidity buffers and manageable working capital metrics."
            )

        return {
            "predicted_13wk_net_cash_flow": round(pred_13wk_cash, 2),
            "daily_cash_target_breach_probability": round(breach_prob, 4),
            "risk_tier": risk_tier,
            "interventions": interventions,
            "key_metrics_assessed": {
                "dso_days": dso,
                "wip_underbillings": wip_under,
                "current_cash": float(payload.get("ending_cash_balance", 0.0)),
            },
        }
