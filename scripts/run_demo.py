"""CLI Demonstration Script for SMB Client Financial Diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import CashFlowGuardInferenceEngine


def main():
    print("=" * 70)
    print("CARDONE PLATFORM REVIEW — PREDICTIVE CASH FLOW GUARD DEMO")
    print("Candidate: Bhupesh Chandra Dimri | Target Role: Finance Manager")
    print("=" * 70)

    engine = CashFlowGuardInferenceEngine(str(project_root / "reports" / "models"))

    scenarios = [
        {
            "name": "Scenario A: High-Growth Commercial Contractor (Facing Severe AR & WIP Slippage)",
            "payload": {
                "industry": "Commercial Contracting",
                "ending_cash_balance": 95000.0,
                "monthly_opex": 160000.0,
                "daily_debt_service": 850.0,
                "weekly_billed_revenue": 52000.0,
                "total_weekly_outflows": 48000.0,
                "ar_balance": 390000.0,
                "dso_days": 68.0,
                "wip_contract_value": 750000.0,
                "wip_underbillings": 68000.0,
                "wip_overbillings": 0.0,
                "wip_cost_incurred": 490000.0,
                "recent_weekly_net_cash_flow": -4500.0,
                "ode_margin": 0.14,
            },
        },
        {
            "name": "Scenario B: Multi-Location Dental Practice (Healthy Liquidity, Stable Margins)",
            "payload": {
                "industry": "Healthcare & Dental",
                "ending_cash_balance": 380000.0,
                "monthly_opex": 110000.0,
                "daily_debt_service": 400.0,
                "weekly_billed_revenue": 38000.0,
                "total_weekly_outflows": 27500.0,
                "ar_balance": 110000.0,
                "dso_days": 28.0,
                "wip_contract_value": 0.0,
                "wip_underbillings": 0.0,
                "wip_overbillings": 0.0,
                "wip_cost_incurred": 0.0,
                "recent_weekly_net_cash_flow": 10500.0,
                "ode_margin": 0.28,
            },
        },
    ]

    for sc in scenarios:
        print(f"\nEvaluating: {sc['name']}")
        res = engine.predict(sc["payload"])
        print(f"  Predicted 13-Week Net Cash Flow: ${res['predicted_13wk_net_cash_flow']:,.2f}")
        print(f"  45-Day Daily Cash Target Breach Risk: {res['daily_cash_target_breach_probability']:.1%}")
        print(f"  Assigned Risk Tier: [{res['risk_tier']}]")
        print("  Actionable Advisory Interventions:")
        for action in res["interventions"]:
            print(f"    - {action}")


if __name__ == "__main__":
    main()
