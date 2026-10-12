"""Streamlit Interactive Application for Platform Review Cash Flow Diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import CashFlowGuardInferenceEngine


def main():
    st.set_page_config(
        page_title="Cardone Platform CashFlow Guard",
        page_icon="💼",
        layout="wide",
    )

    st.title("🛡️ Cardone Platform Review — Predictive Cash Flow Guard")
    st.markdown(
        "**Author:** Bhupesh Chandra Dimri | **Target Role:** Finance Manager (Platform Review) | **Company:** Cardone Ventures"
    )
    st.markdown("---")

    engine = CashFlowGuardInferenceEngine(str(project_root / "reports" / "models"))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🏢 Client Operational & Accounting Inputs")
        industry = st.selectbox(
            "Industry Vertical",
            ["Commercial Contracting", "Professional Services", "Healthcare & Dental", "B2B SaaS & Tech"],
        )
        ending_cash = st.number_input("Ending Cash Balance ($)", min_value=1000.0, max_value=10000000.0, value=95000.0, step=5000.0)
        monthly_opex = st.number_input("Monthly Fixed OpEx ($)", min_value=1000.0, max_value=5000000.0, value=160000.0, step=5000.0)
        daily_debt = st.number_input("Daily Debt Service ($)", min_value=0.0, max_value=50000.0, value=850.0, step=50.0)
        weekly_billed_revenue = st.number_input("Weekly Billed Revenue ($)", min_value=1000.0, max_value=2000000.0, value=52000.0, step=2000.0)
        total_weekly_outflows = st.number_input("Total Weekly Cash Outflows ($)", min_value=1000.0, max_value=2000000.0, value=48000.0, step=2000.0)
        ar_balance = st.number_input("Accounts Receivable Balance ($)", min_value=0.0, max_value=10000000.0, value=390000.0, step=10000.0)
        dso_days = st.slider("Days Sales Outstanding (DSO)", min_value=10.0, max_value=120.0, value=68.0)
        wip_contract_value = st.number_input("Active WIP Contract Value ($)", min_value=0.0, max_value=20000000.0, value=750000.0, step=25000.0)
        wip_underbillings = st.number_input("WIP Underbillings / Unbilled Earned Work ($)", min_value=0.0, max_value=5000000.0, value=68000.0, step=5000.0)
        recent_weekly_net_cash_flow = st.number_input("Recent Weekly Net Cash Flow ($)", min_value=-500000.0, max_value=500000.0, value=-4500.0, step=1000.0)
        ode_margin = st.slider("Owner's Discretionary Earnings (ODE) Margin", min_value=0.05, max_value=0.50, value=0.14)

    payload = {
        "industry": industry,
        "ending_cash_balance": ending_cash,
        "monthly_opex": monthly_opex,
        "daily_debt_service": daily_debt,
        "weekly_billed_revenue": weekly_billed_revenue,
        "total_weekly_outflows": total_weekly_outflows,
        "ar_balance": ar_balance,
        "dso_days": dso_days,
        "wip_contract_value": wip_contract_value,
        "wip_underbillings": wip_underbillings,
        "wip_overbillings": 0.0,
        "wip_cost_incurred": wip_contract_value * 0.65,
        "recent_weekly_net_cash_flow": recent_weekly_net_cash_flow,
        "ode_margin": ode_margin,
    }

    result = engine.predict(payload)

    with col2:
        st.subheader("📊 Model Predictions & Early Warning Diagnostics")
        
        # Risk Badge
        tier = result["risk_tier"]
        color_map = {"LOW": "green", "MODERATE": "orange", "HIGH": "red", "CRITICAL": "darkred"}
        st.markdown(f"### Liquidity Risk Tier: **:{color_map.get(tier, 'blue')}[{tier}]**")

        st.metric(
            label="45-Day Daily Cash Target Breach Probability",
            value=f"{result['daily_cash_target_breach_probability']:.1%}",
            delta="- High Risk" if result['daily_cash_target_breach_probability'] > 0.4 else "Normal Buffer",
        )

        st.metric(
            label="Predicted 13-Week Net Cumulative Cash Flow",
            value=f"${result['predicted_13wk_net_cash_flow']:,.2f}",
        )

        st.markdown("#### 🎯 Prescribed Advisory Interventions (TAPW Protocol)")
        for item in result["interventions"]:
            st.info(f"👉 {item}")

        daily_target = (monthly_opex / 30.0) * 60.0 + (daily_debt * 60.0)
        st.caption(f"Calculated 60-Day Daily Cash Target: ${daily_target:,.2f} | Current Ending Cash: ${ending_cash:,.2f}")


if __name__ == "__main__":
    main()
