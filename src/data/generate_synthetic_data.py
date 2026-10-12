"""Synthetic Data Generation Module for SMB Cash Flow & WIP Analytics.

Simulates weekly operational and accounting time-series data for small-to-medium
businesses ($1M-$35M ARR) across contracting, professional services, healthcare,
and B2B services. Generates ground-truth future cash movements and Daily Cash Target
breaches for leakage-safe machine learning model training.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INDUSTRIES = [
    "Commercial Contracting",
    "Professional Services",
    "Healthcare & Dental",
    "B2B SaaS & Tech",
]


def generate_company_profile(company_id: int, rng: np.random.Generator) -> Dict[str, float | str]:
    """Generates a steady-state business profile for an SMB client."""
    industry = rng.choice(INDUSTRIES, p=[0.35, 0.25, 0.20, 0.20])
    annual_revenue = float(rng.uniform(1.8e6, 28.0e6))
    weekly_rev_base = annual_revenue / 52.0

    # Margin profile based on industry
    if industry == "Commercial Contracting":
        gross_margin = float(rng.uniform(0.24, 0.38))
        target_dso = float(rng.uniform(45.0, 75.0))
        target_dpo = float(rng.uniform(30.0, 45.0))
        wip_intensity = 0.85
    elif industry == "Professional Services":
        gross_margin = float(rng.uniform(0.40, 0.58))
        target_dso = float(rng.uniform(35.0, 60.0))
        target_dpo = float(rng.uniform(20.0, 35.0))
        wip_intensity = 0.40
    elif industry == "Healthcare & Dental":
        gross_margin = float(rng.uniform(0.45, 0.65))
        target_dso = float(rng.uniform(25.0, 45.0))
        target_dpo = float(rng.uniform(25.0, 40.0))
        wip_intensity = 0.05
    else:  # B2B SaaS & Tech
        gross_margin = float(rng.uniform(0.65, 0.82))
        target_dso = float(rng.uniform(30.0, 50.0))
        target_dpo = float(rng.uniform(25.0, 35.0))
        wip_intensity = 0.15

    fixed_opex_weekly = weekly_rev_base * (1.0 - gross_margin) * float(rng.uniform(0.55, 0.75))
    initial_cash = fixed_opex_weekly * float(rng.uniform(6.0, 16.0))
    debt_service_weekly = float(rng.uniform(1200.0, 8500.0))

    return {
        "company_id": f"SMB-{company_id:04d}",
        "industry": industry,
        "annual_revenue": annual_revenue,
        "weekly_rev_base": weekly_rev_base,
        "gross_margin": gross_margin,
        "target_dso": target_dso,
        "target_dpo": target_dpo,
        "wip_intensity": wip_intensity,
        "fixed_opex_weekly": fixed_opex_weekly,
        "initial_cash": initial_cash,
        "debt_service_weekly": debt_service_weekly,
    }


def generate_company_timeseries(
    profile: Dict[str, float | str],
    num_weeks: int,
    start_date: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generates week-by-week cash flow, WIP and balance sheet history for one company."""
    date_range = pd.date_range(start=start_date, periods=num_weeks, freq="W-FRI")
    records: List[Dict[str, float | str | int]] = []

    cash_balance = float(profile["initial_cash"])
    base_rev = float(profile["weekly_rev_base"])
    fixed_opex = float(profile["fixed_opex_weekly"])
    debt_weekly = float(profile["debt_service_weekly"])
    target_dso = float(profile["target_dso"])
    wip_intensity = float(profile["wip_intensity"])

    # Rolling state tracking
    ar_balance = base_rev * (target_dso / 7.0)
    monthly_opex_est = fixed_opex * 4.33
    daily_cash_target = (monthly_opex_est / 30.0) * 60.0 + (debt_weekly / 7.0) * 60.0

    # Trend and seasonality factors
    trend_slope = rng.normal(0.0015, 0.002)

    for w_idx, current_date in enumerate(date_range):
        # Seasonality: higher in Q4, lower in Jan
        seasonality = 1.0 + 0.12 * np.sin(2 * np.pi * (w_idx % 52) / 52.0)
        macro_shock = 1.0 + rng.normal(0.0, 0.08)
        growth = 1.0 + trend_slope * w_idx
        current_billed_rev = max(base_rev * 0.4, base_rev * seasonality * macro_shock * growth)

        # Collections modeled with DSO lag variability
        dso_current = float(np.clip(target_dso + rng.normal(0.0, 6.0), 18.0, 110.0))
        collection_rate = min(1.0, 7.0 / max(dso_current, 10.0))
        collections = (ar_balance * collection_rate) + (current_billed_rev * rng.uniform(0.10, 0.25))
        ar_balance = max(1000.0, ar_balance + current_billed_rev - collections)

        # Outflows
        payroll_outflow = fixed_opex * rng.uniform(0.92, 1.08)
        vendor_outflow = (current_billed_rev * (1.0 - float(profile["gross_margin"]))) * rng.uniform(0.85, 1.15)
        discretionary_owner_draw = max(0.0, base_rev * float(rng.uniform(0.02, 0.08)))
        total_outflows = payroll_outflow + vendor_outflow + debt_weekly + discretionary_owner_draw

        # Current weekly net cash flow
        net_cash_flow = collections - total_outflows
        cash_balance += net_cash_flow

        # Work in Progress (WIP) simulation for project/service businesses
        wip_contract_val = current_billed_rev * 12.0 * wip_intensity
        wip_cost_budget = wip_contract_val * (1.0 - float(profile["gross_margin"]))
        pct_complete = float(np.clip(rng.uniform(0.20, 0.85), 0.05, 0.95))
        wip_cost_incurred = wip_cost_budget * pct_complete * rng.uniform(0.95, 1.18)  # risk of overrun
        wip_earned_rev = (wip_cost_incurred / max(wip_cost_budget, 1.0)) * wip_contract_val
        wip_billings = wip_contract_val * (pct_complete + rng.normal(0.0, 0.08))

        wip_underbillings = max(0.0, wip_earned_rev - wip_billings)  # Asset: unbilled earned work
        wip_overbillings = max(0.0, wip_billings - wip_earned_rev)   # Liability: billed before work

        # Owner's Discretionary Earnings (ODE) proxy
        ode_weekly = max(0.0, current_billed_rev - vendor_outflow - payroll_outflow * 0.70 + discretionary_owner_draw)

        records.append({
            "company_id": profile["company_id"],
            "industry": profile["industry"],
            "week_date": current_date.strftime("%Y-%m-%d"),
            "week_index": w_idx,
            "weekly_billed_revenue": round(current_billed_rev, 2),
            "cash_collections": round(collections, 2),
            "payroll_outflow": round(payroll_outflow, 2),
            "vendor_outflow": round(vendor_outflow, 2),
            "debt_service_weekly": round(debt_weekly, 2),
            "discretionary_owner_draw": round(discretionary_owner_draw, 2),
            "total_weekly_outflows": round(total_outflows, 2),
            "net_cash_flow": round(net_cash_flow, 2),
            "ending_cash_balance": round(cash_balance, 2),
            "ar_balance": round(ar_balance, 2),
            "dso_days": round(dso_current, 1),
            "daily_cash_target": round(daily_cash_target, 2),
            "wip_contract_value": round(wip_contract_val, 2),
            "wip_cost_incurred": round(wip_cost_incurred, 2),
            "wip_earned_revenue": round(wip_earned_rev, 2),
            "wip_billings_to_date": round(wip_billings, 2),
            "wip_underbillings": round(wip_underbillings, 2),
            "wip_overbillings": round(wip_overbillings, 2),
            "ode_weekly_est": round(ode_weekly, 2),
        })

    df = pd.DataFrame(records)

    # Compute Forward-Looking Prediction Targets (isolated for ML training)
    # Target 1: 13-week cumulative forward net cash flow
    df["target_13wk_net_cash_flow"] = (
        df["net_cash_flow"]
        .iloc[::-1]
        .rolling(window=13, min_periods=13)
        .sum()
        .iloc[::-1]
        .shift(-1)
    )

    # Target 2: Minimum cash balance in the next 6 weeks (~45 days)
    min_cash_45d = (
        df["ending_cash_balance"]
        .iloc[::-1]
        .rolling(window=6, min_periods=6)
        .min()
        .iloc[::-1]
        .shift(-1)
    )
    df["target_min_cash_balance_45d"] = min_cash_45d

    # Target 3: Binary breach indicator if minimum cash falls below Daily Cash Target
    df["target_target_breach_45d"] = (
        df["target_min_cash_balance_45d"] < df["daily_cash_target"]
    ).astype(float)

    # Drop trailing weeks that cannot formulate complete 13-week future ground truth
    df = df.dropna(subset=["target_13wk_net_cash_flow", "target_target_breach_45d"]).copy()
    df["target_target_breach_45d"] = df["target_target_breach_45d"].astype(int)

    return df


def generate_full_dataset(
    num_companies: int = 120,
    weeks_history: int = 104,
    start_date: str = "2024-01-05",
    seed: int = 42,
) -> pd.DataFrame:
    """Generates the comprehensive synthetic dataset across all companies."""
    logger.info(
        f"Generating synthetic dataset: {num_companies} companies, {weeks_history} weeks, seed={seed}"
    )
    rng = np.random.default_rng(seed)
    dfs: List[pd.DataFrame] = []

    for i in range(1, num_companies + 1):
        prof = generate_company_profile(i, rng)
        comp_df = generate_company_timeseries(prof, weeks_history, start_date, rng)
        dfs.append(comp_df)

    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"Generated dataset with {len(full_df)} total weekly observations across {num_companies} clients."
    )
    logger.info(
        f"Class balance for 45-day cash breach: {full_df['target_target_breach_45d'].mean():.2%} positive cases."
    )
    return full_df


if __name__ == "__main__":
    df = generate_full_dataset()
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))
