"""Feature Engineering Module for SMB Cash Flow & Risk Forecasting.

Calculates leakage-safe rolling statistics, working capital ratios, WIP health flags,
and liquidity coverage indicators without accessing forward look-ahead periods.
"""

from __future__ import annotations

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    # Liquidity & Coverage Ratios
    "cash_to_daily_target_ratio",
    "burn_coverage_weeks",
    "ar_to_monthly_rev_ratio",
    "dso_days",
    # Rolling Cash Flow Statistics (Lagged)
    "ncf_lag_1",
    "ncf_lag_2",
    "ncf_roll_mean_4w",
    "ncf_roll_std_4w",
    "ncf_roll_mean_8w",
    "collections_roll_mean_4w",
    "outflows_roll_mean_4w",
    # WIP & Margin Leakage Indicators
    "wip_underbilling_ratio",
    "wip_overbilling_ratio",
    "wip_cost_intensity",
    # Profitability & ODE
    "ode_margin_est",
    # Calendar & Seasonality
    "month_sin",
    "month_cos",
    "is_quarter_end",
    # Industry Dummies
    "ind_Contracting",
    "ind_ProfServices",
    "ind_Healthcare",
    "ind_SaaS_Tech",
]


def extract_company_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    """Computes lagging rolling features on a single company's chronological sequence."""
    g = group.sort_values(by="week_date").copy()

    # Lagged cash flows
    g["ncf_lag_1"] = g["net_cash_flow"].shift(1)
    g["ncf_lag_2"] = g["net_cash_flow"].shift(2)

    # Rolling window metrics using closed='left' logic via shift(1) to avoid including current week
    g["ncf_roll_mean_4w"] = g["net_cash_flow"].shift(1).rolling(4, min_periods=2).mean()
    g["ncf_roll_std_4w"] = g["net_cash_flow"].shift(1).rolling(4, min_periods=2).std().fillna(0.0)
    g["ncf_roll_mean_8w"] = g["net_cash_flow"].shift(1).rolling(8, min_periods=3).mean()
    g["collections_roll_mean_4w"] = g["cash_collections"].shift(1).rolling(4, min_periods=2).mean()
    g["outflows_roll_mean_4w"] = g["total_weekly_outflows"].shift(1).rolling(4, min_periods=2).mean()

    return g


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers full predictive feature set across all companies."""
    logger.info("Computing rolling lagged features per company...")
    processed_dfs = []
    for _, group in df.groupby("company_id", group_keys=False):
        processed_dfs.append(extract_company_rolling_features(group))
    feat_df = pd.concat(processed_dfs, ignore_index=True)

    # Fill initial lag values
    feat_df["ncf_lag_1"] = feat_df["ncf_lag_1"].fillna(feat_df["net_cash_flow"])
    feat_df["ncf_lag_2"] = feat_df["ncf_lag_2"].fillna(feat_df["ncf_lag_1"])
    feat_df["ncf_roll_mean_4w"] = feat_df["ncf_roll_mean_4w"].fillna(feat_df["net_cash_flow"])
    feat_df["ncf_roll_mean_8w"] = feat_df["ncf_roll_mean_8w"].fillna(feat_df["ncf_roll_mean_4w"])
    feat_df["collections_roll_mean_4w"] = feat_df["collections_roll_mean_4w"].fillna(feat_df["cash_collections"])
    feat_df["outflows_roll_mean_4w"] = feat_df["outflows_roll_mean_4w"].fillna(feat_df["total_weekly_outflows"])

    # Financial & Working Capital Ratios
    feat_df["cash_to_daily_target_ratio"] = feat_df["ending_cash_balance"] / np.maximum(
        feat_df["daily_cash_target"], 1.0
    )
    feat_df["burn_coverage_weeks"] = feat_df["ending_cash_balance"] / np.maximum(
        feat_df["total_weekly_outflows"], 1.0
    )
    feat_df["ar_to_monthly_rev_ratio"] = feat_df["ar_balance"] / np.maximum(
        feat_df["weekly_billed_revenue"] * 4.33, 1.0
    )

    # WIP Metrics
    contract_val = np.maximum(feat_df["wip_contract_value"], 1.0)
    feat_df["wip_underbilling_ratio"] = feat_df["wip_underbillings"] / contract_val
    feat_df["wip_overbilling_ratio"] = feat_df["wip_overbillings"] / contract_val
    feat_df["wip_cost_intensity"] = feat_df["wip_cost_incurred"] / contract_val

    # Profitability Proxy
    feat_df["ode_margin_est"] = feat_df["ode_weekly_est"] / np.maximum(
        feat_df["weekly_billed_revenue"], 1.0
    )

    # Seasonality
    months = pd.to_datetime(feat_df["week_date"]).dt.month
    feat_df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    feat_df["month_cos"] = np.cos(2 * np.pi * months / 12.0)
    feat_df["is_quarter_end"] = (
        pd.to_datetime(feat_df["week_date"]).dt.is_quarter_end.astype(float)
    )

    # Industry Indicator Columns
    feat_df["ind_Contracting"] = (feat_df["industry"] == "Commercial Contracting").astype(float)
    feat_df["ind_ProfServices"] = (feat_df["industry"] == "Professional Services").astype(float)
    feat_df["ind_Healthcare"] = (feat_df["industry"] == "Healthcare & Dental").astype(float)
    feat_df["ind_SaaS_Tech"] = (feat_df["industry"] == "B2B SaaS & Tech").astype(float)

    logger.info(f"Engineered {len(FEATURE_COLUMNS)} features across {len(feat_df)} rows.")
    return feat_df


def get_feature_and_target_matrices(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extracts X feature matrix, y_regression target, y_classification target, and feature names."""
    feat_df = engineer_features(df)
    X = feat_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_reg = feat_df["target_13wk_net_cash_flow"].to_numpy(dtype=float)
    y_clf = feat_df["target_target_breach_45d"].to_numpy(dtype=int)
    return X, y_reg, y_clf, FEATURE_COLUMNS
