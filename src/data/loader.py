"""Data Loader and Chronological Splitter Module.

Ensures strict zero-lookahead chronological partitioning for financial time-series.
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "company_id",
    "industry",
    "week_date",
    "week_index",
    "weekly_billed_revenue",
    "cash_collections",
    "payroll_outflow",
    "vendor_outflow",
    "total_weekly_outflows",
    "net_cash_flow",
    "ending_cash_balance",
    "ar_balance",
    "dso_days",
    "daily_cash_target",
    "target_13wk_net_cash_flow",
    "target_target_breach_45d",
]


def validate_schema(df: pd.DataFrame) -> bool:
    """Validates that dataframe adheres to required analytical schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df["target_13wk_net_cash_flow"].isnull().any():
        raise ValueError("Found NaN values in regression target column.")
    if df["target_target_breach_45d"].isnull().any():
        raise ValueError("Found NaN values in classification target column.")
    return True


def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads CSV dataset and formats date column."""
    df = pd.read_csv(filepath)
    df["week_date"] = pd.to_datetime(df["week_date"])
    validate_schema(df)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def split_data_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions data into Train, Validation, and Test sets using strict chronological cutoff.

    Prevents look-ahead bias across all companies simultaneously.
    """
    df = df.copy()
    df["week_date"] = pd.to_datetime(df["week_date"])
    sorted_df = df.sort_values(by=["week_date", "company_id"]).reset_index(drop=True)
    unique_dates = sorted_df["week_date"].drop_duplicates().sort_values().to_list()
    total_dates = len(unique_dates)

    train_end_idx = int(total_dates * train_ratio)
    val_end_idx = int(total_dates * (train_ratio + val_ratio))

    train_cutoff_date = unique_dates[train_end_idx - 1]
    val_cutoff_date = unique_dates[val_end_idx - 1]

    train_df = sorted_df[sorted_df["week_date"] <= train_cutoff_date].copy()
    val_df = sorted_df[
        (sorted_df["week_date"] > train_cutoff_date) & (sorted_df["week_date"] <= val_cutoff_date)
    ].copy()
    test_df = sorted_df[sorted_df["week_date"] > val_cutoff_date].copy()

    logger.info(
        f"Chronological split applied:\n"
        f"  Train: {len(train_df)} rows ({train_df['week_date'].min().strftime('%Y-%m-%d')} to {train_df['week_date'].max().strftime('%Y-%m-%d')})\n"
        f"  Val:   {len(val_df)} rows ({val_df['week_date'].min().strftime('%Y-%m-%d')} to {val_df['week_date'].max().strftime('%Y-%m-%d')})\n"
        f"  Test:  {len(test_df)} rows ({test_df['week_date'].min().strftime('%Y-%m-%d')} to {test_df['week_date'].max().strftime('%Y-%m-%d')})"
    )

    return train_df, val_df, test_df
