"""Unit tests for synthetic data generation and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_synthetic_data import (
    generate_company_profile,
    generate_company_timeseries,
    generate_full_dataset,
)
from src.data.loader import split_data_chronological, validate_schema


def test_generate_company_profile():
    rng = np.random.default_rng(42)
    profile = generate_company_profile(1, rng)
    assert profile["company_id"] == "SMB-0001"
    assert profile["annual_revenue"] > 1e6
    assert 0.20 <= profile["gross_margin"] <= 0.85
    assert profile["fixed_opex_weekly"] > 0


def test_generate_company_timeseries():
    rng = np.random.default_rng(42)
    profile = generate_company_profile(1, rng)
    df = generate_company_timeseries(profile, num_weeks=30, start_date="2024-01-05", rng=rng)
    assert len(df) > 0
    assert "target_13wk_net_cash_flow" in df.columns
    assert "target_target_breach_45d" in df.columns
    assert not df["target_13wk_net_cash_flow"].isnull().any()


def test_chronological_split_no_leakage():
    df = generate_full_dataset(num_companies=5, weeks_history=30, seed=42)
    validate_schema(df)
    train_df, val_df, test_df = split_data_chronological(df, train_ratio=0.6, val_ratio=0.2)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    max_train_date = train_df["week_date"].max()
    min_val_date = val_df["week_date"].min()
    max_val_date = val_df["week_date"].max()
    min_test_date = test_df["week_date"].min()

    # Strict chronological boundary check
    assert max_train_date < min_val_date
    assert max_val_date < min_test_date
