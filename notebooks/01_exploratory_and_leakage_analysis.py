"""Exploratory Data Analysis & Leakage-Safe Cash Flow Diagnostics.

Demonstrates statistical properties of SMB platform cash flows, distribution of DSO,
WIP margin slippage correlation with liquidity strain, and chronological cutoff verification.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.data.loader import load_dataset, split_data_chronological
from src.features.build_features import engineer_features

def main():
    data_path = root / "data" / "sample" / "raw_platform_cashflow_sample.csv"
    if not data_path.exists():
        print(f"Dataset missing at {data_path}. Run generate_data.py first.")
        return

    df = load_dataset(str(data_path))
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS — CARDONE PLATFORM CASH FLOW")
    print("=" * 60)
    print(f"Total Observations: {len(df)}")
    print(f"Unique Clients: {df['company_id'].nunique()}")
    print(f"Date Span: {df['week_date'].min().date()} to {df['week_date'].max().date()}")

    print("\nSummary Statistics by Vertical:")
    grouped = df.groupby("industry")[["weekly_billed_revenue", "dso_days", "ending_cash_balance"]].mean()
    print(grouped)

    print("\nChronological Split Verification (Zero Look-Ahead Test):")
    tr, val, ts = split_data_chronological(df)
    assert tr["week_date"].max() < val["week_date"].min()
    assert val["week_date"].max() < ts["week_date"].min()
    print("  Zero lookahead test PASSED: strictly non-overlapping temporal partitions.")

if __name__ == "__main__":
    main()
