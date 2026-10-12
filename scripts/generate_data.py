"""CLI script to generate synthetic SMB platform cash flow data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.generate_synthetic_data import generate_full_dataset
from src.features.build_features import engineer_features


def main():
    print("Generating synthetic platform cash flow and WIP dataset...")
    df = generate_full_dataset(num_companies=100, weeks_history=104, seed=42)

    data_dir = project_root / "data" / "sample"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / "raw_platform_cashflow_sample.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset: {raw_path} ({len(df)} records)")

    print("Engineering features...")
    feat_df = engineer_features(df)
    proc_path = data_dir / "processed_features.csv"
    feat_df.to_csv(proc_path, index=False)
    print(f"Saved processed features: {proc_path} ({len(feat_df)} records)")


if __name__ == "__main__":
    main()
