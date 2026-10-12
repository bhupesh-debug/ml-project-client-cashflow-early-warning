"""CLI script to run model training and cross-validation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_dataset
from src.models.train import train_and_evaluate_all


def main():
    data_path = project_root / "data" / "sample" / "raw_platform_cashflow_sample.csv"
    if not data_path.exists():
        print(f"Data not found at {data_path}. Generating data first...")
        from src.data.generate_synthetic_data import generate_full_dataset
        df = generate_full_dataset(num_companies=100, weeks_history=104, seed=42)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = load_dataset(str(data_path))

    artifacts_dir = project_root / "reports" / "models"
    print("Beginning model training and cross-validation...")
    results = train_and_evaluate_all(df, artifacts_dir=str(artifacts_dir))

    print("\n" + "=" * 50)
    print("TRAINING & EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Test Samples: {results['test_sample_size']}")
    print("\nRegression (13-Week Net Cash Flow):")
    for model_name, metrics in results["regression"].items():
        print(f"  {model_name}: MAE=${metrics['MAE']:,.2f} | RMSE=${metrics['RMSE']:,.2f} | R2={metrics['R2']} | MDA={metrics['MDA']:.2%}")

    print("\nClassification (45-Day Daily Cash Target Breach):")
    for model_name, metrics in results["classification"].items():
        print(f"  {model_name}: ROC-AUC={metrics['ROC_AUC']} | PR-AUC={metrics['PR_AUC']} | F1={metrics['F1']} | Precision={metrics['Precision']} | Recall={metrics['Recall']}")


if __name__ == "__main__":
    main()
