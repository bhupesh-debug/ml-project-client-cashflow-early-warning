"""CLI script to generate evaluation figures and performance reports."""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_dataset, split_data_chronological
from src.features.build_features import FEATURE_COLUMNS, get_feature_and_target_matrices


def main():
    print("Generating evaluation figures and performance visual artifacts...")
    data_path = project_root / "data" / "sample" / "raw_platform_cashflow_sample.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run generate_data.py first.")

    df = load_dataset(str(data_path))
    _, _, test_df = split_data_chronological(df, train_ratio=0.70, val_ratio=0.15)
    X_test, y_reg_test, y_clf_test, feature_names = get_feature_and_target_matrices(test_df)

    models_dir = project_root / "reports" / "models"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    hgbr = joblib.load(models_dir / "cashflow_regressor_hgbr.joblib")
    hgbc = joblib.load(models_dir / "breach_classifier_hgbc.joblib")

    y_reg_pred = hgbr.predict(X_test)
    y_clf_prob = hgbc.predict_proba(X_test)[:, 1]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Actual vs Predicted Cash Flow
    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(y_reg_test / 1e3, y_reg_pred / 1e3, alpha=0.35, color="#1A3A6B", edgecolors="none")
    min_val = min(y_reg_test.min(), y_reg_pred.min()) / 1e3
    max_val = max(y_reg_test.max(), y_reg_pred.max()) / 1e3
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Forecast (1:1)")
    plt.title("Actual vs. Predicted 13-Week Net Cash Flow ($ thousands)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual 13-Week Cumulative Cash Flow ($k)", fontsize=11)
    plt.ylabel("Predicted 13-Week Cumulative Cash Flow ($k)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig1_path = figures_dir / "actual_vs_predicted_cashflow.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Residuals Distribution
    residuals = (y_reg_test - y_reg_pred) / 1e3
    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(residuals, bins=40, color="#C9A84C", edgecolor="#0A1628", alpha=0.85)
    plt.axvline(0, color="#0A1628", linestyle="--", linewidth=1.5)
    plt.title("Forecasting Error Residual Distribution (Actual - Predicted, $k)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Forecast Error ($ thousands)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    fig2_path = figures_dir / "residuals_distribution.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: ROC Curve & PR Curve
    fpr, tpr, _ = roc_curve(y_clf_test, y_clf_prob)
    prec, rec, _ = precision_recall_curve(y_clf_test, y_clf_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax1.plot(fpr, tpr, color="#1A3A6B", lw=2, label="HistGB Classifier")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_title("ROC Curve: 45-Day Daily Cash Breach", fontsize=11, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color="#2E7D32", lw=2, label="PR Curve")
    ax2.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    fig3_path = figures_dir / "roc_pr_curves.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Figure 4: Permutation Feature Importance
    print("Calculating permutation feature importance...")
    perm_imp = permutation_importance(hgbr, X_test, y_reg_test, n_repeats=5, random_state=42)
    sorted_idx = perm_imp.importances_mean.argsort()[-10:]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.barh(np.array(FEATURE_COLUMNS)[sorted_idx], perm_imp.importances_mean[sorted_idx], color="#1A3A6B")
    plt.title("Top 10 Feature Importances (Permutation Metric Drop)", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel("Importance Mean")
    plt.tight_layout()
    fig4_path = figures_dir / "feature_importance.png"
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")


if __name__ == "__main__":
    main()
