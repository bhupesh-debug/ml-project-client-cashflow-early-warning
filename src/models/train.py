"""Model Training & Cross-Validation Pipeline.

Implements chronological time-series cross-validation and trains both baseline
and improved gradient-boosted ensembles for cash flow regression and breach classification.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import split_data_chronological
from src.evaluation.metrics import evaluate_classification, evaluate_regression
from src.features.build_features import FEATURE_COLUMNS, get_feature_and_target_matrices
from src.models.baseline import HeuristicLiquidityRuleClassifier, NaiveRollingRunRateRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_time_series_cv(
    X_train: np.ndarray,
    y_reg_train: np.ndarray,
    y_clf_train: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Runs rolling-origin TimeSeriesSplit cross-validation on the training set."""
    logger.info(f"Conducting {n_splits}-fold Rolling-Origin Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    reg_cv_scores = []
    clf_cv_scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, y_r_tr, y_c_tr = X_train[tr_idx], y_reg_train[tr_idx], y_clf_train[tr_idx]
        X_val, y_r_val, y_c_val = X_train[val_idx], y_reg_train[val_idx], y_clf_train[val_idx]

        # Regressor fold
        reg_model = HistGradientBoostingRegressor(max_iter=100, random_state=42 + fold)
        reg_model.fit(X_tr, y_r_tr)
        preds_r = reg_model.predict(X_val)
        mae = float(np.mean(np.abs(y_r_val - preds_r)))
        reg_cv_scores.append(mae)

        # Classifier fold
        clf_model = HistGradientBoostingClassifier(max_iter=100, random_state=42 + fold)
        clf_model.fit(X_tr, y_c_tr)
        probs_c = clf_model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y_c_val, probs_c))
        clf_cv_scores.append(roc)

    return {
        "reg_cv_mae_mean": round(float(np.mean(reg_cv_scores)), 2),
        "reg_cv_mae_std": round(float(np.std(reg_cv_scores)), 2),
        "clf_cv_roc_mean": round(float(np.mean(clf_cv_scores)), 4),
        "clf_cv_roc_std": round(float(np.std(clf_cv_scores)), 4),
    }


def train_and_evaluate_all(
    df: pd.DataFrame,
    artifacts_dir: str = "reports/models",
) -> Dict[str, Any]:
    """Trains baselines and improved ML models; produces test set evaluation table and serializes artifacts."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Chronological partition
    train_df, val_df, test_df = split_data_chronological(df, train_ratio=0.70, val_ratio=0.15)

    X_train, y_reg_train, y_clf_train, feature_names = get_feature_and_target_matrices(train_df)
    X_val, y_reg_val, y_clf_val, _ = get_feature_and_target_matrices(val_df)
    X_test, y_reg_test, y_clf_test, _ = get_feature_and_target_matrices(test_df)

    cv_results = run_time_series_cv(X_train, y_reg_train, y_clf_train, n_splits=5)
    logger.info(f"CV Results: {cv_results}")

    # =========================================================================
    # 1. REGRESSION MODELS (13-Week Net Cash Flow)
    # =========================================================================
    logger.info("Training Regression Models...")
    # Baseline
    baseline_reg = NaiveRollingRunRateRegressor(ncf_roll_idx=FEATURE_COLUMNS.index("ncf_roll_mean_4w"))
    baseline_reg.fit(X_train, y_reg_train)
    y_reg_pred_base = baseline_reg.predict(X_test)
    eval_reg_base = evaluate_regression(y_reg_test, y_reg_pred_base)

    # Linear Ridge
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=25.0)),
    ])
    ridge_pipeline.fit(X_train, y_reg_train)
    y_reg_pred_ridge = ridge_pipeline.predict(X_test)
    eval_reg_ridge = evaluate_regression(y_reg_test, y_reg_pred_ridge)

    # Improved: HistGradientBoostingRegressor
    hgbr = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    hgbr.fit(X_train, y_reg_train)
    y_reg_pred_hgbr = hgbr.predict(X_test)
    eval_reg_hgbr = evaluate_regression(y_reg_test, y_reg_pred_hgbr)

    # =========================================================================
    # 2. CLASSIFICATION MODELS (45-Day Daily Cash Target Breach)
    # =========================================================================
    logger.info("Training Classification Models...")
    # Baseline Heuristic
    baseline_clf = HeuristicLiquidityRuleClassifier(
        burn_cov_idx=FEATURE_COLUMNS.index("burn_coverage_weeks"),
        threshold_weeks=4.0,
    )
    baseline_clf.fit(X_train, y_clf_train)
    y_clf_pred_base = baseline_clf.predict(X_test)
    y_clf_prob_base = baseline_clf.predict_proba(X_test)[:, 1]
    eval_clf_base = evaluate_classification(y_clf_test, y_clf_pred_base, y_clf_prob_base)

    # Logistic Regression
    logit_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])
    logit_pipeline.fit(X_train, y_clf_train)
    y_clf_pred_logit = logit_pipeline.predict(X_test)
    y_clf_prob_logit = logit_pipeline.predict_proba(X_test)[:, 1]
    eval_clf_logit = evaluate_classification(y_clf_test, y_clf_pred_logit, y_clf_prob_logit)

    # Improved: HistGradientBoostingClassifier
    hgbc = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    hgbc.fit(X_train, y_clf_train)
    y_clf_pred_hgbc = hgbc.predict(X_test)
    y_clf_prob_hgbc = hgbc.predict_proba(X_test)[:, 1]
    eval_clf_hgbc = evaluate_classification(y_clf_test, y_clf_pred_hgbc, y_clf_prob_hgbc)

    # Save artifacts
    logger.info(f"Serializing trained model artifacts to {artifacts_dir}...")
    joblib.dump(hgbr, os.path.join(artifacts_dir, "cashflow_regressor_hgbr.joblib"))
    joblib.dump(ridge_pipeline, os.path.join(artifacts_dir, "cashflow_regressor_ridge.joblib"))
    joblib.dump(hgbc, os.path.join(artifacts_dir, "breach_classifier_hgbc.joblib"))
    joblib.dump(logit_pipeline, os.path.join(artifacts_dir, "breach_classifier_logit.joblib"))

    results_summary = {
        "cross_validation": cv_results,
        "regression": {
            "Naive_Rolling_RunRate": eval_reg_base,
            "Ridge_Linear": eval_reg_ridge,
            "HistGradientBoosting_Ensemble": eval_reg_hgbr,
        },
        "classification": {
            "Heuristic_Rule": eval_clf_base,
            "Logistic_Regression": eval_clf_logit,
            "HistGradientBoosting_Ensemble": eval_clf_hgbc,
        },
        "test_sample_size": int(len(X_test)),
        "feature_count": len(FEATURE_COLUMNS),
    }

    metrics_file = Path(artifacts_dir).parent / "metrics_summary.json"
    with open(metrics_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Saved evaluation metrics to {metrics_file}")
    return results_summary
