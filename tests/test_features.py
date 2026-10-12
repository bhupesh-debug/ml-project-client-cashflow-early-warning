"""Unit tests for feature engineering."""

import numpy as np
import pytest

from src.data.generate_synthetic_data import generate_full_dataset
from src.features.build_features import (
    FEATURE_COLUMNS,
    engineer_features,
    get_feature_and_target_matrices,
)


def test_engineer_features():
    df = generate_full_dataset(num_companies=3, weeks_history=25, seed=42)
    feat_df = engineer_features(df)

    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
        assert not feat_df[col].isnull().any(), f"NaN values in feature column: {col}"
        assert not np.isinf(feat_df[col]).any(), f"Infinite values in feature column: {col}"


def test_get_feature_and_target_matrices():
    df = generate_full_dataset(num_companies=3, weeks_history=25, seed=42)
    X, y_reg, y_clf, names = get_feature_and_target_matrices(df)

    assert X.shape[0] == len(df)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert len(y_reg) == len(df)
    assert len(y_clf) == len(df)
    assert set(np.unique(y_clf)).issubset({0, 1})
