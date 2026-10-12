"""Unit tests for Stage 1 Books-Quality Anomaly Detector."""

import pandas as pd
import pytest
from src.models.books_auditor import BooksAuditor


def test_balance_sheet_gaap_audit():
    auditor = BooksAuditor()

    # Balanced
    res = auditor.audit_balance_sheet(assets=500000.0, liabilities=200000.0, equity=300000.0)
    assert res["is_balanced"] is True
    assert res["imbalance_delta"] == 0.0

    # Imbalanced
    res_err = auditor.audit_balance_sheet(assets=550000.0, liabilities=200000.0, equity=300000.0)
    assert res_err["is_balanced"] is False
    assert res_err["imbalance_delta"] == 50000.0


def test_detect_ledger_anomalies_and_ode():
    auditor = BooksAuditor()

    ledger_data = pd.DataFrame([
        {"account": "6010-Rent", "description": "Office lease payment", "amount": 4500.0},
        {"account": "6020-Payroll", "description": "Bi-weekly regular staff", "amount": 18000.0},
        {"account": "6090-Travel", "description": "Owner personal golf club and travel perk", "amount": 8500.0},
        {"account": "6050-Supplies", "description": "Office paper and staples", "amount": 250.0},
        {"account": "6095-Auto", "description": "Personal luxury vehicle lease for owner", "amount": 1600.0},
    ])

    report = auditor.execute_quality_gate(
        trial_balance={"total_assets": 100000.0, "total_liabilities": 40000.0, "total_equity": 60000.0},
        ledger_df=ledger_data,
    )

    assert report.is_passed is True
    assert report.detected_ode_addbacks >= 10000.0  # 8500 + 1600 = 10100
    assert len(report.flagged_entries) >= 2
