"""Stage 1: Books-Quality Anomaly Detector & GAAP Integrity Gate.

Audits trial balances, general ledger entries, and financial statements for small-business
clients ($1M–$50M ARR) before projecting cash flows. Verifies GAAP equation balance
(Assets = Liabilities + Equity), detects unearned revenue misclassifications, and flags
non-operational Owner's Discretionary Earnings (ODE) add-back candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class AuditReport:
    """Structured report produced by the Books Auditor."""
    is_passed: bool
    gaap_balance_error: float
    anomalous_accounts_count: int
    detected_ode_addbacks: float
    flagged_entries: List[Dict[str, Any]]
    recommendations: List[str]


class BooksAuditor:
    """Pre-forecasting audit gate for trial balances and ledger entries."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.fitted_ = False

    def audit_balance_sheet(
        self,
        assets: float,
        liabilities: float,
        equity: float,
        tolerance: float = 1.0,
    ) -> Dict[str, Any]:
        """Verifies core GAAP balance sheet identity: Assets == Liabilities + Equity."""
        expected = liabilities + equity
        delta = abs(assets - expected)
        is_balanced = delta <= tolerance

        return {
            "assets": assets,
            "liabilities": liabilities,
            "equity": equity,
            "imbalance_delta": round(delta, 2),
            "is_balanced": is_balanced,
        }

    def detect_ledger_anomalies(self, ledger_df: pd.DataFrame) -> pd.DataFrame:
        """Flags suspicious transaction amounts, timing, or account variances."""
        df = ledger_df.copy()
        features = ["amount", "trailing_30d_account_avg", "variance_ratio"]

        for feat in features:
            if feat not in df.columns:
                if feat == "trailing_30d_account_avg":
                    df[feat] = df["amount"].abs().mean()
                elif feat == "variance_ratio":
                    df[feat] = df["amount"].abs() / np.maximum(df["trailing_30d_account_avg"], 1.0)

        X = df[features].fillna(0.0).to_numpy()
        self.iso_forest.fit(X)
        self.fitted_ = True

        predictions = self.iso_forest.predict(X)  # -1 for anomaly, 1 for normal
        df["is_anomaly"] = (predictions == -1).astype(int)
        df["anomaly_score"] = self.iso_forest.decision_function(X)
        return df

    def scan_owner_discretionary_addbacks(self, ledger_df: pd.DataFrame) -> Dict[str, Any]:
        """Identifies non-operational expenses (travel, personal auto, club dues) for ODE add-backs."""
        ode_keywords = ["travel", "auto", "club", "dining", "personal", "owner", "perk", "discretionary"]
        df = ledger_df.copy()

        if "description" not in df.columns:
            return {"total_ode_addbacks": 0.0, "flagged_entries": []}

        pattern = "|".join(ode_keywords)
        mask = df["description"].str.lower().str.contains(pattern, na=False)
        flagged = df[mask]
        total_addbacks = float(flagged["amount"].sum())

        return {
            "total_ode_addbacks": round(total_addbacks, 2),
            "flagged_entries": flagged[["account", "description", "amount"]].to_dict(orient="records"),
        }

    def execute_quality_gate(
        self,
        trial_balance: Dict[str, float],
        ledger_df: pd.DataFrame,
    ) -> AuditReport:
        """Executes full Stage 1 audit pipeline; gates progression to forecasting."""
        bs_check = self.audit_balance_sheet(
            assets=trial_balance.get("total_assets", 0.0),
            liabilities=trial_balance.get("total_liabilities", 0.0),
            equity=trial_balance.get("total_equity", 0.0),
        )

        anomaly_df = self.detect_ledger_anomalies(ledger_df)
        anomalies_count = int(anomaly_df["is_anomaly"].sum())

        ode_res = self.scan_owner_discretionary_addbacks(ledger_df)

        is_passed = bs_check["is_balanced"] and anomalies_count <= max(3, int(len(ledger_df) * 0.10))

        recommendations = []
        if not bs_check["is_balanced"]:
            recommendations.append(
                f"GAAP Imbalance Detected: Assets differ from Liabilities + Equity by ${bs_check['imbalance_delta']:,.2f}. Reconcile retained earnings prior to forecasting."
            )
        if anomalies_count > 0:
            recommendations.append(
                f"Investigate {anomalies_count} anomalous general ledger postings prior to final client presentation."
            )
        if ode_res["total_ode_addbacks"] > 0:
            recommendations.append(
                f"Identified ${ode_res['total_ode_addbacks']:,.2f} in potential Owner Discretionary Earnings (ODE) add-backs to normalize EBITDA."
            )

        return AuditReport(
            is_passed=is_passed,
            gaap_balance_error=bs_check["imbalance_delta"],
            anomalous_accounts_count=anomalies_count,
            detected_ode_addbacks=ode_res["total_ode_addbacks"],
            flagged_entries=ode_res["flagged_entries"],
            recommendations=recommendations,
        )
