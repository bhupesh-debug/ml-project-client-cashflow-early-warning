# Model Card: Cardone Platform CashFlow Guard (CP-PCG)

## Model Details
- **Developer:** Bhupesh Chandra Dimri (MS Accounting Analytics, Pace University; MBA Finance)
- **Target Role Alignment:** Finance Manager (10X360 Platform Review), Cardone Ventures — Scottsdale, AZ
- **Model Version:** 1.0.0
- **Model Architecture:**
  - Continuous Forecast Engine: `HistGradientBoostingRegressor` (Ensemble of gradient-boosted decision trees with histogram binning for non-linear feature interactions).
  - Liquidity Breach Classifier: `HistGradientBoostingClassifier` (Calibrated gradient-boosted binary classifier predicting probability of cash dipping below Daily Cash Target within 45 days).
  - Benchmarks: `NaiveRollingRunRateRegressor` ($13 \times \text{4-week trailing mean}$), `Ridge` Linear Regressor, and `HeuristicLiquidityRuleClassifier`.
- **Frameworks:** Python 3.14, Scikit-Learn 1.9.1, NumPy 2.5, Pandas 3.0, PyTest 9.1.
- **Serialization:** Joblib compressed model binaries (`reports/models/*.joblib`).

---

## Intended Use & Target Users
- **Primary Use Case:** Proactive financial diagnostic intelligence for SMB platform clients ($1.8M–$28M ARR). Predicts 13-week cumulative cash flow and generates an early-warning signal 45 days ahead of an impending Daily Cash Target liquidity crunch.
- **Intended Users:**
  1. **Finance Managers:** Leading the 7-month Platform Review deep-dive and weekly cash-flow pacing calls.
  2. **Core Triad (Marketing Manager & BAM):** Coordinating customer acquisition spend (allowable CAC) and operational delivery against forecasted cash buffers.
  3. **SMB Founders:** Replacing emotional bank-balance checkbook management with normalized accrual and cash pacing.
- **Out-of-Scope Uses:** High-frequency algorithmic intraday trading; automated bankruptcy filings without human forensic review; replacing formal GAAP audits.

---

## Factors & Subpopulations
- **Industry Verticals Simulated:**
  1. *Commercial Contracting:* High WIP intensity, heavy progress billing delays, extended DSO (45–75 days).
  2. *Professional Services:* Moderate WIP intensity, labor utilization dependency, DSO (35–60 days).
  3. *Healthcare & Dental:* Low WIP, high fixed overhead/equipment leases, rapid collections (25–45 days).
  4. *B2B SaaS & Tech Services:* High gross margin, low WIP, recurring billing dynamics.

---

## Training & Evaluation Data Provenance
- **Dataset Provenance:** 100 SMB platform clients across 104 weekly observations (9,100 total rows). Synthetically generated via parameterized stochastic accounting dynamics to protect proprietary client confidentiality.
- **Zero Look-Ahead Partitioning:**
  - *Train Set:* 6,300 weekly records (January 5, 2024 – March 14, 2025; 70% chronological cutoff).
  - *Validation Set:* 1,400 weekly records (March 21, 2025 – June 20, 2025; 15% chronological cutoff).
  - *Out-of-Sample Test Set:* 1,400 weekly records (June 27, 2025 – September 26, 2025; 15% chronological cutoff).
  - **Leakage Prevention:** All rolling statistics (`shift(1)`) are strictly computed on historical periods with closed-left logic. Zero future data is present in feature matrices.

---

## Measured Performance & Benchmark Results

### 1. 13-Week Net Cumulative Cash Flow Regression (Test N = 1,400)
| Model Architecture | MAE ($) | RMSE ($) | R² Score | Mean Directional Acc (MDA) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Rolling Run-Rate Baseline** | $342,894.20 | $500,514.53 | 0.8262 | 89.36% | 217.11% |
| **Ridge Linear Pipeline** | $360,339.35 | $505,737.80 | 0.8226 | 94.43% | 192.44% |
| **HistGradientBoosting Ensemble (Final)** | **$263,271.62** | **$385,354.51** | **0.8970** | **95.64%** | **130.07%** |

*Improvement:* The gradient-boosted ensemble reduces forecast MAE by **$79,622 (23.2% error reduction)** over the standard financial run-rate heuristic, achieving **95.64% directional accuracy**.

### 2. 45-Day Daily Cash Target Breach Classification (Test N = 1,400)
| Model Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Liquidity Rule Baseline** | 0.9943 | 0.9859 | 1.0000 | 0.9929 | 1.0000 | 1.0000 | 0.0044 |
| **Logistic Regression Pipeline** | 0.9979 | 0.9964 | 0.9982 | 0.9973 | 1.0000 | 1.0000 | 0.0043 |
| **HistGradientBoosting Classifier (Final)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0001** |

---

## Feature Drivers & Interpretability
Permutation feature importance confirms that model decisions align with operational financial theory:
1. `burn_coverage_weeks` (Current cash / weekly outflows)
2. `cash_to_daily_target_ratio` (Buffer margin relative to 60-day target)
3. `outflows_roll_mean_4w` (Trailing expenditure commitments)
4. `wip_underbilling_ratio` (Unbilled work tying up liquidity)
5. `dso_days` (Accounts receivable collection velocity)

---

## Limitations, Biases & Responsible Use
1. **Correlation vs. Causation:** The model flags clients at high risk of a cash crunch; however, the model does not fix the cash crunch. The Finance Manager must execute the prescriptive **TAPW protocol** (Target cash enforcement, AR acceleration, AP triage, and WIP milestone billing).
2. **Synthetic Data Boundary:** Performance figures are derived from high-fidelity simulated SMB ledgers. When deploying to live client QuickBooks/NetSuite instances, model retraining and domain adaptation are recommended.
3. **Macroeconomic Shocks:** Unprecedented macro freezes (e.g., pandemic supply-chain halts) require human stress-test scenario adjustments.
