# Dataset Documentation: SMB Platform Cash Flow & WIP Ledger

## Provenance & Overview
This directory contains sample historical weekly operational and accounting records modeled on small-to-medium businesses ($1.8M–$28M ARR). The data simulates realistic transactional outputs from QuickBooks Online, NetSuite, and project-based WIP schedules.

> [!NOTE]
> **Synthetic Data Label:** To respect private company NDA constraints, all datasets in `data/sample/` are algorithmically synthesized via parameter distributions across 100 SMB platform clients over a 104-week horizon (9,100 records). They are not actual private company accounts.

## Schema Definition
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `company_id` | String | Unique identifier for each SMB platform client (`SMB-0001` to `SMB-0100`) |
| `industry` | Categorical | Vertical: Contracting, Professional Services, Healthcare, B2B SaaS |
| `week_date` | Date | Friday weekly date stamp (Jan 2024 to Sep 2025) |
| `weekly_billed_revenue` | Float | Accrual revenue billed during the week ($) |
| `cash_collections` | Float | Cash collected from receivables ($) |
| `payroll_outflow` | Float | Payroll and benefits cash disbursement ($) |
| `vendor_outflow` | Float | Vendor and COGS operational disbursements ($) |
| `ending_cash_balance` | Float | Bank ending cash balance ($) |
| `ar_balance` | Float | Ending Accounts Receivable ledger balance ($) |
| `dso_days` | Float | Days Sales Outstanding metric |
| `daily_cash_target` | Float | Objective 60-day safety liquidity buffer target ($) |
| `wip_contract_value` | Float | Total active contract value ($) |
| `wip_underbillings` | Float | Unbilled earned work asset (Revenue Earned - Billed) |
| `wip_overbillings` | Float | Billed in excess of earned work liability |
| `target_13wk_net_cash_flow` | Float | Ground-truth forward 13-week cumulative net cash flow ($) |
| `target_target_breach_45d` | Binary (0/1) | Ground-truth indicator if cash falls below Daily Cash Target in next 45 days |
