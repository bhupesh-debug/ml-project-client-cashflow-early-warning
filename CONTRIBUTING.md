# Contributing Guidelines

Thank you for your interest in contributing to **Cardone Platform CashFlow Guard**!

## Development Workflow
1. Fork the repository and create your feature branch: `git checkout -b feature/new-financial-metric`
2. Ensure you have installed the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the full test suite before committing:
   ```bash
   pytest tests/ -v
   ```
4. Verify that no future look-ahead bias is introduced into feature engineering (`src/features/build_features.py`). All rolling windows must use trailing slices only (`shift(1)`).
5. Submit a pull request with clear description and test logs.
