# Capacity & Campaign Timeline Forecasting Control Tower (Streamlit)

A Business Analyst-oriented analytics tool that forecasts weekly capacity requirements by team/role and predicts marketing campaign timeline slippage. Includes data quality validation, model backtesting, scenario planning, and monitoring.

## Why this project
Business stakeholders need:
- Staffing plans (capacity vs demand) by team and role
- Early warning for campaign launch delays
- Transparent checks for data integrity and model reliability

## What it demonstrates
- Time-series forecasting (Holt-Winters baseline) + backtesting (MAPE)
- Campaign timeline risk prediction (supervised model + P50/P90 bands)
- Scenario planning (volume, shrinkage, headcount)
- Data integrity checks (schema, nulls, ranges, referential integrity)
- Model monitoring + retraining triggers

## Run locally
> Recommended Python: 3.11–3.12

```bash
pip install -r requirements.txt
streamlit run app.py
