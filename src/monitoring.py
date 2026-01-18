from __future__ import annotations
import pandas as pd
import numpy as np

def weekly_forecast_error(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    a = actual[["week_ending", "team", "role", "required_hours"]].copy()
    f = forecast[["week_ending", "team", "role", "required_hours_forecast"]].copy()
    a["week_ending"] = pd.to_datetime(a["week_ending"])
    f["week_ending"] = pd.to_datetime(f["week_ending"])

    m = a.merge(f, on=["week_ending", "team", "role"], how="inner")
    m["abs_pct_err"] = (m["required_hours"] - m["required_hours_forecast"]).abs() / m["required_hours"].clip(lower=1e-6)
    return m

def retrain_trigger(errors: pd.DataFrame, threshold_mape: float = 0.18) -> dict:
    # simple: aggregate last 4 weeks MAPE
    errors = errors.sort_values("week_ending")
    recent = errors.groupby(["team", "role"]).tail(4)
    mape = recent.groupby(["team", "role"])["abs_pct_err"].mean().reset_index(name="MAPE_4w")
    flagged = mape[mape["MAPE_4w"] >= threshold_mape]
    return {
        "threshold_mape": threshold_mape,
        "flagged_count": int(len(flagged)),
        "flagged": flagged
    }
