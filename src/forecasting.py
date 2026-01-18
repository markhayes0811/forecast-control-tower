from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def _fit_hw(series: pd.Series):
    # weekly series -> season length 52
    model = ExponentialSmoothing(
        series.astype(float),
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated"
    )
    return model.fit(optimized=True)

def forecast_required_hours(intake: pd.DataFrame, horizon_weeks: int = 12) -> pd.DataFrame:
    df = intake.copy()
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    out_rows = []

    for (team, role), g in df.groupby(["team", "role"]):
        g = g.sort_values("week_ending")
        y = g.set_index("week_ending")["required_hours"].asfreq("W-SUN")
        y = y.interpolate(limit_direction="both")

        if len(y) < 60:
            # fallback: naive
            fc = pd.Series([y.iloc[-1]] * horizon_weeks,
                           index=pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon_weeks, freq="W-SUN"))
        else:
            fit = _fit_hw(y)
            fc = fit.forecast(horizon_weeks)

        for dt, val in fc.items():
            out_rows.append({
                "week_ending": dt.date(),
                "team": team,
                "role": role,
                "required_hours_forecast": float(max(0.0, val))
            })

    return pd.DataFrame(out_rows)

def backtest_mape(intake: pd.DataFrame, min_train_weeks: int = 80, test_weeks: int = 12) -> pd.DataFrame:
    df = intake.copy()
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    rows = []

    for (team, role), g in df.groupby(["team", "role"]):
        g = g.sort_values("week_ending")
        y = g.set_index("week_ending")["required_hours"].asfreq("W-SUN").interpolate(limit_direction="both")
        if len(y) < (min_train_weeks + test_weeks):
            continue

        train = y.iloc[:-test_weeks]
        test = y.iloc[-test_weeks:]
        try:
            fit = _fit_hw(train)
            pred = fit.forecast(test_weeks)
        except Exception:
            pred = pd.Series([train.iloc[-1]] * test_weeks, index=test.index)

        mape = float((np.abs((test - pred) / np.maximum(test, 1e-6))).mean())
        rows.append({"team": team, "role": role, "MAPE": mape})

    return pd.DataFrame(rows).sort_values("MAPE")
