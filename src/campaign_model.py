# Predict launch slippage days using a simple, explainable model (RandomForest). Then compute P50/P90 via quantiles of residuals.


from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["size", "dependencies_count", "approvals_count", "holiday_season_start", "channel"]

def prepare_campaign_training(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["planned_start"] = pd.to_datetime(d["planned_start"])
    d["planned_launch"] = pd.to_datetime(d["planned_launch"])
    d["actual_launch"] = pd.to_datetime(d["actual_launch"])
    d["slip_days"] = (d["actual_launch"] - d["planned_launch"]).dt.days.clip(lower=0)

    # One-hot channel
    d = pd.get_dummies(d, columns=["channel"], drop_first=True)
    return d

def train_campaign_model(campaigns: pd.DataFrame, seed: int = 7):
    d = prepare_campaign_training(campaigns[campaigns["status"] == "Launched"])
    y = d["slip_days"].values

    # Build feature matrix
    X_cols = ["size", "dependencies_count", "approvals_count", "holiday_season_start"] + \
             [c for c in d.columns if c.startswith("channel_")]
    X = d[X_cols].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=seed
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    residuals = y_test - pred

    # Use residual quantiles to estimate P50/P90 bands around point prediction
    q50 = float(np.quantile(residuals, 0.50))
    q90 = float(np.quantile(residuals, 0.90))

    return model, X_cols, {"q50_resid": q50, "q90_resid": q90}

def score_campaigns(campaigns: pd.DataFrame, model, X_cols: list[str], bands: dict) -> pd.DataFrame:
    d = prepare_campaign_training(campaigns)
    # Ensure missing one-hot columns exist
    for c in X_cols:
        if c not in d.columns:
            d[c] = 0
    X = d[X_cols].astype(float)

    point = model.predict(X)
    p50 = np.maximum(0, point + bands["q50_resid"])
    p90 = np.maximum(0, point + bands["q90_resid"])

    out = campaigns.copy()
    out["pred_slip_days_point"] = point
    out["pred_slip_days_p50"] = p50
    out["pred_slip_days_p90"] = p90
    out["risk_flag"] = out["pred_slip_days_p90"] >= 7
    return out
