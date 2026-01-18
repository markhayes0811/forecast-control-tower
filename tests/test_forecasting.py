import pandas as pd
from src.generate_data import generate
from src.forecasting import forecast_required_hours, backtest_mape

def test_forecast_output_shape_and_columns():
    tables = generate(seed=7)
    intake = tables["fact_work_intake"]
    horizon = 8

    fc = forecast_required_hours(intake, horizon_weeks=horizon)
    assert isinstance(fc, pd.DataFrame)
    assert set(["week_ending", "team", "role", "required_hours_forecast"]).issubset(fc.columns)
    assert len(fc) > 0
    assert (fc["required_hours_forecast"] >= 0).all()

def test_backtest_returns_mape():
    tables = generate(seed=7)
    intake = tables["fact_work_intake"]

    mape = backtest_mape(intake)
    # If data is sufficient, MAPE should exist for at least some team/role combos
    assert isinstance(mape, pd.DataFrame)
    if len(mape) > 0:
        assert "MAPE" in mape.columns
        assert (mape["MAPE"] >= 0).all()
