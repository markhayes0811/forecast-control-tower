from __future__ import annotations
import numpy as np
import pandas as pd

TEAMS = ["Onboarding", "Fraud Ops", "Disputes", "Servicing"]
ROLES = ["Analyst", "Senior Analyst", "QA"]
CHANNELS = ["Email", "Paid Search", "Display", "Affiliate"]

def _make_calendar(start: str = "2024-01-01", weeks: int = 120) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=weeks * 7, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["year"] = cal["date"].dt.year
    cal["dow"] = cal["date"].dt.dayofweek
    cal["month"] = cal["date"].dt.month
    # Simple holiday proxy: last week of Nov + last 2 weeks of Dec
    cal["holiday_season"] = ((cal["date"].dt.month == 11) & (cal["date"].dt.day >= 20)) | (
        (cal["date"].dt.month == 12) & (cal["date"].dt.day >= 15)
    )
    return cal

def generate(seed: int = 7, start: str = "2024-01-01", weeks: int = 120) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cal = _make_calendar(start=start, weeks=weeks)

    dim_team_role = []
    for t in TEAMS:
        for r in ROLES:
            dim_team_role.append({"team": t, "role": r})
    dim_team_role = pd.DataFrame(dim_team_role)

    # Capacity table (weekly)
    wk = cal.resample("W-SUN", on="date").last().reset_index()
    rows = []
    for _, tr in dim_team_role.iterrows():
        base_hc = {"Analyst": 18, "Senior Analyst": 7, "QA": 4}[tr["role"]]
        team_adj = {"Onboarding": 1.1, "Fraud Ops": 1.2, "Disputes": 0.95, "Servicing": 1.0}[tr["team"]]
        headcount = np.clip(rng.normal(base_hc * team_adj, 1.5, size=len(wk)).round(), 1, None)

        shrinkage = np.clip(rng.normal(0.28, 0.04, size=len(wk)), 0.12, 0.45)
        hours_per_person = 37.5

        for i, wrow in wk.iterrows():
            rows.append({
                "week_ending": wrow["date"].date(),
                "team": tr["team"],
                "role": tr["role"],
                "headcount": int(headcount[i]),
                "hours_per_person": hours_per_person,
                "shrinkage": float(shrinkage[i]),
            })
    fact_capacity = pd.DataFrame(rows)
    fact_capacity["available_hours"] = (
        fact_capacity["headcount"] * fact_capacity["hours_per_person"] * (1 - fact_capacity["shrinkage"])
    )

    # Campaigns (random launches + complexity drivers)
    n_campaigns = 260
    start_dates = rng.choice(cal["date"].values, size=n_campaigns, replace=False)
    start_dates = pd.to_datetime(start_dates).sort_values()

    campaigns = []
    for i, s in enumerate(start_dates):
        channel = rng.choice(CHANNELS)
        size = rng.integers(1, 6)  # 1 small .. 5 big
        deps = rng.integers(0, 4)
        approvals = rng.integers(1, 7)

        planned_duration = int(rng.normal(14 + size * 3, 4))
        planned_launch = s + pd.Timedelta(days=max(5, planned_duration))

        # Actual slippage depends on size, deps, approvals, and holiday season
        holiday = bool(cal.loc[cal["date"] == s, "holiday_season"].iloc[0])
        slip = max(0, int(rng.normal(size * 1.5 + deps * 2 + approvals * 0.8 + (4 if holiday else 0), 3)))
        actual_launch = planned_launch + pd.Timedelta(days=slip)

        status = "Launched" if actual_launch < cal["date"].max() else "Planned"

        campaigns.append({
            "campaign_id": f"C{i+1:04d}",
            "channel": channel,
            "planned_start": s.date(),
            "planned_launch": planned_launch.date(),
            "actual_launch": actual_launch.date(),
            "status": status,
            "size": int(size),
            "dependencies_count": int(deps),
            "approvals_count": int(approvals),
            "holiday_season_start": holiday,
        })
    fact_campaigns = pd.DataFrame(campaigns)

    # Work intake: weekly volumes tied to seasonality + campaigns
    wk_dates = wk["date"].dt.date.values
    intake_rows = []
    for _, tr in dim_team_role.iterrows():
        role_mult = {"Analyst": 1.0, "Senior Analyst": 0.55, "QA": 0.25}[tr["role"]]
        team_base = {"Onboarding": 520, "Fraud Ops": 680, "Disputes": 430, "Servicing": 560}[tr["team"]]
        trend = np.linspace(0.9, 1.1, len(wk_dates))

        for j, we in enumerate(wk_dates):
            # seasonality (annual-ish) + noise
            seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * j / 52)
            holiday_boost = 1.18 if (pd.Timestamp(we).month in [11, 12]) else 1.0

            # campaign-driven spikes: count campaigns starting that week
            week_start = pd.Timestamp(we) - pd.Timedelta(days=6)
            week_end = pd.Timestamp(we)
            c_count = ((pd.to_datetime(fact_campaigns["planned_start"]) >= week_start) &
                       (pd.to_datetime(fact_campaigns["planned_start"]) <= week_end)).sum()
            campaign_boost = 1.0 + min(0.25, c_count / 1000)

            mean_items = team_base * role_mult * seasonal * holiday_boost * campaign_boost * trend[j]
            work_items = max(0, int(rng.normal(mean_items, mean_items * 0.08)))

            aht = {"Analyst": 0.55, "Senior Analyst": 0.75, "QA": 0.35}[tr["role"]]  # hours/item
            aht = float(np.clip(rng.normal(aht, 0.06), 0.2, 1.5))

            intake_rows.append({
                "week_ending": we,
                "team": tr["team"],
                "role": tr["role"],
                "work_items": work_items,
                "avg_handle_time_hours": aht
            })

    fact_work_intake = pd.DataFrame(intake_rows)
    fact_work_intake["required_hours"] = (
        fact_work_intake["work_items"] * fact_work_intake["avg_handle_time_hours"]
    )

    dim_calendar = cal.copy()
    dim_calendar["date"] = dim_calendar["date"].dt.date

    return {
        "dim_calendar": dim_calendar,
        "dim_team_role": dim_team_role,
        "fact_capacity": fact_capacity,
        "fact_work_intake": fact_work_intake,
        "fact_campaigns": fact_campaigns,
    }