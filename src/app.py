from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.generate_data import generate
from src.quality import run_quality_checks
from src.forecasting import forecast_required_hours, backtest_mape
from src.campaign_model import train_campaign_model, score_campaigns
from src.monitoring import weekly_forecast_error, retrain_trigger

st.set_page_config(page_title="Forecast Control Tower", layout="wide")

RISK_BINS = [-1e9, -50, 0, 50, 1e9]
RISK_LABELS = ["Overstaffed", "Balanced", "At Risk", "Critical"]

@st.cache_data
def load_tables(seed: int = 7):
    return generate(seed=seed)

def compute_capacity_gap(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    intake = tables["fact_work_intake"].copy()
    cap = tables["fact_capacity"].copy()
    m = intake.merge(cap, on=["week_ending", "team", "role"], how="left")
    m["gap_hours"] = m["required_hours"] - m["available_hours"]
    m["risk"] = pd.cut(m["gap_hours"], bins=RISK_BINS, labels=RISK_LABELS)
    return m

def apply_scenario(gap: pd.DataFrame, vol_mult: float, shrink_adj: float, hc_adj: int) -> pd.DataFrame:
    g = gap.copy()
    g["required_hours_scn"] = g["required_hours"] * vol_mult
    g["shrinkage_scn"] = (g["shrinkage"] + shrink_adj).clip(0.05, 0.60)
    g["headcount_scn"] = (g["headcount"] + hc_adj).clip(lower=1)
    g["available_hours_scn"] = g["headcount_scn"] * g["hours_per_person"] * (1 - g["shrinkage_scn"])
    g["gap_hours_scn"] = g["required_hours_scn"] - g["available_hours_scn"]
    g["risk_scn"] = pd.cut(g["gap_hours_scn"], bins=RISK_BINS, labels=RISK_LABELS)
    return g

def plot_heatmap(pivot: pd.DataFrame, title: str):
    fig = plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xlabel("Role")
    plt.ylabel("Team")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Avg Gap Hours (Scenario)")
    st.pyplot(fig)

def plot_stacked_risk(dist: pd.DataFrame, title: str):
    # dist columns: team, risk, count
    fig = plt.figure()
    teams = dist["team"].unique()
    bottoms = np.zeros(len(teams))
    for label in RISK_LABELS:
        vals = dist[dist["risk"] == label].set_index("team").reindex(teams)["count"].fillna(0).values
        plt.bar(teams, vals, bottom=bottoms, label=label)
        bottoms += vals
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Weeks (count)")
    plt.legend()
    st.pyplot(fig)

st.title("Capacity & Campaign Timeline Forecasting Control Tower")

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Synthetic data seed", min_value=1, max_value=9999, value=7, step=1)
    horizon = st.slider("Forecast horizon (weeks)", min_value=4, max_value=20, value=12, step=1)

    st.divider()
    st.subheader("What-if scenario")
    scenario_volume = st.slider("Volume multiplier", 0.8, 1.4, 1.0, 0.05)
    scenario_shrink = st.slider("Shrinkage adjustment", -0.10, 0.10, 0.00, 0.01)
    scenario_hc = st.slider("Headcount adjustment", -5, 10, 0, 1)

tables = load_tables(seed)
quality = run_quality_checks(tables)
gap = compute_capacity_gap(tables)
gap_s = apply_scenario(gap, scenario_volume, scenario_shrink, scenario_hc)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview", "Capacity Forecasting", "Campaign Timeline Risk", "Data Quality & Compliance", "Model Monitoring"
])

with tab1:
    st.subheader("Executive Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Risk distribution (Current)")
        curr = gap["risk"].value_counts().reindex(RISK_LABELS).fillna(0).reset_index()
        curr.columns = ["risk", "weeks_count"]
        st.dataframe(curr, use_container_width=True, hide_index=True)

    with col2:
        st.caption("Risk distribution (Scenario)")
        scn = gap_s["risk_scn"].value_counts().reindex(RISK_LABELS).fillna(0).reset_index()
        scn.columns = ["risk", "weeks_count"]
        st.dataframe(scn, use_container_width=True, hide_index=True)

    st.divider()

    # Heatmap: avg scenario gap by team/role
    st.caption("Heatmap: Average scenario capacity gap (hours) by team/role")
    heat = gap_s.groupby(["team", "role"])["gap_hours_scn"].mean().reset_index()
    pivot = heat.pivot(index="team", columns="role", values="gap_hours_scn").reindex(index=sorted(heat["team"].unique()))
    plot_heatmap(pivot, "Avg Scenario Gap Hours by Team/Role")

    st.divider()

    # Team-level stacked risk chart
    st.caption("Stacked risk distribution by team (Scenario)")
    dist = gap_s.groupby(["team", "risk_scn"]).size().reset_index(name="count")
    dist.columns = ["team", "risk", "count"]
    plot_stacked_risk(dist, "Scenario Risk Distribution by Team")

    st.divider()
    st.caption("Top 10 scenario gaps (worst understaffing)")
    top = gap_s.sort_values("gap_hours_scn", ascending=False).head(10)[
        ["week_ending", "team", "role", "required_hours_scn", "available_hours_scn", "gap_hours_scn", "risk_scn"]
    ]
    st.dataframe(top, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Capacity Forecasting (Required Hours)")

    intake = tables["fact_work_intake"].copy()
    fc = forecast_required_hours(intake, horizon_weeks=horizon)

    st.caption("Backtest accuracy by team/role (MAPE)")
    mape = backtest_mape(intake)
    st.dataframe(mape, use_container_width=True, hide_index=True)

    st.divider()

    teams = sorted(intake["team"].unique())
    roles = sorted(intake["role"].unique())
    c1, c2 = st.columns(2)
    with c1:
        team_sel = st.selectbox("Team", teams, index=0, key="team_sel")
    with c2:
        role_sel = st.selectbox("Role", roles, index=0, key="role_sel")

    hist = intake[(intake["team"] == team_sel) & (intake["role"] == role_sel)].copy()
    hist["week_ending"] = pd.to_datetime(hist["week_ending"])
    hist = hist.sort_values("week_ending")

    # Join capacity for same team/role for required vs available plot
    cap = tables["fact_capacity"].copy()
    cap["week_ending"] = pd.to_datetime(cap["week_ending"])
    joined = hist.merge(cap, on=["week_ending", "team", "role"], how="left")
    joined = joined.tail(80)

    fc_one = fc[(fc["team"] == team_sel) & (fc["role"] == role_sel)].copy()
    fc_one["week_ending"] = pd.to_datetime(fc_one["week_ending"])

    st.caption("Historical required hours vs available hours (last ~80 weeks)")
    fig = plt.figure()
    plt.plot(joined["week_ending"], joined["required_hours"], label="Required (actual)")
    plt.plot(joined["week_ending"], joined["available_hours"], label="Available (actual)")
    plt.legend()
    plt.xlabel("Week Ending")
    plt.ylabel("Hours")
    st.pyplot(fig)

    st.caption("Forecasted required hours (next horizon)")
    fig = plt.figure()
    plt.plot(joined["week_ending"], joined["required_hours"], label="Actual")
    plt.plot(fc_one["week_ending"], fc_one["required_hours_forecast"], label="Forecast")
    plt.legend()
    plt.xlabel("Week Ending")
    plt.ylabel("Required Hours")
    st.pyplot(fig)

with tab3:
    st.subheader("Campaign Timeline Risk")

    campaigns = tables["fact_campaigns"].copy()
    model, xcols, bands = train_campaign_model(campaigns)
    scored = score_campaigns(campaigns, model, xcols, bands)

    st.caption("Distribution of predicted P90 launch slippage (days)")
    fig = plt.figure()
    plt.hist(scored["pred_slip_days_p90"].clip(0, 30), bins=20)
    plt.xlabel("Predicted P90 slippage (days) [clipped]")
    plt.ylabel("Campaign count")
    st.pyplot(fig)

    st.divider()

    st.caption("Top-risk upcoming campaigns (P90 slip ≥ 7 days)")
    upcoming = scored[scored["status"] != "Launched"].copy()
    out = upcoming.sort_values("pred_slip_days_p90", ascending=False).head(20)[
        ["campaign_id", "channel", "planned_start", "planned_launch", "size",
         "dependencies_count", "approvals_count", "pred_slip_days_point", "pred_slip_days_p50", "pred_slip_days_p90", "risk_flag"]
    ]
    st.dataframe(out, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Stakeholder-ready recommendations")
    st.markdown(
        "- **Dependencies ≥ 2** and **Approvals ≥ 4**: route to early governance review.\n"
        "- **Size ≥ 4**: add buffer days and pre-book production resources.\n"
        "- **Holiday season start**: treat as elevated timeline risk and prioritize approvals."
    )

with tab4:
    st.subheader("Data Quality & Compliance")
    st.caption("Automated checks (schema, nulls, ranges, referential integrity)")
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Data dictionary (sample)")
    dd = pd.DataFrame([
        {"field": "required_hours", "meaning": "Work items × avg handle time (hours)"},
        {"field": "available_hours", "meaning": "Headcount × hours/person × (1 − shrinkage)"},
        {"field": "gap_hours", "meaning": "Required − Available; positive = understaffing"},
        {"field": "slip_days", "meaning": "Actual launch − planned launch (days), floored at 0"},
    ])
    st.dataframe(dd, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Model Monitoring")

    intake = tables["fact_work_intake"].copy()
    fc = forecast_required_hours(intake, horizon_weeks=horizon)

    # For demo: compare overlap dates (illustrative monitoring)
    aligned = intake.merge(fc, on=["week_ending", "team", "role"], how="inner")
    err = aligned.copy()
    err["abs_pct_err"] = (err["required_hours"] - err["required_hours_forecast"]).abs() / err["required_hours"].clip(lower=1e-6)
    err["week_ending"] = pd.to_datetime(err["week_ending"])

    trig = retrain_trigger(
        errors=err.rename(columns={"abs_pct_err": "abs_pct_err", "required_hours_forecast": "required_hours_forecast"}),
        threshold_mape=0.18
    )
    st.metric("Teams/Roles flagged for retrain (MAPE last 4w ≥ 0.18)", trig["flagged_count"])

    if trig["flagged_count"] > 0:
        st.dataframe(trig["flagged"], use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Absolute % error distribution")
    fig = plt.figure()
    plt.hist(err["abs_pct_err"].clip(0, 1), bins=20)
    plt.xlabel("Absolute % error (clipped 0..1)")
    plt.ylabel("Count")
    st.pyplot(fig)

    st.caption("Average absolute % error over time (all teams/roles)")
    trend = err.groupby("week_ending")["abs_pct_err"].mean().reset_index()
    fig = plt.figure()
    plt.plot(trend["week_ending"], trend["abs_pct_err"])
    plt.xlabel("Week Ending")
    plt.ylabel("Mean abs % error")
    st.pyplot(fig)
