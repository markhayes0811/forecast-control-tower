# Slide 1 — Capacity & Campaign Forecasting Control Tower
**Objective:** Improve staffing decisions and campaign delivery predictability using forecasting + risk analytics.  
**Users:** Ops leaders, team managers, marketing PMs, control partners.  
**Outputs:** Weekly capacity gap forecasts (team/role) + campaign launch slippage risk (P50/P90).

---

# Slide 2 — Business Problem & Decisions
## Capacity
- **Question:** How many staff hours do we need by team/role next 8–12 weeks?
- **Decision:** Allocate headcount and overtime to prevent SLA misses.

## Campaign timelines
- **Question:** Which campaigns are likely to miss launch and why?
- **Decision:** Escalate governance early, adjust launch plans, rebalance workload.

---

# Slide 3 — Data & KPI Definitions (Auditable)
## Core data entities
- Work intake (weekly): work_items, avg_handle_time_hours
- Capacity (weekly): headcount, shrinkage, hours_per_person
- Campaigns: size, approvals_count, dependencies_count, channel, planned/actual dates

## KPIs
- **Required Hours** = work_items × avg_handle_time_hours
- **Available Hours** = headcount × hours_per_person × (1 − shrinkage)
- **Capacity Gap** = Required − Available (positive = understaffed)
- **Campaign Slippage (days)** = actual_launch − planned_launch (floored at 0)

---

# Slide 4 — Modeling Approach & Validation
## Capacity forecasting
- Holt-Winters (ETS) baseline per team/role
- Rolling backtests with MAPE to validate accuracy

## Campaign timeline risk
- Supervised model predicts slippage days
- P50/P90 risk bands for stakeholder-friendly planning

## Controls
- Data quality gate: schema + nulls + ranges + referential integrity
- Documented assumptions and model limitations

---

# Slide 5 — Insights, Monitoring, Next Steps
## What stakeholders get
- Heatmap of scenario capacity gaps by team/role
- Ranked list of critical understaffing weeks
- Top-risk campaigns with drivers and recommendations

## Monitoring
- Weekly forecast error trends
- Retrain triggers when error exceeds threshold

## Next steps (optional upgrades)
- Add exogenous drivers (holidays, backlog, campaign volume)
- Add model registry (MLflow) + stored “prediction-time” forecasts
- Expand scenario planning to cost impacts (overtime vs hiring)
