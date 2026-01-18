from __future__ import annotations
import pandas as pd

def run_quality_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    issues = []

    cap = tables["fact_capacity"]
    intake = tables["fact_work_intake"]
    tr = tables["dim_team_role"]

    # Required columns
    required = {
        "fact_capacity": ["week_ending", "team", "role", "headcount", "available_hours", "shrinkage"],
        "fact_work_intake": ["week_ending", "team", "role", "work_items", "avg_handle_time_hours", "required_hours"],
        "fact_campaigns": ["campaign_id", "planned_start", "planned_launch", "actual_launch", "size"],
    }
    for name, cols in required.items():
        missing = [c for c in cols if c not in tables[name].columns]
        if missing:
            issues.append({"check": "schema", "table": name, "status": "FAIL", "detail": f"Missing {missing}"})

    # Null checks
    for name in ["fact_capacity", "fact_work_intake"]:
        nnull = tables[name].isna().sum().sum()
        if nnull > 0:
            issues.append({"check": "nulls", "table": name, "status": "FAIL", "detail": f"{nnull} null cells"})
        else:
            issues.append({"check": "nulls", "table": name, "status": "PASS", "detail": "No nulls"})

    # Valid ranges
    if (cap["shrinkage"].lt(0).any()) or (cap["shrinkage"].gt(1).any()):
        issues.append({"check": "range", "table": "fact_capacity", "status": "FAIL", "detail": "shrinkage out of [0,1]"})
    else:
        issues.append({"check": "range", "table": "fact_capacity", "status": "PASS", "detail": "shrinkage ok"})

    if (intake["avg_handle_time_hours"].le(0).any()):
        issues.append({"check": "range", "table": "fact_work_intake", "status": "FAIL", "detail": "avg_handle_time_hours <= 0"})
    else:
        issues.append({"check": "range", "table": "fact_work_intake", "status": "PASS", "detail": "handle time ok"})

    # Referential integrity (team/role exists)
    tr_set = set(map(tuple, tr[["team", "role"]].values))
    bad_cap = cap[~cap.apply(lambda r: (r["team"], r["role"]) in tr_set, axis=1)]
    bad_intake = intake[~intake.apply(lambda r: (r["team"], r["role"]) in tr_set, axis=1)]
    if len(bad_cap) or len(bad_intake):
        issues.append({"check": "ref_integrity", "table": "facts", "status": "FAIL",
                       "detail": f"bad_cap={len(bad_cap)}, bad_intake={len(bad_intake)}"})
    else:
        issues.append({"check": "ref_integrity", "table": "facts", "status": "PASS", "detail": "team/role mapping ok"})

    return pd.DataFrame(issues)
