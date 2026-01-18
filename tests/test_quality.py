import pandas as pd
from src.generate_data import generate
from src.quality import run_quality_checks

def test_quality_checks_return_dataframe():
    tables = generate(seed=7)
    issues = run_quality_checks(tables)
    assert isinstance(issues, pd.DataFrame)
    assert set(["check", "table", "status", "detail"]).issubset(issues.columns)

def test_quality_checks_have_passes():
    tables = generate(seed=7)
    issues = run_quality_checks(tables)
    # Expect at least one PASS row in normal synthetic generation
    assert (issues["status"] == "PASS").any()

def test_no_nulls_in_core_tables():
    tables = generate(seed=7)
    cap = tables["fact_capacity"]
    intake = tables["fact_work_intake"]
    assert cap.isna().sum().sum() == 0
    assert intake.isna().sum().sum() == 0
