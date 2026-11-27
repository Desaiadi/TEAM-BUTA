"""Shared date filters and study metadata for all event analyses."""
import pandas as pd
from src import config

EVENT_DATES = {
    "hpd_complaints": "received_date", "hpd_violations": "inspectiondate",
    "bedbugs": "filling_period_end_date", "evictions": "executed_date",
    "crime_historic": "rpt_dt", "crime_current": "rpt_dt",
}


def in_window(values, start=config.STUDY_START_DATE, end=config.STUDY_END_EXCLUSIVE):
    """Boolean mask for local, timezone-naive event dates in [start, end)."""
    dates = pd.to_datetime(values.astype("string"), format="mixed", errors="coerce")
    return dates.ge(pd.Timestamp(start)) & dates.lt(pd.Timestamp(end))


def metadata():
    return {"study_id": config.STUDY_ID, "start": config.STUDY_START_DATE,
            "end": config.STUDY_END_DATE, "as_of": config.STUDY_AS_OF,
            "housing_history_months": config.HOUSING_HISTORY_MONTHS,
            "housing_target_months": config.HOUSING_TARGET_MONTHS,
            "housing_windows": config.HOUSING_WINDOWS,
            "snapshot_note": "PLUTO, ownership and status fields are downloaded snapshots; historical revisions cannot be reconstructed."}


def write_coverage():
    """Audit actual coverage instead of assuming each source reaches the end."""
    import json
    from src.clean_housing import OUT
    dates = {"hpd_complaints": "received_date", "hpd_violations": "inspectiondate",
             "bedbugs": "period_end", "evictions": "executed_date", "crime": "rpt_dt"}
    expected = set(pd.period_range(config.STUDY_START_DATE, config.STUDY_END_DATE, freq="M").astype(str))
    report = {}
    for name, col in dates.items():
        series = pd.read_parquet(OUT[name], columns=[col])[col]
        counts = series.dt.to_period("M").astype(str).value_counts().sort_index()
        report[name] = {"rows": len(series), "first_event": str(series.min()), "last_event": str(series.max()),
                        "monthly_rows": {m: int(n) for m, n in counts.items()},
                        "missing_months": sorted(expected - set(counts.index)) if name != "bedbugs" else [],
                        "frequency": "annual filing periods" if name == "bedbugs" else "events"}
    path = config.REPORTS_DIR / "study_coverage.json"
    path.write_text(json.dumps({"study": metadata(), "datasets": report}, indent=2), encoding="utf-8")
    return report
