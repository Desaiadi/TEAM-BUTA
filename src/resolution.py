"""Predict recorded NYC 311 closure hours, capped at 90 days.

Run ``python -m src.resolution --help`` for downloading, training and prediction.
The source notebook is NYC_311_Resolution_Time_ML.ipynb. Agency closure does
not establish when the physical problem was fixed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from src import config
from src.download import _session, socrata_query
from src.study import in_window

CATEGORIES = ["agency", "complaint_type", "descriptor", "borough", "incident_zip",
              "open_data_channel_type", "location_type"]
CALENDAR = ["filing_month", "filing_weekday", "filing_hour"]
FIELDS = ["unique_key", "created_date", "closed_date", *CATEGORIES]
TARGET = "hours_to_close_capped"
CAP = 2160
TZ = "America/New_York"
CACHE = config.RAW_DIR / "resolution_311"
MODEL = config.PROJECT_ROOT / "models" / "resolution.joblib"
REPORT = config.REPORTS_DIR / "resolution"
MODEL_NAMES = ("linear_regression", "decision_tree", "random_forest", "gradient_boosting")


def local_times(values):
    return pd.to_datetime(values, format="ISO8601", errors="coerce").dt.tz_localize(
        TZ, ambiguous="NaT", nonexistent="NaT")


def prepare(frame, as_of):
    """Use only records with 90 elapsed days of observation at the extraction cutoff.

    A closure after the cutoff is unobserved. Missing closures get the cap only
    after sufficient follow-up; invalid recorded timestamps are excluded.
    """
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize(TZ) if cutoff.tzinfo is None else cutoff.tz_convert(TZ)
    cutoff = min(cutoff, pd.Timestamp(config.STUDY_AS_OF))
    cohort = frame[in_window(frame.created_date)]
    data = cohort.drop_duplicates("unique_key", keep="last").copy()
    opened, closed = local_times(data.created_date), local_times(data.closed_date)
    elapsed = (closed - opened).dt.total_seconds() / 3600
    invalid = (data.unique_key.isna() | opened.isna()
               | (data.closed_date.notna() & closed.isna()) | elapsed.lt(0))
    mature = opened + pd.Timedelta(hours=CAP) <= cutoff
    observed = closed.le(cutoff)
    data["created_at"] = opened.dt.tz_localize(None)
    data[TARGET] = elapsed.where(observed).fillna(CAP).clip(upper=CAP)
    data["closed_within_90d"] = observed & elapsed.le(CAP)
    audit = {"raw_rows": len(frame), "outside_study_rows": len(frame) - len(cohort),
             "duplicates_removed": len(cohort) - len(data),
             "invalid_rows": int(invalid.sum()),
             "insufficient_followup_rows": int((~invalid & ~mature).sum()),
             "usable_rows": int((~invalid & mature).sum()), "as_of": cutoff.isoformat()}
    return data.loc[~invalid & mature].copy(), audit


def features(frame):
    out = frame.reindex(columns=CATEGORIES).copy()
    for col in CATEGORIES:
        out[col] = out[col].astype("string").str.strip().str.upper().replace("", pd.NA).fillna("(MISSING)")
    ts = frame.created_at
    out["filing_month"], out["filing_weekday"], out["filing_hour"] = ts.dt.month, ts.dt.dayofweek, ts.dt.hour
    return out


def make_model(kind="gradient_boosting"):
    if kind not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {kind}")
    # Ordinary regression trees do not understand categorical integer codes.
    # One-hot encoding avoids imposing an arbitrary order on agency/ZIP/etc.
    if kind != "gradient_boosting":
        encoder = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                          max_categories=128, dtype=np.float32), CATEGORIES),
            ("calendar", StandardScaler(), CALENDAR)])
        estimators = {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(max_depth=18, min_samples_leaf=40, random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=18,
                                                    min_samples_leaf=40, n_jobs=4, random_state=42),
        }
        return make_pipeline(encoder, TransformedTargetRegressor(
            regressor=estimators[kind], func=np.log1p, inverse_func=np.expm1))
    encoder = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                      min_frequency=20, max_categories=128), CATEGORIES),
        ("calendar", "passthrough", CALENDAR)])
    tree = HistGradientBoostingRegressor(
        loss="absolute_error", categorical_features=[True] * len(CATEGORIES) + [False] * 3,
        max_iter=200, max_leaf_nodes=31, min_samples_leaf=40, learning_rate=.1,
        early_stopping=False, random_state=42)
    return make_pipeline(encoder, TransformedTargetRegressor(
        regressor=tree, func=np.log1p, inverse_func=np.expm1))


def sample_days(start, end):
    return [p.start_time.replace(day=d) for p in pd.period_range(start, end, freq="M") for d in (5, 15, 25)]


def download(days):
    """Cache complete selected days; never treat these dates as a random sample."""
    CACHE.mkdir(parents=True, exist_ok=True)
    def one(day):
        path = CACHE / f"{day:%Y-%m-%d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        end = day + pd.Timedelta(days=1)
        where = f"created_date >= '{day.isoformat()}' AND created_date < '{end.isoformat()}'"
        with _session() as session:
            data = socrata_query(session, config.NYC_311_URL, FIELDS, where, "created_date,unique_key")
        if data.empty or data.unique_key.isna().any() or data.unique_key.duplicated().any():
            raise ValueError(f"Empty day or invalid request IDs: {day.date()}")
        tmp = path.with_suffix(".tmp")
        data.to_parquet(tmp, index=False)
        tmp.replace(path)
        print(f"Downloaded {day.date()}: {len(data):,} requests", flush=True)
        return data
    with ThreadPoolExecutor(max_workers=4) as pool:
        return pd.concat(pool.map(one, days), ignore_index=True)


def train(frame, as_of=config.STUDY_AS_OF,
          train_start=config.RESOLUTION_TRAIN_START, train_end=config.RESOLUTION_TRAIN_END,
          test_start=config.RESOLUTION_TEST_START, test_end=config.RESOLUTION_TEST_END,
          model_path=MODEL, report_dir=REPORT):
    train_first = pd.Period(train_start, "M").start_time
    train_last = pd.Period(train_end, "M").end_time
    test_first = pd.Period(test_start, "M").start_time
    test_last = pd.Period(test_end, "M").end_time
    if train_first > train_last or test_first > test_last:
        raise ValueError("Start month must precede end month")
    if train_last.tz_localize(TZ) + pd.Timedelta(hours=CAP) >= test_first.tz_localize(TZ):
        raise ValueError("Leave a full 90-day follow-up gap before the test period")
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize(TZ) if cutoff.tzinfo is None else cutoff.tz_convert(TZ)
    cutoff = min(cutoff, pd.Timestamp(config.STUDY_AS_OF))
    if train_first < pd.Timestamp(config.STUDY_START_DATE) or test_last >= pd.Timestamp(config.STUDY_END_EXCLUSIVE):
        raise ValueError("Training and test cohorts must stay inside the shared study window")
    if test_last.tz_localize(TZ) + pd.Timedelta(hours=CAP) > cutoff:
        raise ValueError("The entire test period needs 90 days of follow-up")
    data, audit = prepare(frame, cutoff)
    training = data[data.created_at.between(train_first, train_last)]
    testing = data[data.created_at.between(test_first, test_last)]
    if training.empty or testing.empty:
        raise ValueError("Both training and test periods need usable records")
    X, X_test = features(training), features(testing)
    y, y_test = training[TARGET], testing[TARGET]
    predictions = {}
    fitted = {}
    for name in MODEL_NAMES:
        with threadpool_limits(limits=4):
            estimator = make_model(name).fit(X, y)
            predictions[name] = estimator.predict(X_test).clip(0, CAP)
        fitted[name] = estimator
        print(f"Trained {name}", flush=True)
    boosted = predictions["gradient_boosting"]
    scores = pd.DataFrame([{
        "model": name, "mae_hours": mean_absolute_error(y_test, pred),
        "median_absolute_error_hours": median_absolute_error(y_test, pred),
        "rmse_hours": root_mean_squared_error(y_test, pred)} for name, pred in predictions.items()])
    # Keep the fixed default rather than choosing it using holdout results.
    diagnostics = testing[["unique_key", "agency", "complaint_type", TARGET, "closed_within_90d"]].copy()
    diagnostics["predicted_hours"] = boosted
    diagnostics["signed_error"] = boosted - y_test
    diagnostics["absolute_error"] = diagnostics.signed_error.abs()
    segments = diagnostics.groupby(["agency", "complaint_type"], dropna=False).agg(
        requests=("absolute_error", "size"), mae_hours=("absolute_error", "mean"),
        median_signed_error_hours=("signed_error", "median"))
    metadata = {"target": "Recorded agency closure time; not verified physical resolution",
                "study_id": config.STUDY_ID,
                "cap_hours": CAP, "train_rows": len(training), "test_rows": len(testing),
                "train_start": train_start, "train_end": train_end,
                "test_start": test_start, "test_end": test_end, "audit": audit,
                "default_model": "gradient_boosting", "models": list(MODEL_NAMES),
                "target_transform": "log1p; linear/tree/forest use squared error, boosting uses absolute error",
                "sample": "5th, 15th and 25th of each study month; local input may differ"}
    model_path, report_dir = Path(model_path), Path(report_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted["gradient_boosting"], "models": fitted, "metadata": metadata},
                model_path, compress=3)
    scores.to_csv(report_dir / "metrics.csv", index=False)
    diagnostics.to_parquet(report_dir / "holdout_predictions.parquet", index=False)
    comparison = testing[["unique_key", TARGET]].copy()
    for name, pred in predictions.items():
        comparison[name] = pred
    comparison.to_parquet(report_dir / "model_predictions.parquet", index=False)
    segments.to_csv(report_dir / "errors_by_complaint.csv")
    (report_dir / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scores


def predict(bundle, created_date, *, estimator=None, **values):
    if bundle.get("metadata", {}).get("study_id") != config.STUDY_ID:
        raise ValueError("Saved model uses a different study window; run python -m src.resolution train")
    unexpected = set(values) - set(CATEGORIES)
    if unexpected:
        raise ValueError(f"Unsupported fields: {sorted(unexpected)}")
    opened = pd.Timestamp(created_date)
    if pd.isna(opened):
        raise ValueError("Supply a valid filing timestamp")
    if opened.tzinfo is None:
        opened = opened.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        if pd.isna(opened):
            raise ValueError("Filing timestamp is ambiguous or nonexistent in New York time")
    opened = opened.tz_convert(TZ).tz_localize(None)
    row = pd.DataFrame([{**values, "created_at": opened}])
    model = bundle["model"]
    name = estimator or bundle.get("metadata", {}).get("default_model", "gradient_boosting")
    if estimator is not None:
        if estimator not in bundle.get("models", {}):
            raise ValueError(f"Model not available in this saved bundle: {estimator}")
        model = bundle["models"][estimator]
    with threadpool_limits(limits=4):
        hours = float(model.predict(features(row)).clip(0, CAP)[0])
    return {"estimated_hours": round(hours, 2), "estimated_days": round(hours / 24, 2),
            "model": name, "study_id": config.STUDY_ID, "cap_hours": CAP,
            "prediction_target": "Recorded 311 closure", "uncertainty_interval": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Download the sampled days or train from local CSV/parquet")
    fit.add_argument("--input", type=Path)
    fit.add_argument("--as-of", default=config.STUDY_AS_OF, help="Observation cutoff; never later than study end")
    for name, default in [("train-start", config.RESOLUTION_TRAIN_START), ("train-end", config.RESOLUTION_TRAIN_END),
                          ("test-start", config.RESOLUTION_TEST_START), ("test-end", config.RESOLUTION_TEST_END)]:
        fit.add_argument(f"--{name}", default=default)
    fit.add_argument("--model", type=Path, default=MODEL)
    fit.add_argument("--report-dir", type=Path, default=REPORT)
    score = commands.add_parser("predict", help="Estimate hours for one request")
    score.add_argument("--model", type=Path, default=MODEL)
    score.add_argument("--created-date", required=True)
    score.add_argument("--estimator", choices=MODEL_NAMES, help="Saved algorithm to use; default is gradient boosting")
    for col in CATEGORIES:
        score.add_argument("--" + col.replace("_", "-"))
    args = parser.parse_args()
    if args.command == "predict":
        result = predict(joblib.load(args.model), args.created_date, estimator=args.estimator,
                         **{col: getattr(args, col) for col in CATEGORIES})
        print(json.dumps(result, indent=2))
    else:
        if args.input:
            frame = (pd.read_parquet(args.input) if args.input.suffix == ".parquet"
                     else pd.read_csv(args.input, dtype="string"))
        else:
            days = sample_days(config.STUDY_START_DATE[:7], config.STUDY_END_DATE[:7])
            frame = download(days)
        print(train(frame, args.as_of, args.train_start, args.train_end, args.test_start,
                    args.test_end, args.model, args.report_dir).to_string(index=False))


if __name__ == "__main__":
    main()
