"""Precompute the data behind the app's charts.

    python -m src.insights      (after src.app_data)

Writes two files the Flask app reads at startup:

  app_insights.json        citywide charts for the Insights page:
                           datasets, trends, findings (Q1-Q4) and model evaluation (Q5)
  app_building_history.parquet   per building, per year: complaints by problem type
                           and violations by class (the building page's history chart)

Model evaluation is redone here the honest way (train on Jul–Oct 2025 outcomes, test on Jul–Oct 2026),
so every number on the Insights page matches the notebooks.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config
from src import features_housing as F
from src import landlord_neighbors as LN
from src import model_housing as M
from src.clean_housing import OUT

log = logging.getLogger(__name__)

INSIGHTS_FILE = config.PROCESSED_DIR / "app_insights.json"
HISTORY_FILE = config.PROCESSED_DIR / "app_building_history.parquet"
ERAS = ["<1920", "1920-39", "1940-59", "1960-79", "1980-99", "2000+"]


def _round(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# --------------------------------------------------------------------------
# Datasets and citywide trends
# --------------------------------------------------------------------------
def dataset_table() -> list[dict]:
    """Rows downloaded and rows kept after cleaning, per dataset (parquet metadata only)."""
    def rows(path):
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if path.is_dir():
            files = [f for f in files if config.STUDY_START_DATE[:7] <= f.stem[:7] <= config.STUDY_END_DATE[:7]]
        return sum(pq.ParquetFile(f).metadata.num_rows for f in files if f.exists())

    raw = config.RAW_DIR
    spec = [
        ("HPD housing complaints", raw / "hpd_complaints", OUT["hpd_complaints"], "Problems tenants report"),
        ("HPD violations", raw / "hpd_violations", OUT["hpd_violations"], "Problems inspectors confirmed (A/B/C)"),
        ("HPD registrations", raw / "hpd_registrations.parquet", OUT["landlords"], "Owners and agents → landlords"),
        ("PLUTO tax lots", raw / "pluto.parquet", config.CLEAN_PLUTO_FILE, "Building age, size, location"),
        ("Bedbug filings", raw / "bedbugs.parquet", OUT["bedbugs"], "Infested apartments per year"),
        ("Evictions", raw / "evictions.parquet", OUT["evictions"], "Residential evictions in the study window"),
        ("NYPD complaints", raw / "crime_historic", OUT["crime"], "Reported crime in the study window"),
        ("311 service requests", raw / "resolution_311", None, "Selected days, Jan 2025–Oct 2026"),
    ]
    out = []
    for name, r, c, what in spec:
        n_raw = rows(r) + (rows(raw / "crime_current") if name == "NYPD complaints" else 0)
        out.append({"dataset": name, "raw_rows": n_raw, "clean_rows": rows(c) if c else None, "provides": what})
    return out


def trends() -> dict:
    """Monthly complaints by type, violations by class per year, crime and evictions per month."""
    c = pd.read_parquet(OUT["hpd_complaints"], columns=["complaint_id", "received_date", "problem_type"])
    c = c[c["problem_type"].isin(F.TARGET_TYPES)]
    c["month"] = c["received_date"].dt.to_period("M").astype(str)
    monthly = c.groupby(["month", "problem_type"])["complaint_id"].nunique().unstack(fill_value=0)
    monthly = monthly.reindex(columns=F.TARGET_TYPES, fill_value=0)
    current = config.STUDY_END_EXCLUSIVE[:7]
    monthly = monthly[monthly.index < current]                   # drop the unfinished month

    v = pd.read_parquet(OUT["hpd_violations"], columns=["inspectiondate", "year", "class"])
    v = v[v["inspectiondate"].dt.month <= 10]  # Compare Jan–Oct in both years.
    v_year = pd.crosstab(v["year"], v["class"]).reindex(columns=["A", "B", "C"], fill_value=0)

    cr = pd.read_parquet(OUT["crime"], columns=["rpt_dt", "crime_group"])
    cr["month"] = cr["rpt_dt"].dt.to_period("M").astype(str)
    crime_m = cr[cr["month"] < current].groupby("month").size()

    ev = pd.read_parquet(OUT["evictions"], columns=["executed_date"])
    ev["month"] = ev["executed_date"].dt.to_period("M").astype(str)
    ev_m = ev[ev["month"] < current].groupby("month").size()

    bb = pd.read_parquet(OUT["bedbugs"], columns=["year", "has_infestation"])
    bb_y = bb[bb["year"].isin([2025, 2026])].groupby("year")["has_infestation"].mean()
    return {
        "complaints_monthly": {"months": list(monthly.index),
                               "series": {t: monthly[t].astype(int).tolist() for t in F.TARGET_TYPES}},
        "violations_yearly": {"years": [int(y) for y in v_year.index],
                              "series": {k: v_year[k].astype(int).tolist() for k in ["A", "B", "C"]}},
        "crime_monthly": {"months": list(crime_m.index), "values": crime_m.astype(int).tolist()},
        "evictions_monthly": {"months": list(ev_m.index), "values": ev_m.astype(int).tolist()},
        "bedbug_share_yearly": {"years": [int(y) for y in bb_y.index], "values": [_round(x) for x in bb_y]},
    }


# --------------------------------------------------------------------------
# Findings (Q1-Q4), from Jan–Jun 2025 history → Jul–Oct 2025 outcomes
# --------------------------------------------------------------------------
def findings(train: pd.DataFrame, test: pd.DataFrame, universe: pd.DataFrame, A) -> dict:
    q1, q2, q4 = [], [], []
    port = train[train["has_portfolio"]]
    for t in F.TARGET_TYPES:
        s = F.slug(t)
        had = train[f"c_{s}"] > 0
        q1.append({"type": t, "had": _round(train.loc[had, f"y_{s}"].mean()),
                   "not": _round(train.loc[~had, f"y_{s}"].mean())})
        qq = pd.qcut(port[f"ll_rate_{s}"].rank(method="first"), 4, labels=False)
        rates = port.groupby(qq)[f"y_{s}"].mean()
        q2.append({"type": t, "best": _round(rates.iloc[0]), "worst": _round(rates.iloc[-1])})
        clean = train[train[f"c_{s}"] == 0]
        nb = pd.qcut(clean[f"nb100_{s}"].rank(method="first"), 4, labels=False)
        r = clean.groupby(nb)[f"y_{s}"].mean()
        q4.append({"type": t, "quietest": _round(r.iloc[0]), "busiest": _round(r.iloc[-1])})

    era = train.groupby("era")[[f"y_{F.slug(t)}" for t in F.TARGET_TYPES]].mean().reindex(ERAS)
    pairs = LN.neighbor_pairs(universe, A)
    fair = pairs["both_registered"] & pairs["similar_buildings"]
    agree = LN.outcome_agreement(pairs, test, fair)
    return {
        "q1_history": q1, "q2_landlord_quartiles": q2, "q4_neighborhood": q4,
        "q3_era": {"eras": ERAS, "series": {t: [_round(x) for x in era[f"y_{F.slug(t)}"]] for t in F.TARGET_TYPES}},
        "q2_neighbors": [{"type": t, "same": _round(agree.loc[t].iloc[0]), "different": _round(agree.loc[t].iloc[1])}
                         for t in F.TARGET_TYPES],
    }


# --------------------------------------------------------------------------
# Model evaluation (Q5): train 2025, test 2026
# --------------------------------------------------------------------------
def model_evaluation(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cols = F.feature_columns(train)
    X_tr, X_te = M.to_matrix(train, cols), M.to_matrix(test, cols)
    perf, calib, levels = [], {}, []
    for t in F.TARGET_TYPES:
        s = F.slug(t)
        y_tr, y_te = train[f"y_{s}"], test[f"y_{s}"]
        gbm = M.make_model("gbm").fit(X_tr, y_tr)
        logit = M.make_model("logit").fit(X_tr.fillna(0), y_tr)
        p = gbm.predict_proba(X_te)[:, 1]
        p_lr = logit.predict_proba(X_te.fillna(0))[:, 1]
        base = M.baseline_scores(test, t)
        perf.append({"type": t, "base_rate": _round(y_te.mean()),
                     "auc_gbm": _round(roc_auc_score(y_te, p)), "auc_logit": _round(roc_auc_score(y_te, p_lr)),
                     "auc_baseline": _round(base["auc"]),
                     "ap_gbm": _round(average_precision_score(y_te, p))})
        dec = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"pred": p, "obs": y_te.to_numpy()}).groupby(dec, observed=True).mean()
        calib[t] = {"pred": [_round(x) for x in cal["pred"]], "obs": [_round(x) for x in cal["obs"]]}
        lv = pd.Series(M.risk_level(p, y_tr.mean()), index=test.index)
        obs = test.groupby(lv)[f"y_{s}"].mean()
        levels.append({"type": t, **{k: _round(obs.get(k)) for k in ["Low", "Medium", "High"]}})
        log.info("evaluated %s", t)
    abl = M.ablation(train, test)
    abl_tab = abl.pivot(index="problem", columns="features", values="auc")
    steps = sorted(abl_tab.columns, key=len)
    return {
        "performance": perf, "calibration": calib, "risk_levels": levels,
        "ablation": {"steps": [st.split(" + ")[-1] for st in steps],
                     "series": {t: [_round(abl_tab.loc[t, st]) for st in steps] for t in F.TARGET_TYPES}},
        "n_train": int(len(train)), "n_test": int(len(test)), "n_features": len(cols),
    }


# --------------------------------------------------------------------------
# Per-building history for the building page
# --------------------------------------------------------------------------
def building_history(bbls: pd.Index) -> pd.DataFrame:
    """Long table: bbl, year, series (problem type or 'violations A/B/C'), n."""
    c = pd.read_parquet(OUT["hpd_complaints"], columns=["complaint_id", "bbl", "received_date", "year", "problem_type"])
    c = c[c["bbl"].isin(bbls) & c["problem_type"].isin(F.TARGET_TYPES)]
    c = c[c["received_date"].dt.month <= 10]
    hc = c.groupby(["bbl", "year", "problem_type"])["complaint_id"].nunique().rename("n").reset_index()
    hc = hc.rename(columns={"problem_type": "series"})
    v = pd.read_parquet(OUT["hpd_violations"], columns=["bbl", "inspectiondate", "year", "class"])
    v = v[v["bbl"].isin(bbls) & v["class"].isin(["A", "B", "C"])]
    v = v[v["inspectiondate"].dt.month <= 10]
    hv = v.groupby(["bbl", "year", "class"]).size().rename("n").reset_index()
    hv["series"] = "class " + hv["class"]
    out = pd.concat([hc, hv.drop(columns="class")], ignore_index=True)
    out["year"] = out["year"].astype("int16")
    out["n"] = out["n"].astype("int32")
    return out.sort_values(["bbl", "year"]).reset_index(drop=True)


def build() -> None:
    inp = F.load_inputs()
    universe = F.building_universe(inp["pluto"], inp["landlords"])
    A = F.neighbor_matrix(universe)
    train = pd.read_parquet(str(F.FEATURES_FILE).format(window="train"))
    test = pd.read_parquet(str(F.FEATURES_FILE).format(window="test"))
    from src.study import metadata
    from src.study import write_coverage
    insights = {"study": metadata(), "coverage": write_coverage(), "datasets": dataset_table(), "trends": trends(),
                "findings": findings(train, test, universe, A), "model": model_evaluation(train, test),
                "built": pd.Timestamp.today().strftime("%Y-%m-%d")}
    INSIGHTS_FILE.write_text(json.dumps(insights))
    building_history(pd.Index(universe["bbl"])).to_parquet(HISTORY_FILE, index=False)
    log.info("saved %s and %s", INSIGHTS_FILE.name, HISTORY_FILE.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    build()
