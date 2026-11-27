"""Building features for the shared January 2025–October 2026 study.

Train: January–June 2025 history -> July–October 2025 complaints.
Test: January–June 2026 history -> July–October 2026 complaints.
Forecast: May–October 2026 history -> November 2026–February 2027.
All models use six-month histories and four-month outcome windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from src import config
from src.clean_housing import OUT

TARGET_TYPES = ["heat/hot water", "pests", "leaks/mold", "plumbing", "paint/plaster", "doors/windows", "safety"]
NEIGHBOR_RADIUS_M = 100
CRIME_RADIUS_M = 200          # NYPD geocodes reports to the nearest street segment, so 100 m is too tight
CRIME_GROUPS = ["violent", "property", "drugs/weapons", "harassment"]
FEATURES_FILE = config.PROCESSED_DIR / "building_features_{window}.parquet"


def slug(problem_type: str) -> str:
    """'heat/hot water' -> 'heat_hot_water' (safe column suffix)."""
    return "".join(ch if ch.isalnum() else "_" for ch in problem_type).strip("_").replace("__", "_")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_inputs() -> dict[str, pd.DataFrame]:
    """Read every cleaned dataset needed for features."""
    return {
        "pluto": pd.read_parquet(config.CLEAN_PLUTO_FILE),
        "landlords": pd.read_parquet(OUT["landlords"]),
        "complaints": pd.read_parquet(OUT["hpd_complaints"],
                                      columns=["complaint_id", "bbl", "received_date", "year", "problem_type", "is_emergency"]),
        "violations": pd.read_parquet(OUT["hpd_violations"], columns=["bbl", "inspectiondate", "year", "class", "problem_type"]),
        "bedbugs": pd.read_parquet(OUT["bedbugs"], columns=["bbl", "period_end", "filing_date", "year", "has_infestation"]),
        "evictions": pd.read_parquet(OUT["evictions"], columns=["bbl", "executed_date", "year"]),
        "crime": (pd.read_parquet(OUT["crime"], columns=["rpt_dt", "year", "severity", "crime_group", "in_residence",
                                                         "latitude", "longitude"])
                  if OUT["crime"].exists() else None),
    }


def _to_meters(lat, lon, lat0: float) -> np.ndarray:
    """Project latitude/longitude to local x/y meters (accurate to <1% across NYC)."""
    return np.column_stack([np.asarray(lon) * 111_320 * np.cos(lat0), np.asarray(lat) * 110_540])


def crime_counts(b: pd.DataFrame, crime: pd.DataFrame, years, radius_m: float = CRIME_RADIUS_M, exposure_years=None) -> pd.DataFrame:
    """Reported crimes within `radius_m` of each building over `years`, per year, by group and severity.

    A k-d tree on the crime points answers "how many points within r" for all
    buildings at once (query_ball_point with return_length).
    """
    lat0 = np.radians(b["latitude"].mean())
    xy_b = _to_meters(b["latitude"], b["longitude"], lat0)
    c = crime[crime["year"].isin(years)]
    out = pd.DataFrame(index=b.index)
    subsets = {"crime_total": c, "crime_felony": c[c["severity"] == "FELONY"],
               "crime_in_residence": c[c["in_residence"]]}
    subsets.update({f"crime_{slug(g)}": c[c["crime_group"] == g] for g in CRIME_GROUPS})
    for name, sub in subsets.items():
        if len(sub) == 0:
            out[name] = 0.0
            continue
        tree = cKDTree(_to_meters(sub["latitude"], sub["longitude"], lat0))
        out[name] = tree.query_ball_point(xy_b, radius_m, return_length=True, workers=-1) / (exposure_years if exposure_years is not None else len(years))
    return out


def building_universe(pluto: pd.DataFrame, landlords: pd.DataFrame, min_units: int = 3) -> pd.DataFrame:
    """Residential buildings with 3+ apartments (the ones renters rent in), with traits and landlord."""
    b = pluto[(pluto["unitsres"] >= min_units) & pluto["latitude"].notna() & (pluto["yearbuilt"].isna() | (pluto["yearbuilt"] <= 2025))].copy()
    b["era"] = pd.cut(b["yearbuilt"], [1799, 1919, 1939, 1959, 1979, 1999, 2100],
                      labels=["<1920", "1920-39", "1940-59", "1960-79", "1980-99", "2000+"]).astype("string")
    b["building_age"] = 2025 - b["yearbuilt"]
    cls = b["bldgclass"].astype("string").str[0]
    b["is_elevator"] = cls.eq("D")
    b["is_walkup"] = cls.eq("C")
    b = b.merge(landlords[["bbl", "landlord_id", "portfolio_buildings", "landlord_key_type"]], on="bbl", how="left")
    b["is_registered"] = b["landlord_id"].notna()
    b["landlord_id"] = b["landlord_id"].fillna("L_" + b["bbl"])
    b["portfolio_buildings"] = b["portfolio_buildings"].fillna(1)
    keep = ["bbl", "borough", "zipcode", "latitude", "longitude", "yearbuilt", "building_age", "era", "unitsres",
            "numfloors", "bldgclass", "is_elevator", "is_walkup", "landlord_id", "portfolio_buildings",
            "landlord_key_type", "is_registered"]
    return b[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def counts_by(events: pd.DataFrame, bbls: pd.Index, years, column: str, values, prefix: str,
              distinct: str | None = None) -> pd.DataFrame:
    """Count events per building and category value over `years` (zeros filled)."""
    e = events[events["year"].isin(years) & events["bbl"].isin(bbls) & events[column].isin(values)]
    g = e.groupby(["bbl", column])
    counts = (g[distinct].nunique() if distinct else g.size()).unstack(fill_value=0)
    counts = counts.reindex(index=bbls, columns=values, fill_value=0)
    counts.columns = [f"{prefix}{slug(str(v))}" for v in counts.columns]
    return counts


def neighbor_matrix(b: pd.DataFrame, radius_m: float = NEIGHBOR_RADIUS_M) -> sparse.csr_matrix:
    """Sparse n×n matrix with 1 where two buildings are within `radius_m` (self excluded).

    Latitude/longitude are projected to local meters (good to <1% across NYC),
    then a k-d tree finds every pair within the radius in one pass.
    """
    lat0 = np.radians(b["latitude"].mean())
    xy = np.column_stack([b["longitude"] * 111_320 * np.cos(lat0), b["latitude"] * 110_540])
    pairs = cKDTree(xy).query_pairs(radius_m, output_type="ndarray")
    n = len(b)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def landlord_leave_one_out(own: pd.DataFrame, units: pd.Series, landlord: pd.Series) -> pd.DataFrame:
    """Landlord's complaint rate per 100 apartments in their OTHER buildings, per column of `own`.

    NaN for single-building landlords (no other buildings to learn from).
    """
    df = own.copy()
    df["_units"], df["_ll"] = units.values, landlord.values
    tot = df.groupby("_ll").transform("sum")
    other_units = tot["_units"] - df["_units"]
    rate = (tot[own.columns] - df[own.columns]).div(other_units.where(other_units > 0), axis=0) * 100
    rate.columns = [c.replace("c_", "ll_rate_", 1) for c in own.columns]
    return rate


# --------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------
def build_snapshot(window: str, inp: dict[str, pd.DataFrame], universe: pd.DataFrame | None = None,
                   neighbors: sparse.csr_matrix | None = None) -> pd.DataFrame:
    """Construct a configured six-month history / four-month target snapshot."""
    from src.study import in_window
    if window not in config.HOUSING_WINDOWS:
        raise ValueError(f"Use one of {list(config.HOUSING_WINDOWS)}; annual snapshots are obsolete")
    start, cutoff, end = config.HOUSING_WINDOWS[window]
    b = (universe if universe is not None else building_universe(inp["pluto"], inp["landlords"])).copy()
    A = neighbors if neighbors is not None else neighbor_matrix(b)
    b["building_age"] = pd.Timestamp(cutoff).year - b["yearbuilt"]
    bbls = pd.Index(b["bbl"])
    c = inp["complaints"][in_window(inp["complaints"]["received_date"], start, cutoff)]
    v = inp["violations"][in_window(inp["violations"]["inspectiondate"], start, cutoff)]
    hist = list(range(pd.Timestamp(start).year, pd.Timestamp(cutoff).year + 1))
    own_c = counts_by(c, bbls, hist, "problem_type", TARGET_TYPES, "c_", distinct="complaint_id")
    own_v_cls = counts_by(v, bbls, hist, "class", ["A", "B", "C"], "v_class_")
    own_v_type = counts_by(v, bbls, hist, "problem_type", TARGET_TYPES, "v_")
    bb = inp["bedbugs"]
    # Annual filings are publication-lagged; use the latest filing actually
    # available at cutoff, even if its period ends before the six-month window.
    bb = bb[bb["period_end"].ge(pd.Timestamp(config.STUDY_START_DATE))
            & bb["period_end"].lt(pd.Timestamp(cutoff)) & bb["filing_date"].lt(pd.Timestamp(cutoff))]
    latest_bb = bb.sort_values(["period_end", "filing_date"]).drop_duplicates("bbl", keep="last")
    ev = inp["evictions"]
    ev = ev[in_window(ev["executed_date"], start, cutoff)]
    X = b.set_index("bbl").join(own_c).join(own_v_cls).join(own_v_type)
    X["c_total"] = c.groupby("bbl")["complaint_id"].nunique().reindex(bbls).fillna(0).values
    X["c_emergency"] = c[c["is_emergency"]].groupby("bbl")["complaint_id"].nunique().reindex(bbls).fillna(0).values
    X["c_total_per_unit"] = X["c_total"] / X["unitsres"]
    X["bedbugs_recent_filing"] = latest_bb.set_index("bbl")["has_infestation"].reindex(bbls).astype("boolean").fillna(False).astype(bool).values
    X["bedbug_filing_available"] = bbls.isin(latest_bb["bbl"])
    X["bedbug_years"] = bb[bb["has_infestation"]].groupby("bbl")["year"].nunique().reindex(bbls).fillna(0).values
    X["evictions"] = ev.groupby("bbl").size().reindex(bbls).fillna(0).values
    X["evictions_per_unit"] = X["evictions"] / X["unitsres"]
    ll = landlord_leave_one_out(own_c.reindex(bbls), X["unitsres"], X["landlord_id"])
    ll.index = bbls
    X = X.join(ll)
    X["has_portfolio"] = X["portfolio_buildings"] > 1
    nb_counts = A @ own_c.reindex(bbls).to_numpy(dtype=float)
    for j, t in enumerate(TARGET_TYPES):
        X[f"nb100_{slug(t)}"] = nb_counts[:, j]
    X["nb100_buildings"] = np.asarray(A.sum(axis=1)).ravel()
    if inp.get("crime") is not None:
        cr = inp["crime"][in_window(inp["crime"]["rpt_dt"], start, cutoff)]
        # A missing source month is not a month with zero crime. Annualize
        # over observed months and retain exposure metadata for the app.
        months = cr["rpt_dt"].dt.to_period("M").drop_duplicates()
        observed_days = sum((m.end_time.normalize() + pd.Timedelta(days=1) - m.start_time).days for m in months)
        full_days = (pd.Timestamp(cutoff) - pd.Timestamp(start)).days
        exposure = observed_days / 365.25
        cc = crime_counts(b, cr, hist, exposure_years=exposure if exposure else 1)
        if not observed_days:
            cc[:] = np.nan
        X["crime_exposure_fraction"] = observed_days / full_days
        X["crime_observed_start"] = cr["rpt_dt"].min().strftime("%Y-%m-%d") if len(cr) else "unavailable"
        X["crime_observed_end"] = cr["rpt_dt"].max().strftime("%Y-%m-%d") if len(cr) else "unavailable"
        for col in cc:
            X[col] = cc[col].to_numpy()
    if window != "forecast":
        target = inp["complaints"][in_window(inp["complaints"]["received_date"], cutoff, end)]
        y = counts_by(target, bbls, [pd.Timestamp(cutoff).year], "problem_type", TARGET_TYPES, "y_", distinct="complaint_id")
        X = X.join((y > 0).astype("int8"))
    X["window"] = window
    X["history_start"], X["history_end"] = start, cutoff
    X["target_start"], X["target_end"] = cutoff, end
    return X.reset_index()


def build_tables():
    """Build the training and held-out feature files from cleaned inputs."""
    inp = load_inputs()
    universe = building_universe(inp["pluto"], inp["landlords"])
    A = neighbor_matrix(universe)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for window in ("train", "test"):
        frame = build_snapshot(window, inp, universe, A)
        frame.to_parquet(str(FEATURES_FILE).format(window=window), index=False)
    from src.study import metadata
    import json
    (config.PROCESSED_DIR / "study.json").write_text(json.dumps(metadata(), indent=2), encoding="utf-8")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except ids, coordinates, targets and text labels."""
    drop = {"bbl", "landlord_id", "landlord_key_type", "zipcode", "latitude", "longitude", "bldgclass",
            "yearbuilt", "target_year", "era", "borough", "window", "history_start", "history_end", "target_start", "target_end", "crime_observed_start", "crime_observed_end"}
    return [c for c in df.columns if c not in drop and not c.startswith("y_")]


def target_columns() -> list[str]:
    return [f"y_{slug(t)}" for t in TARGET_TYPES]


if __name__ == "__main__":
    build_tables()
