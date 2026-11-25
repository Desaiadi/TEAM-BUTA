"""Build the one-row-per-building table that the Know Before You Rent model learns from.

For a target year T, every residential building (3+ apartments) gets:

  Building traits   age, era, apartments, floors, elevator vs walk-up, borough   (PLUTO)
  Own history       complaints per problem type, violations by class and type,
                    bedbug infestations, evictions, all in years T-2 and T-1   (HPD, bedbugs, evictions)
  Landlord record   the landlord's OTHER buildings' complaint rates per type in T-2..T-1
                    (leave-one-out, so a building never "sees" its own history twice)  (HPD registrations)
  100 m around it   complaints per type in other buildings within 100 m              (PLUTO coordinates)
  Targets           y_<type> = 1 if the building gets a complaint of that type in year T

Training uses T = 2024 (history 2022-23); testing uses T = 2025 (history 2023-24),
so the model is always judged on a year it has never seen.
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
FEATURES_FILE = config.PROCESSED_DIR / "building_features_{year}.parquet"


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
                                      columns=["complaint_id", "bbl", "year", "problem_type", "is_emergency"]),
        "violations": pd.read_parquet(OUT["hpd_violations"], columns=["bbl", "year", "class", "problem_type"]),
        "bedbugs": pd.read_parquet(OUT["bedbugs"], columns=["bbl", "year", "has_infestation"]),
        "evictions": pd.read_parquet(OUT["evictions"], columns=["bbl", "year"]),
    }


def building_universe(pluto: pd.DataFrame, landlords: pd.DataFrame, min_units: int = 3) -> pd.DataFrame:
    """Residential buildings with 3+ apartments (the ones renters rent in), with traits and landlord."""
    b = pluto[(pluto["unitsres"] >= min_units) & pluto["latitude"].notna()].copy()
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
def build_snapshot(target_year: int, inp: dict[str, pd.DataFrame], universe: pd.DataFrame | None = None,
                   neighbors: sparse.csr_matrix | None = None, history_years: int = 2) -> pd.DataFrame:
    """Features from years target_year-history_years .. target_year-1, targets from target_year."""
    b = universe if universe is not None else building_universe(inp["pluto"], inp["landlords"])
    A = neighbors if neighbors is not None else neighbor_matrix(b)
    bbls = pd.Index(b["bbl"])
    hist = list(range(target_year - history_years, target_year))
    c, v = inp["complaints"], inp["violations"]

    # Own history
    own_c = counts_by(c, bbls, hist, "problem_type", TARGET_TYPES, "c_", distinct="complaint_id")
    total_c = c[c["year"].isin(hist) & c["bbl"].isin(bbls)].groupby("bbl")["complaint_id"].nunique()
    emerg = c[c["year"].isin(hist) & c["is_emergency"]].groupby("bbl")["complaint_id"].nunique()
    own_v_cls = counts_by(v, bbls, hist, "class", ["A", "B", "C"], "v_class_")
    own_v_type = counts_by(v, bbls, hist, "problem_type", TARGET_TYPES, "v_")
    bb = inp["bedbugs"]
    bb_last = bb[bb["year"] == target_year - 1].groupby("bbl")["has_infestation"].max()
    bb_years = bb[bb["year"].isin(hist) & bb["has_infestation"]].groupby("bbl")["year"].nunique()
    ev = inp["evictions"]
    ev_hist = ev[ev["year"].isin(hist)].groupby("bbl").size()

    X = b.set_index("bbl").copy()
    X = X.join(own_c).join(own_v_cls).join(own_v_type)
    X["c_total"] = total_c.reindex(bbls).fillna(0).values
    X["c_emergency"] = emerg.reindex(bbls).fillna(0).values
    X["c_total_per_unit"] = X["c_total"] / X["unitsres"]
    X["bedbugs_last_year"] = bb_last.reindex(bbls).astype("boolean").fillna(False).astype(bool).values
    X["bedbug_years"] = bb_years.reindex(bbls).fillna(0).values
    X["evictions"] = ev_hist.reindex(bbls).fillna(0).values
    X["evictions_per_unit"] = X["evictions"] / X["unitsres"]

    # Landlord record (other buildings only)
    ll = landlord_leave_one_out(own_c.reindex(bbls), X["unitsres"], X["landlord_id"])
    ll.index = bbls
    X = X.join(ll)
    X["has_portfolio"] = X["portfolio_buildings"] > 1

    # 100 m neighborhood: other buildings' complaints, and how many buildings are around
    nb_counts = A @ own_c.reindex(bbls).to_numpy(dtype=float)
    for j, t in enumerate(TARGET_TYPES):
        X[f"nb100_{slug(t)}"] = nb_counts[:, j]
    X["nb100_buildings"] = np.asarray(A.sum(axis=1)).ravel()

    # Targets
    y = counts_by(c, bbls, [target_year], "problem_type", TARGET_TYPES, "y_", distinct="complaint_id")
    X = X.join((y > 0).astype("int8"))
    X["target_year"] = target_year
    return X.reset_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except ids, coordinates, targets and text labels."""
    drop = {"bbl", "landlord_id", "landlord_key_type", "zipcode", "latitude", "longitude", "bldgclass",
            "yearbuilt", "target_year", "era", "borough"}
    return [c for c in df.columns if c not in drop and not c.startswith("y_")]


def target_columns() -> list[str]:
    return [f"y_{slug(t)}" for t in TARGET_TYPES]
