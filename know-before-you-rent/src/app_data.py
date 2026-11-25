"""Precompute every building's renter report for the demo app.

    python -m src.app_data

1. Rebuild the feature table for the most recent history (2025-26) for all
   residential buildings with 3+ apartments.
2. Train the production models on the 2025 table (history 2023-24 → 2025
   outcomes), the latest year with complete outcomes.
3. Score every building: probability and risk level per problem type, plus
   the top three reasons (from the logistic model's contributions).
4. Add the red flags a renter cares about right now: open class C violations,
   bedbugs last year, evictions, crime within 200 m, and the landlord.
5. Save one table the Flask app reads at startup, so the app needs no models.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from src import address, config
from src import features_housing as F
from src import model_housing as M
from src.clean_housing import OUT

log = logging.getLogger(__name__)

APP_FILE = config.PROCESSED_DIR / "app_buildings.parquet"
BASE_RATES_FILE = config.PROCESSED_DIR / "app_base_rates.json"
ADDRESS_FILE = config.PROCESSED_DIR / "app_addresses.parquet"
APP_MODELS = M.MODELS_DIR / "app_models.joblib"
TRAIN_YEAR = 2025          # latest year with complete outcomes
CURRENT_YEAR = 2027        # "next year" for the current history window (2025-26)


def train_production(train: pd.DataFrame, cols: list[str]) -> dict:
    """Gradient boosting for the probabilities, logistic regression for the explanations."""
    X = M.to_matrix(train, cols)
    models = {}
    for p in F.TARGET_TYPES:
        y = train[f"y_{F.slug(p)}"]
        models[(p, "gbm")] = M.make_model("gbm").fit(X, y)
        models[(p, "logit")] = M.make_model("logit").fit(X.fillna(0), y)
        log.info("trained %s", p)
    return models


def top_reasons(logit, X: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Names of the k features pushing each building's risk up the most (vectorized)."""
    scaler, lr = logit.named_steps["standardscaler"], logit.named_steps["logisticregression"]
    contrib = (X.fillna(0).to_numpy() - scaler.mean_) / scaler.scale_ * lr.coef_[0]
    top = np.argsort(-contrib, axis=1)[:, :k]
    names = np.array(X.columns)[top]
    positive = np.take_along_axis(contrib, top, axis=1) > 0
    return np.where(positive, names, None)


def landlord_label(key: pd.Series) -> pd.Series:
    """Human-readable landlord from the portfolio key."""
    kind = key.str.split(":").str[0]
    value = key.str.split(":", n=1).str[1]
    out = value.where(kind != "AGENT_P", value.str.split("|").str[0] + " (managing agent)")
    out = out.where(kind != "OWNER_ADDR", "owner office at " + value.fillna(""))
    return out.str.title()


def red_flags(bbls: pd.Index) -> pd.DataFrame:
    """Current red flags per building."""
    v = pd.read_parquet(OUT["hpd_violations"], columns=["bbl", "class", "is_open", "is_lead"])
    open_v = v[v["is_open"]]
    bb = pd.read_parquet(OUT["bedbugs"], columns=["bbl", "year", "infested_dwelling_unit_count"])
    latest_bb = bb.sort_values("year").drop_duplicates("bbl", keep="last").set_index("bbl")
    ev = pd.read_parquet(OUT["evictions"], columns=["bbl"])
    flags = pd.DataFrame(index=bbls)
    flags["open_class_c"] = open_v[open_v["class"] == "C"].groupby("bbl").size()
    flags["open_violations"] = open_v.groupby("bbl").size()
    flags["open_lead"] = open_v[open_v["is_lead"]].groupby("bbl").size()
    flags["bedbug_year"] = latest_bb["year"]
    flags["bedbug_infested_units"] = latest_bb["infested_dwelling_unit_count"]
    flags["evictions_since_2022"] = ev.groupby("bbl").size()
    count_cols = ["open_class_c", "open_violations", "open_lead", "evictions_since_2022"]
    flags[count_cols] = flags[count_cols].fillna(0).astype(int)
    return flags


def address_aliases(buildings: pd.DataFrame) -> pd.DataFrame:
    """Every way a building's address is written: PLUTO's main address plus HPD registration addresses.

    Many buildings span several house numbers (e.g. 235-245 E 92 St) but PLUTO
    lists only one; HPD registrations record the range, so we add its low,
    high and listed numbers as extra searchable spellings.
    """
    from src import housing as H
    reg = H.load("hpd_registrations", ["boroid", "block", "lot", "housenumber", "lowhousenumber",
                                       "highhousenumber", "streetname"])
    reg["bbl"] = H.bbl_from_parts(reg["boroid"], reg["block"], reg["lot"])
    reg = reg[reg["bbl"].isin(buildings["bbl"])]
    alias = pd.concat([reg[["bbl", c, "streetname"]].rename(columns={c: "number"})
                       for c in ["housenumber", "lowhousenumber", "highhousenumber"]])
    alias = alias.dropna()
    alias["address_key"] = address.normalize_series(alias["number"].astype(str) + " " + alias["streetname"].astype(str))
    main = buildings[["bbl", "address_key"]]
    keys = pd.concat([main, alias[["bbl", "address_key"]]]).dropna().drop_duplicates()
    keys = keys[keys["address_key"].str.len() > 3]
    info = buildings[["bbl", "address", "borough", "zipcode", "unitsres"]]
    return keys.merge(info, on="bbl", how="left")


def build() -> pd.DataFrame:
    inp = F.load_inputs()
    universe = F.building_universe(inp["pluto"], inp["landlords"])
    A = F.neighbor_matrix(universe)
    train = F.build_snapshot(TRAIN_YEAR, inp, universe, A)
    current = F.build_snapshot(CURRENT_YEAR, inp, universe, A)
    cols = F.feature_columns(train)

    models = train_production(train, cols)
    base = {p: float(train[f"y_{F.slug(p)}"].mean()) for p in F.TARGET_TYPES}
    joblib.dump({"features": cols, "base_rates": base, "models": models}, APP_MODELS)

    X = M.to_matrix(current, cols)
    out = current.drop(columns=[c for c in current.columns if c.startswith("y_")]).copy()
    for p in F.TARGET_TYPES:
        s = F.slug(p)
        prob = models[(p, "gbm")].predict_proba(X)[:, 1]
        out[f"p_{s}"] = prob
        out[f"level_{s}"] = M.risk_level(prob, base[p])
        reasons = top_reasons(models[(p, "logit")], X)
        for k in range(reasons.shape[1]):
            out[f"why_{s}_{k + 1}"] = reasons[:, k]

    pluto = inp["pluto"].set_index("bbl")
    out["address"] = out["bbl"].map(pluto["address"]).astype("string").str.title()
    out["address_key"] = address.normalize_series(out["address"])
    ll = pd.read_parquet(OUT["landlords"], columns=["bbl", "landlord_key"]).set_index("bbl")["landlord_key"]
    out["landlord_name"] = landlord_label(out["bbl"].map(ll).astype("string"))
    out = out.join(red_flags(pd.Index(out["bbl"])), on="bbl")
    out["overall_risk"] = out[[f"p_{F.slug(p)}" for p in F.TARGET_TYPES]].mean(axis=1)
    out.to_parquet(APP_FILE, index=False)
    BASE_RATES_FILE.write_text(json.dumps(base, indent=2))
    address_aliases(out).to_parquet(ADDRESS_FILE, index=False)
    log.info("saved %s buildings → %s", f"{len(out):,}", APP_FILE)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    build()
