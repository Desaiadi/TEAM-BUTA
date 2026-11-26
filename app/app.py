"""Team BUTA: the Flask app.

* ``/``                 Explore. Pick one building or two on the 3D map. "Check"
                        shows what to expect inside each building next year (the
                        model's predictions, red flags, history, landlord, crime)
                        and the live 311 picture on the street around it.
* ``/building/<bbl>``   The full report for one building, as a page of its own.
* ``/insights``         The datasets, the findings and the model evaluation.

Explore reads its building data from three JSON endpoints:

* ``/api/nearest?lat=&lng=``   the apartment building nearest a map click
* ``/api/search?q=``           address search, with each building's coordinates
* ``/api/building/<bbl>``      one building's report and chart series

Its 311 data comes from NYC Open Data straight from the browser.

Run from the repository root:

    python app/app.py          # then open http://127.0.0.1:5050

Everything on the building side is precomputed, so the app only loads a few
files at startup and needs no models. That output is not in the repository; if
it is missing the app still starts, Explore keeps its map and live 311
comparison, and it explains which pipeline steps to run.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
app = Flask(__name__)

# Why the building data is unavailable, if it is: "deps" or "data".
# Explore's map and 311 comparison never depend on either.
DATA_STATUS: str | None = None

# A map click snaps to the nearest apartment building within this distance.
SNAP_METERS = 150
M_PER_DEG_LAT = 111_320


# ---------------------------------------------------------------- data (loaded once)
def load_buildings() -> SimpleNamespace | None:
    """Read the precomputed building tables, or return None if they aren't available.

    The pipeline's output is not in the repository and the analysis stack is a
    heavier install than the map needs, so both are treated as normal states
    the app reports rather than as crashes. Imports happen here, not at module
    level, so `/` serves with nothing but Flask and pandas installed.
    """
    global DATA_STATUS
    try:
        # These drag in scipy, scikit-learn, joblib and pyarrow, so they stay out
        # of the module's own imports.
        from src import address, report
        from src.app_data import ADDRESS_FILE, APP_FILE, BASE_RATES_FILE
        from src.features_housing import TARGET_TYPES, slug
        from src.insights import HISTORY_FILE, INSIGHTS_FILE
    except ImportError as exc:
        DATA_STATUS = "deps"
        log.warning("building data disabled, missing dependency: %s", exc)
        return None

    missing = [p for p in (APP_FILE, ADDRESS_FILE, BASE_RATES_FILE, INSIGHTS_FILE, HISTORY_FILE)
               if not p.exists()]
    if missing:
        DATA_STATUS = "data"
        log.warning("building data disabled, %d pipeline file(s) missing, first: %s",
                    len(missing), missing[0])
        return None

    try:
        buildings = pd.read_parquet(APP_FILE)
        # Every spelling of every building's address (PLUTO + HPD registration ranges)
        search_index = pd.read_parquet(ADDRESS_FILE).dropna(subset=["address"])
        history = pd.read_parquet(HISTORY_FILE).set_index("bbl")
    except ImportError as exc:                 # pandas needs pyarrow to read parquet
        DATA_STATUS = "deps"
        log.warning("building data disabled, cannot read parquet: %s", exc)
        return None

    buildings["zipcode"] = buildings["zipcode"].astype("string")
    search_index["zipcode"] = search_index["zipcode"].astype("string")
    units = buildings["unitsres"].sum()
    # Coordinates as plain arrays for the nearest-building lookup; a missing
    # coordinate becomes +inf so it can never be the nearest.
    lat = np.nan_to_num(buildings["latitude"].to_numpy(dtype=float), nan=np.inf)
    lon = np.nan_to_num(buildings["longitude"].to_numpy(dtype=float), nan=np.inf)
    return SimpleNamespace(
        buildings=buildings,
        by_bbl=buildings.set_index("bbl", drop=False),
        search_index=search_index,
        lat=lat,
        lon=lon,
        base_rates=json.loads(BASE_RATES_FILE.read_text()),
        city={
            # citywide complaints per 100 apartments, per problem type
            # (the same scale as the landlord rates)
            "ll_rate": {t: buildings[f"c_{slug(t)}"].sum() / units * 100 for t in TARGET_TYPES},
            "crime": {c[6:]: float(buildings[c].median())
                      for c in buildings.columns if c.startswith("crime_")},
        },
        insights=json.loads(INSIGHTS_FILE.read_text()),   # built by `python -m src.insights`
        history=history,
        history_years=sorted(history["year"].unique().tolist()),
        # Kept so the routes don't need the heavy imports themselves
        address=address,
        report=report,
        target_types=TARGET_TYPES,
    )


D = load_buildings()


def building_charts(bbl: str, data: dict) -> dict:
    """Series for a building's charts, as plain lists."""
    years = D.history_years
    h = D.history.loc[[bbl]] if bbl in D.history.index else D.history.iloc[0:0]
    table = (h.pivot_table(index="year", columns="series", values="n", aggfunc="sum")
             .reindex(years).fillna(0))
    col = lambda name: [int(x) for x in table[name]] if name in table else [0] * len(years)
    return {
        "years": [str(y) + ("*" if y == years[-1] else "") for y in years],
        "complaints": {t: col(t) for t in D.target_types},
        "violations": {k: col(f"class {k}") for k in "ABC"},
        "risk": {"labels": [r["label"].split(" (")[0] for r in data["risks"]],
                 "building": [r["prob"] for r in data["risks"]],
                 "typical": [D.base_rates[r["type"]] for r in data["risks"]]},
        "landlord": {"labels": [r["label"].split(" (")[0] for r in data["landlord_rates"]],
                     "landlord": [None if pd.isna(r["landlord"]) else float(r["landlord"])
                                  for r in data["landlord_rates"]],
                     "city": [float(r["city"]) for r in data["landlord_rates"]]},
        "crime": {"labels": list(data["crime"]),
                  "here": [float(v["here"]) for v in data["crime"].values()],
                  "city": [float(v["city"]) for v in data["crime"].values()]},
    }


def building_payload(bbl: str) -> tuple[dict, list[dict], dict]:
    """One building's report, its landlord's nearby buildings, and its chart series."""
    row = D.by_bbl.loc[bbl]
    data = D.report.building_report(row, D.base_rates, D.city)
    for r in data["risks"]:
        r["typical"] = D.base_rates[r["type"]]   # the share of all buildings with it

    # The landlord's other buildings, for the map (nearest 150 within ~3 km)
    same = D.buildings[(D.buildings["landlord_id"] == row["landlord_id"])
                       & (D.buildings["bbl"] != bbl)]
    if row["portfolio_buildings"] > 1 and len(same):
        d = np.hypot(same["latitude"] - row["latitude"],
                     (same["longitude"] - row["longitude"]) * 0.76)
        same = same.assign(_d=d)[d < 0.03].nsmallest(150, "_d")
    else:
        same = same.iloc[0:0]
    others = [{"lat": r.latitude, "lon": r.longitude, "address": r.address, "bbl": r.bbl,
               "risk": float(r.overall_risk)} for r in same.itertuples()]
    return data, others, building_charts(bbl, data)


def jsonable(x):
    """Plain JSON values: numpy scalars unwrapped, NaN / NA / inf as null.

    Flask would write NaN as a bare token, which browsers refuse to parse.
    """
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if x is None or isinstance(x, (bool, int, str)):
        return x
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    """Explore: one or two buildings on the map, their reports, and the street around them."""
    return render_template("explore.html", data_ready=D is not None, reason=DATA_STATUS,
                           type_order=D.target_types if D is not None else [])


@app.get("/building/<bbl>")
def building(bbl: str):
    if D is None:
        return redirect(url_for("index"))
    if bbl not in D.by_bbl.index:
        abort(404)
    data, others, charts = building_payload(bbl)
    return render_template("building.html", b=data, others=others, charts=charts,
                           type_order=D.target_types)


@app.get("/insights")
def insights():
    if D is None:
        return redirect(url_for("index"))
    return render_template("insights.html", ins=D.insights, type_order=D.target_types)


# The building search used to have a page of its own; it now lives in Explore.
@app.get("/rent")
def rent():
    return redirect(url_for("index"))


@app.get("/search")
def search_page():
    return redirect(url_for("index"))


# ---------------------------------------------------------------- API
def _coords(bbl: str) -> tuple[float | None, float | None]:
    if bbl not in D.by_bbl.index:
        return None, None
    lat, lon = D.by_bbl.at[bbl, "latitude"], D.by_bbl.at[bbl, "longitude"]
    return (None, None) if pd.isna(lat) or pd.isna(lon) else (float(lat), float(lon))


@app.get("/api/search")
def api_search():
    if D is None:
        return jsonify([])
    hits = D.address.search(D.search_index, request.args.get("q", ""), limit=8)
    out = []
    for h in hits.itertuples():
        lat, lon = _coords(h.bbl)
        out.append({"bbl": h.bbl, "address": h.address, "borough": h.borough,
                    "zipcode": h.zipcode, "units": int(h.unitsres), "lat": lat, "lon": lon})
    return jsonify(jsonable(out))


@app.get("/api/nearest")
def api_nearest():
    """The apartment building nearest a point, if there is one within SNAP_METERS."""
    if D is None:
        return jsonify({"found": False, "reason": "no-data"})
    try:
        lat, lng = float(request.args["lat"]), float(request.args["lng"])
    except (KeyError, ValueError):
        abort(400)
    dy = (D.lat - lat) * M_PER_DEG_LAT
    dx = (D.lon - lng) * M_PER_DEG_LAT * math.cos(math.radians(lat))
    dist = np.hypot(dx, dy)
    i = int(np.argmin(dist))
    if not dist[i] <= SNAP_METERS:
        return jsonify({"found": False, "distance_m": None if math.isinf(dist[i]) else round(float(dist[i]))})
    row = D.buildings.iloc[i]
    return jsonify(jsonable({
        "found": True, "bbl": row["bbl"], "address": row["address"], "borough": row["borough"],
        "units": int(row["unitsres"]), "lat": row["latitude"], "lon": row["longitude"],
        "distance_m": round(float(dist[i])),
    }))


@app.get("/api/building/<bbl>")
def api_building(bbl: str):
    if D is None:
        return jsonify({"error": "building data is not built"}), 503
    if bbl not in D.by_bbl.index:
        abort(404)
    data, _, charts = building_payload(bbl)
    return jsonify(jsonable({"report": data, "charts": charts,
                             "url": url_for("building", bbl=bbl)}))


@app.template_filter("pct")
def pct(x):
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


@app.template_filter("num")
def num(x, digits=0):
    return "–" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if D is None:
        log.warning("No processed building data found. Explore's map and 311 comparison "
                    "work; it explains which pipeline steps to run for the building reports.")
    app.run(debug=False, port=5050)
