"""Know Before You Rent: Flask demo.

Run from the subproject folder (after `python -m src.app_data`):

    python app/app.py          # then open http://127.0.0.1:5050

Everything is precomputed, so the app only loads one table at startup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import address, report                           # noqa: E402
from src.app_data import ADDRESS_FILE, APP_FILE, BASE_RATES_FILE   # noqa: E402
from src.features_housing import TARGET_TYPES, slug        # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------- data (loaded once)
BUILDINGS = pd.read_parquet(APP_FILE)
BUILDINGS["zipcode"] = BUILDINGS["zipcode"].astype("string")
BY_BBL = BUILDINGS.set_index("bbl", drop=False)
# Every spelling of every building's address (PLUTO + HPD registration ranges)
SEARCH_INDEX = pd.read_parquet(ADDRESS_FILE).dropna(subset=["address"])
SEARCH_INDEX["zipcode"] = SEARCH_INDEX["zipcode"].astype("string")
BASE_RATES = json.loads(BASE_RATES_FILE.read_text())

units = BUILDINGS["unitsres"].sum()
CITY = {
    # citywide complaints per 100 apartments, per problem type (same scale as the landlord rates)
    "ll_rate": {t: BUILDINGS[f"c_{slug(t)}"].sum() / units * 100 for t in TARGET_TYPES},
    "crime": {c[6:]: float(BUILDINGS[c].median()) for c in BUILDINGS.columns if c.startswith("crime_")},
}


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    examples = (BUILDINGS.dropna(subset=["address"])
                .sort_values("overall_risk")
                .iloc[[-1, len(BUILDINGS) // 2, 0]][["bbl", "address", "borough"]].to_dict("records"))
    return render_template("index.html", examples=examples, n_buildings=len(BUILDINGS))


@app.get("/building/<bbl>")
def building(bbl: str):
    if bbl not in BY_BBL.index:
        abort(404)
    row = BY_BBL.loc[bbl]
    data = report.building_report(row, BASE_RATES, CITY)

    # The landlord's other buildings, for the map (nearest 150 within ~3 km)
    same = BUILDINGS[(BUILDINGS["landlord_id"] == row["landlord_id"]) & (BUILDINGS["bbl"] != bbl)]
    if row["portfolio_buildings"] > 1 and len(same):
        d = np.hypot(same["latitude"] - row["latitude"], (same["longitude"] - row["longitude"]) * 0.76)
        same = same.assign(_d=d)[d < 0.03].nsmallest(150, "_d")
    else:
        same = same.iloc[0:0]
    others = [{"lat": r.latitude, "lon": r.longitude, "address": r.address, "bbl": r.bbl,
               "risk": float(r.overall_risk)} for r in same.itertuples()]
    return render_template("building.html", b=data, others=others)


# ---------------------------------------------------------------- API
@app.get("/api/search")
def api_search():
    q = request.args.get("q", "")
    hits = address.search(SEARCH_INDEX, q, limit=8)
    return jsonify([{"bbl": h.bbl, "address": h.address, "borough": h.borough, "zipcode": h.zipcode,
                     "units": int(h.unitsres)} for h in hits.itertuples()])


@app.get("/search")
def search_page():
    q = request.args.get("q", "")
    hits = address.search(SEARCH_INDEX, q, limit=20)
    if len(hits) == 1:
        return building(hits["bbl"].iat[0])
    return render_template("index.html", results=hits.to_dict("records"), query=q,
                           examples=[], n_buildings=len(BUILDINGS))


@app.template_filter("pct")
def pct(x):
    return "–" if x is None or pd.isna(x) else f"{x:.0%}"


@app.template_filter("num")
def num(x, digits=0):
    return "–" if x is None or pd.isna(x) else f"{x:,.{digits}f}"


if __name__ == "__main__":
    app.run(debug=False, port=5050)
