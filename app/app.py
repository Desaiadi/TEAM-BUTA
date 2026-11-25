"""Know Before You Rent: Flask demo.

Run from the repository root (after `python -m src.app_data`):

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
from src.insights import HISTORY_FILE, INSIGHTS_FILE       # noqa: E402

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
# Chart data (built by `python -m src.insights`)
INSIGHTS = json.loads(INSIGHTS_FILE.read_text())
HISTORY = pd.read_parquet(HISTORY_FILE).set_index("bbl")
HISTORY_YEARS = sorted(HISTORY["year"].unique().tolist())


def building_charts(bbl: str, data: dict) -> dict:
    """Series for the building page's charts, as plain lists."""
    h = HISTORY.loc[[bbl]] if bbl in HISTORY.index else HISTORY.iloc[0:0]
    table = h.pivot_table(index="year", columns="series", values="n", aggfunc="sum").reindex(HISTORY_YEARS).fillna(0)
    col = lambda name: [int(x) for x in table[name]] if name in table else [0] * len(HISTORY_YEARS)
    return {
        "years": [str(y) + ("*" if y == HISTORY_YEARS[-1] else "") for y in HISTORY_YEARS],
        "complaints": {t: col(t) for t in TARGET_TYPES},
        "violations": {k: col(f"class {k}") for k in "ABC"},
        "risk": {"labels": [r["label"].split(" (")[0] for r in data["risks"]],
                 "building": [r["prob"] for r in data["risks"]],
                 "typical": [BASE_RATES[r["type"]] for r in data["risks"]]},
        "landlord": {"labels": [r["label"].split(" (")[0] for r in data["landlord_rates"]],
                     "landlord": [None if pd.isna(r["landlord"]) else float(r["landlord"]) for r in data["landlord_rates"]],
                     "city": [float(r["city"]) for r in data["landlord_rates"]]},
        "crime": {"labels": list(data["crime"]),
                  "here": [float(v["here"]) for v in data["crime"].values()],
                  "city": [float(v["city"]) for v in data["crime"].values()]},
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
    return render_template("building.html", b=data, others=others, charts=building_charts(bbl, data),
                           type_order=TARGET_TYPES)


@app.get("/insights")
def insights():
    return render_template("insights.html", ins=INSIGHTS, type_order=TARGET_TYPES)


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
