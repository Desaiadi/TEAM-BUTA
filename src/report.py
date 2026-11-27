"""Turn one building's precomputed row into a renter-friendly report (used by the Flask app)."""

from __future__ import annotations

import math

import pandas as pd

from src.features_housing import TARGET_TYPES, slug

LABELS = {
    "heat/hot water": "Heat & hot water", "pests": "Pests (roaches, mice, rats)", "leaks/mold": "Leaks & mold",
    "plumbing": "Plumbing", "paint/plaster": "Peeling paint & plaster (lead risk)",
    "doors/windows": "Broken doors & windows", "safety": "Safety (smoke detectors, locks, fire escapes)",
}
TYPE_WORDS = {slug(t): t for t in TARGET_TYPES}


def _type_word(suffix: str) -> str:
    return TYPE_WORDS.get(suffix, suffix.replace("_", " "))


def describe(feature: str, value: float) -> str | None:
    """Plain-English sentence for one model feature, or None if it isn't worth showing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if feature.startswith("c_") and feature not in ("c_total", "c_emergency", "c_total_per_unit"):
        return f"{value:.0f} {_type_word(feature[2:])} complaint(s) in this building in the last 6 months"
    if feature == "c_total":
        return f"{value:.0f} housing complaints in this building in the last 6 months"
    if feature == "c_emergency":
        return f"{value:.0f} emergency complaints in the last 6 months"
    if feature == "c_total_per_unit":
        return f"{value:.1f} complaints per apartment in the last 6 months"
    if feature.startswith("v_class_"):
        return f"{value:.0f} class {feature[-1]} violation(s) issued by inspectors in the last 6 months"
    if feature.startswith("v_"):
        return f"{value:.0f} {_type_word(feature[2:])} violation(s) issued by inspectors in the last 6 months"
    if feature.startswith("ll_rate_"):
        return f"the landlord's other buildings average {value:.1f} {_type_word(feature[8:])} complaints per 100 apartments"
    if feature == "portfolio_buildings":
        return f"the landlord runs {value:.0f} buildings"
    if feature.startswith("nb100_") and feature != "nb100_buildings":
        return f"{value:.0f} {_type_word(feature[6:])} complaints in other buildings within 100 m"
    if feature == "nb100_buildings":
        return f"{value:.0f} other apartment buildings within 100 m"
    if feature == "building_age":
        return f"the building is about {value:.0f} years old"
    if feature == "unitsres":
        return f"a large building ({value:.0f} apartments)"
    if feature == "bedbug_years" or feature == "bedbugs_recent_filing":
        return "bedbugs reported in recent yearly filings"
    if feature.startswith("evictions"):
        return "evictions at this building in the last 6 months"
    if feature.startswith("crime_"):
        return f"{value:.0f} {feature[6:].replace('_', ' ')} crimes reported per year within 200 m"
    return None


def building_report(row: pd.Series, base_rates: dict, city: dict) -> dict:
    """Everything the building page shows, as plain Python values."""
    risks = []
    for t in TARGET_TYPES:
        s = slug(t)
        reasons = [describe(f, row.get(f)) for f in (row.get(f"why_{s}_{k}") for k in (1, 2, 3)) if f]
        risks.append({
            "type": t, "label": LABELS[t], "level": row[f"level_{s}"], "prob": float(row[f"p_{s}"]),
            "times_city": float(row[f"p_{s}"]) / base_rates[t],
            "reasons": [r for r in reasons if r] if row[f"level_{s}"] != "Low" else [],
        })
    risks.sort(key=lambda r: -r["prob"])

    flags = []
    if row["open_class_c"] > 0:
        flags.append(f"{row['open_class_c']} open class C (immediately hazardous) violation(s)")
    if row["open_lead"] > 0:
        flags.append(f"{row['open_lead']} open lead-paint violation(s)")
    if pd.notna(row["bedbug_infested_units"]) and row["bedbug_infested_units"] > 0:
        flags.append(f"bedbugs in {row['bedbug_infested_units']:.0f} apartment(s) in the {row['bedbug_year']:.0f} filing")
    if row["evictions_in_study"] > 0:
        flags.append(f"{row['evictions_in_study']} eviction(s) carried out during Jan 2025–Oct 2026")

    landlord_rates = [{"label": LABELS[t], "landlord": row.get(f"ll_rate_{slug(t)}"), "city": city["ll_rate"][t]}
                      for t in TARGET_TYPES]
    crime = {g: {"here": row.get(f"crime_{g}"), "city": city["crime"].get(g)}
             for g in ["total", "violent", "property", "felony"] if f"crime_{g}" in row}

    return {
        "bbl": row["bbl"], "address": row["address"], "borough": row["borough"], "zipcode": row["zipcode"],
        "yearbuilt": int(row["yearbuilt"]) if pd.notna(row["yearbuilt"]) else "year unknown", "units": int(row["unitsres"]), "floors": row["numfloors"],
        "kind": "Elevator building" if row["is_elevator"] else ("Walk-up" if row["is_walkup"] else "Apartment building"),
        "lat": row["latitude"], "lon": row["longitude"],
        "landlord": row["landlord_name"] if pd.notna(row["landlord_name"]) else None,
        "portfolio": int(row["portfolio_buildings"]), "registered": bool(row["is_registered"]),
        "risks": risks, "flags": flags, "landlord_rates": landlord_rates, "crime": crime,
        "complaints_6m": int(row["c_total"]), "open_violations": int(row["open_violations"]),
        "crime_period": f"{row.get('crime_observed_start', 'unknown')} through {row.get('crime_observed_end', 'unknown')}",
    }
