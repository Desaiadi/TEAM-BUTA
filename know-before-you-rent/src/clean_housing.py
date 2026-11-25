"""Cleaning steps for the Know Before You Rent housing datasets.

Every function takes the raw DataFrame plus a `CleaningReport` (from
src/preprocess.py), records what it changed, and returns a clean frame
keyed by a validated 10-digit BBL. The cleaning notebooks call these one by
one; `clean_all()` runs everything as a script:

    python -m src.clean_housing
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import config, housing as H
from src.preprocess import CleaningReport, normalize_bbl

log = logging.getLogger(__name__)

CLEAN_DIR = config.PROCESSED_DIR
OUT = {name: CLEAN_DIR / f"{name}_clean.parquet"
       for name in ("hpd_complaints", "hpd_violations", "bedbugs", "evictions", "landlords")}

# --------------------------------------------------------------------------
# One vocabulary of renter-facing problem types, shared by complaints and violations
# --------------------------------------------------------------------------
PROBLEM_TYPES = ["heat/hot water", "pests", "leaks/mold", "plumbing", "paint/plaster",
                 "doors/windows", "electric", "safety", "sanitation", "other"]

COMPLAINT_CATEGORY_MAP = {
    "HEAT/HOT WATER": "heat/hot water",
    "WATER LEAK": "leaks/mold",
    "PLUMBING": "plumbing",
    "PAINT/PLASTER": "paint/plaster",
    "DOOR/WINDOW": "doors/windows",
    "ELECTRIC": "electric",
    "SAFETY": "safety",
}
# UNSANITARY CONDITION is split by its minor category
UNSANITARY_MINOR_MAP = {"PESTS": "pests", "MOLD": "leaks/mold"}

VIOLATION_TOPIC_MAP = {
    "heat/hot water": "heat/hot water", "pests": "pests", "mold/leaks": "leaks/mold",
    "plumbing": "plumbing", "lead paint": "paint/plaster", "paint/plaster": "paint/plaster",
    "windows/guards": "doors/windows", "doors/locks": "doors/windows", "electric": "electric",
    "fire safety": "safety",
}


def _start(df: pd.DataFrame, rep: CleaningReport, label: str) -> pd.DataFrame:
    rep.add("load_raw", len(df), len(df), label)
    return df.copy()


def _dedupe(df: pd.DataFrame, key: str, rep: CleaningReport) -> pd.DataFrame:
    n = len(df)
    df = df.drop_duplicates(subset=key, keep="last")
    rep.add(f"drop_duplicate_{key}", n, len(df))
    return df


def _parse_dates(df: pd.DataFrame, cols: list[str], required: str, rep: CleaningReport,
                 start: str = config.HOUSING_START_DATE) -> pd.DataFrame:
    """Parse date columns; drop rows whose `required` date is missing or before `start`."""
    n = len(df)
    for c in cols:
        df[c] = pd.to_datetime(df[c].astype("string"), errors="coerce")
    df = df[df[required].notna() & (df[required] >= pd.Timestamp(start))]
    rep.add("parse_dates", n, len(df), f"missing or pre-{start[:4]} {required}")
    return df


def _valid_bbl(df: pd.DataFrame, rep: CleaningReport, col: str = "bbl") -> pd.DataFrame:
    """Normalize BBL to 10 digits and drop rows without a valid one (the join key for everything)."""
    n = len(df)
    df[col] = normalize_bbl(df[col].astype("string"))
    df = df[df[col].notna()]
    rep.add("valid_bbl", n, len(df), "missing or invalid BBL")
    return df


# --------------------------------------------------------------------------
# HPD complaints
# --------------------------------------------------------------------------
def complaint_problem_type(major: pd.Series, minor: pd.Series) -> pd.Series:
    """Map HPD major/minor categories onto PROBLEM_TYPES."""
    major = major.astype("string").str.upper().str.strip()
    minor = minor.astype("string").str.upper().str.strip()
    out = major.map(COMPLAINT_CATEGORY_MAP)
    unsan = major.eq("UNSANITARY CONDITION")
    out = out.mask(unsan, minor.map(UNSANITARY_MINOR_MAP).fillna("sanitation"))
    return out.fillna("other").astype("string")


def clean_complaints(raw: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    df = _start(raw, rep, f"{raw['complaint_id'].nunique():,} complaints")
    df = _dedupe(df, "problem_id", rep)
    df = _parse_dates(df, ["received_date", "problem_status_date"], "received_date", rep)
    df = _valid_bbl(df, rep)

    n = len(df)
    df["problem_type"] = complaint_problem_type(df["major_category"], df["minor_category"])
    df["is_emergency"] = df["type"].astype("string").str.upper().isin(["EMERGENCY", "IMMEDIATE EMERGENCY"])
    df["is_duplicate_report"] = df["problem_duplicate_flag"].astype("string").eq("Y")
    apt = df["apartment"].astype("string").str.upper().str.strip()
    df["is_building_wide"] = apt.isin(["BLDG", "", "BUILDING"]) | apt.isna() | \
        df["unit_type"].astype("string").str.upper().isin(["BUILDING-WIDE", "PUBLIC AREA"])
    df["is_open"] = df["problem_status"].astype("string").str.upper().eq("OPEN")
    df["year"] = df["received_date"].dt.year.astype("int16")
    rep.add("derive_fields", n, len(df), "problem_type, emergency, duplicate, building-wide, open flags",
            problem_types=df["problem_type"].value_counts().to_dict())

    keep = ["problem_id", "complaint_id", "bbl", "received_date", "year", "apartment", "major_category",
            "minor_category", "problem_code", "problem_type", "is_emergency", "is_duplicate_report",
            "is_building_wide", "is_open", "problem_status_date"]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# HPD violations
# --------------------------------------------------------------------------
def clean_violations(raw: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    df = _start(raw, rep, "")
    df = _dedupe(df, "violationid", rep)
    df = _parse_dates(df, ["inspectiondate", "novissueddate", "currentstatusdate"], "inspectiondate", rep)
    df = _valid_bbl(df, rep)

    n = len(df)
    df["class"] = df["class"].astype("string").str.upper().str.strip()
    df = df[df["class"].isin(["A", "B", "C", "I"])]
    rep.add("valid_class", n, len(df), "class not in A/B/C/I")

    n = len(df)
    df["topic"] = H.violation_topic(df["novdescription"])
    df["problem_type"] = df["topic"].map(VIOLATION_TOPIC_MAP).fillna("other").astype("string")
    df["is_lead"] = df["topic"].eq("lead paint")
    df["is_open"] = df["violationstatus"].astype("string").str.upper().eq("OPEN")
    df["is_rent_impairing"] = df["rentimpairing"].astype("string").str.upper().eq("Y")
    df["year"] = df["inspectiondate"].dt.year.astype("int16")
    rep.add("derive_fields", n, len(df), "topic via regex, problem_type, open/lead/rent-impairing flags",
            tagged_share=round(float((df["topic"] != "other").mean()), 4))

    keep = ["violationid", "bbl", "inspectiondate", "novissueddate", "year", "apartment", "class",
            "ordernumber", "topic", "problem_type", "is_lead", "is_open", "is_rent_impairing", "currentstatus"]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Bedbugs
# --------------------------------------------------------------------------
def clean_bedbugs(raw: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    df = _start(raw, rep, "")
    df = _valid_bbl(df, rep)
    n = len(df)
    counts = ["of_dwelling_units", "infested_dwelling_unit_count", "eradicated_unit_count", "re_infested_dwelling_unit"]
    for c in counts:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["filing_date"] = pd.to_datetime(df["filing_date"].astype("string"), errors="coerce")
    df["period_end"] = pd.to_datetime(df["filling_period_end_date"].astype("string"), errors="coerce")
    bad = (df["infested_dwelling_unit_count"] > df["of_dwelling_units"]) | (df[counts] < 0).any(axis=1) \
        | df["period_end"].isna() | df["of_dwelling_units"].fillna(0).le(0)
    df = df[~bad]
    rep.add("drop_impossible", n, len(df), "infested > units, negative counts, no period or units")

    n = len(df)
    df["year"] = df["period_end"].dt.year.astype("int16")
    # One filing per building per year: keep the latest filed
    df = df.sort_values("filing_date").drop_duplicates(["bbl", "year"], keep="last")
    rep.add("one_filing_per_building_year", n, len(df))
    df["has_infestation"] = df["infested_dwelling_unit_count"] > 0
    df["infested_share"] = df["infested_dwelling_unit_count"] / df["of_dwelling_units"]
    keep = ["bbl", "year", "period_end", "of_dwelling_units", "infested_dwelling_unit_count",
            "eradicated_unit_count", "re_infested_dwelling_unit", "has_infestation", "infested_share"]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Evictions
# --------------------------------------------------------------------------
def clean_evictions(raw: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    df = _start(raw, rep, "")
    n = len(df)
    df = df[df["residential_commercial_ind"].astype("string").str.upper().str.startswith("RES").fillna(False)]
    rep.add("residential_only", n, len(df), "commercial evictions removed")
    df = _parse_dates(df, ["executed_date"], "executed_date", rep)
    df = _valid_bbl(df, rep)
    n = len(df)
    df = df.drop_duplicates(["court_index_number", "docket_number", "bbl", "executed_date"])
    rep.add("drop_duplicates", n, len(df), "same case, docket, building and date")
    df["year"] = df["executed_date"].dt.year.astype("int16")
    return df[["bbl", "executed_date", "year", "eviction_apt_num", "court_index_number"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Landlords: registrations + contacts -> one landlord (portfolio) per building
# --------------------------------------------------------------------------
KEY_PRIORITY = ["AGENT_CO", "AGENT_P", "OWNER_ADDR"]


def landlord_keys(reg: pd.DataFrame, con: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (building, linking key) from the latest registration of each building.

    Keys that link buildings to the same landlord:
      AGENT_CO   managing-agent company name (normalized)
      AGENT_P    managing-agent person + business address (a name alone is too common)
      OWNER_ADDR owner / head-officer business address
    """
    reg = reg.assign(bbl=H.bbl_from_parts(reg["boroid"], reg["block"], reg["lot"]),
                     last=pd.to_datetime(reg["lastregistrationdate"].astype("string"), errors="coerce"))
    latest = reg.dropna(subset=["bbl"]).sort_values("last").drop_duplicates("bbl", keep="last")
    c = con.merge(latest[["registrationid", "bbl"]], on="registrationid", how="inner")
    c["corp"] = H.normalize_name(c["corporationname"])
    c["person"] = H.normalize_name(c["firstname"].astype("string").fillna("") + " " +
                                   c["lastname"].astype("string").fillna(""))
    c["addr"] = H.normalize_address(c["businesshousenumber"], c["businessstreetname"], c["businesszip"])
    agents = c[c["type"] == "Agent"]
    owners = c[c["type"].isin(["CorporateOwner", "HeadOfficer", "IndividualOwner"])]
    keys = pd.concat([
        agents.dropna(subset=["corp"]).assign(key="AGENT_CO:" + agents["corp"]),
        agents.dropna(subset=["person", "addr"]).assign(key="AGENT_P:" + agents["person"] + "|" + agents["addr"]),
        owners.dropna(subset=["addr"]).assign(key="OWNER_ADDR:" + owners["addr"]),
    ])[["bbl", "key"]].dropna().drop_duplicates()
    return keys, latest


def build_portfolios(keys: pd.DataFrame) -> pd.DataFrame:
    """Give every building ONE landlord key, by priority, and group buildings that share it.

    Priority: managing-agent company > agent person at an address > owner's
    office address. We deliberately do not chain keys transitively (A shares
    an agent with B, B shares an office with C, ...): in testing that merged
    18,703 unrelated buildings into one "landlord".
    """
    k = keys.assign(kind=keys["key"].str.split(":").str[0])
    k["rank"] = k["kind"].map({kind: i for i, kind in enumerate(KEY_PRIORITY)})
    # A building listing several keys of the same kind: take the most common key (largest group)
    k["key_size"] = k.groupby("key")["bbl"].transform("nunique")
    best = k.sort_values(["rank", "key_size"], ascending=[True, False]).drop_duplicates("bbl")
    best["landlord_id"] = "L" + best["key"].astype("category").cat.codes.astype(str).str.zfill(6)
    best["portfolio_buildings"] = best.groupby("landlord_id")["bbl"].transform("size")
    return best[["bbl", "landlord_id", "kind", "key", "portfolio_buildings"]].rename(
        columns={"kind": "landlord_key_type", "key": "landlord_key"})


def clean_landlords(reg: pd.DataFrame, con: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """One row per registered building: landlord_id, portfolio size and the main agent name."""
    rep.add("load_raw", len(reg), len(reg), f"{len(con):,} contacts")
    keys, latest = landlord_keys(reg, con)
    rep.add("latest_registration_per_bbl", len(reg), len(latest), "older registrations of the same building")
    port = build_portfolios(keys)
    rep.add("build_portfolios", len(latest), len(port), f"{port['landlord_id'].nunique():,} landlords",
            largest_portfolio=int(port["portfolio_buildings"].max()))

    agent = keys[keys["key"].str.startswith("AGENT_CO:")].drop_duplicates("bbl")
    out = (latest[["bbl", "registrationid", "last", "registrationenddate"]]
           .merge(port, on="bbl", how="left")
           .merge(agent.assign(agent_company=agent["key"].str[9:])[["bbl", "agent_company"]], on="bbl", how="left"))
    out["landlord_id"] = out["landlord_id"].fillna("L_" + out["bbl"])          # building with no usable keys
    out["portfolio_buildings"] = out["portfolio_buildings"].fillna(1).astype(int)
    out["registration_active"] = pd.to_datetime(out["registrationenddate"].astype("string"),
                                                errors="coerce") >= pd.Timestamp.today()
    return out.rename(columns={"last": "last_registration"}).drop(columns="registrationenddate")


# --------------------------------------------------------------------------
def clean_all() -> None:
    """Clean every housing dataset and save to data/processed/ with a JSON report."""
    from src.preprocess import clean_pluto   # PLUTO building data, shared with the 311 pipeline

    pluto, rep_p = clean_pluto()
    pluto.to_parquet(config.CLEAN_PLUTO_FILE, index=False)
    reports = [rep_p.to_dict()]
    jobs = {
        "hpd_complaints": lambda r: clean_complaints(H.load("hpd_complaints"), r),
        "hpd_violations": lambda r: clean_violations(H.load("hpd_violations"), r),
        "bedbugs": lambda r: clean_bedbugs(H.load("bedbugs"), r),
        "evictions": lambda r: clean_evictions(H.load("evictions"), r),
        "landlords": lambda r: clean_landlords(H.load("hpd_registrations"), H.load("hpd_contacts"), r),
    }
    for name, job in jobs.items():
        rep = CleaningReport(name)
        df = job(rep)
        df.to_parquet(OUT[name], index=False)
        reports.append(rep.to_dict())
        log.info("%s: %s rows -> %s", name, f"{len(df):,}", OUT[name].name)
    (config.REPORTS_DIR / "cleaning_report_housing.json").write_text(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    clean_all()
