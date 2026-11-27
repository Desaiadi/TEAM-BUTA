"""Noise and sanitation around each building: the street side of the 100-meter question.

NYC 311 noise and sanitation complaints, January 2025 to October 2026, used in two ways:

1. Analysis (notebooks/14_street_noise_sanitation.ipynb). The city is cut into
   100 m x 100 m cells, and four questions are asked of them:
     Q1  Do cells in the same neighborhood differ more than neighborhoods differ?
     Q2  Are cells next to each other more alike than random cells?
     Q3  When do noise and sanitation complaints happen?
     Q4  Are 2025's hotspots still hotspots in 2026?
2. The app. `StreetIndex.profile()` describes the 100, 200 or 500 m around any point:
   how it ranks in its community district, when complaints happen, what they
   are about and whether it was a hotspot in both years. `compare_report()`
   turns two profiles into a paragraph of plain English.

Run from the project root (after `python -m src.download --only pluto` and
`--only 311street`):

    python -m src.street

A complaint is a report, not a measurement: it shows where people call 311,
which also depends on who lives there and how willing they are to call.
"""

from __future__ import annotations

import json
import logging
import math
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from src import address, config
from src.preprocess import CleaningReport, clean_coordinates, clean_pluto, drop_duplicate_requests

log = logging.getLogger(__name__)

CLEAN_FILE = config.PROCESSED_DIR / "street_complaints.parquet"
BUILDINGS_FILE = config.PROCESSED_DIR / "street_buildings.parquet"
FINDINGS_FILE = config.REPORTS_DIR / "street_findings.json"
CLEANING_FILE = config.REPORTS_DIR / "cleaning_report_street.json"

GROUPS = list(config.STREET_TYPES)                       # ["noise", "sanitation"]
TYPE_TO_GROUP = {t: g for g, types in config.STREET_TYPES.items() for t in types}
COMPARE_YEARS = (2025, 2026)                             # compared on January-October only
COMPARE_LAST_MONTH = 10
HOTSPOT_SHARE = 0.10                                     # hotspot = top 10% of the district
BOROUGH_NAMES = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}

# Words used by the written report, per group.
WORDS = {
    "noise": {"name": "noise", "more": "noisier", "less": "quieter"},
    "sanitation": {"name": "sanitation", "more": "dirtier", "less": "cleaner"},
}


def study_years() -> float:
    """Length of the study window in years (22 months -> 1.83)."""
    start, end = pd.Timestamp(config.STUDY_START_DATE), pd.Timestamp(config.STUDY_END_EXCLUSIVE)
    return ((end.year - start.year) * 12 + end.month - start.month) / 12


# ==========================================================================
# Geometry
# ==========================================================================
LAT0 = math.radians(40.7)            # one fixed reference latitude, so every file uses the same grid


def to_xy(lat, lon) -> np.ndarray:
    """Latitude/longitude -> x/y in meters (flat-earth projection, <1% error across NYC)."""
    return np.column_stack([np.asarray(lon, dtype=float) * 111_320 * math.cos(LAT0),
                            np.asarray(lat, dtype=float) * 110_540])


def cell_ids(xy: np.ndarray, size: float = config.STREET_CELL_M) -> tuple[np.ndarray, np.ndarray]:
    """Integer grid-cell column and row for each point."""
    return (xy[:, 0] // size).astype(np.int64), (xy[:, 1] // size).astype(np.int64)


# ==========================================================================
# 1. Load and clean
# ==========================================================================
def load_raw() -> pd.DataFrame:
    """Every downloaded (complaint type, month) file in one DataFrame."""
    files = sorted(config.RAW_311_STREET_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No street 311 data. Run `python -m src.download --only 311street`.")
    cols = ["unique_key", "created_date", "complaint_type", "descriptor", "incident_address",
            "latitude", "longitude"]
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def tidy_label(text: str) -> str:
    """'Noise: Construction Before/After Hours (NM1)' -> 'Construction Before/After Hours'."""
    text = re.sub(r"\s*\([A-Z0-9]+\)$", "", str(text).strip())
    text = re.sub(r"^Noise[:,]\s*", "", text).strip() or "Other"
    return text[0].upper() + text[1:]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run every cleaning step in order; the report records what each step removed."""
    rep = CleaningReport("311_street")
    df = raw.copy()
    rep.add("load_raw", len(df), len(df), f"{df['complaint_type'].nunique()} complaint types")

    df = drop_duplicate_requests(df, rep)

    n = len(df)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce", format="ISO8601")
    in_window = df["created_date"].ge(config.STUDY_START_DATE) & df["created_date"].lt(config.STUDY_END_EXCLUSIVE)
    df = df[in_window]
    rep.add("study_window", n, len(df), f"created {config.STUDY_START_DATE} to {config.STUDY_END_DATE}")

    n = len(df)
    df["group"] = df["complaint_type"].map(TYPE_TO_GROUP)
    df = df[df["group"].notna()]
    rep.add("known_types", n, len(df), "complaint type listed in config.STREET_TYPES")

    n = len(df)
    df = clean_coordinates(df).dropna(subset=["latitude", "longitude"])
    rep.add("valid_location", n, len(df), "missing coordinates or outside NYC")

    df = drop_same_day_repeats(df, rep)

    ts = df["created_date"]
    df["year"] = ts.dt.year.astype("int16")
    df["month"] = ts.dt.month.astype("int8")
    df["weekday"] = ts.dt.day_of_week.astype("int8")            # Monday = 0
    df["hour"] = ts.dt.hour.astype("int8")
    night_start, night_end = config.NIGHT_HOURS
    df["is_night"] = (df["hour"] >= night_start) | (df["hour"] < night_end)
    df["is_weekend"] = df["weekday"] >= 5
    # Noise: the descriptor says what it was ("Loud Music/Party"); sanitation: the type does ("Rodent").
    raw_label = df["descriptor"].where(df["group"] == "noise", df["complaint_type"])
    df["label"] = raw_label.fillna("Other").map(tidy_label)
    xy = to_xy(df["latitude"], df["longitude"])
    df["x"], df["y"] = xy[:, 0].astype("float32"), xy[:, 1].astype("float32")

    keep = ["unique_key", "created_date", "group", "complaint_type", "label", "incident_address",
            "latitude", "longitude", "x", "y", "year", "month", "weekday", "hour", "is_night", "is_weekend"]
    df = df[keep].sort_values("created_date").reset_index(drop=True)
    for col in ("group", "complaint_type", "label"):
        df[col] = df[col].astype("category")
    rep.add("done", len(df), len(df), ", ".join(f"{g}={(df['group'] == g).sum():,}" for g in GROUPS))
    return df, rep


def drop_same_day_repeats(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Count repeat calls once: one complaint per location, type, descriptor and day.

    A few callers file hundreds of complaints a day about the same address (one
    Bronx address has about 120,000 "Loud Music/Party" calls in the study), which
    would make one person look like a whole noisy block. After this step a count
    means "complaint-days": days on which a location had that kind of complaint.
    """
    n = len(df)
    day = df["created_date"].dt.normalize()
    keys = [df["latitude"], df["longitude"], df["complaint_type"], df["descriptor"].fillna(""), day]
    df = df[~pd.concat(keys, axis=1).duplicated()]
    rep.add("same_day_repeats", n, len(df), "same location, type and descriptor on the same day")
    return df


def compare_period(df: pd.DataFrame) -> pd.Series:
    """2025 or 2026 for January-October rows, NA otherwise (so both years cover the same months)."""
    return df["year"].where(df["month"] <= COMPARE_LAST_MONTH)


# ==========================================================================
# 2. Apartment buildings and their surroundings
# ==========================================================================
def load_apartment_buildings(min_units: int = 3) -> pd.DataFrame:
    """PLUTO residential buildings with 3+ apartments, with their community district."""
    if config.CLEAN_PLUTO_FILE.exists():
        pluto = pd.read_parquet(config.CLEAN_PLUTO_FILE)
    else:
        pluto, _ = clean_pluto()
    b = pluto[(pluto["unitsres"] >= min_units) & pluto["latitude"].notna()].copy()
    b["cd"] = pd.to_numeric(b["cd"], errors="coerce")
    b = b[b["cd"].notna()]
    b["cd"] = b["cd"].astype(int)
    b["district"] = [district_name(cd) for cd in b["cd"]]
    return b[["bbl", "address", "borough", "zipcode", "unitsres", "cd", "district", "latitude", "longitude"]] \
        .reset_index(drop=True)


def district_name(cd: int) -> str:
    """PLUTO community district code -> readable name: 108 -> 'Manhattan Community District 8'."""
    code = str(int(cd))
    return f"{BOROUGH_NAMES.get(code[0], 'NYC')} Community District {int(code[1:])}"


def count_within(tree: cKDTree, xy: np.ndarray, radius: float = config.STREET_RADIUS_M) -> np.ndarray:
    """How many tree points lie within `radius` meters of each row of xy (all rows at once)."""
    return tree.query_ball_point(xy, radius, return_length=True, workers=-1).astype(np.int32)


def col(group: str, what: str = "per_year", radius: int = config.STREET_RADIUS_M) -> str:
    """Column name of a building count: 'noise_per_year' at 100 m, 'noise_per_year_500m' at 500 m."""
    return f"{group}_{what}" if radius == config.STREET_RADIUS_M else f"{group}_{what}_{radius}m"


def building_counts(buildings: pd.DataFrame, complaints: pd.DataFrame,
                    radii: tuple[int, ...] = config.STREET_RADII) -> pd.DataFrame:
    """Complaints within each radius of every apartment building, per group.

    `<group>_per_year` covers the whole study window; `<group>_2025` and
    `<group>_2026` cover January-October of each year. Radii other than 100 m
    get a suffix (see `col`).
    """
    out = buildings.copy()
    xy = to_xy(out["latitude"], out["longitude"])
    period = compare_period(complaints)
    for g in GROUPS:
        sub = complaints[complaints["group"] == g]
        trees = {"per_year": cKDTree(sub[["x", "y"]].to_numpy())}
        for year in COMPARE_YEARS:
            trees[str(year)] = cKDTree(sub.loc[period.loc[sub.index] == year, ["x", "y"]].to_numpy())
        for r in radii:
            for what, tree in trees.items():
                n = count_within(tree, xy, r)
                out[col(g, what, r)] = n / study_years() if what == "per_year" else n
    return out


# ==========================================================================
# 3. The four questions, on a 100 m grid
# ==========================================================================
def cell_table(buildings: pd.DataFrame, complaints: pd.DataFrame) -> pd.DataFrame:
    """One row per 100 m cell that contains an apartment building.

    The cell's district is that of its largest building. Cells with buildings
    but no complaints are kept with zeros: a quiet cell is a result too.
    """
    bx, by = cell_ids(to_xy(buildings["latitude"], buildings["longitude"]))
    b = buildings.assign(cx=bx, cy=by).sort_values("unitsres", ascending=False)
    cells = (b.groupby(["cx", "cy"])
              .agg(cd=("cd", "first"), district=("district", "first"),
                   buildings=("bbl", "size"), units=("unitsres", "sum"))
              .reset_index())

    cx, cy = cell_ids(complaints[["x", "y"]].to_numpy())
    c = pd.DataFrame({"cx": cx, "cy": cy, "group": complaints["group"].astype(str).to_numpy(),
                      "period": compare_period(complaints).to_numpy()})
    total = c.groupby(["cx", "cy", "group"]).size().unstack("group", fill_value=0)
    total = (total / study_years()).add_suffix("_per_year")
    by_year = (c.dropna(subset=["period"]).astype({"period": int})
                .groupby(["cx", "cy", "group", "period"]).size()
                .unstack(["group", "period"], fill_value=0))
    by_year.columns = [f"{g}_{y}" for g, y in by_year.columns]

    cells = cells.merge(total.reset_index(), on=["cx", "cy"], how="left") \
                 .merge(by_year.reset_index(), on=["cx", "cy"], how="left")
    expected = [f"{g}_per_year" for g in GROUPS] + [f"{g}_{y}" for g in GROUPS for y in COMPARE_YEARS]
    return cells.reindex(columns=[*cells.columns.difference(expected, sort=False), *expected]).fillna(0)


def log_rate(cells: pd.DataFrame, group: str) -> pd.Series:
    """log(1 + complaints per year). Counts are very skewed (a few cells have thousands),
    so differences are compared on a log scale: a gap of 0.69 means one cell has twice the other's."""
    return np.log1p(cells[f"{group}_per_year"])


def within_between(cells: pd.DataFrame, group: str) -> dict:
    """Q1. Split the variation between cells into a within-district and a between-district part.

    within share = sum of (cell - its district's mean)^2 / sum of (cell - city mean)^2.
    A share near 1 means district averages hide most of the differences.
    """
    v = log_rate(cells, group)
    district_mean = v.groupby(cells["cd"]).transform("mean")
    total = float(((v - v.mean()) ** 2).sum())
    within = float(((v - district_mean) ** 2).sum())
    return {"cells": int(len(cells)), "districts": int(cells["cd"].nunique()),
            "within_share": within / total, "between_share": 1 - within / total}


def _pair_gaps(cells: pd.DataFrame, values: pd.Series, step: int) -> np.ndarray:
    """|difference| between each cell and the cells `step` cells east and north of it."""
    v = cells[["cx", "cy"]].assign(v=values.to_numpy())
    gaps = []
    for dx, dy in ((step, 0), (0, step)):
        moved = v.assign(cx=v["cx"] - dx, cy=v["cy"] - dy)
        pairs = v.merge(moved, on=["cx", "cy"], suffixes=("", "_other"))
        gaps.append((pairs["v"] - pairs["v_other"]).abs().to_numpy())
    return np.concatenate(gaps)


def neighbor_gaps(cells: pd.DataFrame, group: str, max_steps: int = 10,
                  n_boot: int = 200, seed: int = 0) -> dict:
    """Q2. Are nearby cells more alike than random cells?

    For distances of 1 to max_steps cells (100 m to 1 km): the mean gap between
    cells that far apart, with a bootstrap 95% interval. Two baselines: random
    pairs inside the same district (values shuffled within each district) and
    random pairs anywhere in the city.
    """
    rng = np.random.default_rng(seed)
    v = log_rate(cells, group)
    rows = []
    for step in range(1, max_steps + 1):
        gaps = _pair_gaps(cells, v, step)
        boot = [rng.choice(gaps, size=len(gaps)).mean() for _ in range(n_boot)]
        rows.append({"distance_m": step * config.STREET_CELL_M, "pairs": int(len(gaps)),
                     "mean_gap": float(gaps.mean()),
                     "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5))})

    shuffled_in_district = v.groupby(cells["cd"]).transform(lambda s: rng.permutation(s.to_numpy()))
    same_district = float((v - shuffled_in_district).abs().mean())
    citywide = float((v - rng.permutation(v.to_numpy())).abs().mean())
    adjacent = rows[0]["mean_gap"]
    return {"by_distance": rows, "random_same_district": same_district, "random_citywide": citywide,
            "adjacent_vs_random_district": adjacent / same_district}


def timing(complaints: pd.DataFrame) -> dict:
    """Q3. When complaints happen: hour x weekday counts, and night/weekend shares by type."""
    out = {}
    for g in GROUPS:
        sub = complaints[complaints["group"] == g]
        grid = (sub.groupby(["weekday", "hour"]).size().unstack("hour", fill_value=0)
                   .reindex(index=range(7), columns=range(24), fill_value=0))
        by_label = sub.groupby("label", observed=True).agg(n=("unique_key", "size"),
                                                           night=("is_night", "mean"),
                                                           weekend=("is_weekend", "mean"))
        by_label = by_label.sort_values("n", ascending=False).head(8)
        monthly = sub.groupby(sub["created_date"].dt.to_period("M")).size()
        out[g] = {"total": int(len(sub)), "night_share": float(sub["is_night"].mean()),
                  "weekend_share": float(sub["is_weekend"].mean()),
                  "hour_weekday": grid.to_numpy().tolist(),
                  "by_type": {k: {"n": int(r.n), "night": float(r.night), "weekend": float(r.weekend)}
                              for k, r in by_label.iterrows()},
                  "monthly": {str(k): int(n) for k, n in monthly.items()}}
    return out


def persistence(cells: pd.DataFrame, group: str) -> dict:
    """Q4. Do the same cells stay at the top from January-October 2025 to January-October 2026?"""
    a, b = (cells[f"{group}_{y}"] for y in COMPARE_YEARS)
    active = (a + b) > 0
    rho = spearmanr(a[active], b[active]).statistic
    top_a = a >= a.quantile(1 - HOTSPOT_SHARE)
    top_b = b >= b.quantile(1 - HOTSPOT_SHARE)
    return {"cells_with_complaints": int(active.sum()), "spearman": float(rho),
            "hotspots_2025": int(top_a.sum()), "still_hotspot_2026": float(top_b[top_a].mean()),
            "chance_level": float(top_b.mean())}


def findings(cells: pd.DataFrame, complaints: pd.DataFrame) -> dict:
    """All four answers in one JSON-ready dict (saved to reports/street_findings.json)."""
    return {
        "study": {"start": config.STUDY_START_DATE, "end": config.STUDY_END_DATE,
                  "cell_m": config.STREET_CELL_M, "radius_m": config.STREET_RADIUS_M,
                  "types": config.STREET_TYPES},
        "q1_within_between": {g: within_between(cells, g) for g in GROUPS},
        "q2_neighbor_gaps": {g: neighbor_gaps(cells, g) for g in GROUPS},
        "q3_timing": timing(complaints),
        "q4_persistence": {g: persistence(cells, g) for g in GROUPS},
    }


# ==========================================================================
# 4. The 100 m around one point (used by the app and the notebook)
# ==========================================================================
def midrank_percentile(sorted_values: np.ndarray, value: float) -> float:
    """Share of the reference below `value`, counting ties as half (so many zeros don't skew it)."""
    if len(sorted_values) == 0:
        return float("nan")
    below = np.searchsorted(sorted_values, value, side="left")
    not_above = np.searchsorted(sorted_values, value, side="right")
    return float((below + not_above) / 2 / len(sorted_values))


class StreetIndex:
    """Answers "what is reported within 100, 200 or 500 m of this point?" in milliseconds.

    Holds every cleaned complaint in a k-d tree, plus every apartment building's
    own counts at the same radius as the yardstick for "noisier than X% of the district".
    """

    def __init__(self, complaints: pd.DataFrame, buildings: pd.DataFrame):
        self.tree = cKDTree(complaints[["x", "y"]].to_numpy())
        self.group = complaints["group"].astype(str).to_numpy()
        self.label = complaints["label"].astype(str).to_numpy()
        self.hour = complaints["hour"].to_numpy()
        self.is_night = complaints["is_night"].to_numpy()
        self.is_weekend = complaints["is_weekend"].to_numpy()
        self.period = compare_period(complaints).fillna(0).astype(int).to_numpy()
        self.years = study_years()

        self.buildings = buildings.reset_index(drop=True)
        self.building_tree = cKDTree(to_xy(buildings["latitude"], buildings["longitude"]))
        # The radii whose yardstick was built (an older buildings file may have only 100 m)
        self.radii = tuple(r for r in config.STREET_RADII if all(col(g, "per_year", r) in buildings for g in GROUPS))
        self.reference, self.hotspot_cut, self.median = {}, {}, {}
        for r in self.radii:
            for g in GROUPS:
                for cd, part in [*buildings.groupby("cd"), ("city", buildings)]:
                    values = part[col(g, "per_year", r)]
                    self.reference[(cd, g, r)] = np.sort(values.to_numpy())
                    self.median[(cd, g, r)] = float(values.median())
                    for y in COMPARE_YEARS:
                        self.hotspot_cut[(cd, g, y, r)] = float(part[col(g, str(y), r)].quantile(1 - HOTSPOT_SHARE))

    @classmethod
    def load(cls) -> "StreetIndex":
        """Build the index from the files `python -m src.street` writes."""
        cols = ["group", "label", "x", "y", "year", "month", "hour", "is_night", "is_weekend"]
        return cls(pd.read_parquet(CLEAN_FILE, columns=cols), pd.read_parquet(BUILDINGS_FILE))

    def nearest_building(self, lat: float, lon: float) -> pd.Series:
        """The apartment building closest to the point (gives the point its district)."""
        _, i = self.building_tree.query(to_xy([lat], [lon])[0])
        return self.buildings.iloc[int(i)]

    def profile(self, lat: float, lon: float, radius: float = config.STREET_RADIUS_M) -> dict:
        """Everything the report needs about the `radius` meters around one point."""
        if radius not in self.radii:
            raise ValueError(f"radius must be one of {self.radii}, not {radius}")
        near = self.nearest_building(lat, lon)
        cd = int(near["cd"])
        idx = np.asarray(self.tree.query_ball_point(to_xy([lat], [lon])[0], radius), dtype=np.int64)
        out = {"lat": lat, "lon": lon, "radius_m": radius, "district": near["district"],
               "nearest_address": near["address"], "groups": {}}
        for g in GROUPS:
            mine = idx[self.group[idx] == g]
            n = int(len(mine))
            per_year = n / self.years
            labels = pd.Series(self.label[mine]).value_counts().head(3)
            by_year = {y: int((self.period[mine] == y).sum()) for y in COMPARE_YEARS}
            out["groups"][g] = {
                "count": n,
                "per_year": per_year,
                "district_median": self.median[(cd, g, radius)],
                "city_median": self.median[("city", g, radius)],
                "pct_district": midrank_percentile(self.reference[(cd, g, radius)], per_year),
                "pct_city": midrank_percentile(self.reference[("city", g, radius)], per_year),
                "night_share": float(self.is_night[mine].mean()) if n else None,
                "weekend_share": float(self.is_weekend[mine].mean()) if n else None,
                "by_hour": np.bincount(self.hour[mine], minlength=24).tolist(),
                "top": [{"label": k, "count": int(v), "share": v / n} for k, v in labels.items()],
                "by_year": by_year,
                "hotspot": {y: bool(by_year[y] > 0 and by_year[y] >= self.hotspot_cut[(cd, g, y, radius)])
                            for y in COMPARE_YEARS},
            }
        return out


def distance_m(a: dict, b: dict) -> float:
    """Straight-line distance between two profiles' points, in meters."""
    return float(np.hypot(*(to_xy([a["lat"]], [a["lon"]])[0] - to_xy([b["lat"]], [b["lon"]])[0])))


# ==========================================================================
# 5. Plain-English report
# ==========================================================================
MIN_PER_YEAR = 5        # below this, both places are "quiet" and ratios mean little
SIMILAR = 1.25          # within 25% of each other counts as "about the same"
def caveat(radius: int = config.STREET_RADIUS_M) -> str:
    """The note under the report: what the numbers are and are not."""
    return (f"Based on NYC 311 complaints within {radius} m, "
            f"{pd.Timestamp(config.STUDY_START_DATE):%B %Y} to {pd.Timestamp(config.STUDY_END_DATE):%B %Y}; "
            "repeat calls about the same thing at the same address on the same day count once. "
            "Complaints are reports, not measured noise or dirt, and some neighborhoods report more than others.")


CAVEAT = caveat()


def _rank_phrase(g: str, pct: float) -> str:
    """0.85 -> 'noisier than 85% of apartment buildings in its district'."""
    w = WORDS[g]
    pct = min(max(pct, 0.01), 0.99)       # never "100%": the yardstick is a sample, not every place
    if pct >= 0.5:
        return f"{w['more']} than {pct:.0%} of apartment buildings in its district"
    return f"{w['less']} than {1 - pct:.0%} of apartment buildings in its district"


def _times(a: float, b: float, what: str) -> str:
    """How much bigger a is than b, in words: 'about three times as many noise complaints as',
    'about 40% more noise complaints than'."""
    ratio = a / b if b else float("inf")
    if ratio >= 10:
        return f"more than ten times as many {what} as"
    if ratio >= 1.95:
        words = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}
        return f"about {words.get(round(ratio), f'{round(ratio)} times')} as many {what} as"
    return f"about {ratio - 1:.0%} more {what} than"


def _differs(a: float, b: float) -> bool:
    """Enough complaints, and far enough apart, to call one place busier than the other."""
    return max(a, b) >= MIN_PER_YEAR and max(a, b) / max(min(a, b), 1e-9) >= SIMILAR


def _detail(g: str, ga: dict, gb: dict, name_a: str, name_b: str) -> str:
    """', mostly loud music/party, about 60% of it at night': what the complaints are, and when."""
    tops = [(name, gr["top"][0]["label"].lower()) for name, gr in ((name_a, ga), (name_b, gb))
            if gr["count"] >= 10 and gr["top"]]
    if not tops:
        return ""
    if len(tops) == 2 and tops[0][1] != tops[1][1]:
        out = f", mostly {tops[0][1]} around {tops[0][0]} and {tops[1][1]} around {tops[1][0]}"
    elif len(tops) == 2:
        out = f", mostly {tops[0][1]} in both places"
    else:
        out = f", mostly {tops[0][1]} around {tops[0][0]}"
    if g == "noise" and min(ga["count"], gb["count"]) >= 20:
        na, nb = ga["night_share"], gb["night_share"]
        out += (f", about {(na + nb) / 2:.0%} of it at night" if abs(na - nb) < 0.10 else
                f", with {na:.0%} of {name_a}'s and {nb:.0%} of {name_b}'s at night")
    return out


def _group_sentence(g: str, ga: dict, gb: dict, name_a: str, name_b: str) -> str:
    """One sentence per group: who gets more, by how much, of what kind."""
    a, b = ga["per_year"], gb["per_year"]
    name = WORDS[g]["name"]
    if max(a, b) < MIN_PER_YEAR:
        return f"Both get very few {name} complaints ({a:,.0f} and {b:,.0f} a year)."
    detail = _detail(g, ga, gb, name_a, name_b)
    if not _differs(a, b):
        return f"{name.capitalize()} complaints are about the same ({a:,.0f} and {b:,.0f} a year){detail}."
    hi, lo = (name_a, name_b) if a > b else (name_b, name_a)
    return (f"{hi} gets {_times(max(a, b), min(a, b), name + ' complaints')} {lo} "
            f"({max(a, b):,.0f} versus {min(a, b):,.0f} a year){detail}.")


def _trend_sentence(g: str, gr: dict, name: str) -> str | None:
    """Hotspot in both years, or a clear change between January-October 2025 and 2026."""
    y0, y1 = COMPARE_YEARS
    n0, n1 = gr["by_year"][y0], gr["by_year"][y1]
    if gr["hotspot"][y0] and gr["hotspot"][y1]:
        return f"{name} was a {WORDS[g]['name']} hotspot (top 10% of its district) in both {y0} and {y1}."
    for change, word in ((n1 <= 0.7 * n0, WORDS[g]["less"]), (n1 >= 1.4 * n0, WORDS[g]["more"])):
        if max(n0, n1) >= 20 and change:
            return (f"{name} has been getting {word}: {n0} {WORDS[g]['name']} complaints in "
                    f"January–October {y0}, {n1} in {y1}.")
    return None


def _verdict(pa: dict, pb: dict, name_a: str, name_b: str) -> str:
    """The opening: where the two places are, and which one suits a renter who wants quiet, clean streets."""
    dist = distance_m(pa, pb)
    where = (pa["district"] if pa["district"] == pb["district"]
             else f"{pa['district']} and {pb['district']}")
    better = {g: (name_a if pa["groups"][g]["per_year"] < pb["groups"][g]["per_year"] else name_b)
              for g in GROUPS if _differs(pa["groups"][g]["per_year"], pb["groups"][g]["per_year"])}
    these = f"These two buildings, {dist:,.0f} meters apart in {where},"
    if not better:
        return f"{these} are much alike on noise and sanitation complaints."
    if len(set(better.values())) == 1:
        return f"{these} are not alike: {next(iter(better.values()))} looks like the better choice for quiet, clean streets."
    return f"{these} are a trade-off: {better['noise']} is quieter, while {better['sanitation']} is cleaner."


def compare_report(pa: dict, pb: dict, name_a: str = "Building A", name_b: str = "Building B") -> str:
    """Two profiles -> one short paragraph: the verdict first, then the reasons.

    Each sentence is picked by a rule from the numbers. Rankings, hours and the
    full list of complaint types are left to the cards and charts beside it.
    """
    ga = {g: pa["groups"][g] for g in GROUPS}
    gb = {g: pb["groups"][g] for g in GROUPS}
    sentences = [
        _verdict(pa, pb, name_a, name_b),
        *[_group_sentence(g, ga[g], gb[g], name_a, name_b) for g in GROUPS],
        *[_trend_sentence(g, gr, name) for g in GROUPS for name, gr in ((name_a, ga[g]), (name_b, gb[g]))],
    ]
    return " ".join(s for s in sentences if s)


def profile_report(p: dict, name: str = "This building") -> str:
    """One profile -> one short paragraph (used when only one building is picked)."""
    parts = []
    for g in GROUPS:
        gr = p["groups"][g]
        if gr["per_year"] < MIN_PER_YEAR:
            parts.append(f"gets very few {WORDS[g]['name']} complaints (about {gr['per_year']:,.0f} a year)")
            continue
        detail = f", mostly {gr['top'][0]['label'].lower()}" if gr["top"] else ""
        if g == "noise" and gr["count"] >= 20:
            detail += f", {gr['night_share']:.0%} of it at night"
        parts.append(f"is {_rank_phrase(g, gr['pct_district'])} "
                     f"({gr['per_year']:,.0f} {WORDS[g]['name']} complaints a year within {p['radius_m']} m{detail})")
    sentences = [f"{name}, in {p['district']}, " + ", and ".join(parts) + "."]
    sentences += [_trend_sentence(g, p["groups"][g], name) for g in GROUPS]
    return " ".join(s for s in sentences if s)


# ==========================================================================
# 6. Notebook helper: compare two addresses
# ==========================================================================
def find_building(index: StreetIndex, query: str) -> pd.Series:
    """The apartment building whose address best matches `query`.

    A borough after a comma narrows the search: '235 East 92nd Street, Manhattan'
    (several boroughs have an East 92nd Street).
    """
    b = index.buildings
    if "address_key" not in b:
        b["address_key"] = address.normalize_series(b["address"])
    street_part, _, borough = query.rpartition(",")
    if street_part and borough.strip().upper() in config.VALID_BOROUGHS:
        b, query = b[b["borough"] == borough.strip().upper()], street_part
    hits = address.search(b, query, limit=1)
    if hits.empty:
        raise ValueError(f"No apartment building found for {query!r}")
    return hits.iloc[0]


def compare_addresses(index: StreetIndex, query_a: str, query_b: str,
                      radius: int = config.STREET_RADIUS_M) -> tuple[dict, dict, str]:
    """Look up two addresses, profile both and write the paragraph."""
    a, b = find_building(index, query_a), find_building(index, query_b)
    pa = index.profile(a["latitude"], a["longitude"], radius)
    pb = index.profile(b["latitude"], b["longitude"], radius)
    pa["name"], pb["name"] = a["address"].title(), b["address"].title()
    return pa, pb, compare_report(pa, pb, pa["name"], pb["name"])


# ==========================================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    complaints, rep = clean(load_raw())
    complaints.to_parquet(CLEAN_FILE, index=False)
    CLEANING_FILE.write_text(json.dumps({**rep.to_dict(),
                                         "by_type": complaints["complaint_type"].value_counts().to_dict()},
                                        indent=2, default=str))
    log.info("saved %s complaints to %s", f"{len(complaints):,}", CLEAN_FILE)

    buildings = building_counts(load_apartment_buildings(), complaints)
    buildings.to_parquet(BUILDINGS_FILE, index=False)
    log.info("saved %s m counts for %s apartment buildings",
             "/".join(map(str, config.STREET_RADII)), f"{len(buildings):,}")

    cells = cell_table(buildings, complaints)
    FINDINGS_FILE.write_text(json.dumps(findings(cells, complaints), indent=2))
    log.info("saved findings for %s cells to %s", f"{len(cells):,}", FINDINGS_FILE)


if __name__ == "__main__":
    main()
