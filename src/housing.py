"""Load and inspect the housing datasets for "Know Before You Rent".

Datasets (downloaded by `python -m src.download --only housing`):
    hpd_complaints     data/raw/hpd_complaints/YYYY-MM.parquet  (one row per problem)
    hpd_violations     data/raw/hpd_violations/YYYY-MM.parquet  (one row per violation)
    hpd_registrations  data/raw/hpd_registrations.parquet       (one row per registered building)
    hpd_contacts       data/raw/hpd_contacts.parquet            (owners/agents per registration)
    bedbugs            data/raw/bedbugs.parquet                 (yearly filing per building)
    evictions          data/raw/evictions.parquet               (one row per executed eviction)

All raw columns are strings, exactly as the API returned them. These helpers
are shared by the exploration and cleaning notebooks.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def dataset_files(name: str) -> list[Path]:
    """Parquet file(s) of a downloaded dataset: monthly files or a single file."""
    folder, single = config.RAW_DIR / name, config.RAW_DIR / f"{name}.parquet"
    files = sorted(folder.glob("*.parquet")) if folder.is_dir() else ([single] if single.exists() else [])
    if not files:
        raise FileNotFoundError(f"{name} not downloaded. Run `python -m src.download --only housing`.")
    return files


def load(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a dataset (optionally only some columns) with memory-light pyarrow strings."""
    parts = []
    for f in dataset_files(name):
        available = pq.read_schema(f).names
        cols = None if columns is None else [c for c in columns if c in available]
        parts.append(pd.read_parquet(f, columns=cols, dtype_backend="pyarrow").reindex(columns=columns)
                     if columns else pd.read_parquet(f, dtype_backend="pyarrow"))
    return pd.concat(parts, ignore_index=True)


def clean_pluto_table() -> pd.DataFrame:
    """Cleaned PLUTO, created on first use.

    The exploration notebooks (01-04) need clean building data before the
    cleaning notebooks run, so PLUTO is cleaned and cached here the first
    time anyone asks for it. Notebook 08 shows the same cleaning step by step.
    """
    if not config.CLEAN_PLUTO_FILE.exists():
        from src.preprocess import clean_pluto
        pluto, _ = clean_pluto()
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pluto.to_parquet(config.CLEAN_PLUTO_FILE, index=False)
    return pd.read_parquet(config.CLEAN_PLUTO_FILE)


def monthly_rows(name: str) -> pd.Series:
    """Rows per month from parquet metadata (instant, loads no data)."""
    return pd.Series({f.stem: pq.ParquetFile(f).metadata.num_rows for f in dataset_files(name)}, name="rows")


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: missing share, distinct values and an example value."""
    return pd.DataFrame({
        "missing_%": df.isna().mean().mul(100).round(1),
        "distinct": df.nunique(),
        "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    })


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
def bbl_from_parts(boro, block, lot) -> pd.Series:
    """Build a 10-digit BBL from borough (1-5), block and lot columns."""
    b = pd.Series(boro, dtype="string").str.strip()
    blk = pd.to_numeric(pd.Series(block), errors="coerce")
    lt = pd.to_numeric(pd.Series(lot), errors="coerce")
    ok = b.str.fullmatch(r"[1-5]") & blk.between(1, 99999) & lt.between(1, 9999)
    out = b + blk.fillna(0).astype(int).astype(str).str.zfill(5) + lt.fillna(0).astype(int).astype(str).str.zfill(4)
    return out.where(ok.fillna(False))


def coverage(keys: pd.Series, reference: set) -> float:
    """Share of non-missing keys that appear in a reference set (e.g. PLUTO BBLs)."""
    k = keys.dropna()
    return float(k.isin(reference).mean()) if len(k) else np.nan


# --------------------------------------------------------------------------
# Concentration and persistence (used to preview "does history predict?")
# --------------------------------------------------------------------------
def concentration(counts: pd.Series, shares=(0.01, 0.05, 0.10)) -> pd.Series:
    """Share of all events produced by the top x% of units (e.g. buildings)."""
    s = counts.sort_values(ascending=False)
    total = s.sum()
    return pd.Series({f"top {int(p * 100)}%": s.head(max(1, int(len(s) * p))).sum() / total for p in shares})


def persistence(events: pd.DataFrame, key: str, period: str, first, second) -> pd.DataFrame:
    """P(problem in `second` period | problem or not in `first`), over keys seen in either.

    A crude first look at whether a building's past predicts its future.
    """
    a = set(events.loc[events[period] == first, key])
    b = set(events.loc[events[period] == second, key])
    universe = a | b
    had = pd.Series({k: k in a for k in universe})
    again = pd.Series({k: k in b for k in universe})
    out = pd.DataFrame({"buildings": had.value_counts(),
                        f"share with problem in {second}": again.groupby(had).mean()})
    out.index = out.index.map({True: f"had problem in {first}", False: f"no problem in {first}"})
    return out


# --------------------------------------------------------------------------
# Text: violation descriptions and owner names
# --------------------------------------------------------------------------
VIOLATION_TOPICS = {
    "heat/hot water": r"\bHEAT|HOT WATER|BOILER",
    "pests": r"ROACH|MICE|\bRATS?\b|RODENT|VERMIN|BED ?BUG|INSECT|PEST",
    "mold/leaks": r"\bMOLD|MOULD|LEAK|WATER DAMAGE|DAMP",
    "lead paint": r"LEAD",
    "paint/plaster": r"PAINT|PLASTER",
    "fire safety": r"SMOKE|CARBON MONOXIDE|FIRE|SPRINKLER|EGRESS",
    "windows/guards": r"WINDOW",
    "doors/locks": r"DOOR|LOCK",
    "electric": r"ELECTRIC|OUTLET|WIRING|LIGHT",
    "plumbing": r"PLUMB|TOILET|SINK|FAUCET|DRAIN|BASIN|BATH",
    "admin (registration/signs)": r"REGISTRATION|\bPOST|\bFILE\b|\bSIGNS?\b",
}


def violation_topic(text: pd.Series) -> pd.Series:
    """First matching topic for each violation description (order of VIOLATION_TOPICS), else 'other'."""
    t = text.astype("string").str.upper()
    out = pd.Series("other", index=t.index, dtype="string")
    assigned = pd.Series(False, index=t.index)
    for topic, pattern in VIOLATION_TOPICS.items():
        hit = t.str.contains(pattern, regex=True, na=False) & ~assigned
        out[hit] = topic
        assigned |= hit
    return out.where(t.notna())


_SUFFIXES = r"\b(LLC|L L C|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LP|L P|LLP|ASSOC|ASSOCIATES|HDFC|THE)\b"


def normalize_name(name: pd.Series) -> pd.Series:
    """Uppercase, drop punctuation and company suffixes, collapse spaces.

    'A.B.C. Realty, L.L.C.' and 'ABC REALTY LLC' both become 'ABC REALTY'.
    """
    s = name.astype("string").str.upper()
    s = s.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\b([A-Z])\s(?=[A-Z]\b)", r"\1", regex=True)
    s = s.str.replace(_SUFFIXES, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.eq(""))


def normalize_address(number: pd.Series, street: pd.Series, zipcode: pd.Series) -> pd.Series:
    """'123  W. 45th St.' + zip -> '123 W 45TH ST 10036' for matching business addresses."""
    parts = (number.astype("string").fillna("") + " " + street.astype("string").fillna("") + " "
             + zipcode.astype("string").fillna("").str[:5])
    s = parts.str.upper().str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.str.len() < 6)
