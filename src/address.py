"""Turn the many ways people write NYC addresses into one searchable form.

    "235 East 92nd Street"  ->  "235 E 92 ST"
    "235 E. 92 St, Apt 4B"  ->  "235 E 92 ST"
    "1 W 72nd St"           ->  "1 W 72 ST"

Used to index PLUTO's building addresses and to normalize what a renter
types into the demo's search box, so both sides meet in the middle.
"""

from __future__ import annotations

import re

import pandas as pd

WORDS = {
    "EAST": "E", "WEST": "W", "NORTH": "N", "SOUTH": "S",
    "STREET": "ST", "STR": "ST", "AVENUE": "AVE", "AV": "AVE", "ROAD": "RD", "PLACE": "PL",
    "BOULEVARD": "BLVD", "BLVRD": "BLVD", "PARKWAY": "PKWY", "DRIVE": "DR", "LANE": "LN", "COURT": "CT",
    "TERRACE": "TER", "HIGHWAY": "HWY", "EXPRESSWAY": "EXPY", "SQUARE": "SQ", "TURNPIKE": "TPKE",
    "CONCOURSE": "CONC", "SAINT": "ST", "FORT": "FT", "MOUNT": "MT",
}
_UNIT = re.compile(r"(\b\d+(ST|ND|RD|TH)?\s+(FL|FLOOR)\b|\b(APT|APARTMENT|UNIT|FL|FLOOR|STE|SUITE)\b|#).*$")
_ORDINAL = re.compile(r"\b(\d+)(ST|ND|RD|TH)\b")


def normalize(address: str) -> str:
    """Normalize one address string (see module docstring)."""
    s = str(address).upper()
    s = s.split(",")[0]                         # drop ", Brooklyn, NY 11201"
    s = re.sub(r"[^\w\s#-]", " ", s)            # punctuation
    s = _UNIT.sub("", s)                        # apartment / floor
    s = re.sub(r"\b\d{5}\b", "", s)             # ZIP codes
    s = _ORDINAL.sub(r"\1", s)                  # 92ND -> 92
    words = [WORDS.get(w, w) for w in s.split()]
    return " ".join(words).strip()


def normalize_series(addresses: pd.Series) -> pd.Series:
    """Vectorized wrapper for a whole column."""
    return addresses.fillna("").map(normalize)


def search(index: pd.DataFrame, query: str, limit: int = 10) -> pd.DataFrame:
    """Find buildings whose normalized address matches the query.

    Ranking: exact match first, then addresses that start with the query,
    then addresses that contain every word of the query. Ties go to bigger
    buildings (more apartments, so more likely the one people mean).
    """
    q = normalize(query)
    if not q:
        return index.iloc[0:0]
    hits = _match_all_words(index, q.split())
    hits["_rank"] = 2
    hits.loc[hits["address_key"].str.startswith(q), "_rank"] = 1
    hits.loc[hits["address_key"] == q, "_rank"] = 0
    if hits.empty:
        hits = _nearest_house_numbers(index, q)
    hits = hits.sort_values(["_rank", "unitsres"], ascending=[True, False]).drop_duplicates("bbl")
    return hits.head(limit).drop(columns="_rank")


def _match_all_words(index: pd.DataFrame, words: list[str]) -> pd.DataFrame:
    """Rows whose address contains every word of the query (as whole words)."""
    mask = pd.Series(True, index=index.index)
    for w in words:
        mask &= index["address_key"].str.contains(rf"(?:^|\s){re.escape(w)}(?:\s|$)", regex=True)
    return index[mask].copy()


def _nearest_house_numbers(index: pd.DataFrame, q: str) -> pd.DataFrame:
    """Fallback: same street, closest house numbers (e.g. 237 E 92 ST → 235 and 241 E 92 ST)."""
    m = re.match(r"^(\d+)\s+(.+)$", q)
    if not m:
        return index.iloc[0:0].assign(_rank=3)
    number, street = int(m.group(1)), m.group(2)
    same_street = index[index["address_key"].str.endswith(" " + street)].copy()
    nums = pd.to_numeric(same_street["address_key"].str.extract(r"^(\d+)\s", expand=False), errors="coerce")
    same_street["_rank"] = 3 + (nums - number).abs() / 1e6     # nearest numbers first
    return same_street[nums.notna()]
