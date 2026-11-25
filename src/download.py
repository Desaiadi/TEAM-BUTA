"""Download the raw data for the project.

Sources (all NYC Open Data, via the Socrata API):
  - PLUTO tax lots: building age, apartments, location, address.
  - Housing datasets: HPD complaints, violations, registrations and contacts,
    bedbug filings, evictions and NYPD complaints (HOUSING_DATASETS in config).
  - NYC 311 service requests: the full table month by month, or a subset of
    complaint types. Used by the city response-time work.
Dated data is fetched month by month and cached, so an interrupted run
resumes where it stopped.

Usage (from the project root):
    python -m src.download                  # PLUTO + housing datasets (what the project needs)
    python -m src.download --only pluto     # one source: pluto | housing | 311 | 311full
    python -m src.download --only housing   # HPD, bedbugs, evictions, NYPD complaints
    python -m src.download --only 311full   # the complete 311 table (~23M rows, ~1.5 GB)
    python -m src.download --force      # ignore the cache and re-download
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _session() -> requests.Session:
    """Return a requests session, adding the Socrata app token if one is set."""
    s = requests.Session()
    token = os.environ.get("NYC_APP_TOKEN")
    if token:
        s.headers["X-App-Token"] = token
    return s


def _get_with_retry(session: requests.Session, url: str, params: dict) -> requests.Response:
    """GET a URL, retrying with exponential backoff on errors and throttling."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            log.warning("HTTP %s (attempt %d): %s", resp.status_code, attempt, resp.text[:200])
        except requests.RequestException as exc:
            log.warning("Request failed (attempt %d): %s", attempt, exc)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Giving up on {url} after {config.MAX_RETRIES} attempts")


def socrata_query(session: requests.Session, url: str, select: list[str] | None,
                  where: str | None = None, order: str | None = None) -> pd.DataFrame:
    """Page through a Socrata (NYC Open Data) endpoint and return every row.

    All columns are read as strings; type conversion belongs in preprocessing,
    so the raw files stay exactly what the API returned. select=None returns
    every column.
    """
    frames, offset = [], 0
    while True:
        params = {"$limit": config.PAGE_SIZE, "$offset": offset}
        if select:
            params["$select"] = ",".join(select)
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        resp = _get_with_retry(session, url, params)
        page = pd.read_csv(io.StringIO(resp.text), dtype=str)
        frames.append(page)
        if len(page) < config.PAGE_SIZE:
            break
        offset += config.PAGE_SIZE
    df = pd.concat(frames, ignore_index=True)
    # A column that is empty for the whole page is dropped by the API, so restore it.
    return df.reindex(columns=select) if select else df


# --------------------------------------------------------------------------
# 1. NYC 311
# --------------------------------------------------------------------------
def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month between start and end (inclusive)."""
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def _chunk_path(complaint_type: str, month: pd.Timestamp):
    """Cache-file path for one (complaint type, month) chunk."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in complaint_type).strip("_")
    return config.RAW_311_DIR / safe / f"{month:%Y-%m}.parquet"


def _download_311_chunk(complaint_type: str, month: pd.Timestamp, force: bool) -> tuple[str, int]:
    """Download one month of one complaint type unless it is already cached."""
    path = _chunk_path(complaint_type, month)
    if path.exists() and not force:
        return f"{complaint_type} {month:%Y-%m} (cached)", 0
    nxt = month + pd.offsets.MonthBegin(1)
    ct = complaint_type.replace("'", "''")
    where = (f"complaint_type = '{ct}' AND created_date >= '{month:%Y-%m-%d}T00:00:00' "
             f"AND created_date < '{nxt:%Y-%m-%d}T00:00:00'")
    df = socrata_query(_session(), config.NYC_311_URL, config.COLUMNS_311, where, order="unique_key")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"{complaint_type} {month:%Y-%m}", len(df)


def download_311(force: bool = False) -> None:
    """Download every (complaint type, month) chunk in parallel threads."""
    months = _month_starts(config.START_DATE, config.END_DATE)
    jobs = [(ct, m) for ct in config.COMPLAINT_TYPES for m in months]
    log.info("311: %d chunks (%d types x %d months)", len(jobs), len(config.COMPLAINT_TYPES), len(months))
    total = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(_download_311_chunk, ct, m, force): (ct, m) for ct, m in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            label, n = fut.result()
            total += n
            log.info("[%d/%d] %s: %s rows", i, len(jobs), label, f"{n:,}")
    log.info("311 done: %s new rows downloaded", f"{total:,}")


def _download_311_full_month(month: pd.Timestamp, force: bool) -> tuple[str, int]:
    """Download every row and column created in one month, unless cached."""
    path = config.RAW_311_FULL_DIR / f"{month:%Y-%m}.parquet"
    if path.exists() and not force:
        return f"{month:%Y-%m} (cached)", 0
    nxt = month + pd.offsets.MonthBegin(1)
    where = f"created_date >= '{month:%Y-%m-%d}T00:00:00' AND created_date < '{nxt:%Y-%m-%d}T00:00:00'"
    df = socrata_query(_session(), config.NYC_311_URL, None, where, order="unique_key")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(path)   # only a complete month ever gets the final name
    return f"{month:%Y-%m}", len(df)


def download_311_full(force: bool = False) -> None:
    """Download the complete 311 table (2020 to the current month), one file per month.

    The current, still-growing month is always re-downloaded so it stays fresh.
    """
    months = _month_starts(config.FULL_START_DATE, pd.Timestamp.today().normalize())
    current = pd.Timestamp.today().normalize().replace(day=1)
    log.info("311 full: %d months", len(months))
    total = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(_download_311_full_month, m, force or m == current) for m in months]
        for i, fut in enumerate(as_completed(futures), 1):
            label, n = fut.result()
            total += n
            log.info("[%d/%d] %s: %s rows", i, len(months), label, f"{n:,}")
    log.info("311 full done: %s new rows downloaded", f"{total:,}")


# --------------------------------------------------------------------------
# Housing datasets (HPD complaints/violations/registrations, bedbugs, evictions)
# --------------------------------------------------------------------------
def _write_parquet_atomic(df: pd.DataFrame, path) -> None:
    """Write via a temp name so an interrupted run never leaves a half file under the real name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(path)


def _download_dataset_month(name: str, month: pd.Timestamp, force: bool) -> tuple[str, int]:
    """Download one month of a dated housing dataset unless cached."""
    dataset_id, columns, date_col = config.HOUSING_DATASETS[name][:3]
    path = config.RAW_DIR / name / f"{month:%Y-%m}.parquet"
    if path.exists() and not force:
        return f"{name} {month:%Y-%m} (cached)", 0
    nxt = month + pd.offsets.MonthBegin(1)
    where = f"{date_col} >= '{month:%Y-%m-%d}T00:00:00' AND {date_col} < '{nxt:%Y-%m-%d}T00:00:00'"
    url = f"https://data.cityofnewyork.us/resource/{dataset_id}.csv"
    df = socrata_query(_session(), url, columns, where, order=":id")
    _write_parquet_atomic(df, path)
    return f"{name} {month:%Y-%m}", len(df)


def download_housing(force: bool = False, only: list[str] | None = None) -> None:
    """Download the housing datasets used by Know Before You Rent.

    Dated datasets (HPD complaints, violations) are fetched month by month from
    HOUSING_START_DATE to today into data/raw/<name>/YYYY-MM.parquet; the
    current month is always refreshed. Undated ones (registrations, contacts,
    bedbugs, evictions) are fetched whole into data/raw/<name>.parquet.
    """
    names = only or list(config.HOUSING_DATASETS)
    current = pd.Timestamp.today().normalize().replace(day=1)
    jobs = []
    for name in names:
        spec = config.HOUSING_DATASETS[name]
        dataset_id, columns, date_col = spec[:3]
        start = spec[3] if len(spec) > 3 else config.HOUSING_START_DATE
        if date_col:
            jobs += [(name, m) for m in _month_starts(start, pd.Timestamp.today().normalize())]
            continue
        path = config.RAW_DIR / f"{name}.parquet"
        if path.exists() and not force:
            log.info("%s: cached", name)
            continue
        url = f"https://data.cityofnewyork.us/resource/{dataset_id}.csv"
        df = socrata_query(_session(), url, columns, order=":id")
        _write_parquet_atomic(df, path)
        log.info("%s: %s rows saved", name, f"{len(df):,}")

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(_download_dataset_month, n, m, force or m == current) for n, m in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            label, n = fut.result()
            log.info("[%d/%d] %s: %s rows", i, len(jobs), label, f"{n:,}")


# --------------------------------------------------------------------------
# PLUTO
# --------------------------------------------------------------------------
def download_pluto(force: bool = False) -> None:
    """Download the PLUTO columns needed for building-level denominators."""
    if config.RAW_PLUTO_FILE.exists() and not force:
        log.info("PLUTO: cached at %s", config.RAW_PLUTO_FILE)
        return
    df = socrata_query(_session(), config.PLUTO_URL, config.COLUMNS_PLUTO, order="bbl")
    config.RAW_PLUTO_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.RAW_PLUTO_FILE, index=False)
    log.info("PLUTO: %s tax lots saved", f"{len(df):,}")


SOURCES = {"pluto": download_pluto, "housing": download_housing, "311": download_311, "311full": download_311_full}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=SOURCES, help="download a single source")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    for name in (["pluto", "housing"] if args.only is None else [args.only]):
        SOURCES[name](force=args.force)


if __name__ == "__main__":
    main()
