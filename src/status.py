#!/usr/bin/env python3
"""
Burlington VT Beach Closure Tracker Scraper
https://www.burlingtonvt.gov/1219/Beach-Closure-Tracker

Fetches beach status from the Burlington VT ArcGIS REST endpoint.
No browser or JS rendering needed — plain JSON response.
Outputs results to data/beach_status.parquet.
"""

import logging
import time
from datetime import timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

ARCGIS_URL = (
    "https://maps.burlingtonvt.gov/arcgis/rest/services/BTV_Beach_Status/MapServer/0/query"
    "?where=1%3D1"
    "&outFields=LocationName,CyanobacteriaDescription,ResultDateTime,DisplayOrder,Notes"
    "&returnGeometry=false"
    "&outSR=4326"
    "&orderByFields=DisplayOrder%20ASC"
    "&f=geojson"
)
DATA_PATH = Path("data/beach_status.parquet")


def scrape_beach_statuses(timeout: int = 45, retries: int = 3) -> list[dict]:
    """
    Fetch raw beach status records from the Burlington VT ArcGIS REST endpoint.

    Parameters
    ----------
    timeout : request timeout in seconds (default 45 — GIS server can be slow)
    retries : number of retry attempts on timeout (default 3, with backoff)

    Returns
    -------
    List of property dicts, one per beach location, e.g.:
      [{"LocationName": "Leddy Beach", "CyanobacteriaDescription": "Open", ...}, ...]

    Raises
    ------
    requests.exceptions.Timeout       if all retry attempts time out
    requests.exceptions.RequestException  on non-retryable HTTP errors
    RuntimeError                       if the response contains no features
    """
    log.info(f"Loading {ARCGIS_URL[:80]}...")

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(ARCGIS_URL, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if not features:
                raise RuntimeError(
                    "ArcGIS response contained no features — endpoint may have changed"
                )

            beaches = [f.get("properties", {}) for f in features]
            log.info(f"Found data for {len(beaches)} locations.")
            return beaches

        except requests.exceptions.Timeout as e:
            last_exc = e
            wait = 5 * attempt
            log.warning(f"Attempt {attempt}/{retries} timed out — retrying in {wait}s...")
            time.sleep(wait)

        except requests.exceptions.RequestException:
            # Non-timeout errors (4xx, 5xx, connection error) — don't retry
            raise

    raise requests.exceptions.Timeout(
        f"Beach scrape failed after {retries} attempts: {last_exc}"
    )


def clean_status(beaches: list[dict]) -> pd.DataFrame:
    """
    Normalize raw ArcGIS property dicts into a tidy DataFrame.

    Input fields
    ------------
    LocationName              : beach name string
    CyanobacteriaDescription  : status string e.g. "Open", "Closed", "Alert"
    ResultDateTime            : Unix timestamp in milliseconds (UTC)
    Notes                     : free-text notes, e.g. E. coli closure details

    Returns
    -------
    DataFrame with columns: [beach_name, status, notes, updated_at, recorded_at]
    """
    beach_status = (
        pd.DataFrame(beaches)
        .assign(
            beach_name  = lambda df: df["LocationName"],
            status      = lambda df: df["CyanobacteriaDescription"],
            notes       = lambda df: df.get("Notes", pd.NA),
            # ResultDateTime is Unix ms UTC — convert to tz-aware datetime
            updated_at  = lambda df: pd.to_datetime(df["ResultDateTime"], unit="ms", utc=True),
            recorded_at = pd.Timestamp.now(tz=timezone.utc),
        )
        .filter(items=["beach_name", "status", "notes", "updated_at", "recorded_at"])
    )
    return beach_status


def status_to_parquet(beach_status: pd.DataFrame, path: Path = DATA_PATH) -> None:
    """
    Append new beach status rows to the parquet file, deduplicating on the way.

    Handles mixed tz-naive / tz-aware timestamp columns that can appear when
    concatenating rows written by the old Playwright scraper (tz-naive) with
    rows from the ArcGIS scraper (tz-aware UTC).
    """
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, beach_status], ignore_index=True)
    else:
        combined = beach_status.copy()

    # Normalize timestamp columns to tz-aware UTC — old parquet rows may be
    # tz-naive, which causes sort_values to fail when mixed with tz-aware rows
    for col in ("updated_at", "recorded_at"):
        if col not in combined.columns:
            continue
        # Coerce to datetime first — column may be object dtype if old parquet
        # rows had mixed or null values
        combined[col] = pd.to_datetime(combined[col], utc=False, errors="coerce")
        if combined[col].dt.tz is None:
            combined[col] = combined[col].dt.tz_localize(
                "UTC", ambiguous="NaT", nonexistent="NaT"
            )
        else:
            combined[col] = combined[col].dt.tz_convert("UTC")

    before = len(combined)
    combined = (
        combined
        .drop_duplicates(subset=["beach_name", "status", "updated_at", "recorded_at"])
        .sort_values(["beach_name", "updated_at", "recorded_at"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    after = len(combined)
    log.info(f"[Beach Status] Saved {after - before} new rows to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    log.info(f"[Beach Status] Parquet file now has {after} total rows | {path}")


def load_cached_status(path: Path = DATA_PATH) -> pd.DataFrame | None:
    """
    Load the most recent beach status from the parquet cache.
    Returns None if no cache exists yet.
    Used as a fallback when scrape_beach_statuses() fails.
    """
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    # Return only the latest recorded snapshot per beach
    return (
        df.sort_values("recorded_at", ascending=False)
        .groupby("beach_name", as_index=False)
        .first()
    )


def run(path: Path = DATA_PATH) -> pd.DataFrame | None:
    """
    Main entry point: scrape → clean → persist → return.

    On timeout or scrape failure, logs the error and falls back to the most
    recent cached status from parquet so callers always get a usable result.

    Returns
    -------
    DataFrame of current beach statuses, or None if scrape failed and no
    cache exists.
    """
    try:
        beaches = scrape_beach_statuses()
        beach_status = clean_status(beaches)
        status_to_parquet(beach_status, path=path)
        return beach_status

    except Exception as e:
        log.error(f"Beach scrape failed: {e}")
        cached = load_cached_status(path=path)
        if cached is not None:
            log.warning(
                f"Using cached beach status from last successful scrape "
                f"(recorded_at: {cached['recorded_at'].max()})"
            )
        else:
            log.warning("No cached beach status available")
        return cached


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run()
    if result is not None:
        print(result.to_string(index=False))