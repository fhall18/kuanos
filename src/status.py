#!/usr/bin/env python3
"""
Burlington VT Beach Closure Tracker Scraper
https://www.burlingtonvt.gov/1219/Beach-Closure-Tracker

Fetches beach status from the Burlington VT ArcGIS REST endpoint.
No browser or JS rendering needed — plain JSON response.
Outputs results to data/beach_status.parquet.
"""

import logging
from datetime import timezone
from pathlib import Path

import pandas as pd
import requests

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


def clean_status(beaches: list[dict]) -> pd.DataFrame:
    beach_status = (
        pd.DataFrame(beaches)
        .assign(
            beach_name  = lambda df: df["LocationName"],
            status      = lambda df: df["CyanobacteriaDescription"],
            notes       = lambda df: df["Notes"],
            # ResultDateTime is Unix ms UTC — convert to tz-aware datetime
            updated_at  = lambda df: pd.to_datetime(df["ResultDateTime"], unit="ms", utc=True),
            recorded_at = pd.Timestamp.now(tz=timezone.utc),
        )
        .filter(items=["beach_name", "status", "notes", "updated_at", "recorded_at"])
    )
    return beach_status


def status_to_parquet(beach_status: pd.DataFrame, path: Path = DATA_PATH) -> None:
    """Save the beach status DataFrame to a Parquet file."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, beach_status], ignore_index=True)
    else:
        combined = beach_status.copy()

    # Normalize timestamp columns to tz-aware UTC — existing parquet rows from
    # the old scraper may be tz-naive, which causes sort_values to fail when
    # mixed with the tz-aware timestamps the ArcGIS scraper produces
    for col in ("updated_at", "recorded_at"):
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], utc=False).dt.tz_localize(
                "UTC", ambiguous="NaT", nonexistent="NaT"
            ) if combined[col].dt.tz is None else combined[col].dt.tz_convert("UTC")

    before = len(combined)
    combined = (
        combined
        .drop_duplicates(subset=["beach_name", "status", "updated_at", "recorded_at"])
        .sort_values(["beach_name", "updated_at", "recorded_at"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    after = len(combined)
    logging.info(f"[Beach Status] Saved {after - before} new rows to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    logging.info(f"[Beach Status] Parquet file now has {after} total rows | {path}")


def scrape_beach_statuses() -> list[dict]:
    logging.info(f"Loading {ARCGIS_URL}...")
    resp = requests.get(ARCGIS_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    features = data.get("features", [])
    if not features:
        raise RuntimeError("ArcGIS response contained no features — endpoint may have changed")

    beaches = [f.get("properties", {}) for f in features]
    logging.info(f"Found data for {len(beaches)} locations.")
    return beaches

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    beaches = scrape_beach_statuses()
    beach_status = clean_status(beaches)
    status_to_parquet(beach_status)