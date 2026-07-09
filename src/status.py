#!/usr/bin/env python3
"""
Burlington VT Beach Closure Tracker Scraper
https://www.burlingtonvt.gov/1219/Beach-Closure-Tracker

Uses Playwright async API to render the page and extract beach statuses from
the dynamic widget (data-widgetcontainerid="5d5bbbdc-6623-45b1-845c-9d64e35cf1e3").
Outputs results to data/beach_status.parquet.
"""

import logging
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright

URL = "https://www.burlingtonvt.gov/1219/Beach-Closure-Tracker"
WIDGET_ID = "5d5bbbdc-6623-45b1-845c-9d64e35cf1e3"
DATA_PATH = Path("data/beach_status.parquet")


def clean_status(beaches: list[dict]) -> pd.DataFrame:
    beach_status = (
        pd.DataFrame(beaches)
        .assign(
            beach_name=lambda df: df['Beach'].apply(lambda s: s.split('\n')[0]),
            updated=lambda df: df['Beach'].apply(lambda s: s.split('\n')[2]).str.rstrip("Updated: "),
            updated_at=lambda df: pd.to_datetime(df["updated"].str.replace("Updated: ", "")), 
            recorded_at=pd.Timestamp.now(tz="UTC"),
            status=lambda df: df['Status']
        )
        .filter(items=['beach_name', 'status', 'updated_at', 'recorded_at'])
    )
    return beach_status


def status_to_parquet(beach_status: pd.DataFrame, path: Path) -> None:
    """Save the beach status DataFrame to a Parquet file."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, beach_status], ignore_index=True)
    else:
        combined = beach_status.copy()

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


async def scrape_beach_statuses():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logging.info(f"Loading {URL}...")
        await page.goto(URL, wait_until="networkidle", timeout=30000)

        # Wait for the widget container to be present
        widget = page.locator(f'[data-widgetcontainerid="{WIDGET_ID}"]')
        try:
            await widget.wait_for(timeout=15000)
        except Exception:
            raise RuntimeError(
                f"Widget '{WIDGET_ID}' not found — the page structure may have changed."
            )

        # Give any inner JS a moment to finish rendering
        await page.wait_for_timeout(2000)

        beaches = []

        # Try to find a table inside the widget
        tables = widget.locator("table")
        table_count = await tables.count()

        if table_count > 0:
            table = tables.first
            headers = [
                (await th.inner_text()).strip()
                for th in await table.locator("thead th, thead td").all()
            ]
            if not headers:
                headers = [
                    (await td.inner_text()).strip()
                    for td in await table.locator("tr").first.locator("td, th").all()
                ]

            rows = await table.locator("tbody tr").all()
            if not rows:
                all_rows = await table.locator("tr").all()
                rows = all_rows[1:] if len(all_rows) > 1 else []

            for row in rows:
                cells = [(await td.inner_text()).strip() for td in await row.locator("td, th").all()]
                if cells:
                    if headers and len(headers) == len(cells):
                        beaches.append(dict(zip(headers, cells)))
                    else:
                        beaches.append({"raw": " | ".join(cells)})

        # If no table found, fall back to all text content in the widget
        if not beaches:
            text = await widget.inner_text()
            for line in text.splitlines():
                line = line.strip()
                if line:
                    beaches.append({"raw": line})

        await browser.close()
        return beaches