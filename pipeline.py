import asyncio
import logging
from src.fetch import fetch_weather_forecast
from src.etl import transform, load_to_parquet
from src.inference import run_inference
from src.status import scrape_beach_statuses, clean_status, status_to_parquet, DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Config — swap these for your location ────────────────────────────────────
LATITUDE  = 44.4759   # Burlington, VT
LONGITUDE = -73.2121
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Pipeline start")

    # Extract
    raw = fetch_weather_forecast(LATITUDE, LONGITUDE)
    log.info(f"Fetched {len(raw)} forecast rows")

    # Transform
    log.info("Starting data transformation")
    transformed = transform(raw)

    # Load
    total_rows = load_to_parquet(transformed)
    log.info(f"Parquet file now has {total_rows} total rows")

    # Inference
    results = run_inference(transformed)
    log.info(f"Inference results: {results}")

    # Beach status
    log.info("Scraping beach statuses")
    try:
        beaches = asyncio.run(scrape_beach_statuses())
        log.info(f"Found {len(beaches)} beach entries")
        beach_status = clean_status(beaches)
        log.info(f"Last updated: {beach_status['updated_at'].max()}")
        status_to_parquet(beach_status, DATA_PATH)
    except Exception as e:
        log.error(f"Beach scrape failed: {e}")

    log.info("Pipeline complete")

if __name__ == "__main__":
    main()