import asyncio
import logging
from src.fetch import fetch_weather_forecast
from src.etl import transform, load_to_parquet
from src.inference import run_inference
from src.status import run as run_beach_status

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
        beach_status = run_beach_status()
        if beach_status is not None:
            log.info(f"Last updated: {beach_status['updated_at'].max()}")
        else:
            log.warning("No beach status available")
    except Exception as e:
        log.error(f"Beach scrape failed: {e}")

    log.info("Pipeline complete")

if __name__ == "__main__":
    main()