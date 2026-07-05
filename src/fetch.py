import requests
import pandas as pd
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_session(
    retries: int = 3,
    backoff_factor: float = 2.0,
) -> requests.Session:
    """
    Returns a requests Session that automatically retries on
    connection errors, timeouts, and 5xx responses.

    Waits between retries: 2s → 4s → 8s (backoff_factor * 2^retry)
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_weather_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetch a 7-day hourly forecast from Open-Meteo with retry logic."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude':           latitude,
        'longitude':          longitude,
        'hourly':             (
            'temperature_2m,relative_humidity_2m,precipitation,'
            'wind_speed_10m,wind_direction_10m,cloud_cover,'
            'shortwave_radiation,direct_normal_irradiance,'
            'diffuse_radiation,uv_index'
        ),
        'temperature_unit':   'fahrenheit',
        'wind_speed_unit':    'mph',
        'precipitation_unit': 'inch',
        'timezone':           'auto',
        'forecast_days':      7,
    }

    session = _make_session(retries=3, backoff_factor=2.0)

    try:
        # Split connect vs. read timeout — connect should be fast,
        # read can take longer on slow responses
        response = session.get(url, params=params, timeout=(10, 60))
        response.raise_for_status()
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError(
            "Could not connect to api.open-meteo.com — check your network "
            "or VPN. The SSL handshake timed out."
        )
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            "Connected to api.open-meteo.com but the response took too long. "
            "Try again — the API may be under load."
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Network error reaching api.open-meteo.com: {e}\n"
            "Check DNS, firewall, or proxy settings."
        )

    data   = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "datetime":              hourly["time"],
        "temp_f":                hourly["temperature_2m"],
        "relative_humidity_pct": hourly["relative_humidity_2m"],
        "precipitation_in":      hourly["precipitation"],
        "wind_speed_mph":        hourly["wind_speed_10m"],
        "wind_direction_deg":    hourly["wind_direction_10m"],
        "cloud_cover_pct":       hourly["cloud_cover"],
        "shortwave_radiation":   hourly["shortwave_radiation"],
        "direct_normal_irr":     hourly["direct_normal_irradiance"],
        "diffuse_radiation":     hourly["diffuse_radiation"],
        "uv_index":              hourly["uv_index"],
        "fetched_at":            datetime.now(timezone.utc).isoformat(),
        "latitude":              latitude,
        "longitude":             longitude,
        "utc_offset_seconds":    data.get("utc_offset_seconds", 0),
        "timezone_abbrev":       data.get("timezone_abbreviation", "UTC"),
    })

    return df