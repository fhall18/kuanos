import requests
import pandas as pd
from datetime import datetime, timezone

def fetch_weather_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Fetch a 7-day hourly forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,cloud_cover,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,uv_index',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'precipitation_unit': 'inch',
        'timezone': 'auto',
        'forecast_days': 7,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime":              hourly["time"],
        "temp_f":                hourly["temperature_2m"],
        "relative_humidity_pct": hourly["relative_humidity_2m"],
        "precipitation_in":      hourly["precipitation"],
        "wind_speed_mph":        hourly["wind_speed_10m"],
        "wind_direction_deg":    hourly["wind_direction_10m"],
        "cloud_cover_pct":       hourly["cloud_cover"],
        "shortwave_radiation":   hourly["shortwave_radiation"],   # W/m²
        "direct_normal_irr":     hourly["direct_normal_irradiance"],  # W/m²
        "diffuse_radiation":     hourly["diffuse_radiation"],     # W/m²
        "uv_index":              hourly["uv_index"],
        "fetched_at":            datetime.now(timezone.utc).isoformat(),
        "latitude":              latitude,
        "longitude":             longitude,
        "utc_offset_seconds":    data.get("utc_offset_seconds", 0),
        "timezone_abbrev":       data.get("timezone_abbreviation", "UTC"),
    })

    return df