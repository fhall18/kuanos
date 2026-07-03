import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/weather_history.csv")

def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"]   = pd.to_datetime(df["datetime"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])

    df["date"]         = df["datetime"].dt.date
    df["hour"]         = df["datetime"].dt.hour
    df["day_of_week"]  = df["datetime"].dt.day_name()
    df["is_daytime"]   = df["hour"].between(6, 20)
    df["wind_cardinal"] = df["wind_direction_deg"].apply(_deg_to_cardinal)
    df["total_irradiance"] = df["shortwave_radiation"] + df["diffuse_radiation"]
    df["is_sunny"]  = (df["shortwave_radiation"] > 200) & (df["cloud_cover_pct"] < 40)
    df["uv_risk"]   = df["uv_index"].apply(_uv_risk_label)
    df["is_raining"] = df["precipitation_in"] > 0.0
    df["feels_like_f"] = df.apply(
        lambda r: _heat_index_f(r["temp_f"], r["relative_humidity_pct"])
                  if r["temp_f"] >= 80
                  else _wind_chill_f(r["temp_f"], r["wind_speed_mph"]),
        axis=1,
    )
    return df


def load_to_csv(new_data: pd.DataFrame) -> pd.DataFrame:
    """
    Append new forecast rows to the historical CSV.

    Deduplication key: (datetime, latitude, longitude, fetched_at)
    ─ This prevents the same pipeline run from writing twice,
      but intentionally KEEPS multiple forecasts for the same
      datetime that were pulled on different days.

    Over time the CSV will look like:
      datetime            fetched_at           temp_f  ...
      2026-07-04 14:00    2026-07-01 08:00     82.1
      2026-07-04 14:00    2026-07-02 08:00     81.4   ← same hour, different pull
      2026-07-04 14:00    2026-07-03 08:00     83.0
      2026-07-04 14:00    2026-07-04 08:00     82.8   ← actual day-of forecast
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH, parse_dates=["datetime", "fetched_at"])
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data.copy()

    before = len(combined)
    combined = (
        combined
        # Only drop true exact duplicates (same pull, same row)
        .drop_duplicates(subset=["datetime", "latitude", "longitude", "fetched_at"])
        .sort_values(["datetime", "fetched_at"])
        .reset_index(drop=True)
    )
    after = len(combined)

    combined.to_csv(DATA_PATH, index=False)
    print(
        f"[ETL] +{after - (before - len(new_data))} new rows appended | "
        f"{after} total rows | {DATA_PATH}"
    )
    return combined


# ── Helpers (unchanged) ───────────────────────────────────────────────────────

def _deg_to_cardinal(deg: float) -> str:
    if pd.isna(deg): return "N/A"
    directions = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                  "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return directions[round(deg / 22.5) % 16]

def _uv_risk_label(uv: float) -> str:
    if pd.isna(uv): return "Unknown"
    if uv < 3:      return "Low"
    if uv < 6:      return "Moderate"
    if uv < 8:      return "High"
    if uv < 11:     return "Very High"
    return "Extreme"

def _heat_index_f(t: float, rh: float) -> float:
    hi = (-42.379 + 2.04901523*t + 10.14333127*rh
          - 0.22475541*t*rh - 6.83783e-3*t**2
          - 5.481717e-2*rh**2 + 1.22874e-3*t**2*rh
          + 8.5282e-4*t*rh**2 - 1.99e-6*t**2*rh**2)
    return round(hi, 1)

def _wind_chill_f(t: float, ws: float) -> float:
    if t > 50 or ws < 3: return round(t, 1)
    wc = (35.74 + 0.6215*t - 35.75*(ws**0.16) + 0.4275*t*(ws**0.16))
    return round(wc, 1)
