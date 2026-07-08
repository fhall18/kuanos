import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/weather_history.parquet")

def transform(df: pd.DataFrame) -> pd.DataFrame:
    
    historical_df = pd.read_parquet(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

    historical_df = (
        historical_df
        .sort_values("fetched_at", ascending=False)
        .drop_duplicates(subset=["datetime", "latitude", "longitude"], keep="first")
        .loc[lambda df: pd.to_datetime(df["datetime"]) > pd.Timestamp.now(tz="UTC").tz_convert(None) - pd.Timedelta(days=4)]
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    daily_avg = (
        pd.concat([historical_df, df], ignore_index=True)
        .sort_values("fetched_at", ascending=False)
        .drop_duplicates(subset=["datetime", "latitude", "longitude"], keep="first")
        .assign(
            datetime_local=lambda df: pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York"),
            date=lambda df: df.datetime_local.dt.date
            )
        .groupby('date')[['temp_f',
                        'wind_speed_mph',
                        'shortwave_radiation',
                        'direct_normal_irr',
                        'diffuse_radiation',
                        ]]
        .mean()
        .shift(1)  # shift by 1 day
        .rename(columns={'temp_f': 'prior_day_temp', 
                        'wind_speed_mph': 'prior_day_wind_speed',
                        'shortwave_radiation': 'prior_day_shortwave_radiation',
                        'direct_normal_irr': 'prior_day_direct_normal_irradiance',
                        'diffuse_radiation': 'prior_day_diffuse_radiation'})
    )

    weather_transformed = (
        df.copy()
        .assign(
            datetime=lambda df: pd.to_datetime(df["datetime"]),
            datetime_local=lambda df: pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York"),
            date=lambda df: df.datetime_local.dt.date,
            year=lambda df: df.datetime_local.dt.year,
            month=lambda df: df.datetime_local.dt.month,
            day_number=lambda df: df.datetime_local.dt.day_of_year,
            # 1-day rolling mean, sum, and median based on discretion
            rolling_temperature_24h=lambda df: df["temp_f"].transform(
                        lambda x: x.rolling(window=24, min_periods=1).mean()
                        ),
            rolling_precipitation_24h=lambda df: df["precipitation_in"].transform(
                        lambda x: x.rolling(window=24, min_periods=1).sum()
                        ),
            rolling_wind_direction_24h=lambda df: df["wind_direction_deg"].transform(
                        lambda x: x.rolling(window=24, min_periods=1).median()
                        ),
            rolling_wind_speed_24h=lambda df: df["wind_speed_mph"].transform(
                lambda x: x.rolling(window=24, min_periods=1).mean()
                ),
            # 3-day rolling mean, sum, and median based on discretion
            rolling_temperature_72h=lambda df: df["temp_f"].transform(
                        lambda x: x.rolling(window=72, min_periods=1).mean()
                        ),
            rolling_precipitation_72h=lambda df: df["precipitation_in"].transform(
                        lambda x: x.rolling(window=72, min_periods=1).sum()
                        ),
            rolling_wind_direction_72h=lambda df: df["wind_direction_deg"].transform(
                        lambda x: x.rolling(window=72, min_periods=1).median()
                        ),
            rolling_wind_speed_72h=lambda df: df["wind_speed_mph"].transform(
                lambda x: x.rolling(window=72, min_periods=1).mean()
                ),
        )
        .merge(daily_avg, on='date', how='left')
    )
    
    return weather_transformed


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
        existing = pd.read_parquet(DATA_PATH)
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

    combined.to_parquet(DATA_PATH, index=False)
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
