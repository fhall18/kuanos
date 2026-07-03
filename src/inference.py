import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

RESULTS_PATH = Path("data/inference_results.json")

def run_inference(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["datetime"]   = pd.to_datetime(df["datetime"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    now = pd.Timestamp.utcnow().tz_localize(None)

    results = {}

    # ── Use only the latest pull for forward-looking inference ────────────────
    # (the CSV has many pulls per datetime; we want today's freshest forecast)
    latest_pull = df["fetched_at"].max()
    latest_df   = df[df["fetched_at"] == latest_pull].copy()

    results["latest_pull"] = str(latest_pull)
    results["unique_pulls"] = int(df["fetched_at"].nunique())

    past   = latest_df[latest_df["datetime"] < now]
    future = latest_df[latest_df["datetime"] >= now].head(168)

    # ── Temperature trend ─────────────────────────────────────────────────────
    if len(past) >= 12:
        past = past.copy()
        past["t_idx"] = (
            (past["datetime"] - past["datetime"].min()).dt.total_seconds() / 3600
        )
        model = LinearRegression().fit(past[["t_idx"]], past["temp_f"])
        slope = round(float(model.coef_[0]), 4)
        results["temp_trend_f_per_hour"] = slope
        results["trend_direction"]        = "warming" if slope > 0 else "cooling"

    # ── Next-24h summary ──────────────────────────────────────────────────────
    next_24h = future[future["datetime"] <= now + pd.Timedelta(hours=24)]
    if not next_24h.empty:
        results["next_24h"] = {
            "temp_max_f":          round(float(next_24h["temp_f"].max()), 1),
            "temp_min_f":          round(float(next_24h["temp_f"].min()), 1),
            "avg_humidity_pct":    round(float(next_24h["relative_humidity_pct"].mean()), 1),
            "total_precip_in":     round(float(next_24h["precipitation_in"].sum()), 3),
            "hours_raining":       int(next_24h["is_raining"].sum()),
            "max_wind_mph":        round(float(next_24h["wind_speed_mph"].max()), 1),
            "avg_cloud_cover_pct": round(float(next_24h["cloud_cover_pct"].mean()), 1),
            "peak_uv_index":       round(float(next_24h["uv_index"].max()), 1),
            "peak_uv_risk":        next_24h.loc[next_24h["uv_index"].idxmax(), "uv_risk"],
            "peak_solar_w_m2":     round(float(next_24h["shortwave_radiation"].max()), 1),
        }

    # ── 7-day outlook ─────────────────────────────────────────────────────────
    if not future.empty:
        daily_groups = future.groupby(future["datetime"].dt.date)
        results["seven_day_outlook"] = [
            {
                "date":            str(date),
                "temp_max_f":      round(float(g["temp_f"].max()), 1),
                "temp_min_f":      round(float(g["temp_f"].min()), 1),
                "total_precip_in": round(float(g["precipitation_in"].sum()), 3),
                "hours_raining":   int(g["is_raining"].sum()),
                "peak_uv_index":   round(float(g["uv_index"].max()), 1),
                "peak_solar_w_m2": round(float(g["shortwave_radiation"].max()), 1),
                "dominant_wind":   g["wind_cardinal"].mode()[0],
            }
            for date, g in daily_groups
        ]

    # ── Anomaly detection (hour-aware z-score) ────────────────────────────────
    if len(past) >= 48 and not future.empty:
        past_c   = past.copy()
        future_c = future.copy()
        past_c["hour"]   = past_c["datetime"].dt.hour
        future_c["hour"] = future_c["datetime"].dt.hour

        hourly_stats = (
            past_c.groupby("hour")["temp_f"]
            .agg(["mean", "std"])
            .reset_index()
        )
        future_c = future_c.merge(hourly_stats, on="hour", how="left")
        future_c["z"] = (
            (future_c["temp_f"] - future_c["mean"]) / (future_c["std"] + 1e-6)
        )
        anomalous = future_c[future_c["z"].abs() > 2]
        results["anomaly_detection"] = {
            "anomalous_hours": len(anomalous),
            "max_z_score":     round(float(future_c["z"].abs().max()), 2),
            "is_anomalous":    len(anomalous) > 0,
        }

    # ── Forecast accuracy (bonus: only possible because we keep history) ───────
    # Compare what past pulls predicted for hours that have now passed,
    # vs. the most-recent (day-of) forecast for those same hours.
    accuracy = _compute_forecast_accuracy(df, now)
    if accuracy:
        results["forecast_accuracy"] = accuracy

    results["generated_at"]   = pd.Timestamp.utcnow().isoformat()
    results["total_records"]  = len(df)
    results["forecast_hours"] = len(future)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Inference] Results written → {RESULTS_PATH}")
    return results


def _compute_forecast_accuracy(df: pd.DataFrame, now: pd.Timestamp) -> dict:
    """
    For each past hour, compare what the day-of forecast predicted (fetched_at
    closest to that datetime) against what earlier pulls predicted.
    Returns MAE by how many days in advance the forecast was made.
    """
    past_hours = df[df["datetime"] < now].copy()
    if past_hours.empty or past_hours["fetched_at"].nunique() < 2:
        return {}

    # "Ground truth" = the fetch made closest to (but before) each datetime
    past_hours["lead_hours"] = (
        (past_hours["datetime"] - past_hours["fetched_at"])
        .dt.total_seconds() / 3600
    ).round(0)

    # Only keep rows where the forecast was made before the valid datetime
    past_hours = past_hours[past_hours["lead_hours"] > 0]

    # Bin into forecast lead buckets
    bins   = [0, 24, 48, 72, 120, 168]
    labels = ["<1 day", "1-2 days", "2-3 days", "3-5 days", "5-7 days"]
    past_hours["lead_bucket"] = pd.cut(
        past_hours["lead_hours"], bins=bins, labels=labels, right=True
    )

    # For each (datetime, location) pick the minimum-lead row as "ground truth"
    ground_truth = (
        past_hours.loc[past_hours.groupby(["datetime", "latitude", "longitude"])
                       ["lead_hours"].idxmin()]
        [["datetime", "latitude", "longitude", "temp_f"]]
        .rename(columns={"temp_f": "temp_truth"})
    )

    merged = past_hours.merge(
        ground_truth, on=["datetime", "latitude", "longitude"], how="left"
    )
    merged["abs_error"] = (merged["temp_f"] - merged["temp_truth"]).abs()

    accuracy = {}
    for bucket, grp in merged.groupby("lead_bucket", observed=True):
        accuracy[str(bucket)] = {
            "mae_f":       round(float(grp["abs_error"].mean()), 2),
            "sample_hours": int(len(grp)),
        }

    return accuracy
