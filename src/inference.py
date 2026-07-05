import lightgbm as lgb
from pathlib import Path
import pandas as pd
import numpy as np
import json

REGRESSION_MODEL_PATH  = Path("model/gbm_regression_model.txt")
REGRESSION_CONFIG_PATH = Path("model/config_regression.json")

CLASSIFICATION_MODEL_PATH  = Path("model/gbm_classification_model.txt")
CLASSIFICATION_CONFIG_PATH = Path("model/config_classification.json")

PREDICTIONS_PATH = Path("data/predictions.csv")

def _load_gbm() -> tuple:
    """
    Load the trained GBM model and its feature config.
    Returns (None, None) if no model has been trained yet —
    the rest of inference continues normally without it.
    """
    if not REGRESSION_MODEL_PATH.exists():
        print("[GBM] No model found — run src/train.py first")
        return None, None

    model  = lgb.Booster(model_file=str(REGRESSION_MODEL_PATH))
    config = json.loads(REGRESSION_CONFIG_PATH.read_text()) if REGRESSION_CONFIG_PATH.exists() else {}
    print(f"[GBM] Model loaded | trained at {config.get('trained_at', 'unknown')}")
    return model, config

# ── Generate predictions ──────────────────────────────────────────────────────
def _gbm_predictions(df: pd.DataFrame, model, config: dict) -> pd.DataFrame:

    df = (
        df.copy()
        .loc[lambda df: pd.to_datetime(df["datetime"]) > pd.Timestamp.now(tz="UTC").floor("h").tz_convert(None)]
    )

    features = config.get("features", [])
    missing  = [f for f in features if f not in df.columns]
    if missing:
        print(f"[GBM] Skipping — missing features: {missing}")
        return pd.DataFrame()

    valid = df.dropna(subset=features)
    preds = model.predict(valid[features])

    out = (
        valid[features + ["datetime", "datetime_local"]]
        .copy()
        .assign(
            predicted_at=pd.Timestamp.now(tz="UTC"),
            raw_preds=preds,
            target=lambda df: np.round(df["raw_preds"], 1)
        )
    )
    return out

def _predictions_to_csv(predictions: pd.DataFrame, path: Path) -> None:
    """
    Append predictions to a CSV file, creating it if it doesn't exist.
    """
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["datetime"])
        combined = pd.concat([existing, predictions]).drop_duplicates(subset=["datetime"], keep="last")
        combined.to_csv(path, index=False)
    else:
        predictions.to_csv(path, index=False)

def run_inference(df: pd.DataFrame) -> int:
    """
    Run inference on the given DataFrame using the trained GBM model.
    Returns the number of predictions made.
    """
    model, config = _load_gbm()
    if model is None:
        return 0

    predictions = _gbm_predictions(df, model, config)
    _predictions_to_csv(predictions, PREDICTIONS_PATH)

    return len(predictions)
