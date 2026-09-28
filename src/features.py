"""Feature engineering for the Atmos modeling layer.

LEAKAGE POLICY (critical):

* TARGET: `o3_exceed` = 1 if the basin daily MAX 8-h O3 across monitors
  exceeded 70 ppb (the current US 8-hour NAAQS level). "Exceedance day"
  is defined by ANY monitor exceeding, matching regulatory practice;
  basin-mean O3 never enters the feature set (it is essentially the same
  measurement as the target and would be near-circular).

* BASELINE FEATURES (same-day, cross-parameter): basin-mean NO2, TEMP,
  WIND plus calendar attributes (month sin/cos, day-of-week). These are
  measured/known operationally for the day and are NOT the target
  measurement (basin-mean O3 and O3_max are excluded).

* ENGINEERED FEATURES: physically motivated interactions and rolling
  statistics computed ONLY from past days (lag>=1). Same-day values are
  available operationally by evening (regulatory daily summaries publish
  same-day), so same-day NO2/TEMP/WIND are admissible predictors for a
  *same-evening screening* use case; anything rolling uses lag>=1 data
  exclusively.

* Any feature that used O3 statistics (mean or max, same-day or lagged)
  as input is excluded from the model feature set by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED  # noqa: E402

BASELINE_FEATURES = [
    "NO2", "TEMP", "WIND",
    "month_sin", "month_cos", "dow_is_weekend",
]

ENGINEERED_FEATURES = [
    # physics-guided interactions
    "temp_x_no2",
    "no2_per_wind",
    "no2_x_weekend",
    # persistence of pollution (lag-1 only: past information)
    "no2_lag1", "temp_lag1",
    "no2_roll3_mean", "no2_roll3_max", "temp_roll3_mean",
    # warm-season indicator (May-Sep, the climatological ozone season;
    # pure calendar - contains no O3 measurement)
    "is_warm_season",
]

ALL_FEATURES = BASELINE_FEATURES + ENGINEERED_FEATURES


def build_modeling_frame(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---- target ---------------------------------------------------------
    df["o3_exceed"] = (df["O3_max"] > 70).astype(float)

    # ---- calendar (deterministic, known in advance) ---------------------
    m = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    df["dow_is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_warm_season"] = m.isin([5, 6, 7, 8, 9]).astype(int)

    # ---- same-day interactions (no target information) -------------------
    df["temp_x_no2"] = df["TEMP"] * df["NO2"] / 1000.0       # scale for stability
    df["no2_per_wind"] = df["NO2"] / df["WIND"].replace(0, np.nan)
    df["no2_x_weekend"] = df["NO2"] * df["dow_is_weekend"]

    # ---- persistence features: strictly past-only (lag >= 1) -------------
    for col, lag in [("NO2", 1), ("TEMP", 1)]:
        df[f"{col.lower()}_lag{lag}"] = df[col].shift(lag)
    df["no2_roll3_mean"] = df["NO2"].shift(1).rolling(3).mean()
    df["no2_roll3_max"] = df["NO2"].shift(1).rolling(3).max()
    df["temp_roll3_mean"] = df["TEMP"].shift(1).rolling(3).mean()

    return df


def load_modeling_frame() -> pd.DataFrame:
    panel = pd.read_csv(DATA_PROCESSED / "daily_panel.csv", parse_dates=["date"])
    return build_modeling_frame(panel)


def frame_with_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Alias kept for notebook/dashboard use; builds features from a panel."""
    return build_modeling_frame(panel)
