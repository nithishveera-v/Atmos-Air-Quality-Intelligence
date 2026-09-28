"""Forecasting module for the Atmos project.

TEMPORAL INTEGRITY: forecasts use the genuine daily time index of the
AQS panel (2021-2025), aggregated to weekly means. There are NO random
or synthetic timestamps anywhere in this project.

Design:
    * Series: basin-mean PM2.5, weekly mean (weeks with < 4 valid daily
      values are excluded, then interior gaps are time-interpolated;
      excluded/interpolated counts are reported).
    * Split: CHRONOLOGICAL - train through 2024-12-31, test = 2025
      (~53 weeks). No shuffling, no look-ahead: models are fitted on the
      training window only and produce a multi-step forecast over the
      entire test window.
    * Models: Holt (damped trend, no seasonality), Holt-Winters additive
      (52-week seasonality), and a seasonal-naive baseline (value from
      52 weeks earlier) as the honesty floor.
    * Metrics: MAE, RMSE, MAPE (on the holdout), computed from executed
      code only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED  # noqa: E402

SEASONAL_PERIOD = 52
MIN_DAYS_PER_WEEK = 4


def build_weekly_series(panel: pd.DataFrame, col: str = "PM25") -> pd.Series:
    d = panel[["date", col]].dropna().copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    weekly = d.resample("W-SUN").agg(["mean", "count"])[col]
    weekly.loc[weekly["count"] < MIN_DAYS_PER_WEEK, "mean"] = np.nan
    s = weekly["mean"].asfreq("W-SUN")
    n_before = s.notna().sum()
    s = s.interpolate(method="time", limit_direction="both")
    s.index = pd.DatetimeIndex(s.index, freq="W-SUN")
    return s.rename(f"{col}_weekly_mean"), {"weeks_total": len(s),
                                            "weeks_valid_before_interp": int(n_before)}


def train_test_split_chrono(s: pd.Series, test_start: str = "2025-01-01"):
    train = s[s.index < pd.Timestamp(test_start)]
    test = s[s.index >= pd.Timestamp(test_start)]
    return train, test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    return {
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mape_pct": round(float(100 * np.mean(np.abs(err / y_true))), 2),
    }


def seasonal_naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Baseline: repeat the last full seasonal cycle of the training data."""
    vals = train.values[-SEASONAL_PERIOD:]
    reps = int(np.ceil(horizon / SEASONAL_PERIOD))
    return np.tile(vals, reps)[:horizon]


def fit_forecasts(train: pd.Series, horizon: int) -> dict:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

    out = {}
    holt = Holt(train, damped_trend=True, initialization_method="estimated").fit()
    out["holt_damped"] = np.asarray(holt.forecast(horizon))

    hw = ExponentialSmoothing(
        train, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated").fit()
    out["holt_winters_add"] = np.asarray(hw.forecast(horizon))

    out["seasonal_naive"] = seasonal_naive_forecast(train, horizon)
    out["fitted"] = {"holt_damped": holt, "holt_winters_add": hw}
    return out


def run_forecasting(panel: pd.DataFrame, col: str = "PM25") -> dict:
    s, info = build_weekly_series(panel, col)
    train, test = train_test_split_chrono(s)
    horizon = len(test)
    fits = fit_forecasts(train, horizon)

    results = {
        "series": f"{col} weekly basin mean",
        "series_info": info,
        "train_range": (str(train.index.min().date()), str(train.index.max().date())),
        "test_range": (str(test.index.min().date()), str(test.index.max().date())),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "seasonal_period": SEASONAL_PERIOD,
        "metrics": {},
        "predictions": {name: np.asarray(pred).tolist()
                        for name, pred in fits.items() if name != "fitted"},
        "test_actual": test.values.tolist(),
        "test_dates": [str(d.date()) for d in test.index],
        "train_series": train.tolist(),
        "train_dates": [str(d.date()) for d in train.index],
    }
    y = test.values
    for name in ("seasonal_naive", "holt_damped", "holt_winters_add"):
        results["metrics"][name] = _metrics(y, np.asarray(results["predictions"][name]))

    best = min(results["metrics"], key=lambda k: results["metrics"][k]["rmse"])
    results["best_by_rmse"] = best
    results["interpretation"] = (
        f"Best holdout RMSE: {best} ({results['metrics'][best]['rmse']} ug/m3, "
        f"vs seasonal-naive {results['metrics']['seasonal_naive']['rmse']}). "
        f"Forecasts are multi-step over a full held-out year; accuracy is "
        f"limited by wildfire episodes and meteorological variability that "
        f"exponential smoothing cannot anticipate. This is a statistical "
        f"baseline demonstration of the syllabus forecasting topic, not an "
        f"operational air-quality forecast system.")
    return results
