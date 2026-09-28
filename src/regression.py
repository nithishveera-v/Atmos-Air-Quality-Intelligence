"""Linear regression module for the Atmos project (syllabus topic: Linear
Regression, with regularized and nonlinear reference models).

TASK (continuous target): estimate the basin daily MAX 8-hour O3 level
(`O3_max`, ppb) from the same day's cross-pollutant network means (NO2, TEMP,
WIND) and calendar attributes.

* This is an ESTIMATION / nowcast-framing task, not forecasting: same-day
  co-pollutant and meteorological variables are admissible because regulatory
  daily summaries are published same-day; O3 variables (mean or max) are
  excluded by the same leakage rule as the classification module - the target
  is built from O3, so no O3 measurement may be a feature.
* Operational motivation: `O3_max` is the daily peak across up to 44 monitors.
  When the peak monitor reports late or fails, an estimate of the basin maximum
  from the rest of the network is genuinely useful. This module quantifies how
  well the linear part of that relationship performs.

TEMPORAL INTEGRITY: identical chronological split as src/ml_models.py
(train < CHRONO_CUTOFF, test = calendar 2025); no shuffling, no look-ahead,
no tuning on the test year. Scalers are fitted on the training fold only.

Models:
    1. DummyRegressor (mean)      - the honest floor for a continuous target
    2. LinearRegression (OLS)     - syllabus topic; standardized features so
                                    coefficients are per-standard-deviation
                                    effects; inference (SE, t, p, 95% CI) via
                                    statsmodels OLS on the same standardized
                                    design (identical point estimates)
    3. Ridge (alpha = 1.0)        - L2-regularized variant, chosen A PRIORI
                                    (no test-set tuning)
    4. RandomForestRegressor      - nonlinear reference point (same family as
                                    the classification module, fixed seed)

Metrics: MAE, MSE, RMSE, R2 on the 2025 holdout; train RMSE reported for the
overfit gap. Durbin-Watson on OLS residuals is reported because daily air
quality is autocorrelated - the independence assumption behind the OLS
standard errors does not fully hold, and p-values are read with that caveat.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, RANDOM_SEED  # noqa: E402
from src.features import ALL_FEATURES  # noqa: E402
from src.ml_models import CHRONO_CUTOFF  # noqa: E402

TARGET = "O3_max"
RIDGE_ALPHA = 1.0  # fixed a priori; not tuned on the test year


def chronological_split(df: pd.DataFrame, features: list[str],
                        target: str = TARGET):
    """Same temporal contract as the classification module."""
    train = df[df["date"] < pd.Timestamp(CHRONO_CUTOFF)]
    test = df[df["date"] >= pd.Timestamp(CHRONO_CUTOFF)]
    ok_tr = train[features].notna().all(axis=1) & train[target].notna()
    ok_te = test[features].notna().all(axis=1) & test[target].notna()
    return (train.loc[ok_tr, features], train.loc[ok_tr, target],
            test.loc[ok_te, features], test.loc[ok_te, target],
            test.loc[ok_te, "date"])


def _reg_metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def run_regression(df: pd.DataFrame) -> dict:
    Xtr, ytr, Xte, yte, dates_te = chronological_split(df, ALL_FEATURES)

    # ------------------------------------------------------------- models --
    # scaler fitted on the TRAINING fold only (Pipeline / pre-fit)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    ols = LinearRegression().fit(Xtr_s, ytr)
    ridge = Pipeline([("scaler", StandardScaler()),
                      ("ridge", Ridge(alpha=RIDGE_ALPHA))]).fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                               random_state=RANDOM_SEED, n_jobs=-1
                               ).fit(Xtr_s, ytr)

    preds = {
        "dummy_mean": dummy.predict(Xte),
        "ols": ols.predict(Xte_s),
        "ridge": ridge.predict(Xte),
        "random_forest": rf.predict(Xte_s),
    }

    # ----------------------------------------------- OLS inference table --
    # statsmodels OLS on the SAME standardized train design -> identical
    # point estimates to sklearn LinearRegression; adds SE / t / p / CI.
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson

    Xtr_sm = sm.add_constant(Xtr_s, has_constant="add")
    ols_sm = sm.OLS(np.asarray(ytr, dtype=float), Xtr_sm).fit()
    names = ["const"] + list(ALL_FEATURES)
    coef_table = pd.DataFrame({
        "feature": names,
        "coef": ols_sm.params,
        "std_err": ols_sm.bse,
        "t_stat": ols_sm.tvalues,
        "p_value": ols_sm.pvalues,
        "ci_low": ols_sm.conf_int()[:, 0],
        "ci_high": ols_sm.conf_int()[:, 1],
    }).round(4)

    resid_train = np.asarray(ytr) - ols.predict(Xtr_s)
    dw = float(durbin_watson(resid_train))

    # ------------------------------------------------------------- report --
    train_preds = {
        "ols": ols.predict(Xtr_s),
        "ridge": ridge.predict(Xtr),
        "random_forest": rf.predict(Xtr_s),
    }
    metrics = {}
    for name, p in preds.items():
        metrics[name] = _reg_metrics(yte, p)
        if name in train_preds:  # overfit gap: train vs test RMSE
            rmse_tr = float(np.sqrt(np.mean((ytr - train_preds[name]) ** 2)))
            metrics[name]["rmse_train"] = round(rmse_tr, 3)

    best = min((k for k in metrics if k != "dummy_mean"),
               key=lambda k: metrics[k]["rmse"])

    results = {
        "task": ("estimate same-day basin daily-max 8h O3 (ppb) from "
                 "cross-pollutant network means + calendar (estimation, "
                 "not forecasting)"),
        "target": TARGET,
        "split": {"train_end": CHRONO_CUTOFF,
                  "test_period": "2025-01-01..2025-12-31",
                  "seed": RANDOM_SEED},
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "features": list(ALL_FEATURES),
        "metrics": metrics,
        "best_by_rmse": best,
        "ols_inference": coef_table.to_dict(orient="records"),
        "durbin_watson_train_residuals": round(dw, 3),
        "predictions": {k: np.asarray(v).tolist() for k, v in preds.items()},
        "test_actual": np.asarray(yte).tolist(),
        "test_dates": [str(d.date()) for d in dates_te],
        "interpretation": (
            f"Best holdout RMSE: {best} ({metrics[best]['rmse']} ppb vs mean-baseline "
            f"{metrics['dummy_mean']['rmse']}). OLS coefficients are per-SD effects "
            f"from the training fold; Durbin-Watson = {dw:.2f} confirms residual "
            f"autocorrelation, so OLS p-values (independence assumption) should be "
            f"read with caution. Associations describe this dataset; they do not "
            f"establish causation."),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "regression_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    for name, m in metrics.items():
        print(f"[REG] {name:14s} MAE={m['mae']:6.2f}  RMSE={m['rmse']:6.2f}  "
              f"R2={m['r2']:.3f}")
    print(f"[REG] best_by_rmse = {best} | Durbin-Watson (train residuals) = {dw:.2f}")
    return results


def load_modeling_frame_for_regression() -> pd.DataFrame:
    from src.features import load_modeling_frame
    return load_modeling_frame()


if __name__ == "__main__":
    run_regression(load_modeling_frame_for_regression())
