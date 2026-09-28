"""Focused tests for the Atmos analysis core.

Run:  ./.venv/Scripts/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import ALL_FEATURES, BASELINE_FEATURES, build_modeling_frame
from src.ml_models import CHRONO_CUTOFF, _chronological_split
from src.config import RANDOM_SEED


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data/processed/daily_panel.csv",
                       parse_dates=["date"])


@pytest.fixture(scope="module")
def modeling(panel) -> pd.DataFrame:
    return build_modeling_frame(panel)


# --------------------------------------------------------------------- #
# leakage guards
# --------------------------------------------------------------------- #
def test_no_o3_columns_in_features():
    """The target is derived from O3; no O3 variable may be a model feature."""
    forbidden = [f for f in ALL_FEATURES if "O3" in f.upper() or "o3_" in f.lower()]
    assert forbidden == []


def test_lag_features_are_past_only(modeling):
    """Rolling/lag features must use shift(1): the first day must be NaN."""
    assert pd.isna(modeling["no2_lag1"].iloc[0])
    assert pd.isna(modeling["no2_roll3_mean"].iloc[0])
    # and lag-1 must equal the previous day's value (not the same day)
    valid = modeling.dropna(subset=["no2_lag1"]).head(50)
    assert (valid["no2_lag1"].values ==
            modeling.loc[valid.index - 1, "NO2"].values).all()


def test_target_uses_max_not_mean(modeling):
    """Target must be defined on basin-max O3 (>70 ppb), not basin mean."""
    expected = (modeling["O3_max"] > 70).astype(float)
    assert (modeling["o3_exceed"] == expected).all()


def test_chronological_split_no_shuffling(modeling):
    train = modeling[modeling["date"] < CHRONO_CUTOFF]
    test = modeling[modeling["date"] >= CHRONO_CUTOFF]
    assert train["date"].max() < pd.Timestamp(CHRONO_CUTOFF)
    assert test["date"].min() >= pd.Timestamp(CHRONO_CUTOFF)
    Xtr, ytr, Xte, yte = _chronological_split(modeling, BASELINE_FEATURES)
    assert len(Xtr) > 0 and len(Xte) > 0
    assert set(ytr.unique()) <= {0, 1}


# --------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------- #
def test_rf_reproducible_with_seed(modeling):
    from sklearn.ensemble import RandomForestClassifier
    Xtr, ytr, Xte, _ = _chronological_split(modeling, BASELINE_FEATURES)
    preds = []
    for _ in range(2):
        m = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                   n_jobs=1).fit(Xtr, ytr)
        preds.append(m.predict_proba(Xte)[:, 1])
    assert np.allclose(preds[0], preds[1])


# --------------------------------------------------------------------- #
# statistical sanity
# --------------------------------------------------------------------- #
def test_t_ci_wider_than_z_ci():
    from src.statistics import ci_for_mean
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(10, 3, 60))
    r = ci_for_mean(s)
    assert r["width_t"] > r["width_z"]


def test_independent_rule_has_lift_one():
    from src.association_rules import mine_rules
    n = 2000
    rng = np.random.default_rng(1)
    bins = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n),
        "high_temp": rng.random(n) < 0.25,
        "low_wind": rng.random(n) < 0.25,
        "high_no2": rng.random(n) < 0.25,
        "high_pm25": rng.random(n) < 0.25,
        "o3_exceed": rng.random(n) < 0.15,
    })
    rules = mine_rules(bins)
    single = rules[~rules["antecedent"].str.contains("&")]
    # under independence, confidence ~ target prevalence -> lift ~ 1
    # (tolerance reflects sampling noise at n=2000, p=0.15)
    assert (single["lift"] - 1).abs().max() < 0.35
    # a bug detector: no rule may show a spuriously strong association
    assert single["lift"].max() < 2.0


def test_forecast_split_is_chronological(panel):
    from src.forecasting import build_weekly_series, train_test_split_chrono
    s, _ = build_weekly_series(panel)
    train, test = train_test_split_chrono(s)
    assert train.index.max() < test.index.min()
    assert test.index.min().year == 2025


# --------------------------------------------------------------------- #
# regression module (continuous O3_max estimation)
# --------------------------------------------------------------------- #
def test_regression_module_end_to_end(modeling):
    """Chronological split, no O3 features, and OLS beats the mean baseline."""
    from src.regression import TARGET, chronological_split, run_regression

    # temporal contract: identical cutoff; test fold = calendar 2025
    assert TARGET == "O3_max"
    Xtr, ytr, Xte, yte, dates_te = chronological_split(modeling, ALL_FEATURES)
    assert len(Xtr) > 0 and len(Xte) > 0
    assert dates_te.min().year == 2025 and dates_te.max().year == 2025

    # performance floor: fitted models must beat predicting the train mean
    res = run_regression(modeling)
    m = res["metrics"]
    assert m["ols"]["rmse"] < m["dummy_mean"]["rmse"]
    assert m["ols"]["r2"] > 0.3
    for name in ["dummy_mean", "ols", "ridge", "random_forest"]:
        assert {"mae", "mse", "rmse", "r2"} <= set(m[name])
