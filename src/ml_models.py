"""Model training and evaluation for the Atmos screening model.

Reproducibility: fixed RANDOM_SEED (src/config), CHRONOLOGICAL split
(no shuffling across time), and every metric written to outputs/metrics
is produced by executing this module.

Models:
    1. majority-class baseline  (the honest floor for imbalanced data)
    2. Gaussian Naive Bayes     (syllabus: Naive Bayes)
    3. Random Forest            (syllabus: Random Forest), class_weight balanced

Evaluation reflects imbalance: PR-AUC and recall/precision at the
operating threshold are primary; accuracy and ROC-AUC reported but not
relied upon. Threshold 0.5 for NB/RF (documented).

Scaling is NOT needed for tree models; NB is scale-sensitive, so a
StandardScaler is fitted on the TRAINING FOLD ONLY inside a sklearn
Pipeline (no preprocessing fitted on test data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, accuracy_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR, MODELS_DIR, RANDOM_SEED  # noqa: E402
from src.features import ALL_FEATURES, BASELINE_FEATURES, ENGINEERED_FEATURES  # noqa: E402

CHRONO_CUTOFF = "2025-01-01"  # train 2021-2024, test 2025 (calendar holdout)


def _chronological_split(df: pd.DataFrame, features: list[str]):
    train = df[df["date"] < CHRONO_CUTOFF]
    test = df[df["date"] >= CHRONO_CUTOFF]
    Xtr, ytr = train[features], train["o3_exceed"]
    Xte, yte = test[features], test["o3_exceed"]
    ok = Xtr.notna().all(axis=1) & ytr.notna()
    okte = Xte.notna().all(axis=1) & yte.notna()
    return Xtr[ok], ytr[ok].astype(int), Xte[okte], yte[okte].astype(int)


def _classification_report_dict(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_prob)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _make_models() -> dict:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        "naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("nb", GaussianNB()),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_SEED, n_jobs=-1),
    }


def run_model_comparison(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict:
    results: dict = {"split": {"train_end": CHRONO_CUTOFF,
                               "test_period": "2025-01-01..2025-12-31",
                               "seed": RANDOM_SEED}}
    models = _make_models()
    for set_name, feats in feature_sets.items():
        Xtr, ytr, Xte, yte = _chronological_split(df, feats)
        for model_name, model in models.items():
            model.fit(Xtr, ytr)
            prob = model.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            rep = _classification_report_dict(yte, pred, prob)
            rep["n_features"] = len(feats)
            results[f"{set_name}/{model_name}"] = rep
            print(f"[ML] {set_name:10s} {model_name:18s} "
                  f"PR-AUC={rep['pr_auc']:.3f} ROC-AUC={rep.get('roc_auc')} "
                  f"recall={rep['recall']} precision={rep['precision']}")
    return results


def feature_importance_rf(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    Xtr, ytr, _, _ = _chronological_split(df, features)
    rf = _make_models()["random_forest"]
    rf.fit(Xtr, ytr)
    imp = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp["cumulative"] = imp["importance"].cumsum()
    return imp


def save_artifacts(results: dict, importance: pd.DataFrame) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "model_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    importance.to_csv(METRICS_DIR / "rf_feature_importance.csv", index=False)


def run_full_ml() -> tuple[dict, pd.DataFrame]:
    from src.features import load_modeling_frame
    df = load_modeling_frame()
    feature_sets = {
        "baseline": BASELINE_FEATURES,
        "engineered": ALL_FEATURES,
    }
    results = run_model_comparison(df, feature_sets)
    importance = feature_importance_rf(df, ALL_FEATURES)
    save_artifacts(results, importance)
    return results, importance


if __name__ == "__main__":
    run_full_ml()
