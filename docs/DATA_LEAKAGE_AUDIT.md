# DATA LEAKAGE AUDIT — Atmos

**Scope:** every path by which target or future information could contaminate
model inputs, evaluation, or reported results. Status column uses
**PASS** (checked, no leakage) / **FIXED** (found and corrected during
development) / **N/A with justification**.

## 1. Target definition

| Item | Status | Notes |
|---|---|---|
| Target = `o3_exceed` = (basin daily **max** O3 > 70 ppb) | PASS | Defined in `src/features.py::build_modeling_frame`; grounded in the current US 8-hour NAAQS |
| No O3 variable (mean or max, same-day or lagged) used as a feature | PASS | Enforced by construction and by `tests/test_core.py::test_no_o3_columns_in_features`; basin O3 is the same measurement as the target and is excluded |
| Earlier design (target = basin **mean** O3 > 70) had 0 positive days | FIXED | Mean-across-sites dilutes peaks; target redefined to basin max, matching regulatory "any monitor exceeds" practice — no positive class existed before the fix |

## 2. Feature construction

| Item | Status | Notes |
|---|---|---|
| Rolling/lag features computed from **past only** | PASS | All use `shift(1)`; first row is NaN; verified by `test_lag_features_are_past_only` |
| Same-day cross-parameter features (NO2, TEMP, WIND, calendar) | PASS with justification | Admissible for the stated use case (same-evening screening after daily summaries publish); they are not the target measurement |
| Calendar features (month sin/cos, weekend, warm-season flag) | PASS | Pure calendar; `is_warm_season` renamed from `is_o3_season` so the leakage guard is unambiguous |
| Interactions (temp×NO2, NO2/wind, NO2×weekend) | PASS | Built from admissible inputs only |
| Missing-value policy | PASS | Rows with any missing feature are dropped per split; no imputation fitted on test data |

## 3. Preprocessing / split discipline

| Item | Status | Notes |
|---|---|---|
| Chronological split (train < 2025-01-01 ≤ test) | PASS | `CHRONO_CUTOFF` in `src/ml_models.py`; no shuffling across time; verified by `test_chronological_split_no_shuffling` |
| Scaler fitted on training fold only | PASS | `StandardScaler` inside sklearn `Pipeline` (NB model); never fit on test |
| TF-IDF vectorizer fitted on training fold only | PASS | NLP pipeline: vectorizer + NB fit on train split only |
| Feature selection using full dataset before split | N/A | No pre-split feature-selection step exists; importance is computed on the train fold; engineered-vs-baseline comparison uses identical splits for both sets |
| Duplicate records across train/test | PASS | One row per calendar day; split is by date so no entity can appear on both sides |
| Threshold/tuning leakage | PASS | Threshold 0.5 fixed a priori; **no hyperparameter search was performed on the test year** |

## 4. Label / outcome leakage outside the primary model

| Item | Status | Notes |
|---|---|---|
| NLP labels | PASS | Labels are official Federal Register document metadata (`type`); no text was generated from labels |
| NLP vectorization | PASS | TF-IDF fitted on train only (inside Pipeline) |
| NLP corpus separation | PASS | Secondary corpus explicitly excluded from the sensor-model layer |
| Association-rule bins | PASS | Quantile thresholds computed on the analysis subset without reference to the outcome item's distribution; outcome included only as an item, never as a feature of anything |
| Clustering | PASS | Unsupervised; no labels used in fitting; scaler fitted on the (unlabeled) analysis matrix — no supervised objective exists to leak |

## 5. Performance-plausibility review

| Result | Plausibility check | Status |
|---|---|---|
| NB PR-AUC 0.639 / RF 0.585 (2025 holdout, prevalence 11.8%) | Well above the 0.118 floor, below suspicious levels; confusion matrices show realistic FP/FN trade-offs; engineered set does NOT outperform baseline (a leakage signature would usually show inflated engineered results) | PASS |
| NLP accuracy 0.953 | Rule vs Notice differ in procedural vocabulary; rates of this kind are typical for document-type classification; confusion matrix shows a realistic 14/300 FP pattern | PASS |
| Forecast RMSE 3.33 (Holt-Winters) vs 4.17 (seasonal naive) | Modest, honest improvement; no test-period fitting | PASS |

## 6. Known residual limitations (documented, not hidden)

1. Same-day meteorology/pollutant features assume the same-evening screening
   use case; for a *morning-of* forecast the same-day values would be
   unavailable and the lag-only feature set should be used instead.
2. Basin-max O3 uses the monitoring network as deployed; site openings/
   closings across 2021-2025 slightly change network sensitivity to peaks.
3. Autocorrelation of daily series means test-window metrics have wider
   uncertainty than independent-sample formulas would suggest.
