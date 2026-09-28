# METHODOLOGY — Atmos

## Architecture

```
data/raw (EPA AQS national daily CSVs, downloaded once)
  |  src/etl.py  EXTRACT -> TRANSFORM (documented rules) -> VALIDATE -> LOAD
  v
data/processed/  site-day CSVs + basin daily panel + quality report
  |  src/db.py   SQLite schema, load, SQL demonstrations
  v
database/atmos_airquality.db   (sites, observations, daily_panel, metadata)
  |
  |-- src/statistics.py, hypothesis_tests.py   inference layer
  |-- src/features.py -> src/ml_models.py      screening model (chronological holdout)
  |-- src/regression.py                        continuous O3_max estimation (OLS + inference)
  |-- src/clustering.py, association_rules.py  unsupervised + co-occurrence
  |-- src/nlp_module.py                        SECONDARY corpus (Federal Register)
  |-- src/forecasting.py                       weekly PM2.5, chronological holdout
  v
notebooks/ (narrative)  outputs/ (metrics)  dashboard/app.py (Streamlit + Plotly)
```

## ETL methodology

Extract reads the national daily CSVs; transform filters to the South Coast
counties, applies the empirically verified per-parameter rules
(docs/DATA_DICTIONARY.md), resolves co-located instruments (POC) with a
deterministic preference (reference duration > Observation Percent > lowest
POC), and standardizes units with documented conversions. Validate produces
a quality report (missingness, negatives, duplicates, site coverage,
quantiles). Load writes processed CSVs; src/db.py builds an idempotent SQLite
mirror. No step silently modifies values; every rule is traceable to a check
performed on the raw files.

## Statistical methodology

Descriptive statistics with quartiles/IQR and skew/kurtosis; normality
assessed by Jarque-Bera plus skew/kurtosis before and after a log transform.
Z-score screening (|z| > 3) flags daily spikes. Confidence intervals for
means use the T-distribution (sigma estimated) with Z-intervals reported for
comparison. Hypothesis tests pre-specify H0/H1, check assumptions, run
distribution-free counterparts (Mann-Whitney, Kruskal-Wallis), and report
Cohen's d / eta-squared / Cramer's V alongside p-values. Interpretations
explicitly separate statistical from practical significance; autocorrelation
is acknowledged as inflating nominal significance.

## ML methodology

Task: screen whether the basin daily-max 8-h O3 exceeds 70 ppb. Chronological
split (2021-2024 train / 2025 test) mirrors deployment; no random splits of a
time series. Three models (majority baseline, Gaussian NB in a scaling
Pipeline, Random Forest with class_weight='balanced_subsample', seed=42).
Primary metric PR-AUC given 11.8% prevalence; recall/precision trade-offs and
confusion matrices reported. Baseline vs engineered feature sets compared
identically; no tuning on the test year. Feature importance is descriptive of
the model, not causal.

## Unsupervised methodology

Days described by standardized (PM2.5, O3, NO2, TEMP, WIND) profiles. PCA
quantifies structure (explained variance, loadings). Ward hierarchical
clustering with silhouette evaluated over k=2..8; the chosen k is reported
with its score and the dendrogram. Clusters are interpreted post-hoc as
patterns in this dataset, not objective categories.

## Association methodology

Condition bins from documented quantiles (>=75th pct high; <=25th pct low
wind) on the complete-data subset; the outcome (O3 exceedance) enters only
as an item. 1- and 2-item antecedents with >=5% support; support, confidence,
lift reported; interpretation restricted to co-occurrence language.

## NLP methodology (secondary corpus)

Federal Register EPA documents (Rule vs Notice), cleaned/tokenized with
stopword removal; word frequencies and WordCloud for corpus description;
TF-IDF (train-fold fit only) + Multinomial NB for classification with
macro precision/recall/F1 and a confusion matrix. Clearly labeled as a
secondary dataset, separate from all sensor analysis.

## Forecasting methodology

Genuine daily AQS time index aggregated to weekly basin-mean PM2.5 (weeks
with <4 valid days excluded, then time-interpolated; counts reported).
Chronological holdout (train <=2024, test = 2025) with multi-step forecasts
from Holt (damped) and Holt-Winters (additive, 52-week season) against a
seasonal-naive baseline; MAE/RMSE/MAPE on the holdout. Presented as a
statistical syllabus demonstration, not an operational forecast.
