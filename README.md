# Atmos — Urban Air Quality & Health-Risk Intelligence

An end-to-end Data Science platform for historical air-quality analytics over
**US EPA AQS monitoring records** (California South Coast air basin,
2021–2025): data engineering → SQLite → statistics → hypothesis testing →
leakage-safe machine learning → unsupervised regime discovery → association
analysis → a clearly separated NLP module → forecasting → an interactive
**Streamlit + Plotly** dashboard.

> **What this is:** a reproducible, offline analytics study of public
> monitoring data.
> **What this is not:** a live monitoring system, an operational forecast
> service, or a source of causal claims.

---

## 1. Problem statement

Ground-level ozone and PM2.5 harm health; regulators need to know which
conditions produce unhealthy days and how predictable those days are. This
project builds a portfolio-grade analytics pipeline that answers:

1. Which multi-pollutant regimes characterize the basin?
2. Which conditions co-occur with days where **any monitor** exceeds the
   70 ppb 8-hour ozone NAAQS?
3. How well can those days be **screened one day ahead** without data leakage?
4. Can weekly PM2.5 be forecast usefully with statistical methods?

## 2. Datasets (all real, all public)

| Role | Dataset | Source | License | Size |
|---|---|---|---|---|
| **Primary** | EPA AQS daily summaries: PM2.5 (88101), O3 (44201), NO2 (42602), temperature, wind — 2021–2025 | https://aqs.epa.gov/aqsweb/airdata/ | US Gov public domain | 25 files, ~3.3 GB, 8.4M national rows → 274k study rows, 1,826-day basin panel |
| **Secondary (NLP)** | Federal Register — EPA documents matching "air quality", 2021–2025 | https://www.federalregister.gov/developers/documentation/api/v1 | Public domain | 1,200 documents (745 Rule / 455 Notice) |

No synthetic data is used anywhere in the project. There is no generated
dataset and no fabricated value; every number in this README comes from
executed code in `src/` and `notebooks/`.

**Study area:** South Coast counties — Los Angeles (06037), Orange (06059),
Riverside (06065), San Bernardino (06071). Key raw fields and the full ETL
rule set are documented in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## 3. Architecture

```
data/raw (EPA AQS national daily CSVs, downloaded once)
  |  src/etl.py        EXTRACT -> TRANSFORM -> VALIDATE -> LOAD
  v
data/processed/        site-day CSVs + basin daily panel + quality report
  |  src/db.py         SQLite schema + load + SQL demonstrations
  v
database/atmos_airquality.db    (sites, observations, daily_panel, metadata)
  |
  |-- src/statistics.py, hypothesis_tests.py      inference layer
  |-- src/features.py -> src/ml_models.py         screening model (2025 holdout)
  |-- src/clustering.py, association_rules.py     regimes + co-occurrence
  |-- src/nlp_module.py                           SECONDARY corpus (NLP)
  |-- src/forecasting.py                          weekly PM2.5 forecasting
  v
notebooks/ (narrative)      outputs/ (metrics, tables)      dashboard/app.py
```

## 4. Installation

```bash
# Python 3.12+ recommended (project developed on 3.13, Windows)
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

## 5. Usage (reproducible pipeline)

```bash
# 1) raw data (~3.3 GB; re-downloadable, git-ignored)
./.venv/Scripts/python scripts/download_data.py

# 2) ETL: extract -> transform -> validate -> processed CSVs + quality report
./.venv/Scripts/python -m src.etl

# 3) SQLite database + SQL demonstrations -> outputs/tables/
./.venv/Scripts/python -m src.db

# 4) classification artifacts -> outputs/metrics/
./.venv/Scripts/python -m src.ml_models

# 5) linear-regression module (continuous O3_max estimation) -> outputs/metrics/
./.venv/Scripts/python -m src.regression

# 6) tests (leakage guards, split integrity, reproducibility, sanity)
./.venv/Scripts/python -m pytest tests/ -q

# 7) regenerate + execute the six demonstration notebooks
./.venv/Scripts/python scripts/make_notebooks.py
./.venv/Scripts/python -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# 8) dashboard
./.venv/Scripts/python -m streamlit run dashboard/app.py
```

## 6. Methodology (summary)

* **ETL** — every filter rule was verified against the raw files before being
  encoded (e.g. PM2.5 `'1 HOUR'` vs `'24-HR BLK AVG'` agree to 0.24 µg/m³ on
  shared site-days; `'24 HOUR'` rows differ by 1.56 µg/m³ and are excluded).
  Co-located instruments resolved deterministically; units converted with
  documented factors; days with <2 reporting sites set to NaN.
* **Statistics** — T-based confidence intervals (Z reported for comparison),
  Jarque–Bera normality analysis with log transform, Z-score spike screening,
  pre-specified tests (Welch t, Mann–Whitney, ANOVA/Kruskal–Wallis, χ²) with
  effect sizes and explicit statistical-vs-practical interpretation.
* **ML** — target = basin daily-**max** O3 > 70 ppb (regulatory "any monitor"
  definition); **chronological** split (2021–24 train / 2025 test); majority
  baseline, Gaussian NB, Random Forest; PR-AUC primary at 11.8% prevalence;
  **no O3 variable may be a feature** (enforced by a unit test); scalers and
  vectorizers fit on train folds only; no tuning on the test year.
* **Regression** — continuous counterpart of the screening task: same-day
  basin daily-max O3 (ppb) estimated with OLS (statsmodels inference: per-SD
  coefficients, p-values, Durbin–Watson on residuals) against a mean
  baseline, Ridge (alpha fixed a priori, never tuned on the test year) and a
  random-forest reference; identical chronological split and leakage rules.
* **Unsupervised** — standardized profiles, PCA, Ward clustering, k chosen by
  silhouette over k=2..8 (not hard-coded), clusters reported as patterns.
* **Association rules** — documented quantile bins; support/confidence/lift;
  co-occurrence language only.
* **NLP (secondary)** — real Federal Register corpus; Rule-vs-Notice with
  TF-IDF + Multinomial NB; WordCloud; train-fold-only vectorization.
* **Forecasting** — genuine weekly series, chronological holdout, Holt /
  Holt-Winters vs seasonal-naive, MAE/RMSE/MAPE.

Full details: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## 7. Results (all from executed code, 2025 holdout unless noted)

**Screening model (task: basin-max O3 > 70 ppb; prevalence 11.8%)**

| Model | Features | PR-AUC | ROC-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Majority baseline | – | 0.118 | – | 0.00 | – |
| Gaussian NB | baseline (6) | **0.639** | 0.937 | 0.953 | 0.376 |
| Random Forest | baseline (6) | 0.585 | 0.929 | 0.628 | 0.500 |
| Gaussian NB | engineered (14) | 0.597 | 0.926 | 0.977 | 0.336 |
| Random Forest | engineered (14) | 0.516 | 0.920 | 0.465 | 0.500 |

Honest finding: **engineered features did not improve the holdout** — reported
as found, not tuned away.

**Hypothesis tests** — weekday NO2 > weekend (13.7 vs 10.9 ppb, p≈9e−26,
d=0.54, medium); winter-vs-summer PM2.5 statistically significant but
practically small (p=0.029, d=−0.15); season explains only ~3% of PM2.5
variance (η²=0.034); exceedance×season **not** significant (p=0.39, V=0.04)
— a genuine null finding.

**Regimes & associations** — PCA PC1+PC2 ≈ 78% of variance; silhouette
selects k=2 (cool/stagnant/combustion-heavy vs warm/ventilated days).
`high_temp & high_pm25` days co-occur with O3 exceedance at **lift 4.5**;
`low_wind` days are *negatively* associated in this basin (cool-season
inversion days) — associations, not causation.

**Regression (OLS level estimation, 2025 holdout)** — R² **0.690**, RMSE
**6.71** ppb vs mean-baseline 12.05 (dummy R² ≈ 0); Ridge identical (0.690),
random forest 0.693 — the O3–meteorology relationship is largely linear.
Significant per-SD coefficients: TEMP **+3.29** ppb (p=0.010), WIND **−1.32**
(p=2e−4), weekend **−1.33** (p=0.009); NO2 not significant once temperature
and seasonality are included. Durbin–Watson 0.92 → autocorrelated residuals,
so OLS p-values are approximate (documented in the artifact and dashboard).

**NLP** — 95.3% accuracy (F1 0.949) Rule-vs-Notice on 300 held-out documents.

**Forecasting** — Holt-Winters RMSE **3.33** vs seasonal-naive 4.17 µg/m³ on
the 2025 weekly holdout; bounded skill, limited by wildfire episodes.

See `outputs/metrics/`, the executed notebooks, and the dashboard for full
numbers; `docs/DATA_LEAKAGE_AUDIT.md` documents the leakage review behind
every figure.

## 8. Dashboard

`streamlit run dashboard/app.py` — 12 sections: Executive Overview · Data
Quality · Exploratory Analysis · Statistical Intelligence (CIs, tests,
z-scores) · Risk Screening & Model Performance (with an interactive
what-if screen) · Feature Importance · Regression (OLS level estimation
with full inference) · Regimes (PCA + clustering) ·
Association Analysis · Text Intelligence (NLP + WordCloud) · Forecasting ·
Key Findings. Interactive **Plotly** throughout; parameter/year filters;
explicitly labeled as historical analytics.

## 9. Project structure

```
data/{raw,processed,external}   database/               docs/ (5 documents)
notebooks/ (6, executed)        src/ (13 modules)       outputs/{metrics,tables,plots}
scripts/ (download, NLP fetch, notebook generator, import warm-up)
tests/ (pytest)                 dashboard/app.py        README.md
```

## 10. Reproducibility

* Fixed seed (42) for every model; unit test asserts bit-identical RF outputs.
* Data is re-downloadable from the URLs above (versioned by year; AQS files
  are periodically revised by EPA — re-running ETL reflects the current files).
* Package versions: `requirements.txt`; developed on Python 3.13 / Windows.
* ETL → DB → ML → tests run top-to-bottom in minutes (after download).
* Windows Application Control may briefly block first imports of freshly
  installed wheels; `scripts/warm_imports.py` documents the retry workaround.

## 11. Limitations & data integrity

Documented honestly in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md):
network bias, basin-mean dilution, autocorrelation-inflated significance,
single holdout year, non-causal inference throughout, bounded forecasting
skill, and the clear separation of the NLP secondary corpus. The project
contains no fabricated data, no synthetic results presented as real, and no
claim of live monitoring.
