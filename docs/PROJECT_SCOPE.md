# PROJECT SCOPE — Atmos

## Problem statement

Ground-level ozone and fine particulate matter (PM2.5) harm human health, and
regulators must understand *which conditions* produce unhealthy days and
*how predictable* those days are. This project builds a reproducible,
software-only analytics platform over five years of EPA monitoring records
for California's South Coast air basin (Los Angeles, Orange, Riverside, San
Bernardino counties), answering:

1. What multi-pollutant conditions and regimes characterize the basin?
2. Which conditions co-occur with days on which any monitor exceeds the
   70 ppb 8-hour ozone standard?
3. How well can such exceedance days be screened one day ahead using
   admissible, non-leaking features?
4. Can weekly PM2.5 be forecast usefully with statistical methods?

## Objectives

* Honest, reproducible ETL from raw public data into SQLite.
* Correct statistical inference (effect sizes + significance, assumptions stated).
* A leakage-safe ML screening model with an explicit chronological holdout.
* Unsupervised regime analysis (PCA + hierarchical clustering, validated k).
* Association-rule mining interpreted strictly as co-occurrence.
* Clearly separated secondary modules: NLP (Federal Register corpus) and
  forecasting (genuine weekly time series).
* A Streamlit + Plotly decision-support dashboard over the finished analysis.

## In scope

South Coast basin, 2021-2025, parameters PM2.5 / O3 / NO2 / TEMP / WIND,
basin-daily and site-day granularity, the analyses above.

## Out of scope

* Real-time data ingestion, live alerting, or operational forecasting.
* Exposure/health-outcome modeling (no health records in scope).
* Causal claims of any kind.
* Chemical transport modeling or emission-inventory work.

## Stakeholders / users

Air-quality analysts, public-health planners, students of applied data
science, and reviewers assessing end-to-end DS engineering practice.

## Success criteria

The project is complete when every reported number is produced by executed
code, all tests pass, the dashboard runs locally, and the documented
methodology and limitations accurately describe what was done (see
docs/DATA_LEAKAGE_AUDIT.md and docs/LIMITATIONS.md).
