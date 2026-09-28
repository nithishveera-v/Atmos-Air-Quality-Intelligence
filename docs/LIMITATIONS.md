# LIMITATIONS — Atmos

## Dataset limitations

* **Monitoring-network bias:** sites are not spatially representative; basin
  aggregates weight monitored locations, not population exposure. Near-road
  PM2.5 sites began reporting during the study window, so network composition
  shifts slightly across years.
* **Basin-mean dilution:** averaging 34-47 sites mutes local peaks; basin-max
  metrics partially compensate but depend on which sites operated that day.
* **Missingness:** per-site coverage is incomplete (the ETL report
  quantifies it); the >=2-sites-per-day rule prevents single-instrument days
  but cannot remove network-level gaps.
* **Five-year window:** short for trend detection; conclusions describe this
  period and basin, not long-term trajectories.
* **Exceptional events:** wildfire smoke days are included as measured (flagged
  via Event Type) and materially shape tails; excluding them would change
  results and is left as an explicit analysis choice, not a silent one.

## Statistical limitations

* Daily series are autocorrelated; nominal p-values overstate effective
  sample size (acknowledged per test; rankings unaffected).
* Observational data: every association (correlations, rules, feature
  importance) is non-causal.
* Multiple tests were run without correction; p-values near 0.05 (e.g.
  winter-vs-summer PM2.5) should be read with that in mind.

## Model limitations

* The screening model uses the monitoring-era feature set; the same-evening
  use case is stated explicitly, and lag-only features would be required for
  a morning-of forecast.
* Engineered features did not improve the 2025 holdout; the reported model
  set is deliberately untuned, so absolute numbers are not state-of-the-art.
* 2025 is a single holdout year; metric uncertainty across years is not
  quantified (a rolling-origin evaluation is a natural extension).

## NLP & forecasting limitations

* The NLP corpus is a secondary dataset (Federal Register documents); the
  Rule/Notice task is a demonstration of the syllabus module, and its
  vocabulary differs from operational incident text.
* Forecasting is statistical (exponential smoothing) with genuinely limited
  skill against wildfire-driven variability; it is not an operational
  air-quality forecast and does not use meteorological forecast inputs.

## Deployment limitations

* Everything is offline and historical: no live feeds, no alerting, and no
  claim of real-time monitoring anywhere in the project.
* Reproducibility requires re-downloading the raw AQS files (URLs in the
  README); Windows Defender/Application Control on the development machine
  intermittently delayed first import of new compiled wheels
  (scripts/warm_imports.py documents the workaround) - unrelated to project code.
