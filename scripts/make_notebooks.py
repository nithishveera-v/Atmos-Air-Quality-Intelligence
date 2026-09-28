"""Generate the Atmos demonstration notebooks.

Notebooks are the narrative layer: they explain the analytical process and
call the reusable logic in src/ (no duplicated implementations). The
generated files are executed afterwards with nbconvert to guarantee they
run end-to-end.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = """\
import sys, pathlib
root = pathlib.Path().resolve()
while root.parent != root and not (root / "src").exists():
    root = root.parent
sys.path.insert(0, str(root))
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
import matplotlib.pyplot as plt
pd.set_option("display.width", 120)
# make plotly render statically in exported notebooks
import plotly.io as pio
pio.renderers.default = "notebook"
panel = pd.read_csv(root / "data" / "processed" / "daily_panel.csv", parse_dates=["date"])
panel["date"] = panel["date"].dt.tz_localize(None)
"""


def nb(title: str, intro: str, cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    book = nbf.v4.new_notebook()
    book.cells = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"),
        nbf.v4.new_code_cell(SETUP),
    ]
    for kind, src in cells:
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                          else nbf.v4.new_code_cell(src))
    return book


NOTEBOOKS: dict[str, nbf.NotebookNode] = {}

NOTEBOOKS["01_data_ingestion_and_quality.ipynb"] = nb(
    "01 - Data Ingestion & Data Quality",
    """**Dataset:** US EPA AQS daily summaries (public domain),
https://aqs.epa.gov/aqsweb/airdata/ - files `daily_88101_YYYY.csv` (PM2.5),
`daily_44201_YYYY.csv` (O3), `daily_42602_YYYY.csv` (NO2), `daily_TEMP_YYYY.csv`,
`daily_WIND_YYYY.csv`, 2021-2025. Study area: California South Coast counties
(Los Angeles 037, Orange 059, Riverside 065, San Bernardino 071).

The full ETL (extract -> transform -> validate -> load) lives in `src/etl.py`;
this notebook documents the decisions and inspects the outputs. To re-run the
ETL from raw data: `python -m src.etl`.""",
    [
        ("md", """## ETL decisions encoded in src/etl.py

* **PM2.5 sample durations:** `'1 HOUR'` rows (daily means of ~24 hourly
  samples from continuous monitors) were verified against the filter-based
  reference `'24-HR BLK AVG'` on shared site-days: mean absolute difference
  0.24 ug/m3 -> both retained; `'24 HOUR'` rows differ systematically
  (1.56 ug/m3) -> excluded as a distinct monitoring objective.
* **Co-located instruments (POC):** one record per site/day; the
  reference-method duration wins, then higher `Observation Percent`,
  then lowest POC (deterministic).
* **Units:** PM2.5 ug/m3; O3 ppm -> ppb (x1000); NO2 ppb; TEMP degF; WIND knots
  (wind direction rows excluded).
* **Basin daily panel:** mean across reporting sites; days with < 2 reporting
  sites are set to NaN; basin daily MAX also kept for O3/PM2.5 because a NAAQS
  exceedance day is defined by ANY monitor exceeding."""),
        ("code", """\
from src.config import DATA_PROCESSED
panel = pd.read_csv(DATA_PROCESSED / "daily_panel.csv", parse_dates=["date"])
quality = pd.read_csv(DATA_PROCESSED / "data_quality_report.csv")
quality"""),
        ("code", """\
print("panel days:", len(panel), "| range:", panel["date"].min().date(), "->", panel["date"].max().date())
panel[["PM25","O3","NO2","TEMP","WIND"]].describe().round(2)"""),
        ("code", """\
import plotly.express as px
fig = px.histogram(panel, x="n_O3_sites", nbins=30, title="O3 monitors reporting per day")
fig.show()"""),
        ("md", """## Findings & caveats

* Missing values in the basin panel are negligible (<0.2% for PM2.5, 0 for the
  others after the >=2-site rule); per-site missingness is larger and is
  documented in the quality report.
* The PM2.5 network (20 reference-grade sites) is sparser than the O3 network
  (44 sites); near-road PM2.5 sites run only since 2022, shaping coverage."""),
    ])

NOTEBOOKS["02_eda_and_statistics.ipynb"] = nb(
    "02 - Exploratory Analysis & Statistics",
    """Descriptive statistics, normal-distribution analysis, Z-score anomaly
screening, T vs Z confidence intervals, and correlation analysis -
implemented in `src/statistics.py` and applied to the basin daily panel.""",
    [
        ("code", """\
from src.statistics import run_statistics
res = run_statistics(panel)
pd.DataFrame(res["descriptive"]).T.round(2)"""),
        ("code", """\
n = res["pm25_normality"]
print(f"PM2.5 skew raw={n['skew_raw']} vs log={n['skew_log']}; "
      f"Jarque-Bera p(raw)={n['jb_p_raw']:.1e}, p(log)={n['jb_p_log']:.1e}")
print(n["interpretation"])
fig = px.histogram(panel, x="PM25", nbins=60, marginal="box",
                   title="PM2.5 daily basin mean - right-skewed")
fig.show()"""),
        ("code", """\
res["seasonal_ci_pm25"]"""),
        ("code", """\
import plotly.graph_objects as go
ci = res["seasonal_ci_pm25"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ci["season"], y=ci["mean"], mode="markers", name="mean"))
fig.add_trace(go.Scatter(x=ci["season"], y=ci["ci_high_t"], mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=ci["season"], y=ci["ci_low_t"], mode="lines", line=dict(width=0),
                         fill="tonexty", fillcolor="rgba(42,157,143,0.25)", name="95% CI (t)"))
fig.update_layout(title="Mean PM2.5 by season with T-based 95% CI", yaxis_title="ug/m3")
fig.show()
print("T intervals are slightly wider than Z intervals because sigma is estimated:",
      ci[["season", "ci_width_t"]].to_string(index=False))"""),
        ("code", """\
res["correlations"]"""),
        ("code", """\
za = panel.set_index("date")["PM25"]
z = (za - za.mean()) / za.std(ddof=1)
anom = z[z.abs() > 3]
fig = go.Figure(go.Scatter(x=z.index, y=z.values, mode="lines", name="z"))
fig.add_trace(go.Scatter(x=anom.index, y=anom.values, mode="markers", marker=dict(color="red"),
                         name="|z|>3"))
fig.update_layout(title=f"PM2.5 z-scores ({len(anom)} anomaly days)", yaxis_title="z")
fig.show()
print(res["pm25_anomalies_z3"]["top_spikes"][:3])"""),
        ("md", """## Findings & caveats

* Raw PM2.5 is strongly right-skewed; the log transform reduces skew but the
  series is a source mixture, so exact normality is neither expected nor found.
* O3-TEMP (+0.69) and NO2-WIND (-0.66) correlations match atmospheric physics;
  correlation is association, not causation.
* Z-score screening flags wildfire-episode days; global z-scores are
  conservative because fire days inflate the sample SD."""),
    ])

NOTEBOOKS["03_hypothesis_testing.ipynb"] = nb(
    "03 - Hypothesis Testing",
    """Four pre-specified tests with explicit H0/H1, assumptions, effect sizes,
and practical-significance interpretation - implemented in
`src/hypothesis_tests.py`.""",
    [
        ("code", """\
from src.hypothesis_tests import run_hypothesis_tests
tests = run_hypothesis_tests(panel)
for key in ["winter_vs_summer_pm25", "weekday_vs_weekend_no2",
            "anova_seasons_pm25", "chi2_exceedance_season"]:
    r = tests[key]
    print("=" * 70)
    print("QUESTION:", r["question"])
    print("H0:", r["h0"])
    print("H1:", r["h1"])
    print("TEST:", r["test"], "| statistic =", r["statistic"], "| p =", f"{r['p_value']:.3e}")
    for k, v in r.items():
        if k.startswith("effect_size"):
            print(k, "=", v)
    print("INTERPRETATION:", tests["interpretation"][key])"""),
        ("md", """## Statistical vs practical significance

The winter-summer PM2.5 difference is statistically significant at
alpha=0.05 but Cohen's d is small: with 450+ days per group, tiny
differences become detectable. The chi-square test for exceedance x
season is NOT significant - an honest null finding consistent with
exceedances clustering in warm episodes across multiple seasons.

Limitation: daily air quality is autocorrelated, so effective sample
sizes are smaller than nominal n; this can inflate significance (noted
in every test's assumptions)."""),
    ])

NOTEBOOKS["04_features_and_machine_learning.ipynb"] = nb(
    "04 - Feature Engineering & Machine Learning",
    """Target: `o3_exceed` = basin daily max 8-h O3 > 70 ppb (current NAAQS).
Chronological split: train 2021-2024, test 2025. Baseline vs engineered
feature sets compared honestly - implemented in `src/features.py` and
`src/ml_models.py`.""",
    [
        ("md", """**Leakage policy:** no O3 variable (mean or max, same-day or lagged) may be
a feature - the target is built from O3_max, so basin O3 is excluded by
construction. Rolling/lag features use shift(1) (past only). The NB scaler
is fitted inside a Pipeline on the training fold only."""),
        ("code", """\
from src.features import BASELINE_FEATURES, ENGINEERED_FEATURES, ALL_FEATURES, load_modeling_frame
print("baseline:", BASELINE_FEATURES)
print("engineered:", ENGINEERED_FEATURES)"""),
        ("code", """\
from src.ml_models import run_model_comparison
dfm = load_modeling_frame()
results = run_model_comparison(dfm, {"baseline": BASELINE_FEATURES, "engineered": ALL_FEATURES})"""),
        ("code", """\
rows = [{ "set": k.split("/")[0], "model": k.split("/")[1],
          **{m: v.get(m) for m in ["pr_auc", "roc_auc", "recall", "precision", "f1"]}}
        for k, v in results.items() if "/" in k]
pd.DataFrame(rows).sort_values("pr_auc", ascending=False)"""),
        ("code", """\
from src.ml_models import feature_importance_rf
imp = feature_importance_rf(dfm, ALL_FEATURES)
fig = px.bar(imp.head(12), x="importance", y="feature", orientation="h",
             title="Random Forest feature importance (contribution to predictions, not causation)")
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.show()"""),
        ("md", """## Regression view: estimating the O3 *level* (continuous target)

Classification above predicts the *category* (exceed / not). The regression
module (`src/regression.py`) estimates the *level*: basin daily-max O3 (ppb)
from the same features - the continuous counterpart (Linear Regression
syllabus topic). Same leakage policy (no O3 variables as features), same
chronological split; OLS coefficients come with statsmodels inference
(per-SD effects; Durbin-Watson flags residual autocorrelation)."""),
        ("code", """\
from src.regression import run_regression
reg = run_regression(dfm)
pd.DataFrame(reg["metrics"]).T[["mae", "rmse", "rmse_train", "r2"]]"""),
        ("code", """\
coef = pd.DataFrame(reg["ols_inference"])
sig = coef[coef["p_value"] < 0.05].copy()
print("Significant OLS coefficients (per-SD effects, train fold):")
print(sig[["feature", "coef", "p_value"]].to_string(index=False))
print()
print(f"Durbin-Watson (train residuals) = {reg['durbin_watson_train_residuals']} "
      "-> autocorrelated residuals; p-values are approximate.")"""),
        ("code", """\
fig = go.Figure()
fig.add_trace(go.Scatter(x=reg["test_actual"], y=reg["predictions"]["ols"],
                         mode="markers", opacity=0.5, name="OLS predicted"))
lims = [30, 130]
fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash"),
                         showlegend=False))
fig.update_layout(title="Basin daily-max O3: observed vs OLS estimate (2025 holdout)",
                  xaxis_title="observed (ppb)", yaxis_title="predicted (ppb)")
fig.show()"""),
        ("md", """## Findings & caveats

* With 11.8% prevalence, the majority baseline scores PR-AUC 0.118; NB
  reaches 0.639 (high recall / low precision) and RF 0.585 on the engineered
  set's baseline-feature counterpart.
* **Engineered features did not improve the 2025 holdout** (RF PR-AUC 0.585
  -> 0.516). This is reported as found: same-day interactions add little
  beyond temperature/NO2/calendar for next-day screening; no tuning was
  applied to rescue them.
* **Regression:** OLS R-squared ~0.69 (RMSE ~6.7 ppb) vs a mean baseline of
  ~12.1; temperature (+) and wind (-) carry the signal; NO2 is not
  significant once temperature and seasonality are included. The linear
  model captures nearly all of the achievable accuracy (RF ~6.7). This is
  an *estimation* task on same-day network data - association, not
  causation, and not a stand-alone forecast.
* Metrics come from the chronological 2025 holdout only - no random
  splitting of a time series."""),
    ])

NOTEBOOKS["05_unsupervised_and_association.ipynb"] = nb(
    "05 - PCA, Hierarchical Clustering & Association Rules",
    """Do days fall into recognizable multi-pollutant regimes? Ward hierarchical
clustering on standardized profiles with silhouette-based k selection; PCA
for structure visualization; condition-bin association rules (support /
confidence / lift) - `src/clustering.py`, `src/association_rules.py`.""",
    [
        ("code", """\
from src.features import build_modeling_frame
from src.clustering import run_clustering
mf = build_modeling_frame(panel)
clus = run_clustering(mf)
print("EVR:", [round(x, 3) for x in clus["pca"]["explained_variance_ratio"]])
print("silhouette:", clus["hierarchical"]["silhouette_by_k"])
print("chosen k =", clus["hierarchical"]["best_k"])"""),
        ("code", """\
Z = clus["pca"]["scores"]
fig = px.scatter(x=Z[:, 0], y=Z[:, 1], color=clus["labels"].astype(str), opacity=0.6,
                 labels={"x": "PC1", "y": "PC2"}, title="Days in PC space by cluster")
fig.show()
clus["cluster_profile"]"""),
        ("code", """\
from scipy.cluster.hierarchy import dendrogram
fig, ax = plt.subplots(figsize=(11, 4))
dendrogram(clus["hierarchical"]["linkage_matrix"], truncate_mode="lastp", p=30, ax=ax, no_labels=True)
ax.set_title("Ward dendrogram (truncated to last 30 merges)")
plt.show()"""),
        ("code", """\
from src.association_rules import make_bins, mine_rules, interpret_rules
rules = mine_rules(make_bins(mf))
rules"""),
        ("code", "print(interpret_rules(rules))"),
        ("md", """## Findings & caveats

* Two regimes (silhouette-selected k=2): cool, stagnant, combustion-influenced
  days vs warm, ventilated days with higher O3. Silhouette values are modest
  (~0.33) - honest labeling: regimes overlap; clusters are patterns in this
  dataset, not objective categories.
* Association rules are CO-OCCURRENCE only: high-temp (and high-temp &
  high-PM2.5) days co-occur with O3 exceedance at lift > 3; low-wind days are
  NEGATIVELY associated in this basin (cool-season inversion days). Nothing
  here implies causation."""),
    ])

NOTEBOOKS["06_nlp_and_forecasting.ipynb"] = nb(
    "06 - NLP (Secondary Corpus) & Forecasting",
    """Two clearly-separated syllabus modules: (1) NLP on a real secondary corpus -
1,200 Federal Register EPA documents (public domain) - Rule vs Notice
classification with Naive Bayes + WordCloud; (2) forecasting of weekly
basin PM2.5 with chronological holdout - `src/nlp_module.py`,
`src/forecasting.py`.""",
    [
        ("md", """**Why a secondary dataset?** The primary AQS sensor data has no legitimate
text field, and fabricating maintenance-style notes from labels would create
artificial text-label leakage. The Federal Register corpus is real, public
domain, and thematically coherent; it is NOT part of the sensor analysis."""),
        ("code", """\
from src.nlp_module import run_nlp, make_wordcloud
nlp = run_nlp()
r = nlp["results"]
print({k: v for k, v in r.items() if k not in ("word_freq_all_top10", "distinctive_note")})"""),
        ("code", """\
wc = make_wordcloud(nlp["tables"]["freq_all"])
fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(wc.to_array(), interpolation="bilinear")
ax.axis("off")
ax.set_title("Most frequent words (stopwords removed)")
plt.show()"""),
        ("code", """\
from src.forecasting import run_forecasting
fc = run_forecasting(panel)
fig = go.Figure()
fig.add_trace(go.Scatter(x=fc["train_dates"], y=fc["train_series"], name="train"))
fig.add_trace(go.Scatter(x=fc["test_dates"], y=fc["test_actual"], name="actual 2025"))
best = fc["best_by_rmse"]
fig.add_trace(go.Scatter(x=fc["test_dates"], y=fc["predictions"][best], name=f"forecast ({best})",
                         line=dict(dash="dash")))
fig.update_layout(title="Weekly PM2.5 - chronological holdout forecast", yaxis_title="ug/m3")
fig.show()
pd.DataFrame(fc["metrics"]).T"""),
        ("code", "print(fc['interpretation'])"),
        ("md", """## Findings & caveats

* NB distinguishes Rule vs Notice at ~95% accuracy from procedural vocabulary;
  labels are official metadata (no label leakage; TF-IDF fitted on train only).
* Holt-Winters beats the seasonal-naive baseline on the 2025 holdout
  (RMSE ~3.3 vs ~4.2 ug/m3) - real but bounded skill: wildfire episodes are
  unpredictable from history alone. No fake timestamps, no randomly split
  time series."""),
    ])

if __name__ == "__main__":
    import sys as _sys
    only = set(_sys.argv[1:])  # optional: regenerate a subset by filename
    for name, book in NOTEBOOKS.items():
        if only and name not in only and not any(a in name for a in only):
            continue
        path = OUT / name
        nbf.write(book, str(path))
        print("wrote", path.name)
