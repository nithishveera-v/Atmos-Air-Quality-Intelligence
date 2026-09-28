"""Atmos - Urban Air Quality & Health-Risk Intelligence dashboard.

HISTORICAL ANALYTICS ONLY. This is an offline analysis of EPA AQS
monitoring records (2021-2025, California South Coast). It is NOT a
live monitoring system and displays no real-time data.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_PATH, METRICS_DIR  # noqa: E402
from src.features import build_modeling_frame  # noqa: E402

st.set_page_config(page_title="Atmos | Air Quality Intelligence",
                   page_icon="wind", layout="wide")


# --------------------------------------------------------------------- #
# cached data loading
# --------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading analysis data...")
def load_all():
    panel = pd.read_csv(PROJECT_ROOT / "data/processed/daily_panel.csv",
                        parse_dates=["date"])
    q = pd.read_csv(PROJECT_ROOT / "data/processed/data_quality_report.csv")
    frames = {p: pd.read_csv(PROJECT_ROOT / f"data/processed/{p.lower()}_southcoast.csv",
                             parse_dates=["date"])
              for p in ["PM25", "O3", "NO2"]}
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    sites = pd.read_sql("SELECT * FROM sites", con)
    con.close()
    modeling = build_modeling_frame(panel)
    return panel, q, frames, sites, modeling


@st.cache_data(show_spinner="Running models...")
def run_analysis():
    from src.statistics import run_statistics
    from src.hypothesis_tests import run_hypothesis_tests
    from src.clustering import run_clustering
    from src.association_rules import make_bins, mine_rules
    from src.forecasting import run_forecasting
    from src.nlp_module import run_nlp
    from src.regression import run_regression

    panel, _, _, _, modeling = load_all()
    stats = run_statistics(panel)
    tests = run_hypothesis_tests(panel)
    clus = run_clustering(modeling)
    rules = mine_rules(make_bins(modeling))
    fc = run_forecasting(panel)
    nlp = run_nlp()
    reg = run_regression(modeling)
    return stats, tests, clus, rules, fc, nlp, reg


panel, quality, site_frames, sites, modeling = load_all()
stats, tests, clus, rules, fc, nlp, reg = run_analysis()

PARAM_META = {
    "PM25": {"label": "PM2.5", "unit": "ug/m3"},
    "O3": {"label": "Ozone", "unit": "ppb"},
    "NO2": {"label": "NO2", "unit": "ppb"},
    "TEMP": {"label": "Temperature", "unit": "degF"},
    "WIND": {"label": "Wind speed", "unit": "knots"},
}

# --------------------------------------------------------------------- #
SIDEBAR = st.sidebar
SIDEBAR.title("Atmos")
SIDEBAR.caption("Urban Air Quality & Health-Risk Intelligence - historical "
                "analytics on EPA AQS monitoring data (South Coast, 2021-2025). "
                "Not a live monitoring system.")
page = SIDEBAR.radio("Section", [
    "1. Executive Overview",
    "2. Data Quality",
    "3. Exploratory Analysis",
    "4. Statistical Intelligence",
    "5. Risk Screening & Model Performance",
    "6. Feature Importance",
    "7. Regression: O3 Level Estimation",
    "8. Regimes: PCA & Clustering",
    "9. Association Analysis",
    "10. Text Intelligence (NLP)",
    "11. Forecasting",
    "12. Key Findings",
])

# filters ---------------------------------------------------------------
PARAM = SIDEBAR.selectbox("Parameter", list(PARAM_META), format_func=lambda p: PARAM_META[p]["label"])
YEAR_MIN, YEAR_MAX = int(panel["date"].dt.year.min()), int(panel["date"].dt.year.max())
YEARS_SEL = SIDEBAR.slider("Years", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
d0 = pd.Timestamp(f"{YEARS_SEL[0]}-01-01")
d1 = pd.Timestamp(f"{YEARS_SEL[1]}-12-31")
panel_f = panel[(panel["date"] >= d0) & (panel["date"] <= d1)]

PRIMARY = "wind"
ACCENT = "wind"  # colors resolved via continuous scales below

# --------------------------------------------------------------------- #
if page.startswith("1."):
    st.title("Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days analyzed", f"{len(panel_f):,}")
    c2.metric(f"{PARAM_META[PARAM]['label']} mean",
              f"{panel_f[PARAM].mean():.1f} {PARAM_META[PARAM]['unit']}")
    c3.metric("O3 exceedance days (max > 70 ppb)",
              f"{int((panel_f['O3_max'] > 70).sum()):,}")
    c4.metric("Monitors in study", f"{len(sites):,}")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        for p in ["PM25", "O3", "NO2"]:
            s = panel_f.set_index("date")[p].rolling(30, min_periods=10).mean()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=PARAM_META[p]["label"],
                                     opacity=0.8))
        fig.update_layout(title="30-day rolling means (basin average)",
                          yaxis_title="concentration", height=420,
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width='stretch')
    with col2:
        ex = panel_f.set_index("date")["PM25"].resample("ME").mean()
        fig = px.bar(x=ex.index.strftime("%Y-%m"), y=ex.values,
                     labels={"x": "month", "y": "ug/m3"},
                     title="Monthly mean PM2.5", color=ex.values,
                     color_continuous_scale="Viridis")
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, width='stretch')

    st.info("Scope: California South Coast (Los Angeles, Orange, Riverside, "
            "San Bernardino counties). Data: US EPA AQS daily summaries. "
            "All values are historical monitoring records, not forecasts or live feeds.")

elif page.startswith("2."):
    st.title("Data Quality")
    st.caption("Produced by the ETL validation step (src/etl.py::quality_report).")
    st.dataframe(quality, width='stretch', hide_index=True)
    st.subheader("Per-day monitor coverage")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(panel_f, x="n_PM25_sites", nbins=30,
                           title="PM2.5 monitors reporting per day")
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.histogram(panel_f, x="n_O3_sites", nbins=30,
                           title="O3 monitors reporting per day")
        st.plotly_chart(fig, width='stretch')
    st.subheader("Missing values in the daily panel")
    miss = panel_f[[p for p in PARAM_META]].isna().mean().mul(100).round(2)
    st.bar_chart(miss.rename("percent missing"))

elif page.startswith("3."):
    st.title("Exploratory Analysis")
    p = PARAM
    unit = PARAM_META[p]["unit"]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(panel_f, x=p, nbins=60, title=f"{PARAM_META[p]['label']} distribution ({unit})",
                           marginal="box", color_discrete_sequence=["#2a9d8f"])
        st.plotly_chart(fig, width='stretch')
    with c2:
        monthly = panel_f.set_index("date")[p].resample("ME").mean()
        fig = px.line(x=monthly.index, y=monthly.values, labels={"x": "month", "y": unit},
                      title=f"{PARAM_META[p]['label']} monthly mean")
        st.plotly_chart(fig, width='stretch')
    st.subheader("Pairwise relationships")
    vars_sel = ["PM25", "O3", "NO2", "TEMP", "WIND"]
    fig = px.scatter_matrix(panel_f.sample(min(800, len(panel_f)), random_state=42),
                            dimensions=vars_sel, height=700)
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig, width='stretch')
    st.subheader("Highest days on record (basin mean)")
    top = panel_f.nlargest(10, p)[["date", p, "O3_max", "PM25_max"]]
    top[p] = top[p].round(1)
    st.dataframe(top, width='stretch', hide_index=True)

elif page.startswith("4."):
    st.title("Statistical Intelligence")
    st.subheader("Seasonal confidence intervals - PM2.5 (T vs Z distribution)")
    ci = stats["seasonal_ci_pm25"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ci["season"], y=ci["mean"], mode="markers",
                             marker=dict(size=12, color="#264653"), name="mean"))
    fig.add_trace(go.Scatter(x=ci["season"], y=ci["ci_high_t"], mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=ci["season"], y=ci["ci_low_t"],
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,157,143,0.25)", name="95% CI (t)"))
    fig.update_layout(yaxis_title="PM2.5 (ug/m3)", height=380,
                      title="Mean PM2.5 by season with 95% confidence intervals")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(ci, width='stretch', hide_index=True)
    st.caption("T-intervals (used) are slightly wider than Z-intervals (comparison "
               "only) because sigma is estimated from the sample.")

    st.subheader("Hypothesis tests")
    for key in ["winter_vs_summer_pm25", "weekday_vs_weekend_no2",
                "anova_seasons_pm25", "chi2_exceedance_season"]:
        r = tests[key]
        with st.expander(f"{r['question']}"):
            st.markdown(f"**H0:** {r['h0']}  \n**H1:** {r['h1']}")
            st.markdown(f"**Test:** {r['test']}")
            mcol = st.columns(4)
            mcol[0].metric("statistic", r["statistic"])
            mcol[1].metric("p-value", f"{r['p_value']:.2e}")
            eff = {k: v for k, v in r.items() if k.startswith("effect")}
            mcol[2].metric(list(eff)[0].replace("effect_size_", ""), list(eff.values())[0])
            mcol[3].metric("alpha", r["alpha"])
            st.markdown(f"**Interpretation:** {tests['interpretation'][key]}")
            if "assumptions" in r:
                st.caption("Assumptions: " + r["assumptions"])

    st.subheader("Z-score anomaly screening (|z| > 3)")
    s = panel_f.set_index("date")["PM25"]
    mu, sd = s.mean(), s.std(ddof=1)
    z = (s - mu) / sd
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z.index, y=z.values, mode="lines", name="z"))
    thr = 3.0
    anom = z[z.abs() > thr]
    fig.add_hline(y=thr, line_dash="dash", annotation_text="+3")
    fig.add_hline(y=-thr, line_dash="dash")
    fig.add_trace(go.Scatter(x=anom.index, y=anom.values, mode="markers",
                             marker=dict(color="red", size=8), name="anomaly"))
    fig.update_layout(height=380, title=f"PM2.5 daily z-scores ({len(anom)} flagged days)")
    st.plotly_chart(fig, width='stretch')

elif page.startswith("5."):
    st.title("Risk Screening & Model Performance")
    st.caption("Task: will tomorrow's basin-max 8-hour O3 exceed the 70 ppb NAAQS? "
               "Chronological split: trained on 2021-2024, evaluated on all of 2025.")
    metrics_path = METRICS_DIR / "model_comparison.json"
    mc = json.loads(metrics_path.read_text())
    rows = []
    for k, v in mc.items():
        if "/" in k:
            rows.append({"feature_set": k.split("/")[0], "model": k.split("/")[1], **{
                kk: v.get(kk) for kk in ["pr_auc", "roc_auc", "recall", "precision", "f1"]}})
    mdf = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    st.dataframe(mdf, width='stretch', hide_index=True)
    st.caption("PR-AUC is the primary metric (prevalence 11.8% in the test year); "
               "the majority-class floor is 0.118. Accuracy is deliberately not "
               "relied upon for this imbalanced problem.")

    c1, c2 = st.columns(2)
    with c1:
        nb = mc["baseline/naive_bayes"]["confusion"]
        fig = px.imshow([[nb["tn"], nb["fp"]], [nb["fn"], nb["tp"]]],
                        x=["pred: no", "pred: exceed"], y=["actual: no", "actual: exceed"],
                        text_auto=True, color_continuous_scale="Blues",
                        title="Naive Bayes - confusion (2025 holdout)")
        st.plotly_chart(fig, width='stretch')
    with c2:
        rf = mc["baseline/random_forest"]["confusion"]
        fig = px.imshow([[rf["tn"], rf["fp"]], [rf["fn"], rf["tp"]]],
                        x=["pred: no", "pred: exceed"], y=["actual: no", "actual: exceed"],
                        text_auto=True, color_continuous_scale="Blues",
                        title="Random Forest - confusion (2025 holdout)")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Try the screening model")
    c1, c2, c3 = st.columns(3)
    no2 = c1.number_input("NO2 (ppb)", 1.0, 80.0, 15.0)
    temp = c2.number_input("Temperature (degF)", 35.0, 115.0, 70.0)
    wind = c3.number_input("Wind speed (knots)", 0.5, 20.0, 4.0)
    month = c1.selectbox("Month", list(range(1, 13)),
                         format_func=lambda m: pd.Timestamp(2021, m, 1).strftime("%B"))
    weekend = c2.checkbox("Weekend")
    if st.button("Screen tomorrow's risk"):
        import math
        row = pd.DataFrame([{
            "NO2": no2, "TEMP": temp, "WIND": wind,
            "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
            "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
            "dow_is_weekend": int(weekend),
        }])
        from src.ml_models import _make_models, _chronological_split
        dfm = modeling.dropna(subset=["NO2", "TEMP", "WIND"] + [
            c for c in ["month_sin", "month_cos", "dow_is_weekend"]])
        Xtr, ytr, _, _ = _chronological_split(dfm, ["NO2", "TEMP", "WIND", "month_sin",
                                                    "month_cos", "dow_is_weekend"])
        m = _make_models()["random_forest"].fit(Xtr, ytr)
        prob = float(m.predict_proba(row)[:, 1][0])
        st.metric("Estimated exceedance probability", f"{prob:.2f}")
        st.caption("Demonstration of the trained model on user-supplied operating "
                   "conditions; persistence/lag features are unavailable for "
                   "hypothetical inputs and are excluded here.")

elif page.startswith("6."):
    st.title("Feature Importance")
    imp = pd.read_csv(METRICS_DIR / "rf_feature_importance.csv")
    fig = px.bar(imp.head(12), x="importance", y="feature", orientation="h",
                 title="Random Forest feature importance (all features, train fold)")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=480)
    st.plotly_chart(fig, width='stretch')
    st.caption("Importance reflects each variable's contribution to the model's "
               "predictions in this dataset. It does NOT establish causation.")

elif page.startswith("7."):
    st.title("Regression: O3 Level Estimation")
    st.caption("Continuous counterpart of the screening model: estimate the basin "
               "daily-max 8-hour O3 level (ppb) from same-day cross-pollutant "
               "network means + calendar. Chronological split (train 2021-2024, "
               "test = 2025); no O3 variable is a feature; this is an estimation "
               "task on historical data, not a live forecast.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OLS R2 (2025 holdout)", f"{reg['metrics']['ols']['r2']:.3f}")
    c2.metric("OLS RMSE", f"{reg['metrics']['ols']['rmse']:.2f} ppb")
    c3.metric("Mean-baseline RMSE", f"{reg['metrics']['dummy_mean']['rmse']:.2f} ppb")
    c4.metric("Random forest RMSE", f"{reg['metrics']['random_forest']['rmse']:.2f} ppb")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reg["test_actual"], y=reg["predictions"]["ols"],
                             mode="markers", opacity=0.45, name="OLS estimate"))
    lo = float(min(reg["test_actual"])) - 5
    hi = float(max(reg["test_actual"])) + 5
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(dash="dash"), showlegend=False))
    fig.update_layout(title="Observed vs estimated basin daily-max O3 (2025 holdout)",
                      xaxis_title="observed (ppb)", yaxis_title="estimated (ppb)",
                      height=440)
    st.plotly_chart(fig, width='stretch')

    st.subheader("OLS inference (per-standard-deviation coefficients, train fold)")
    coef = pd.DataFrame(reg["ols_inference"])
    coef["significant_at_0.05"] = coef["p_value"] < 0.05
    st.dataframe(coef, width='stretch', hide_index=True)
    st.caption(f"Durbin-Watson on training residuals = "
               f"{reg['durbin_watson_train_residuals']:.2f}: residuals are "
               "autocorrelated, so OLS standard errors (independence assumption) "
               "are optimistic and p-values are approximate. Ridge (alpha=1.0, "
               "fixed a priori) matches OLS; the random forest improves RMSE by "
               "less than 1% - the relationship is largely linear. These are "
               "associations in this dataset, not causal effects.")

elif page.startswith("8."):
    st.title("Regimes: PCA & Hierarchical Clustering")
    p = clus["pca"]
    evr = p["explained_variance_ratio"]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric("Days analyzed", f"{clus['n_days_used']:,}")
        st.metric("PCs for 90% variance", p["n_components_for_90pct"])
        st.metric("Chosen k (silhouette)", clus["hierarchical"]["best_k"])
        st.caption("Silhouette by k: " +
                   ", ".join(f"k={k}: {v:.3f}" for k, v in clus["hierarchical"]["silhouette_by_k"].items()))
    with c2:
        fig = px.line(x=list(range(1, len(evr) + 1)), y=np.cumsum(evr),
                      markers=True, labels={"x": "n components", "y": "cumulative variance"},
                      title="PCA explained variance")
        fig.add_hline(y=0.9, line_dash="dash")
        st.plotly_chart(fig, width='stretch')

    Z = p["scores"][:, :2]
    fig = px.scatter(x=Z[:, 0], y=Z[:, 1], color=clus["labels"].astype(str),
                     labels={"x": f"PC1 ({evr[0]:.0%})", "y": f"PC2 ({evr[1]:.0%})"},
                     title="Days in PC space, colored by cluster", opacity=0.6)
    st.plotly_chart(fig, width='stretch')
    st.subheader("Cluster profiles (basin means per cluster)")
    st.dataframe(clus["cluster_profile"], width='stretch')
    st.caption("Clusters are data-driven patterns in this dataset (Ward linkage; "
               "k chosen by silhouette, not hard-coded) - not objectively existing "
               "categories of days.")

elif page.startswith("9."):
    st.title("Association Analysis")
    st.caption("CO-OCCURRENCE ONLY - support/confidence/lift describe how condition "
               "bins co-occur in this dataset; nothing here implies causation. Bins: "
               ">=75th pct = high (TEMP, NO2, PM2.5); <=25th pct = low (WIND).")
    fig = px.bar(rules, x="lift", y="antecedent", orientation="h",
                 color="confidence", color_continuous_scale="Viridis",
                 title="Condition bins -> O3 exceedance (lift, colored by confidence)")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=460)
    st.plotly_chart(fig, width='stretch')
    st.dataframe(rules, width='stretch', hide_index=True)

elif page.startswith("10."):
    st.title("Text Intelligence (NLP) - Secondary Dataset")
    st.warning("SECONDARY DATASET: 1,200 U.S. Federal Register EPA documents "
               "('air quality', 2021-2025, public domain). The primary sensor data "
               "has no text field; this module is separate from the air-quality "
               "analysis above.")
    r = nlp["results"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", r["n_docs"])
    c2.metric("Accuracy (Rule vs Notice)", f"{r['accuracy']:.3f}")
    c3.metric("F1 (macro)", r["f1_macro"])
    c4.metric("Train/Test", f"{r['n_train']}/{r['n_test']}")
    c1, c2 = st.columns(2)
    with c1:
        tab = pd.DataFrame({"word": [w["word"] for w in r["word_freq_all_top10"]],
                            "count": [w["count"] for w in r["word_freq_all_top10"]]})
        fig = px.bar(tab, x="count", y="word", orientation="h",
                     title="Top words (stopwords removed)")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, width='stretch')
    with c2:
        wc = __import__("src.nlp_module", fromlist=["make_wordcloud"]).make_wordcloud(
            nlp["tables"]["freq_all"])
        if wc is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.imshow(wc.to_array(), interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("wordcloud package not installed")

elif page.startswith("11."):
    st.title("Forecasting")
    st.caption("Weekly basin-mean PM2.5, chronological holdout (train through 2024, "
               "test = 2025). Multi-step forecast over the full test year.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["train_dates"], y=fc["train_series"],
                             name="train (2021-2024)", line=dict(color="#264653")))
    fig.add_trace(go.Scatter(x=fc["test_dates"], y=fc["test_actual"],
                             name="actual (2025)", line=dict(color="#2a9d8f")))
    best = fc["best_by_rmse"]
    fig.add_trace(go.Scatter(x=fc["test_dates"], y=fc["predictions"][best],
                             name=f"forecast: {best}", line=dict(color="#e76f51", dash="dash")))
    fig.update_layout(yaxis_title="PM2.5 (ug/m3)", height=460,
                      title=f"Weekly PM2.5: holdout forecast ({best} wins on RMSE)")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(pd.DataFrame(fc["metrics"]).T.rename_axis("model"), width='stretch')
    st.info(fc["interpretation"])

elif page.startswith("12."):
    st.title("Key Findings")
    st.markdown(f"""
1. **Two regimes dominate** the basin: warm, ventilated days (higher O3, lower
   PM2.5/NO2) vs cool, stagnant, combustion-influenced days (higher PM2.5 and NO2).
   Clustering (k=2, silhouette-selected) plus PC1/PC2 = {clus['pca']['explained_variance_ratio'][0]:.0%}/{clus['pca']['explained_variance_ratio'][1]:.0%} of variance.
2. **Ozone exceedance risk is temperature-driven**: high-temp days co-occur with
   exceedance at lift {float(rules.iloc[1]['lift']):.1f} (association, not causation);
   the RF model leans on temperature and the ozone-season indicator.
3. **Traffic signal is real but modest**: weekday NO2 averages {tests['weekday_vs_weekend_no2']['group_means']['weekday']} vs weekend
   {tests['weekday_vs_weekend_no2']['group_means']['weekend']} ppb (p={tests['weekday_vs_weekend_no2']['p_value']:.1e}, d={tests['weekday_vs_weekend_no2']['effect_size_cohens_d']}).
4. **Statistically significant is not practically large**: winter vs summer PM2.5
   differs by only {abs(tests['winter_vs_summer_pm25']['group_means']['winter'] - tests['winter_vs_summer_pm25']['group_means']['summer']):.1f} ug/m3 (d={tests['winter_vs_summer_pm25']['effect_size_cohens_d']}); most variance is day-to-day, not seasonal.
5. **Exceedance days are not evenly seasonal** (chi-square p={tests['chi2_exceedance_season']['p_value']:.2f}) - they concentrate
   in warm episodes, consistent with the association analysis.
6. **Engineered features did not improve the 2025 holdout** (RF PR-AUC 0.585 ->
   0.516): same-day physics-guided interactions add little beyond
   temperature/NO2/calendar; reported honestly rather than tuned away.
7. **Forecasting is possible but bounded**: Holt-Winters beats the seasonal-naive
   baseline on the 2025 holdout (RMSE {fc['metrics']['holt_winters_add']['rmse']} vs
   {fc['metrics']['seasonal_naive']['rmse']} ug/m3) - wildfire variability limits accuracy.
8. **Honest limitation**: near-road PM2.5 sites and exceptional-event (wildfire)
   days materially shape tails; the NLP module uses a clearly separated
   secondary corpus.
9. **The O3-meteorology relationship is largely linear**: OLS estimates the
   basin daily-max O3 level at R2 {reg['metrics']['ols']['r2']:.2f} (RMSE
   {reg['metrics']['ols']['rmse']:.1f} ppb vs {reg['metrics']['dummy_mean']['rmse']:.1f}
   for the mean baseline) while the random forest adds almost nothing
   ({reg['metrics']['random_forest']['rmse']:.1f}). Temperature (+) and wind (-) are
   the significant per-SD drivers; residuals are autocorrelated (Durbin-Watson
   {reg['durbin_watson_train_residuals']:.2f}), so OLS p-values are approximate.
   Same-day estimation - association, not causation.
""")
    st.caption("All numbers on this page are produced by executed code in src/ and "
               "cached analysis runs; see docs/ for methodology and the leakage audit.")
