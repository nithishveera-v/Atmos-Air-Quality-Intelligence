"""Statistical analysis module for the Atmos project.

Implements the syllabus statistics topics on the basin-level daily panel:

* descriptive statistics (mean/median/SD/quartiles/IQR + the syllabus
  'mode' applied to the *categorical* attributes where it is meaningful)
* Z-score anomaly/spike detection (documented threshold: |z| > 3)
* normal-distribution analysis: raw vs log-transformed PM2.5
* confidence intervals for the mean using BOTH the Z-distribution
  (sigma known assumption, sigma estimated from the sample — reported as
  an approximation for comparison) and the T-distribution (default,
  honest choice for real data with estimated sigma)
* correlation analysis (Pearson + Spearman with interpretation notes)

All functions are pure (no I/O side effects); the notebook and dashboard
call these and persist their own outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Z-quantile for the conventional 95% confidence level.
Z95 = stats.norm.ppf(0.975)  # 1.959964


# --------------------------------------------------------------------- #
# descriptive statistics
# --------------------------------------------------------------------- #
def describe_series(s: pd.Series) -> dict:
    """Descriptive statistics for one numeric series."""
    s = s.dropna()
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "sd": float(s.std(ddof=1)),
        "variance": float(s.var(ddof=1)),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "min": float(s.min()),
        "max": float(s.max()),
        "skew": float(stats.skew(s)),
        "excess_kurtosis": float(stats.kurtosis(s)),
    }


def mode_of_categorical(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Mode of categorical attributes (the statistically meaningful use of
    'mode' here — the modal value of a series of raw measurements has no
    practical interpretation for continuous sensor data)."""
    out = {}
    for c in columns:
        if c in df.columns:
            out[c] = df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else None
    return pd.Series(out, name="mode")


# --------------------------------------------------------------------- #
# z-score anomaly detection
# --------------------------------------------------------------------- #
def zscore_anomalies(s: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    """Flag |z| > threshold as statistical spikes.

    Note documented everywhere these results are used: Z-scores use the
    global mean/SD, so firestorm days (which drive the SD up) are flagged
    conservatively; this is a screening tool, not a classification.
    """
    s = s.dropna()
    mu, sd = s.mean(), s.std(ddof=1)
    z = (s - mu) / sd
    flags = z.abs() > threshold
    return pd.DataFrame({
        "date": s.index,
        "value": s.values,
        "z": z.values,
        "is_anomaly": flags.values,
    })


def anomaly_summary(s: pd.Series, threshold: float = 3.0) -> dict:
    za = zscore_anomalies(s, threshold)
    n = len(za)
    a = za[za["is_anomaly"]]
    return {
        "n_days": n,
        "threshold": threshold,
        "n_anomalies": int(a["is_anomaly"].sum()),
        "pct": round(100 * a["is_anomaly"].mean(), 3) if n else 0.0,
        "mean_z": round(float(za["z"].mean()), 3),
        "top_spikes": za.sort_values("z", ascending=False).head(5)[
            ["date", "value", "z"]].to_dict("records"),
    }


# --------------------------------------------------------------------- #
# confidence intervals: Z-distribution vs T-distribution
# symmetric two-sided CI for the mean
# --------------------------------------------------------------------- #
def ci_for_mean(s: pd.Series, conf: float = 0.95) -> dict:
    """Two-sided CI for the mean: T-distribution (used, sigma estimated)
    and Z-distribution (reported for comparison, sigma-known assumption)."""
    s = s.dropna().astype(float)
    n = s.size
    mean = s.mean()
    sd = s.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf((1 + conf) / 2, df=n - 1)
    zcrit = stats.norm.ppf((1 + conf) / 2)
    return {
        "n": int(n),
        "mean": round(float(mean), 3),
        "sd": round(float(sd), 3),
        "se": round(float(se), 5),
        "t_crit": round(float(tcrit), 4),
        "z_crit": round(float(zcrit), 4),
        "ci_t": (round(float(mean - tcrit * se), 3), round(float(mean + tcrit * se), 3)),
        "ci_z": (round(float(mean - zcrit * se), 3), round(float(mean + zcrit * se), 3)),
        "width_t": round(float(2 * tcrit * se), 3),
        "width_z": round(float(2 * zcrit * se), 3),
        "conf": conf,
    }


def seasonal_ci_table(panel: pd.DataFrame, col: str = "PM25", conf: float = 0.95) -> pd.DataFrame:
    """Mean PM2.5 CI per calendar season (meteorological seasons)."""
    d = panel.dropna(subset=[col]).copy()
    d["month"] = d["date"].dt.month
    bins = [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
    labels = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]
    d["season"] = np.select(
        [d["month"].isin(b) for b in bins], labels, default="Winter (DJF)")
    rows = []
    for season, g in d.groupby("season"):
        r = ci_for_mean(g[col], conf)
        rows.append({
            "season": season, "n": r["n"], "mean": r["mean"], "sd": r["sd"],
            "ci_low_t": r["ci_t"][0], "ci_high_t": r["ci_t"][1],
            "ci_low_z": r["ci_z"][0], "ci_high_z": r["ci_z"][1],
            "ci_width_t": r["width_t"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# normal-distribution analysis
# Jarque-Bera + skew/kurtosis; log transform comparison
# --------------------------------------------------------------------- #
def normality_analysis(s: pd.Series) -> dict:
    s = s.dropna().astype(float)
    s_pos = s[s > 0]
    log_s = np.log(s_pos)
    jb_raw = stats.jarque_bera(s)
    jb_log = stats.jarque_bera(log_s)
    return {
        "n": int(s.size),
        "skew_raw": round(float(stats.skew(s)), 3),
        "excess_kurt_raw": round(float(stats.kurtosis(s)), 3),
        "jb_stat_raw": round(float(jb_raw.statistic), 1),
        "jb_p_raw": float(jb_raw.pvalue),
        "skew_log": round(float(stats.skew(log_s)), 3),
        "excess_kurt_log": round(float(stats.kurtosis(log_s)), 3),
        "jb_stat_log": round(float(jb_log.statistic), 1),
        "jb_p_log": float(jb_log.pvalue),
        "interpretation": (
            "Raw PM2.5 is right-skewed; the log transform reduces skew. "
            "PM2.5 is generated by a mixture of sources and conditions, so "
            "exact normality is not expected; the log transform is used "
            "where a symmetric approximation is needed."
        ),
    }


# --------------------------------------------------------------------- #
# correlation
# --------------------------------------------------------------------- #
def correlation_table(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pearson and Spearman correlations with pairwise n."""
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = panel[[a, b]].dropna()
            pr, pp = stats.pearsonr(d[a], d[b])
            sr, sp = stats.spearmanr(d[a], d[b])
            out.append({
                "var_a": a, "var_b": b, "n": len(d),
                "pearson_r": round(float(pr), 3), "pearson_p": float(pp),
                "spearman_rho": round(float(sr), 3), "spearman_p": float(sp),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------- #
# runner (used by notebook / verification script)
# --------------------------------------------------------------------- #
def run_statistics(panel: pd.DataFrame) -> dict:
    results = {}
    results["descriptive"] = {p: describe_series(panel[p])
                              for p in ["PM25", "O3", "NO2", "TEMP", "WIND"]}
    results["pm25_normality"] = normality_analysis(panel["PM25"])
    results["pm25_anomalies_z3"] = anomaly_summary(panel.set_index("date")["PM25"], 3.0)
    results["seasonal_ci_pm25"] = seasonal_ci_table(panel, "PM25")
    results["correlations"] = correlation_table(
        panel, ["PM25", "O3", "NO2", "TEMP", "WIND"])
    return results
