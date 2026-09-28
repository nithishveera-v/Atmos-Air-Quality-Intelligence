"""Association rule mining for the Atmos project (support/confidence/lift).

CO-OCCURRENCE ANALYSIS ONLY. Rules describe how condition bins co-occur
in this dataset; they do NOT establish that any condition causes another.

BIN CONSTRUCTION (fully documented; quantiles computed on the analysis
subset, no target information used to define bins):

    high_temp : basin-mean TEMP >= its 75th percentile
    low_wind  : basin-mean WIND <= its 25th percentile
    high_no2  : basin-mean NO2  >= its 75th percentile
    high_pm25 : basin-mean PM2.5 >= its 75th percentile
    o3_exceed : basin daily MAX O3 > 70 ppb (the outcome of interest;
                included as an ITEM so its co-occurrence with condition
                bins can be examined, never as a feature of anything)

Rules mined: 1- and 2-item condition antecedents -> o3_exceed consequent,
minimum antecedent support 5% of days. Only lift > 1 indicates a positive
association; the report wording below enforces association language.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

CONDITION_ITEMS = ["high_temp", "low_wind", "high_no2", "high_pm25"]
TARGET = "o3_exceed"


def make_bins(panel: pd.DataFrame) -> pd.DataFrame:
    # drop incomplete days FIRST, then binarize on complete data so that
    # quantile thresholds are computed on observed values only
    d = panel.dropna(subset=["TEMP", "WIND", "NO2", "PM25", "O3_max"]).copy()
    b = pd.DataFrame(index=d.index)
    b["date"] = d["date"]
    b["high_temp"] = d["TEMP"] >= d["TEMP"].quantile(0.75)
    b["low_wind"] = d["WIND"] <= d["WIND"].quantile(0.25)
    b["high_no2"] = d["NO2"] >= d["NO2"].quantile(0.75)
    b["high_pm25"] = d["PM25"] >= d["PM25"].quantile(0.75)
    b[TARGET] = d["O3_max"] > 70
    return b.reset_index(drop=True)


def mine_rules(bins: pd.DataFrame, min_support: float = 0.05) -> pd.DataFrame:
    n = len(bins)
    target_sup = float(bins[TARGET].mean())
    rules = []
    for r in (1, 2):
        for ante in combinations(CONDITION_ITEMS, r):
            ante_sup = float(bins[list(ante)].all(axis=1).mean())
            if ante_sup < min_support:
                continue
            both_sup = float(bins[list(ante) + [TARGET]].all(axis=1).mean())
            conf = both_sup / ante_sup if ante_sup else 0.0
            lift = conf / target_sup if target_sup else 0.0
            rules.append({
                "antecedent": " & ".join(ante),
                "antecedent_support": round(ante_sup, 4),
                "consequent": TARGET,
                "support": round(both_sup, 4),
                "confidence": round(conf, 4),
                "lift": round(lift, 3),
                "n_days": int(round(both_sup * n)),
            })
    return pd.DataFrame(rules).sort_values("lift", ascending=False).reset_index(drop=True)


def interpret_rules(rules: pd.DataFrame) -> str:
    top = rules[rules["lift"] > 1].head(5)
    if top.empty:
        return ("No positive associations above lift 1.0 were found at the "
                "documented thresholds.")
    lines = ["Top associations (CO-OCCURRENCE, not causation):"]
    for _, r in top.iterrows():
        lines.append(
            f"  {r['antecedent']} -> {r['consequent']}: support={r['support']}, "
            f"confidence={r['confidence']}, lift={r['lift']} "
            f"(lift>1 means the conditions co-occur more often than "
            f"independence would predict in THIS dataset)")
    return "\n".join(lines)
