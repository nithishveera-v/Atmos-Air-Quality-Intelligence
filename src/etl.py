"""ETL pipeline for the Atmos project.

EXTRACT   Read EPA AQS daily-summary CSVs (data/raw).
TRANSFORM Filter to the California South Coast study area, apply the
          documented per-parameter validity rules, resolve co-located
          instruments (POC dedup), standardise units.
VALIDATE  Compute a data-quality report (missingness, negatives, duplicates).
LOAD      Write analysis-ready CSVs to data/processed (SQLite load is in
          src/db.py).

Every rule below was verified against the raw files before being encoded:

* PM2.5 (88101): rows carry three sample durations in CA. Empirical
  verification on 2023 South Coast data (scripts in project history)
  showed: (a) '1 HOUR' rows are daily means of ~24 hourly samples
  (Observation Count ~ 24) and agree with the filter-based reference
  '24-HR BLK AVG' on shared site-days to within 0.24 ug/m3 mean absolute
  difference, so both are retained as daily PM2.5; (b) '24 HOUR' rows
  differ systematically from the reference (1.56 ug/m3 mean absolute
  difference on shared site-days, higher average level) and are EXCLUDED
  as a distinct, non-comparable monitoring objective. When both a
  '24-HR BLK AVG' (reference method) and a '1 HOUR' row exist for the
  same site/day, the reference row wins in the POC dedup step.
* O3 (44201): CA rows are uniformly '8-HR RUN AVG BEGIN HOUR'.
* NO2 (42602): CA rows are uniformly daily aggregates ('1 HOUR' label
  denotes the daily maximum 1-hour value).
* TEMP (62101): daily maximum temperature, degrees Fahrenheit.
* WIND: the daily_WIND file mixes wind speed (61103, knots) with wind
  direction (61104, compass degrees); only 61103 is retained.
* POC: 'Parameter Occurrence Code' distinguishes co-located instruments
  at the same site. When several POCs report for the same site/day the
  record with the higher 'Observation Percent' is kept (better
  completeness); ties resolve to the lowest POC deterministically.
* Units: PM2.5 ug/m3, O3 converted ppm -> ppb (*1000), NO2 ppb,
  TEMP degF, WIND knots. Conversions are documented, not silent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_PROCESSED,
    DATA_RAW,
    SOUTH_COAST_COUNTY_CODES,
    STATE_CODE,
    YEARS,
)

STUDY_COUNTIES = {f"06{c}" for c in SOUTH_COAST_COUNTY_CODES}

# (parameter label, AQS parameter code, sample-duration rule, unit conversion)
PARAM_SPECS: dict[str, dict] = {
    "PM25": {
        "code": "88101",
        "durations_allowed": ("24-HR BLK AVG", "1 HOUR"),
        "duration_priority": {"24-HR BLK AVG": 0, "1 HOUR": 1},
        "unit": "ug/m3",
        "scale": 1.0,
    },
    "O3": {
        "code": "44201",
        "durations_allowed": ("8-HR RUN AVG BEGIN HOUR",),
        "duration_priority": {},
        "unit": "ppb",
        "scale": 1000.0,
    },
    "NO2": {"code": "42602", "durations_allowed": None, "duration_priority": {}, "unit": "ppb", "scale": 1.0},
    "TEMP": {"code": "62101", "durations_allowed": None, "duration_priority": {}, "unit": "degF", "scale": 1.0},
    "WIND": {"code": "61103", "durations_allowed": None, "duration_priority": {}, "unit": "knots", "scale": 1.0},
}

RAW_FILES = {
    "PM25": "daily_88101_{y}.csv",
    "O3": "daily_44201_{y}.csv",
    "NO2": "daily_42602_{y}.csv",
    "TEMP": "daily_TEMP_{y}.csv",
    "WIND": "daily_WIND_{y}.csv",
}

KEEP_COLS = [
    "State Code", "County Code", "Site Num", "Latitude", "Longitude",
    "Parameter Code", "POC", "Date Local", "Event Type", "Sample Duration",
    "Observation Count", "Observation Percent", "Arithmetic Mean", "AQI",
    "Local Site Name", "County Name", "CBSA Name",
]

VALUE_COL = "Arithmetic Mean"
MISSING_SENTINELS = (-999.0, -99.0, -1.0)


# --------------------------------------------------------------------- #
# EXTRACT
# --------------------------------------------------------------------- #
def extract_raw(param: str) -> pd.DataFrame:
    """Read the raw national CSVs for one parameter across all study years."""
    spec = PARAM_SPECS[param]
    frames = []
    for year in YEARS:
        path = DATA_RAW / RAW_FILES[param].format(y=year)
        if not path.exists():
            raise FileNotFoundError(f"missing raw file: {path}")
        df = pd.read_csv(path, usecols=KEEP_COLS, low_memory=False)
        df = df[df["Parameter Code"] == int(spec["code"])]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- #
# TRANSFORM
# --------------------------------------------------------------------- #
def to_study_area(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to California South Coast counties.

    Keys are built as zero-padded FIPS: state (2) + county (3), e.g. 06037
    = Los Angeles County. Cast through Int64 first so float-typed columns
    (as produced by some readers) do not yield keys like '6.0'.
    """
    state = df["State Code"].astype("Int64").astype(str).str.zfill(2)
    county = df["County Code"].astype("Int64").astype(str).str.zfill(3)
    key = state + county
    return df[key.isin(STUDY_COUNTIES)].copy()


def apply_param_rules(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """Apply the documented per-parameter validity rules."""
    spec = PARAM_SPECS[param]
    out = df.copy()
    allowed = spec.get("durations_allowed")
    if allowed is not None and "Sample Duration" in out.columns:
        out = out[out["Sample Duration"].isin(allowed)]
    return out


def dedup_poc(df: pd.DataFrame, param: str) -> tuple[pd.DataFrame, int]:
    """Resolve co-located instruments and duplicate durations.

    Preference order per site/day: (1) reference-method duration
    ('duration_priority' in PARAM_SPECS, lower wins), (2) higher
    Observation Percent, (3) lowest POC (deterministic).
    Returns (frame, number of rows dropped).
    """
    before = len(df)
    if len(df) == 0:
        return df, 0
    out = df.copy()
    out["_dur_rank"] = out["Sample Duration"].map(PARAM_SPECS[param]["duration_priority"]).fillna(9)
    out = out.sort_values(
        ["County Code", "Site Num", "Date Local", "_dur_rank", "Observation Percent", "POC"],
        ascending=[True, True, True, True, False, True],
    )
    out = out.drop_duplicates(subset=["County Code", "Site Num", "Date Local"], keep="first")
    deduped = out.drop(columns="_dur_rank")
    return deduped, before - len(deduped)


def standardise(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """Rename the value column, apply unit conversion, clean types."""
    spec = PARAM_SPECS[param]
    out = df.copy()
    out["date"] = pd.to_datetime(out["Date Local"])
    value = out[VALUE_COL].astype(float)
    for sentinel in MISSING_SENTINELS:
        value = value.mask(value <= sentinel)
    out[f"{param}_value"] = value * spec["scale"]
    out["event_type"] = out["Event Type"].fillna("None")
    return out


# --------------------------------------------------------------------- #
# VALIDATE
# --------------------------------------------------------------------- #
def quality_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-parameter data-quality summary of the transformed study data."""
    rows = []
    for param, df in frames.items():
        vcol = f"{param}_value"
        d = df.dropna(subset=[vcol])
        rows.append({
            "parameter": param,
            "rows_transformed": len(df),
            "rows_with_value": len(d),
            "pct_missing_value": round(100 * (1 - len(d) / max(len(df), 1)), 2),
            "n_sites": d.groupby(["County Code", "Site Num"]).ngroups,
            "date_min": d["date"].min().date().isoformat(),
            "date_max": d["date"].max().date().isoformat(),
            "n_negative_values": int((d[vcol] < 0).sum()),
            "n_zero_values": int((d[vcol] == 0).sum()),
            "n_duplicate_site_days": int(d.duplicated(["County Code", "Site Num", "date"]).sum()),
            "n_exceptional_event_rows": int((d["event_type"] != "None").sum()),
            "value_p01": round(d[vcol].quantile(0.01), 3),
            "value_p50": round(d[vcol].quantile(0.50), 3),
            "value_p99": round(d[vcol].quantile(0.99), 3),
            "value_max": round(d[vcol].max(), 3),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# LOAD
# --------------------------------------------------------------------- #
def load_processed(frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = {}
    for param, df in frames.items():
        path = DATA_PROCESSED / f"{param.lower()}_southcoast.csv"
        df.to_csv(path, index=False)
        out[param] = path
    return out


def build_daily_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the basin-level daily modeling panel (one row per day).

    Aggregation rule (documented, not silent): the daily basin value for a
    parameter is the mean across South Coast sites reporting that day; a
    day requires at least MIN_SITES_PER_DAY reporting sites, otherwise the
    value is NaN (avoiding days represented by a single instrument).
    n_<param>_sites records per-day site coverage.
    """
    MIN_SITES_PER_DAY = 2
    parts = {}
    for param, df in frames.items():
        vcol = f"{param}_value"
        d = df.dropna(subset=[vcol])
        g = d.groupby(d["date"].dt.normalize()).agg(
            **{f"{param}": (vcol, "mean"),
               f"n_{param}_sites": (vcol, "size")}
        )
        parts[param] = g
    # Basin-level daily MAX for the pollutants (peak-exposure metric:
    # a NAAQS exceedance day is defined by ANY monitor exceeding, so the
    # max across sites is the policy-relevant aggregate; means dilute peaks
    # across the coastal-inland gradient).
    for param in ("O3", "PM25"):
        vcol = f"{param}_value"
        d = frames[param].dropna(subset=[vcol])
        gmax = d.groupby(d["date"].dt.normalize()).agg(
            **{f"{param}_max": (vcol, "max"),
               f"n_{param}_max_sites": (vcol, "size")})
        parts[f"{param}#max"] = gmax
    panel = pd.concat(parts.values(), axis=1)
    for param in frames:
        low = panel[f"n_{param}_sites"] < MIN_SITES_PER_DAY
        panel.loc[low, param] = np.nan
        panel.loc[low, f"n_{param}_sites"] = np.nan
    for param in ("O3", "PM25"):
        low = panel[f"n_{param}_max_sites"] < MIN_SITES_PER_DAY
        panel.loc[low, f"{param}_max"] = np.nan
        panel.loc[low, f"n_{param}_max_sites"] = np.nan
    panel.index.name = "date"
    return panel.reset_index()


def run_etl() -> dict[str, pd.DataFrame]:
    """Run the full ETL for all parameters; return transformed frames."""
    frames: dict[str, pd.DataFrame] = {}
    for param in PARAM_SPECS:
        print(f"[ETL] {param}: extracting...")
        raw = extract_raw(param)
        print(f"[ETL] {param}: {len(raw):,} national rows")
        study = to_study_area(raw)
        study = apply_param_rules(study, param)
        deduped, dropped = dedup_poc(study, param)
        std = standardise(deduped, param)
        print(f"[ETL] {param}: {len(std):,} study rows "
              f"({dropped:,} co-located duplicates resolved)")
        frames[param] = std
    report = quality_report(frames)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    load_processed(frames)
    report.to_csv(DATA_PROCESSED / "data_quality_report.csv", index=False)
    panel = build_daily_panel(frames)
    panel.to_csv(DATA_PROCESSED / "daily_panel.csv", index=False)
    print(f"\n[ETL] Daily panel: {len(panel)} days, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")
    print("[ETL] Panel non-null coverage (%):")
    for param in frames:
        print(f"  {param}: {100 * panel[param].notna().mean():.1f}")
    print("\n[ETL] Data quality report:")
    cols = ["parameter", "rows_with_value", "n_sites", "pct_missing_value",
            "n_duplicate_site_days", "value_p50", "value_max"]
    print(report[cols].to_string(index=False))
    return frames


if __name__ == "__main__":
    run_etl()
