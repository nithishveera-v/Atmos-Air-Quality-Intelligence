"""SQLite layer for the Atmos project.

Design
------
SCHEMA (kept minimal — only tables that are actually used):

    sites            dimension: one row per unique monitoring site
    observations     fact: one row per site/day/parameter measurement
    daily_panel      basin-level daily aggregate produced by src/etl.py
    analysis_metadata provenance: ETL run info, row counts, scope

The load is idempotent: tables are dropped and recreated on each run so
the database always mirrors the current processed CSVs.

This module also demonstrates the SQL techniques required by the
syllabus (table creation, loading, SELECT, filtering, aggregation,
joins) via demo_queries(); results are written to outputs/tables/.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_PATH, METRICS_DIR, OUTPUTS_DIR, TABLES_DIR  # noqa: E402
from src.etl import PARAM_SPECS  # noqa: E402

UNITS = {p: spec["unit"] for p, spec in PARAM_SPECS.items()}

DDL = """
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS sites;
DROP TABLE IF EXISTS daily_panel;
DROP TABLE IF EXISTS analysis_metadata;

CREATE TABLE sites (
    site_key       TEXT PRIMARY KEY,      -- e.g. '06037-1103'
    state_code     TEXT NOT NULL,
    county_code    TEXT NOT NULL,
    site_num       TEXT NOT NULL,
    latitude       REAL,
    longitude      REAL,
    local_site_name TEXT,
    county_name    TEXT,
    cbsa_name      TEXT
);

CREATE TABLE observations (
    obs_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    site_key           TEXT NOT NULL REFERENCES sites(site_key),
    parameter          TEXT NOT NULL,     -- PM25 / O3 / NO2 / TEMP / WIND
    date               TEXT NOT NULL,     -- ISO date
    value              REAL,
    unit               TEXT NOT NULL,
    event_type         TEXT,              -- exceptional-event flag
    observation_percent REAL,
    aqi                INTEGER
);

CREATE TABLE daily_panel (
    date            TEXT PRIMARY KEY,      -- ISO date, basin aggregate
    PM25            REAL,
    n_PM25_sites    INTEGER,
    O3              REAL,
    n_O3_sites      INTEGER,
    NO2             REAL,
    n_NO2_sites     INTEGER,
    TEMP            REAL,
    n_TEMP_sites    INTEGER,
    WIND            REAL,
    n_WIND_sites    INTEGER,
    O3_max          REAL,
    n_O3_max_sites  INTEGER,
    PM25_max        REAL,
    n_PM25_max_sites INTEGER
);

CREATE TABLE analysis_metadata (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE INDEX idx_obs_param_date ON observations(parameter, date);
CREATE INDEX idx_obs_site       ON observations(site_key);
"""


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _native(row: tuple) -> tuple:
    """Convert numpy/pandas scalars to Python natives for sqlite3."""
    out = []
    for v in row:
        if v is None:
            out.append(None)
        elif isinstance(v, float) and pd.isna(v):
            out.append(None)
        elif pd.isna(v):
            out.append(None)
        elif hasattr(v, "item"):
            out.append(v.item())
        else:
            out.append(v)
    return tuple(out)


def _site_key(county_code, site_num) -> str:
    return f"{int(county_code):03d}-{int(site_num):04d}"


# --------------------------------------------------------------------- #
# schema + load
# --------------------------------------------------------------------- #
def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)


def load_sites(conn: sqlite3.Connection, frames: dict[str, pd.DataFrame]) -> int:
    """Build the site dimension from the unique sites in all frames."""
    cols = ["State Code", "County Code", "Site Num", "Latitude", "Longitude",
            "Local Site Name", "County Name", "CBSA Name"]
    parts = [df[cols].drop_duplicates() for df in frames.values()]
    sites = pd.concat(parts).drop_duplicates(
        subset=["County Code", "Site Num"], keep="first")
    rows = []
    # positional tuple: 0=StateCode 1=County 2=Site 3=Lat 4=Lon 5=Name 6=CountyName 7=CBSA
    for r in sites[cols].itertuples(index=False, name=None):
        rows.append((
            _site_key(r[1], r[2]),
            "06", str(int(r[1])).zfill(3), str(int(r[2])).zfill(4),
            float(r[3]) if pd.notna(r[3]) else None,
            float(r[4]) if pd.notna(r[4]) else None,
            r[5], r[6], r[7],
        ))
    conn.executemany(
        "INSERT INTO sites (site_key, state_code, county_code, site_num, "
        "latitude, longitude, local_site_name, county_name, cbsa_name) "
        "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def load_observations(conn: sqlite3.Connection, frames: dict[str, pd.DataFrame]) -> int:
    """Insert one row per site/day/parameter; skip measurements without a value."""
    total = 0
    for param, df in frames.items():
        d = df[df[f"{param}_value"].notna()]
        idx = {c: i for i, c in enumerate(d.columns)}
        rows = []
        for r in d.itertuples(index=False, name=None):
            rows.append((
                _site_key(r[idx["County Code"]], r[idx["Site Num"]]),
                param,
                r[idx["date"]].strftime("%Y-%m-%d"),
                float(r[idx[f"{param}_value"]]),
                UNITS[param],
                r[idx["event_type"]],
                float(r[idx["Observation Percent"]]) if pd.notna(r[idx["Observation Percent"]]) else None,
                int(r[idx["AQI"]]) if pd.notna(r[idx["AQI"]]) else None,
            ))
        conn.executemany(
            "INSERT INTO observations (site_key, parameter, date, value, unit, "
            "event_type, observation_percent, aqi) VALUES (?,?,?,?,?,?,?,?)",
            rows)
        total += len(rows)
        print(f"[DB] observations loaded for {param}: {len(rows):,}")
    return total


def load_daily_panel(conn: sqlite3.Connection, panel: pd.DataFrame) -> int:
    cols = list(panel.columns)
    date_col = cols.index("date")
    rows = []
    for r in panel.itertuples(index=False, name=None):
        row = (str(r[date_col])[:10],) + tuple(r[:date_col]) + tuple(r[date_col + 1:])
        rows.append(_native(row))
    placeholders = ",".join("?" * len(cols))
    conn.executemany(f"INSERT INTO daily_panel VALUES ({placeholders})", rows)
    return len(rows)


def load_metadata(conn: sqlite3.Connection, n_obs: int, n_sites: int, n_days: int) -> None:
    meta = {
        "project": "Atmos - Urban Air Quality & Health-Risk Intelligence",
        "source": "US EPA AQS daily summaries, https://aqs.epa.gov/aqsweb/airdata/",
        "study_area": "California South Coast: counties 037, 059, 065, 071",
        "period": "2021-01-01 to 2025-12-31",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "n_sites": str(n_sites),
        "n_observations": str(n_obs),
        "n_panel_days": str(n_days),
        "units": "; ".join(f"{k}={v}" for k, v in UNITS.items()),
    }
    conn.executemany("INSERT INTO analysis_metadata VALUES (?,?)", list(meta.items()))


# --------------------------------------------------------------------- #
# SQL demonstrations (syllabus: select / filter / aggregate / join)
# --------------------------------------------------------------------- #
DEMO_QUERIES: list[tuple[str, str]] = [
    ("SELECT with filtering: top-10 highest daily-basin PM2.5 days",
     """SELECT date, ROUND(pm25, 1) AS pm25_ugm3, ROUND(o3, 1) AS o3_ppb
        FROM daily_panel WHERE pm25 IS NOT NULL
        ORDER BY pm25 DESC LIMIT 10;"""),
    ("JOIN + aggregation: sites ranked by 5-year mean PM2.5",
     """SELECT s.county_name, s.local_site_name, COUNT(*) AS n_days,
               ROUND(AVG(o.value), 2) AS mean_pm25
        FROM observations o JOIN sites s ON s.site_key = o.site_key
        WHERE o.parameter = 'PM25'
        GROUP BY o.site_key ORDER BY mean_pm25 DESC LIMIT 10;"""),
    ("Aggregation: monthly mean by parameter (basin level)",
     """SELECT parameter, substr(date, 1, 7) AS month,
               ROUND(AVG(value), 2) AS mean_value, COUNT(*) AS n
        FROM observations WHERE parameter IN ('PM25','O3','NO2')
        GROUP BY parameter, month ORDER BY parameter, month;"""),
    ("Conditional aggregation: exceptional-event share by year (PM25)",
     """SELECT substr(date, 1, 4) AS year,
               COUNT(*) AS total_days,
               SUM(CASE WHEN event_type != 'None' THEN 1 ELSE 0 END) AS event_days,
               ROUND(100.0 * SUM(CASE WHEN event_type != 'None' THEN 1 ELSE 0 END)
                     / COUNT(*), 1) AS event_pct
        FROM observations WHERE parameter = 'PM25'
        GROUP BY year ORDER BY year;"""),
    ("JOIN + HAVING: O3 sites with >1500 valid days, ranked by peak",
     """SELECT s.county_name, s.local_site_name, COUNT(*) AS n_days,
               ROUND(MAX(o.value), 1) AS max_o3_ppb
        FROM observations o JOIN sites s ON s.site_key = o.site_key
        WHERE o.parameter = 'O3'
        GROUP BY o.site_key HAVING COUNT(*) > 1500
        ORDER BY max_o3_ppb DESC LIMIT 8;"""),
]


def demo_queries(conn: sqlite3.Connection) -> str:
    lines = ["Atmos SQL demonstrations", "=" * 60, ""]
    for title, sql in DEMO_QUERIES:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        lines.append(title)
        lines.append("-" * len(title))
        lines.append(" | ".join(cols))
        for r in rows:
            lines.append(" | ".join("" if v is None else str(v) for v in r))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------- #
def build_database() -> None:
    from src.config import DATA_PROCESSED  # local import: avoids ETL side effects
    frames = {}
    for param in PARAM_SPECS:
        frames[param] = pd.read_csv(
            DATA_PROCESSED / f"{param.lower()}_southcoast.csv",
            parse_dates=["date"])
    panel = pd.read_csv(DATA_PROCESSED / "daily_panel.csv", parse_dates=["date"])

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        create_schema(conn)
        n_sites = load_sites(conn, frames)
        print(f"[DB] sites loaded: {n_sites:,}")
        n_obs = load_observations(conn, frames)
        n_days = load_daily_panel(conn, panel)
        print(f"[DB] daily_panel rows loaded: {n_days:,}")
        load_metadata(conn, n_obs, n_sites, n_days)
        conn.commit()

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        report = demo_queries(conn)
        (TABLES_DIR / "sql_demonstrations.txt").write_text(report, encoding="utf-8")

    print(f"[DB] database written -> {DB_PATH}")
    print(f"[DB] SQL demonstration results -> outputs/tables/sql_demonstrations.txt")


if __name__ == "__main__":
    build_database()
