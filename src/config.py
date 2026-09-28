"""Central configuration for the Atmos project.

All paths are derived from the project root (the folder containing this
package's parent), so the project runs on any machine and operating system
without hard-coded locations.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

DATABASE_DIR = PROJECT_ROOT / "database"
DB_PATH = DATABASE_DIR / "atmos_airquality.db"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
TABLES_DIR = OUTPUTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

# ------------------------------------------------------- study scope -----
# Phase 1 decision: California / South Coast air basin, recent 5 years,
# EPA AQS daily summary files (national files, filtered to the study area).
STATE_CODE = "CA"
YEARS = (2021, 2022, 2023, 2024, 2025)

# AQS parameter codes for the daily summary files:
#   88101 = PM2.5 local conditions, 44201 = Ozone, 42602 = Nitrogen dioxide,
#   62101 = Temperature, 61103 = Wind speed
PARAMETER_CODES = {
    "PM25": "88101",
    "O3": "44201",
    "NO2": "42602",
    "TEMP": "62101",
    "WIND": "61103",
}
AQS_BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"

# South Coast air basin core counties (county codes within California):
# Los Angeles (037), Orange (059), Riverside (065), San Bernardino (071).
SOUTH_COAST_COUNTY_CODES = ("037", "059", "065", "071")

# ------------------------------------------------------- reproducibility --
RANDOM_SEED = 42

for _p in (DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, DATABASE_DIR, MODELS_DIR,
           PLOTS_DIR, METRICS_DIR, TABLES_DIR):
    _p.mkdir(parents=True, exist_ok=True)
