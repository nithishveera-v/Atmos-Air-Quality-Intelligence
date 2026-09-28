# DATA DICTIONARY — Atmos

## Raw source files (data/raw/, US EPA AQS, public domain)

Downloaded from https://aqs.epa.gov/aqsweb/airdata/ :
`daily_88101_YYYY.csv` (PM2.5), `daily_44201_YYYY.csv` (O3),
`daily_42602_YYYY.csv` (NO2), `daily_TEMP_YYYY.csv` (temperature),
`daily_WIND_YYYY.csv` (wind), 2021-2025. 29 columns each; key fields:

| Field | Meaning | Notes |
|---|---|---|
| State Code / County Code / Site Num | monitoring site identifier | South Coast filter: 06037, 06059, 06065, 06071 |
| Parameter Code | AQS parameter | 88101 PM2.5, 44201 O3, 42602 NO2, 62101 TEMP, 61103 wind speed |
| POC | Parameter Occurrence Code (co-located instruments) | resolved by dedup rule below |
| Date Local | local calendar date | becomes the time index |
| Sample Duration | averaging period label | PM2.5 rule below |
| Event Type | exceptional-event flag | 'Included' = wildfire-type episode included by agency |
| Observation Count / Percent | completeness of the daily sample | used in dedup preference |
| Arithmetic Mean | daily aggregate value | the measurement used |
| AQI | EPA Air Quality Index for the day/parameter | retained in DB |

## Parameter-specific rules (verified empirically, encoded in src/etl.py)

| Parameter | Retained duration(s) | Excluded | Conversion |
|---|---|---|---|
| PM2.5 | '24-HR BLK AVG' (reference) and '1 HOUR' (continuous daily mean; agrees to 0.24 ug/m3 on shared site-days) | '24 HOUR' (distinct objective, 1.56 ug/m3 disagreement) | ug/m3 |
| O3 | '8-HR RUN AVG BEGIN HOUR' | - | ppm -> ppb (x1000) |
| NO2 | daily aggregate rows | - | ppb |
| TEMP | daily maximum | - | degF |
| WIND | 61103 speed rows | 61104 direction rows | knots |

## Processed tables

### {param}_southcoast.csv (site-day level, one row per site/day)
`date`, `PM25_value`/`O3_value`/... (converted value, NaN if missing),
`event_type`, `Observation Percent`, `AQI`, site identity columns.

### daily_panel.csv (basin-day level, one row per calendar day)

| Column | Meaning |
|---|---|
| PM25, O3, NO2, TEMP, WIND | basin mean across reporting sites (NaN if < 2 sites) |
| O3_max, PM25_max | basin daily MAX across sites (peak metric; target basis) |
| n_<param>_sites / n_<param>_max_sites | number of sites contributing that day |

### SQLite (database/atmos_airquality.db)
`sites` (dimension), `observations` (site/day/parameter fact),
`daily_panel` (mirror of the panel), `analysis_metadata` (provenance).

## Modeling layer (src/features.py)

| Field | Role | Definition |
|---|---|---|
| o3_exceed | TARGET | 1 if O3_max > 70 ppb |
| NO2, TEMP, WIND | baseline features | same-day basin means |
| month_sin/cos, dow_is_weekend, is_warm_season | baseline features | pure calendar |
| temp_x_no2, no2_per_wind, no2_x_weekend | engineered | same-day interactions |
| no2_lag1, temp_lag1, no2_roll3_mean/max, temp_roll3_mean | engineered | shift(1) past-only persistence |

## Secondary NLP corpus (data/external/federal_register_epa_airquality.json)

1,200 EPA documents, Federal Register API, 2021-2025; fields:
`document_number`, `title`, `type` (Rule/Notice - the label), `abstract`,
`publication_date`. Public domain.
