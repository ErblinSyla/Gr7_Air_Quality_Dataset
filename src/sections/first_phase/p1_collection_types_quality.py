"""p1 report section: data collection, data types and data quality."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, by_name, md

Q = CFG["quality"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    q, flags, bronze = r.q, r.flags, r.bronze
    neg, st = q["negatives"], q["stuck"]
    small_majority = neg.filter(pl.col("pct_small") >= 50)["pollutant"].cast(pl.String).to_list()
    small_minority = neg.filter(pl.col("pct_small") < 50)["pollutant"].cast(pl.String).to_list()
    neg_text = (f"For {', '.join(small_majority)} most negatives are small values near zero (instrument noise → 0)"
                + (f"; for {', '.join(small_minority)} most are large negatives (errors → null)" if small_minority
                   else ""))
    stuck_text = (f"{int(st['of_which_zero'].sum()):,} of {int(st['stuck_hours'].sum()):,} stuck hours "
                  f"({100 * st['of_which_zero'].sum() / st['stuck_hours'].sum():.0f}%) are zero; most for "
                  + ", ".join(f"{p} ({n:,})" for p, n in st.sort("stuck_hours", descending=True)
                              .select("pollutant", "stuck_hours").head(3).rows()))
    inv_years = r.inv_years
    return f"""
## 2. Data collection, data types and data quality

### 2.1 Data collection
**Source:** EEA Air Quality Download Service (the official download service of the European Environment Agency),
country XK, E1a series (verified), hourly measurements from the stations of the IHMK monitoring network (official
sensors). Before collecting, the three questions from the lecture were checked: **accessible** (public download, no
API key), **reliable** (official institution, verified data) and **legal** (CC BY 4.0 licence, © European Environment
Agency; no personal data).

{CFG['expected']['sampling_points']} Parquet files (one per station × pollutant) are read at once with
`pl.scan_parquet`.
Checks: {bronze.height:,} rows, {bronze['sampling_point_id'].n_unique()} sampling points,
{bronze['station_id'].n_unique()} stations; the pollutant code in the ID equals the `Pollutant` column; the file name
equals the sampling point ID. The EEA metadata has {r.meta.height:,} rows for the whole of Europe; the
{r.stations_raw.height} rows for Kosovo are kept. Size: {r.size_raw:.1f} MB raw → {r.size_bronze:.1f} MB bronze (Parquet
zstd, partitioned by pollutant).

### 2.2 Data types
**Type of data set:** record data (every row has the same attributes) that is also ordered in time: an hourly time
series for every station × pollutant (spatio-temporal data). **Data object** = one measurement (station, pollutant,
hour); **attributes** = the columns below.

`Value` arrives as Decimal(38,18) → Float64 (0 conversion failures). Station, pollutant and instrument are extracted
from `Samplingpoint`. Raw time is fixed UTC+1 without daylight saving time → `ts_utc` (the time key) and `ts_local`
(Europe/Belgrade, with daylight saving time, for daily features). Text columns → Categorical, pollutant → Enum.
The attribute type defines which operations are meaningful: **nominal** only =, ≠; **ordinal** also <, >;
**interval** also +, − (time: differences are meaningful, ratios are not); **ratio** also ×, ÷ (concentration: true
zero, 40 µg/m³ is twice 20). Numeric codes (`Pollutant`, `Validity`) are nominal even though they are stored as
numbers: their mean is meaningless.
{md(r.dictionary)}

### 2.3 Data quality (nothing has been changed yet)
**Summary by data quality dimension:**
{md(r.dims)}

**Completeness (missing values).** Explicit nulls in `Value`: {bronze['value_raw'].null_count()}. However,
{int(q['completeness_series']['hours_without_row'].sum()):,} hours have **no row at all** within the period of their
series (implicit missing values, invisible until the hourly grid is built in p3). `DataCapture` is 100% empty.
{md(q['completeness_series'].sort('pct_without_row', descending=True), 12)}

**Validity** (% of rows by Validity code):
{md(q['validity'])}
{figs['invalid']}
Mean share of invalid values by year: {', '.join(f"{c}: {v:.1f}%" for c, v in inv_years)}. The trend is increasing:
from {inv_years[0][1]:.1f}% in {inv_years[0][0]} to {inv_years[-1][1]:.1f}% in {inv_years[-1][0]} (worst year:
{max(inv_years, key=lambda x: x[1])[0]}). Worst station (mean over years): {r.worst[0]} ({r.worst[1]:.1f}%).

**Accuracy: noise and outliers.** The flags are computed, not applied (totals):
{md(flags)}
- **Sentinel −9900** (disguised missing value): a code for "no value", not a measurement; the raw mean of all values
  is {bronze['value_raw'].mean():,.0f}, while the mean of the usable raw values is {r.usable_mean:.2f}.
- **Negatives with Validity 1/2** (noise or error; in the canonical unit, the "small → 0" threshold is
  {CFG['pollutants'][1]['neg_zero']} µg/m³, {CFG['pollutants'][10]['neg_zero']} mg/m³ for CO):
{md(neg)}
  {neg_text}. PM10 and PM2.5 have no valid negatives.
- **Stuck sensor** (the same value for ≥ {Q['stuck_min_hours']} hours, valid values only):
{md(st)}
  {stuck_text} → long zero runs are judged per pollutant in p2.
- **Isolated spikes (noise):** a rise above P99.9 of the hourly changes followed immediately by a fall
  (`tables/p1_spike_thresholds.csv`).
- **Statistical outliers** (outside [Q1 − {Q['outlier_iqr_k']}·IQR, Q3 + {Q['outlier_iqr_k']}·IQR] per pollutant):
  {int(flags['f_outlier_stat'][0]):,} ({100 * flags['f_outlier_stat'][0] / r.usable_n:.1f}% of usable values).
  The distributions are right-skewed (many hours with little pollution, few hours with a lot), so the rule mainly
  catches high-pollution winter hours. As the lecture notes, outliers may be legitimate values (real pollution
  episodes), so they are only flagged, never removed; true errors are caught by the rules above.

**Consistency.**
- PM2.5 > PM10 in the same hour: {int(flags['f_pm25_gt_pm10'][0]) // 2:,} hours; outside the tolerance
  max({Q['pm_tolerance']['abs']} µg/m³, {int(Q['pm_tolerance']['rel'] * 100)}%): {int(flags['f_pm_out_tol'][0]) // 2:,}.
- Declared units: {md(q['units'])}
- **NOx.** Chemically NOx = NO2 + NO. In µg/m³ expressed as NO2: NOx ≈ NO2 + 1.53·NO, so the ratio should be ≈ 1. The
  ratio is ≈ {r.nox_year['median'].median():.4f} = 1/{by_name('ppb_to_ugm3')['NOx']} → **NOx is in ppb even though it is
  labelled µg/m³**. However, {r.mismatch.height} station-years (years:
  {', '.join(str(y) for y in r.mismatch['year'].unique().sort())}) have medians outside the ppb range:
{md(r.mismatch)}
{figs['nox']}
  Before mid-November 2019 the ratio is not constant (no single unit explains the values). Handling: see p2.

**Timeliness.** Days from measurement to publication of the result (`ResultTime`):
{md(q['timeliness'])}

**Duplicates.** {int(q['uniqueness']['duplicate_keys'][0])} duplicate keys (station, pollutant, hour): every measurement
appears exactly once.

**Uninformative columns** (candidates for removal in p5):
{md(q['uninformative_columns'])}
"""
