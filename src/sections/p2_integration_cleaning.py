"""p2 report section: data integration and data cleaning."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.p2_integration_cleaning import NOX_FACTOR
from utils import CFG, md

Q = CFG["quality"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    counts, gas = r.counts, {p: (b, a) for p, b, a in r.gas_summary.rows()}
    return f"""
## 3. Data integration and cleaning

### 3.1 Integration of the station metadata
Integration combines two sources: the measurements (93 files) and the EEA metadata. Following the lecture, three
problems are solved:
- **Schema integration:** the same attribute has different names (`Air Quality Station EoI Code` in the metadata ≡
  the station inside `Samplingpoint` in the measurements) → both become `station_id`; the other columns get uniform
  names.
- **Entity identification:** the same station appears with several names and codes (e.g. names with quotes or HTML
  codes such as `Dardhisht&#235;` = Dardhishtë) → all variants are linked to one `station_id`.
- **Data value conflicts:** the metadata has {r.stations_raw.height} rows for {r.stations.height} stations, coming from
  several record sets with contradictory values for the same station:
{md(r.conflicts)}
**Record set analysis:**
{md(r.sets)}
The `2024_municipality` record set (the 2025 import with the Municipality column filled) labels **every station
"background"**, including XK0002A (Pristina, Rilindja) which the other sources call "traffic", and gives the
municipality **"Podgorica"** (the capital of Montenegro) for {r.podgorica} stations in Kosovo: signs of an unfinished
template.
Yet this record set has the correct coordinates for Pristina, while the other record sets place both Pristina stations
up to {r.pristina_far:.0f} km away. So **no record set is reliable for all attributes**, and the rule "most recent
year" does not
work (in 2024 there are two record sets that contradict each other).

**Conflict resolution, one rule per attribute:**
- **Coordinates:** among all candidates, the one closest to the centre of the station's town/village is taken (an
  independent geographic check in `config.yaml`); a distance > {CFG['stations']['max_distance_km']} km stops the
  pipeline.
- **Altitude:** from the same record as the coordinates (so both describe the same place); 0 m → null.
- **Type and area:** majority vote among the trusted sources (`legacy`, `2024`, `2017`), without `2024_municipality`;
  ties go to the most recent report.
- **Name and national code:** from the 2024 report; names are cleaned (quotes, spaces, HTML codes) and all variants
  are kept.

Coordinate candidates and their distance from the town (`tables/p2_metadata_coordinate_candidates.csv`), e.g. Pristina:
{md(r.coords.filter(pl.col("station_id").is_in(["XK0001A", "XK0002A"])))}
Result: {r.stations.height} stations (every station within {r.stations['distance_km'].max()} km of its town):
{md(r.stations.drop("name_variants", "national_code_variants"))}

### 3.2 Unit conflict (harmonisation before cleaning, so thresholds are applied in the right unit)
The lecture gives the example of different scales for the same attribute (metric versus British units). Here, p1
showed that NOx is in ppb (ratio ≈ {1 / NOX_FACTOR:.3f} = 1/{NOX_FACTOR}), while NO2 and NO are in µg/m³.
**Decision:** NOx × {NOX_FACTOR} → µg/m³ (as NO2), everywhere. CO stays in mg/m³ (standard practice). Before
mid-November 2019 the ratio matches no unit, so NOx in that period is flagged **inconsistent** using two sources of
evidence: (a) the period: every hour before the week from which the station's weekly median stays in the ppb range
(`consistent_from`); (b) the hour: after conversion, NOx/(NO2 + 1.53·NO) deviates from 1 by more than
{int(Q['nox_tolerance'] * 100)}%.
{md(r.periods)}
Flagged NOx hours: {counts['nox_period']:,} from the period + {counts['nox_hourly_only']:,} from the hourly rule only.
**Verification:** after conversion and cleaning, the median ratio of every station × year is within 1 ± 5%
({r.check['ok'].sum()}/{r.check.height} station-years; overall median {r.after['r'].median():.4f}).
{figs['nox']}

### 3.2.1 Gas analyzer change in November 2019 (inconsistent data)
Question: are NO and NO2 in that period correct (they would be used to recompute NOx)? The 2019 level is compared with
the seasonal normal of the same station in the same week of the year (median 2020–2024); particles (PM2.5), which come
from the same combustion sources but are measured by different instruments, serve as a control:
{md(r.gas_summary)}
{figs['gas']}
Before November **NO is ~{gas['NO'][0]}× and NO2 ~{gas['NO2'][0]}× above normal**, while PM2.5 and O3 are normal; the
return to normal happens at the end of October / in November 2019, at the same moment NOx becomes
consistent. A sudden change that affects only the gases of the NOx analyzer and not the particles is not a real change
in pollution but an **analyzer replacement or recalibration** (lecture: faulty instruments as a source of incorrect
values).
**Decision:** NO and NO2 before the station's `consistent_from` → `null_calibration` ({counts['calibration']:,} hours).
Consequence: 2019 NOx is **not recomputed** from NO2 + 1.53·NO, because it would inherit the same error (the
recomputation would give a mean of {r.nox_recomputed:.0f} µg/m³ versus {r.nox_measured:.0f} for the NOx measured
after the
change).

**Why NO/NO2 are removed but CO/SO2 are not:** a level change alone does not prove an instrument error, because
pollution can change for real. Two independent pieces of evidence are therefore required: (1) **physical proof** that
the instrument was not working correctly, and (2) a level change at the same moment. NO and NO2 are measured by the
same analyzer as NOx, and for this triple there is physical proof (NOx ≠ NO2 + 1.53·NO before the change) plus a
level change. CO and SO2 also show a level change (SO2 {gas['SO2'][0]}× → {gas['SO2'][1]}×,
CO {gas['CO'][0]}× → {gas['CO'][1]}×), but they are measured by other analyzers and have no physical check; SO2 also
depends on the operation of the power plants, which changes for real. Deleting possibly correct values would be
another error, so they are **kept** with the flag `f_gas_period_flag` ({counts['gas_flag']:,} hours) and treated as a
documented limitation: analyses of CO and SO2 can exclude this period through the flag.

### 3.3 Data cleaning
Cleaning follows the process from the lecture: (1) **discrepancy detection** using metadata (validity codes, physical
limits of each pollutant), domain knowledge (sentinel −9900, negatives, stuck sensors) and data auditing (the
relations PM2.5 ≤ PM10 and NOx ≈ NO2 + 1.53·NO); (2) **transformation**: the rules are applied in order and every
value receives only the first reason that applies; (3) **iteration**: every step is checked again (e.g. the NOx ratio
after conversion). **No row is deleted**: the value becomes null, the reason is stored in `clean_action` and the
original in `value_raw`.
{md(r.actions)}
**Kept on purpose:** Validity 2 below the detection limit ({counts['below_dl']:,}; removing them would bias the mean
upwards), long plausible zero runs for CO/O3/SO2/NO ({counts['stuck_zero_kept']:,}; at night these gases really fall
close to zero), PM2.5 > PM10 within tolerance ({counts['pm_within_tol']:,} rows, measurement uncertainty), statistical
outliers ({counts['outliers']:,}; real pollution episodes: the lecture stresses that outliers may be legitimate).

**PM2.5 > PM10 outside tolerance:** {r.pm_null_hours} hours ({100 * r.pm_null_hours / r.pm_hours:.2f}% of PM2.5 hours);
**both** values become null, because there is no objective basis to know which instrument was wrong, and any choice
would be a guess. The loss is negligible and these hours are filled by interpolation in p3 when the gap is short.

**Small negatives → 0:** they are instrument noise around zero. How the mean would change if they became null instead
of 0:
{md(r.neg)}
Removing them would raise the mean (up to {r.neg['change_pct'].max()}% for
{r.neg.sort('change_pct')['pollutant'][-1]}), because it removes exactly the small values. The "→ 0" policy keeps the
lower end of the distribution.

**Checks:** 0 negative values, 0 above the physical limit, rows {r.n_before:,} → {r.n_clean:,} (none lost in the
join with the stations), every null has a reason.

**Statistics before/after** ("before" = values with Validity 1/2 in the canonical unit, without sentinel/absurd values):
{md(r.stats, 16)}
{figs['hist']}
"""
