"""p4 report section: aggregation and sampling."""

from __future__ import annotations

from types import SimpleNamespace

from utils import CFG, md

AG, S = CFG["aggregation"], CFG["sampling"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    g, s = r.agg, r.smp
    strat, srs = s.reps["stratified"], s.reps["SRSWOR"]
    strat_dev = max(strat["max_station_deviation_pp"], strat["max_season_deviation_pp"])
    srs_dev = max(srs["max_station_deviation_pp"], srs["max_season_deviation_pp"])
    return f"""
## 5. Aggregation and sampling

### 5.1 Aggregation
Aggregation combines many objects into one (lecture): it reduces the data, changes the scale (station → zone → Kosovo)
and gives more stable data (the daily mean varies less than the hourly value).

**Daily** (by local date, because the daily life of people and traffic follows local time): mean, maximum and number of
valid hours. **Completeness rule:** the daily mean is computed only when there are ≥ {AG['min_hours_per_day']} valid
hours (75%); otherwise it is null, because a "mean" from few hours (e.g. only the night) would be biased.
**Monthly:** mean of the valid days, only when ≥ {int(AG['min_days_per_month_frac'] * 100)}% of the days are valid.
**Annual:** mean and coverage (valid hours / hours of the year).

Aggregation is done twice: from the **measured** (official) values and from the **imputed** values. Valid days gained
thanks to imputation:
{md(g.gained)}

**Concept hierarchy and data cube.** The data is organised in two hierarchies: time (hour → day → month → year) and
space (station → zone → Kosovo). Moving up the hierarchy (*roll up*) replaces many values by one more general value;
moving down (*drill down*) returns to the details. Following the lecture, the smallest level that is enough for the
task is used: e.g. to compare towns, the annual mean of the zone is enough. Size reduction at every level:
{md(g.sizes)}
PM2.5 roll-up (annual mean of the valid days, µg/m³) from station to zone and Kosovo
(`tables/p4_rollup_station_zone_kosovo.csv` has every pollutant and level):
{md(g.pm_roll)}

**Annual coverage:** {g.n_low} of {g.n_years} station-pollutant-years are below {int(AG['min_annual_coverage'] * 100)}%:
for them `mean_official` = null and only `mean` is kept as informative, because an annual mean from less than 90% of
the hours does not represent the year (e.g. a whole season may be missing). Full table: `tables/p4_annual_coverage.csv`.
{md(g.coverage, 12)}
**Heating season (October–April) versus the rest of the year** (mean of the valid days):
{md(g.heating)}
Particles (PM2.5, PM10), CO and NO are much higher in the heating season (burning of wood and coal, and thermal
inversion in winter); O3 is higher outside it (it is formed by sunlight). SO2 changes little over the year, because
its main source is industry and the power plants, not household heating.
{figs['daily']}

### 5.2 Sampling
Population: {s.n_pop:,} station-hours (wide format, after imputation). Following the lecture, a sample works almost
as well as the whole data set only if it is **representative**: it has approximately the same properties as the
population. Three methods are compared with {int(S['fraction'] * 100)}% (seed {S['seed']}):
- **SRSWOR** (simple random sample without replacement): every row is selected at most once;
- **SRSWR** (simple random sample with replacement): the same row can be selected more than once;
- **stratified** by station × season: the same share from every stratum, so no group is left out.

Properties of the sample versus the population:
{md(s.dist)}
Representation of the strata (maximum deviation of a station's/season's share from the population, in percentage
points) and rows selected more than once:
{md(s.rep)}
**Decision:** the **stratified** sample is kept (`data/gold/sample_10pct.parquet`). All three methods preserve the
mean (maximum deviation: SRSWOR {s.dev['SRSWOR']:.2f}%, SRSWR {s.dev['SRSWR']:.2f}%, stratified
{s.dev['stratified']:.2f}%), but SRSWR has {s.reps['SRSWR']['repeated_rows']:,} repeated rows (duplicated information),
and only the stratified sample guarantees the shares of stations and seasons (deviation ≤ {strat_dev} pp versus
{srs_dev} pp for SRSWOR). This matters because pollution varies strongly by station and season.
"""
