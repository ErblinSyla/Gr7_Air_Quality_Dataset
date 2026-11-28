"""o2 report section: contextual anomalies (normal values in the wrong context: station, month, day type, weather)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o2_contextual_anomalies import CONTEXTS, REGIONAL
from utils import CFG, md

O = CFG["outliers"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    """Markdown text of the contextual anomaly section; every number comes from the results of the step."""
    best = r.best
    k_text = f", n_neighbors = {best['param']}" if best["method"] == "LOF" else ""
    robust = dict(r.robust.select("context", "f1").iter_rows())
    lof_best = r.evaluation.filter(pl.col("method") == "LOF").group_by("context").agg(pl.col("f1").max())
    lof_f1 = dict(lof_best.iter_rows())
    none = r.robust.filter(pl.col("context") == "none").row(0, named=True)
    glob = r.global_by_type.select(pl.col("wrong season", "wrong station").max()).row(0, named=True)
    alarms = r.flags.filter("contextual_anomaly")
    low = alarms.filter(pl.col("value") < pl.col("usual_value")).height
    peak = r.by_month.sort("days", descending=True).row(0, named=True)
    rep = r.repeated.row(0, named=True) if r.repeated.height else None
    top = r.top.head(12).with_columns(pl.col("value", "usual_value", "context_residual", "score_contextual").round(2))
    day_type = "station × month × day type"
    repeated_text = "" if rep is None else (
        f"The anomalies are not spread at random: {r.repeated.height} station-pollutant-months have three or more "
        f"contextual anomalies, the most {rep['days']} days of {rep['context_driver']} at {rep['station_id']} in "
        f"{rep['month']} ({rep['share_low']:.0%} of them lower than usual). Several unusual days of the same pollutant "
        "at the same station in a row are typical of an instrument problem (e.g. an O3 analyzer that measures too "
        "low for weeks), not of a single real event.")
    return f"""
## 2. Contextual anomalies

A **contextual anomaly** is a value that is anomalous only in its context (lecture: 20 °C is normal in May but not in
January). The lecture separates **contextual attributes**, which define the context, from **behavioural attributes**,
whose values are judged. Here the behavioural attributes are the six daily means of section 1, and four contextual
attributes are tested: the **station** (every station has its own sources and level), the **month** (heating season,
sunshine for O3), the **day type** (weekday or weekend, because of traffic) and the **regional level of the day**
(the other stations on the same date, a proxy for the weather). A day with summer values in January, or with the values
of an industrial station at a background station, is globally normal, so the point methods of section 1 cannot see it.

### 2.1 Method: remove the context, then detect
1. **Context profile:** for every context group (e.g. station XK0001A in January) the median and the MAD of
   log(1 + value) of every pollutant: what is usual for that station in that month. The log makes the skewed values
   more symmetric; the median and the MAD are not pulled by the outliers of the group.
2. **Residual:** every value becomes a robust z-score within its group, (log value − median) / (1.4826 · MAD), i.e.
   how unusual it is *for its context*.
3. **Detection on the residuals** with two methods: the **contextual robust z-score** (largest \\|residual\\| of the
   day, statistical, one pollutant at a time) and **LOF on the residual vectors** (the best point method of section
   1, all pollutants together), with the same {O["threshold_percentile"]:g}th-percentile threshold.

**Regional level (weather proxy).** The data set contains no meteorological data, but the weather – wind, thermal
inversion, rain, transported dust – moves all stations of Kosovo together. For the context "{REGIONAL}" every log value
is first compared with the mean log value of the *other* stations on the same day (at least three); the residual of
this difference within the station × month group shows what is special about the station on that day, after the
weather of the day has been taken out. Real meteorological data (e.g. ERA5 reanalysis) would be the next step.

The context is the main parameter, from none to the richest: {", ".join(f"**{c}**" for c in CONTEXTS)}; for LOF also
n_neighbors ({", ".join(map(str, O["contextual_lof_k"]))}).

### 2.2 Evaluation with injected contextual anomalies
As in section 1, {round(O["injected_share"] * 100):g}% known anomalies are added to a copy of the data, now of two
contextual types: **wrong season** (the real values of the same station from a day 150–215 days away, e.g. summer
values in winter) and **wrong station** (the real values of another station on the same date). Every injected day is
a real day, so its values are normal for the data set as a whole.

Contextual robust z-score, by context:
{md(r.robust)}
LOF on the residuals, F1 by n_neighbors (rows) and context (columns):
{md(r.lof)}
The same for the threshold-free PR-AUC (a random score gets ≈ {O["injected_share"]:.2f}):
{md(r.lof_auc)}
{figs["context"]}
**Without context both methods find nothing** (best recall of any global setting: wrong season {glob["wrong season"]:.3f},
wrong station {glob["wrong station"]:.3f}; the global robust z-score has ROC-AUC {none["roc_auc"]:.2f} and PR-AUC
{none["pr_auc"]:.3f}, exactly the values of a random score): this confirms the lecture – a contextual anomaly cannot be
detected by looking at the values alone. Every contextual attribute is then judged by what it adds (best F1 of LOF):
- **station** ({lof_f1["station"]:.3f}) and **month** (station × month {lof_f1["station × month"]:.3f}) are the basis:
  every station and every season has its own normal level;
- **day type** ({lof_f1[day_type]:.3f}) {"adds nothing" if lof_f1[day_type] <= lof_f1["station × month"] else "helps a little"}:
  the weekday/weekend difference is small in daily means, and splitting every group in two leaves fewer days to
  estimate what is usual;
- the **regional level** ({lof_f1[REGIONAL]:.3f}) helps most: a large part of the day-to-day variation is weather that
  all stations share, and once it is removed, a day that does not belong to its station stands out clearly.

LOF on the residuals is clearly better than the robust z-score (F1 {best["f1"]:.3f} vs {max(robust.values()):.3f}),
because a wrong day changes several pollutants at once only a little, and LOF sees the whole vector while the robust
z-score sees one pollutant at a time. Recall of the best setting of each method by type:
{md(r.by_type)}
**Choice: {best["method"]} on the residuals, context {best["context"]}{k_text}** (F1 = {best["f1"]:.3f}, PR-AUC =
{best["pr_auc"]:.3f}, {best["pr_auc"] / O["injected_share"]:.0f}× a random score). The highest PR-AUC is reached by
{r.best_auc["method"]}, context {r.best_auc["context"]}, n_neighbors = {r.best_auc["param"]} ({r.best_auc["pr_auc"]:.3f}),
{"the same context, so the threshold-free measure confirms the choice" if r.best_auc["context"] == best["context"]
 else "another context, which is discussed below"}. The F1 stays lower than for the point anomalies, because a real day
from another month or station is often similar to a normal day of the right context (e.g. April and May, or two
background stations); only clearly different contexts are found.

### 2.3 Contextual anomalies in the real data
The chosen setting marks **{r.n_alarms} station-days**; **{r.n_only_contextual}** of them are *not* point anomalies of
section 1, so the context adds anomalies that the global methods do not see. Regional events (e.g. the dust of March
2020, which raised all stations together) are *not* contextual anomalies here, because the regional level removes them.
{figs["usual"]}
{low} of the {r.n_alarms} days are **lower** than expected for their context, the others higher. The most frequent
pollutant is shown below, and the most anomalies fall in month {peak["month"]} ({peak["days"]} days):
{md(r.by_driver)}
{md(r.by_month)}
Station-pollutant-months with three or more contextual anomalies (`tables/o2_repeated_anomalies.csv`):
{md(r.repeated, 10)}
{repeated_text}

The strongest contextual anomalies that are not point anomalies (`tables/o2_top_contextual.csv`; the usual value is
what is expected for the station, the month and the regional level of that day):
{md(top)}
Two kinds stand out. **Too low for the context:** e.g. O3 of a few µg/m³ on a summer day when about 50–65 µg/m³ is
expected, or SO2 near zero at a station that usually measures 15–20 µg/m³. A low value is never a global outlier, but it
is a typical sign of an analyzer that stopped working properly (or a real, very clean day). **Too high for the
context:** e.g. PM2.5 or O3 several times what the station and the other stations of the day suggest, which points to a
local source (a fire, a local emission). Both kinds are flagged for a check, not removed.
"""
