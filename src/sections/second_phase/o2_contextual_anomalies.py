"""o2 report section: contextual anomalies (normal values in the wrong context: station and month)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, md

O = CFG["outliers"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    """Markdown text of the contextual anomaly section; every number comes from the results of the step."""
    best = r.best
    k_text = f", n_neighbors = {best['param']}" if best["method"] == "LOF" else ""
    robust = dict(r.robust.select("context", "f1").iter_rows())
    none = r.robust.filter(pl.col("context") == "none").row(0, named=True)
    glob = r.global_by_type.select(pl.col("wrong season", "wrong station").max()).row(0, named=True)
    alarms = r.flags.filter("contextual_anomaly")
    low = alarms.filter(pl.col("value") < pl.col("usual_value")).height
    peak = r.by_month.sort("days", descending=True).row(0, named=True)
    top = r.top.head(12).with_columns(pl.col("value", "usual_value", "context_residual", "score_contextual").round(2))
    return f"""
## 2. Contextual anomalies

A **contextual anomaly** is a value that is anomalous only in its context (lecture: 20 °C is normal in May but not in
January). The lecture separates **contextual attributes**, which define the context, from **behavioural attributes**,
whose values are judged. Here the contextual attributes are the **station** (every station has its own sources and
level) and the **month** (heating season, sunshine for O3); the behavioural attributes are the six daily means of
section 1. A day with summer values in January, or with the values of an industrial station at a background station,
is globally normal, so the point methods of section 1 cannot see it.

### 2.1 Method: remove the context, then detect
1. **Context profile:** for every context group (e.g. station XK0001A in January) the median and the MAD of
   log(1 + value) of every pollutant: what is usual for that station in that month. The log makes the skewed values
   more symmetric; the median and the MAD are not pulled by the outliers of the group.
2. **Residual:** every value becomes a robust z-score within its group, (log value − median) / (1.4826 · MAD), i.e.
   how unusual it is *for its context*.
3. **Detection on the residuals** with two methods: the **contextual robust z-score** (largest \\|residual\\| of the
   day, statistical, one pollutant at a time) and **LOF on the residual vectors** (the best point method of section
   1, all pollutants together), with the same {O["threshold_percentile"]:g}th-percentile threshold.

The context granularity is the main parameter: **none** (global, as in section 1), **station**, **station × season**
or **station × month**; for LOF also n_neighbors ({", ".join(map(str, O["contextual_lof_k"]))}).

### 2.2 Evaluation with injected contextual anomalies
As in section 1, {round(O["injected_share"] * 100):g}% known anomalies are added to a copy of the data, now of two
contextual types: **wrong season** (the real values of the same station from a day 150–215 days away, e.g. summer
values in winter) and **wrong station** (the real values of another station on the same date). Every injected day is
a real day, so its values are normal for the data set as a whole.

Contextual robust z-score, F1 by context:
{md(r.robust)}
LOF on the residuals, F1 by n_neighbors (rows) and context (columns):
{md(r.lof)}
The same for the threshold-free PR-AUC (a random score gets ≈ {O["injected_share"]:.2f}):
{md(r.lof_auc)}
{figs["context"]}
**Without context both methods find nothing** (best recall of any global setting: wrong season {glob["wrong season"]:.3f},
wrong station {glob["wrong station"]:.3f}; the global robust z-score has ROC-AUC {none["roc_auc"]:.2f} and PR-AUC
{none["pr_auc"]:.3f}, exactly the values of a random score): this confirms the lecture – a contextual anomaly cannot be detected by
looking at the values alone. The station alone already helps, adding the season or the month helps much more, and
**station × month** gives the best result. LOF on the residuals is clearly better than
the robust z-score (F1 {best["f1"]:.3f} vs {max(robust.values()):.3f}), because a wrong day changes several pollutants
at once only a little, and LOF sees the whole vector while the robust z-score sees one pollutant at a time.
Recall of the best setting of each method by type:
{md(r.by_type)}
**Choice: {best["method"]} on the residuals, context {best["context"]}{k_text}** (F1 = {best["f1"]:.3f}, PR-AUC =
{best["pr_auc"]:.3f}, {best["pr_auc"] / O["injected_share"]:.0f}× a random score). The highest PR-AUC is reached by
{r.best_auc["method"]}, context {r.best_auc["context"]}, n_neighbors = {r.best_auc["param"]} ({r.best_auc["pr_auc"]:.3f}),
{"the same context, so the threshold-free measure confirms the choice" if r.best_auc["context"] == best["context"]
 else "another context, which is discussed below"}. The F1 is lower
than for the point anomalies, because a real day from another month or station is often similar to a normal day of the
right context (e.g. April and May, or two background stations); only clearly different contexts are found.

### 2.3 Contextual anomalies in the real data
The chosen setting marks **{r.n_alarms} station-days**; **{r.n_only_contextual}** of them are *not* point anomalies of
section 1, so the context adds anomalies that the global methods do not see.
{figs["usual"]}
{low} of the {r.n_alarms} days are **lower** than usual for their context, the others higher. The most frequent
pollutant is shown below, and the most anomalies fall in month {peak["month"]} ({peak["days"]} days):
{md(r.by_driver)}
{md(r.by_month)}
The strongest contextual anomalies that are not point anomalies (`tables/o2_top_contextual.csv`):
{md(top)}
Two kinds stand out. **Too low for the context:** e.g. O3 of a few µg/m³ on a July day when 60–70 µg/m³ is usual, or
SO2 near zero at a station that usually measures 10–20 µg/m³. A low value is never a global outlier, but it is a
typical sign of an analyzer that stopped working properly (or a real, very clean day). **Too high for the context:**
e.g. SO2 or CO several times the usual summer value at an industrial station, which points to a local emission event.
Both kinds are flagged for a check, not removed.
"""
