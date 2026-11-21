"""o1 report section: outlier detection with KNN distance, LOF and K-Means (detection only, no removal)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o1_outlier_detection import FEATURES
from utils import CFG, md

O = CFG["outliers"]
EVAL_COLS = ["k", "precision", "recall", "f1", "f1_std"]


def knn_metric_comparison(knn: pl.DataFrame) -> tuple[pl.DataFrame, bool]:
    """F1 of Euclidean and Manhattan distance side by side, and whether Euclidean is better for every k."""
    wide = knn.pivot(on="metric", index="k", values="f1").sort("k")
    return wide, bool((wide["euclidean"] > wide["manhattan"]).all())


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    """Markdown text of the outlier detection section; every number comes from the results of the step."""
    b = {row["method"]: row for row in r.best.iter_rows(named=True)}
    p, pct_out = O["threshold_percentile"], 100 - O["threshold_percentile"]
    lo, hi = O["injected_factor"]
    metrics, euclid_wins = knn_metric_comparison(r.knn)
    th = r.threshold.filter(pl.col("method") == "LOF").sort("percentile")
    th95, th99 = (th.filter(pl.col("percentile") == x).row(0, named=True) for x in (95, p))
    k_lof = r.lof.sort("k")
    top_episode = r.episodes.row(0, named=True) if r.episodes.height else None
    best_table = r.best.select("method", "metric", "k", "precision", "recall", "f1", "f1_std")
    lof_vs = {m: b["LOF"]["f1"] / b[m]["f1"] for m in ("KNN", "K-Means", "univariate")}
    top = r.top.head(15).with_columns(pl.col(*FEATURES, "driver_z", "score_knn", "score_lof", "score_kmeans")
                                      .round(2))
    episode_text = "" if top_episode is None else (
        f"The largest episode is **{top_episode['date']}**: {top_episode['stations']} stations at once, mean PM10 "
        f"{top_episode['PM10']:.0f} µg/m³ with a PM2.5/PM10 ratio of only {top_episode['pm25_pm10_ratio']:.2f} "
        f"(median of all station-days: {r.ratio_median:.2f}). Many stations together and a low share of fine "
        "particles is the pattern of transported coarse dust, not of local smoke from heating (which has a high "
        "ratio); such a day is a real event, not a sensor error.")
    return f"""
## 1. Outlier detection

An **outlier** is an observation that differs strongly from the rest of the data. It may be an error (a broken or
miscalibrated sensor) or a real but rare event (a pollution episode). This phase only **detects** outliers: every
station-day keeps its values, and the result is a score and a 0/1 flag per method
(`data/gold/outlier_flags.parquet`). Whether an outlier is removed or kept is a later decision that needs the reason
for the outlier.

Three methods from the exercises are applied, each with the same threshold rule (a point is an outlier when its score
is above the {p:g}th percentile of the scores):

| Method | Type | Score of a point | Parameter analysed |
|---|---|---|---|
| KNN distance | distance-based | mean distance to its k nearest neighbours, D_i = (1/k)·Σ d(i, NN_j) | k, distance metric |
| Local Outlier Factor (LOF) | density-based | local density of the neighbours / local density of the point | n_neighbors |
| K-Means | clustering-based | distance to the nearest centroid | K (number of clusters) |

### 1.1 Feature space
One point = one **station-day**: the daily means of {", ".join(FEATURES)} (step p4, measured values only, valid days
with ≥ {CFG["aggregation"]["min_hours_per_day"]} hours). NO and NOx are not used, because they repeat the
information of NO2 (NOx = NO2 + 1.53·NO, phase I). Only days with all six pollutants are kept, because a distance
needs every coordinate: **{r.n_days:,} station-days from {r.n_stations} stations**. Daily means are used instead of
hours because a day is the unit of the limit values and of the episodes, the hourly noise was already treated in
phase I, and the methods search the neighbours of every point, which is fast for 13,000 points but slow for the
4 million hourly values.

Before the distances are computed every pollutant is standardized with the **z-score** (StandardScaler, mean 0,
standard deviation 1), as in the exercise: CO is around 1 mg/m³ and PM10 up to 200 µg/m³, so without scaling the
distance would be decided almost only by PM10 and PM2.5.
{md(r.space)}

### 1.2 How the parameters are evaluated
The real data has no labels that say which days are anomalies. The approach of the exercises is therefore used: to a
copy of the data **known anomalies** are added, {r.n_injected} points ({O["injected_share"]:.0%}). Each one is a real
station-day in which one random pollutant is multiplied by a factor between {lo:g} and {hi:g}, as a miscalibrated
analyzer would do. The other five pollutants stay real, so many of these anomalies are hard: the changed value alone
may still be in the normal range, only the combination with the other pollutants is unusual. Every setting is
evaluated on {len(O["eval_seeds"])} different injections (seeds {", ".join(map(str, O["eval_seeds"]))}); the tables
show the mean and the standard deviation of F1 between them.

- **Precision** = TP / (TP + FP): of the points marked as outliers, the share that are injected anomalies;
- **Recall** = TP / (TP + FN): of the injected anomalies, the share that was found;
- **F1** = 2·P·R / (P + R): the balance of the two, used to choose the best parameter.

The precision is a lower bound: the real data also contains real outliers (pollution episodes), which are counted
as false positives although they are genuinely unusual. A univariate baseline (the largest |z| of the day over the
six pollutants) is evaluated in the same way, to see what the multivariate methods add.

### 1.3 KNN distance: choice of k and of the distance metric
The score is the mean distance to the k nearest neighbours (k + 1 neighbours are searched, because the nearest one is
the point itself). k = 1 is not used: two similar anomalies close to each other would hide each other.
{md(r.knn.select("metric", *EVAL_COLS))}
{figs["knn"]}
F1 falls as k grows: with many neighbours the score becomes the distance to the whole region, so an anomaly that sits
just outside a dense group looks as normal as a point on the edge of the group. **Euclidean versus Manhattan**
(the exercise task): {"Euclidean is better for every k" if euclid_wins else "neither metric is better for every k"}.
{md(metrics)}
The Manhattan distance adds the differences of all six pollutants, so the small natural differences of the five
unchanged pollutants weigh as much as the one changed pollutant; the Euclidean distance squares the differences and
lets the one large difference dominate, which is exactly the shape of these anomalies.
**Choice: k = {b["KNN"]["k"]}, {b["KNN"]["metric"]} distance** (F1 = {b["KNN"]["f1"]:.3f}).

### 1.4 Local Outlier Factor: choice of n_neighbors
LOF compares the local density of a point with the density of its neighbours: LOF ≈ 1 normal, LOF > 1 the point is in a
sparser region than its neighbours. Unlike the KNN distance it is relative, which matters here because the stations
have very different levels: a day that is normal for an industrial station in Obiliq can be unusual for a background
station, and LOF judges every point by its own neighbourhood.
{md(r.lof.select(*EVAL_COLS))}
{figs["lof"]}
With n_neighbors = {k_lof["k"][0]} the density is estimated from too few points and is noisy (F1 = {k_lof["f1"][0]:.3f});
between 10 and 50 the F1 is almost constant, and with {k_lof["k"][-1]} neighbours the neighbourhood mixes different
groups and F1 falls again ({k_lof["f1"][-1]:.3f}). As in the exercise with make_blobs, the result is not very sensitive
to the exact value once the neighbourhood is large enough. **Choice: n_neighbors = {b["LOF"]["k"]}** (F1 =
{b["LOF"]["f1"]:.3f}, the smallest of the values with the highest F1, with the smallest spread between the injections).

### 1.5 K-Means: choice of K
Every point is assigned to the nearest of K centroids; the score is the distance to it. The inertia (sum of squared
distances to the own centroid) is used for the elbow method.
{md(r.kmeans.select("k", "inertia", "precision", "recall", "f1", "f1_std"))}
{figs["kmeans"]}
The inertia falls steadily without a clear elbow: the station-days do not form a few separate groups but a continuous
cloud (from clean summer days to polluted winter days). With few clusters a centroid stands in the middle of a large,
spread group, so the most distant points are simply the most polluted days and the hard anomalies are missed
(the same effect as K = 2 in the exercise). More clusters describe the cloud better and F1 grows, but slowly.
**Choice: K = {b["K-Means"]["k"]}** (F1 = {b["K-Means"]["f1"]:.3f}); even so K-Means stays the weakest of the three
methods, because a centroid describes a group by a single point and cannot follow its shape.

### 1.6 Choice of the threshold
The exercises use the 95th percentile (5% outliers). For the best setting of every method:
{md(r.threshold)}
{figs["threshold"]}
A lower threshold finds more anomalies but marks many normal days (LOF at the 95th percentile: recall
{th95["recall"]:.2f}, precision {th95["precision"]:.2f}); a higher threshold marks fewer but more certain days.
**Choice: the {p:g}th percentile** (top {pct_out:g}%, about {round(r.n_days * pct_out / 100)} station-days per method),
where precision and recall are balanced (LOF: {th99["precision"]:.2f} / {th99["recall"]:.2f}). It is also a number of
days that a person can still check one by one. The 95th percentile of the exercise would mark about
{round(r.n_days * 0.05)} days, most of them ordinary winter days.

### 1.7 Comparison of the methods
Best setting of every method at the {p:g}th percentile:
{md(best_table)}
**LOF is the best method** (F1 {b["LOF"]["f1"]:.3f}, {lof_vs["KNN"]:.1f}× KNN, {lof_vs["K-Means"]:.1f}× K-Means and
{lof_vs["univariate"]:.1f}× the univariate baseline). All three multivariate methods beat the univariate rule,
because they compare the whole day and not each pollutant alone. Recall by the size of the error and by the changed
pollutant:
{md(r.by_factor)}
{md(r.by_pollutant)}
Larger errors are found more easily by every method. Errors in PM2.5, PM10 and O3 are found best: PM2.5 and PM10 are
strongly correlated and O3 is inversely related to NO2, so a change in one of them breaks a relation that the methods
see. Errors in CO and SO2 are found worst, because these pollutants already vary strongly from station to station
(SO2 near the power plants, CO from heating), so ×{lo:g}–{hi:g} is still inside their normal spread.

### 1.8 Outliers in the real data
The three methods run with their chosen parameters on the {r.n_days:,} real station-days (no injected points):
{figs["pca"]}
Number of methods that flag the same day, and the overlap of every pair of methods:
{md(r.n_by_methods)}
{md(r.agreement)}
The methods agree only partly, because they define "different" differently (far away, less dense, far from a
centroid). A day flagged by **at least two methods** is treated as a reliable outlier: **{r.n_consensus} station-days**.
Of them, {r.n_multivariate_only} have no pollutant with |z| > 3, so they are outliers only because of the combination
of the pollutants and a univariate rule (|z| > 3 for one pollutant) would not find them.

Per station (share of its days) and per month:
{md(r.by_station)}
{figs["station_month"]}
The pollutant with the largest |z| on these days (the one that pulls the day away from the others):
{md(r.by_driver)}
Dates on which the outliers appear at several stations at once:
{md(r.episodes, 10)}
{episode_text}

The strongest outliers (flagged by all three methods, sorted by LOF; all in `tables/o1_top_outliers.csv`):
{md(top)}
The list contains both kinds of outliers: **real events** (regional dust and winter smog days on several stations at
once; SO2 episodes at a single station, e.g. Mitrovica in May 2020, while NO2 and CO stay normal, which points to
one local source) and **suspicious values**
(e.g. a daily CO mean of exactly 0 on a day with very high particles and NO2, which a working CO analyzer would not
measure). This is why the outliers are only flagged: deleting them would remove the most important pollution
episodes of the data set together with the errors.

### 1.9 Decisions
| Decision | Reason |
|---|---|
| Station-days of six standardized daily means | the unit of the episodes and of the limit values; z-score so that every pollutant counts the same |
| Parameters chosen by F1 on injected anomalies, mean of {len(O["eval_seeds"])} injections | the real data has no labels; several injections make the choice independent of one random draw |
| KNN k = {b["KNN"]["k"]} ({b["KNN"]["metric"]}), LOF n_neighbors = {b["LOF"]["k"]}, K-Means K = {b["K-Means"]["k"]} | highest F1 of every method (sections 1.3–1.5) |
| Threshold: {p:g}th percentile | balance of precision and recall; a number of days that can be checked |
| An outlier = flagged by ≥ 2 methods | the methods see different kinds of outliers; agreement makes the flag more reliable |
| Outliers are flagged, not removed | many are real pollution episodes; removal needs the reason for every outlier |
"""
