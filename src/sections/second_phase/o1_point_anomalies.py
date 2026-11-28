"""o1 report section: point anomalies of the station-days with five methods from the lectures and exercises."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o1_point_anomalies import BASELINE, FEATURES, INJECTED, METHODS
from utils import CFG, md

O = CFG["outliers"]
EVAL_COLS = ["precision", "recall", "f1", "f1_std", "pr_auc", "roc_auc"]
DUST_PAPER = "https://acp.copernicus.org/articles/22/4047/2022/"  # independent source for the March 2020 dust event


def with_k(d: pl.DataFrame, label: str = "k") -> pl.DataFrame:
    """Evaluation table with the numeric parameter shown as an integer (k, n_neighbors, K)."""
    return d.select(pl.col("param").cast(pl.Int64).alias(label), *EVAL_COLS)


def f1_pivot(d: pl.DataFrame, index: str) -> pl.DataFrame:
    """F1 with one row per parameter value and one column per variant (compact view of a grid)."""
    return d.pivot(on="variant", index="param", values="f1").sort("param").rename({"param": index})


def best_for(by_type: pl.DataFrame, kind: str) -> tuple[str, float]:
    """The method with the highest recall for one type of injected anomaly."""
    row = by_type.filter(pl.col("type") == kind).select(*METHODS).row(0, named=True)
    method = max(row, key=row.get)
    return method, row[method]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    """Markdown text of the point anomaly section; every number comes from the results of the step."""
    b = {row["method"]: row for row in r.best.iter_rows(named=True)}
    p, pct_out = O["threshold_percentile"], 100 - O["threshold_percentile"]
    lo, hi = O["injected_factor"]
    knn = f1_pivot(r.knn, "k").with_columns(pl.col("k").cast(pl.Int64))
    euclid_wins = bool((knn["euclidean, mean"] > knn["manhattan, mean"]).all())
    mean_wins = bool((knn["euclidean, mean"] >= knn["euclidean, kth"]).all())
    lof = r.lof.sort("param")
    maha = dict(r.mahalanobis.select("variant", "f1").iter_rows())
    chi2_pct = dict(r.chi2.filter(pl.col("alpha") == O["chi2_alpha"][0]).select("covariance", "flagged_pct")
                    .iter_rows())
    svm_worst = r.ocsvm.sort("f1").row(0, named=True)
    th = r.threshold.filter(pl.col("method") == "LOF").sort("percentile")
    th95, th99 = (th.filter(pl.col("percentile") == x).row(0, named=True) for x in (95, p))
    ranking = r.best.sort("f1", descending=True)
    by_type = {row["type"]: row for row in r.by_type.iter_rows(named=True)}
    rules = {row["rule"]: row for row in r.rules.iter_rows(named=True)}
    rules_type = {row["rule"]: row for row in r.rules_by_type.iter_rows(named=True)}
    union, majority = rules["LOF or Mahalanobis"], rules["≥ 3 of 5 methods"]
    best_pm, best_zero = best_for(r.by_type, "PM2.5 > PM10"), best_for(r.by_type, "sensor at zero")
    lof_types = [t for t in INJECTED if best_for(r.by_type, t)[0] == "LOF"]
    same = bool(r.auc["same_choice"].all())
    order_f1 = r.auc.sort("f1", descending=True)["method"].to_list()
    order_auc = r.auc.sort("pr_auc", descending=True)["method"].to_list()
    roc = r.auc.sort("roc_auc")
    auc_text = (("PR-AUC chooses **the same setting for every method**" if same else
                 "PR-AUC chooses a different setting for some methods") + " and ranks the methods "
                + ("**in the same order** as F1" if order_f1 == order_auc else "in a different order than F1")
                + f" ({' > '.join(order_auc)}; a random score would get {O['injected_share']:.2f}). The decision therefore"
                " does not depend on the chosen threshold. ROC-AUC gives the same picture but squeezes the methods into"
                f" a narrow range ({roc['roc_auc'][0]:.2f}–{roc['roc_auc'][-1]:.2f}) because of the many normal days,"
                " which is why PR-AUC is the more useful threshold-free measure here.")
    forest = b["Isolation Forest"]
    forest_grid = (r.iforest.pivot(on="variant", index="param", values="f1").sort("param")
                   .with_columns(pl.col("param").cast(pl.Int64)).rename({"param": "max_samples"}))
    sens = r.sensitivity
    sens_auc = sens.pivot(on="injected_pct", index="method", values="pr_auc")
    sens_p = sens.pivot(on="injected_pct", index="method", values="best_percentile")
    sens_auc = sens_auc.rename({c: f"PR-AUC at {c}%" for c in sens_auc.columns if c != "method"})
    sens_p = sens_p.rename({c: f"best percentile at {c}%" for c in sens_p.columns if c != "method"})
    leaders = sens.sort("pr_auc", descending=True).group_by("injected_pct", maintain_order=True).first()
    same_leader = leaders["method"].n_unique() == 1
    lof_moves = sens.filter(pl.col("method") == "LOF").sort("injected_pct")["best_percentile"].to_list()
    lof15 = r.lof_share.filter(pl.col("lof_above") == 1.5).row(0, named=True)
    top = r.top.head(15).with_columns(pl.col(*FEATURES, "driver_z", "score_lof").round(2))
    skewed = ", ".join(f"{f} {s:.1f}" for f, s in r.skew.items())
    episode = r.episodes.row(0, named=True) if r.episodes.height else None
    dust = episode is not None and str(episode["date"]).startswith("2020-03-2")
    episode_text = "" if episode is None else (
        f"The largest episode is **{episode['date']}**: {episode['stations']} stations at once, mean PM10 "
        f"{episode['PM10']:.0f} µg/m³ with a PM2.5/PM10 ratio of only {episode['pm25_pm10_ratio']:.2f} "
        f"(median of all station-days: {r.ratio_median:.2f}). Many stations together and a low share of fine "
        "particles is the pattern of transported coarse dust, not of local smoke from heating (which has a high "
        "ratio); such a day is a real event, not a sensor error." + (
            " **External check:** the episode is documented in the literature as an exceptional incursion of"
            " mineral dust from Central Asia (Aralkum desert) over southern Europe on 27–30 March 2020, traced with"
            f" the Copernicus CAMS dust forecasts ([Tositti et al., 2022, Atmos. Chem. Phys. 22, 4047]({DUST_PAPER}));"
            " the detection without any labels found exactly these days." if dust else ""))
    return f"""
## 1. Point anomalies

A **point anomaly** is a single observation that differs from the rest of the data (lecture: o₁, o₂ in the figure of
N₁ and N₂). Here an observation is one **station-day**, and it is anomalous when its combination of pollutant levels is
unusual. One method from each of the four families of the lecture, plus the one-class SVM, is applied with the same
threshold rule (a point is an outlier when its score is above the {p:g}th percentile of the scores). Isolation
Forest, the most widely used method in practice, and the univariate z-score are evaluated as comparisons, but they do
not take part in the final decision:

| Method | Family | Score of a point | Parameters analysed |
|---|---|---|---|
| KNN distance | proximity-based | mean distance to the k nearest neighbours (exercise) or distance to the k-th (lecture) | k, distance metric, mean vs k-th |
| Local Outlier Factor (LOF) | density-based | density of the neighbours / density of the point | n_neighbors |
| K-Means | clustering-based | distance to the nearest centroid | K, elbow |
| Mahalanobis distance | statistical (multivariate Gaussian) | (x − x̄) S⁻¹ (x − x̄)ᵀ | classic vs robust covariance, log values, χ² threshold |
| One-class SVM | boundary around the normal data | distance outside the learned boundary | gamma, nu |
| Isolation Forest (comparison) | isolation by random trees (not in the lectures) | how few random splits isolate the point | max_samples, n_estimators |
| z-score (baseline) | statistical (univariate) | largest \\|z\\| of the day over the pollutants | – |

### 1.1 Feature space
One point = one **station-day**: the daily means of {", ".join(FEATURES)} (step p4, measured values only, valid days
with ≥ {CFG["aggregation"]["min_hours_per_day"]} hours). NO and NOx are not used, because they repeat the
information of NO2 (NOx = NO2 + 1.53·NO, phase I). Only days with all six pollutants are kept, because a distance
needs every coordinate: **{r.n_days:,} station-days from {r.n_stations} stations**. Daily means are used because a day
is the unit of the limit values and of the episodes, and the distance-based methods have O(n²) complexity (lecture);
the hourly series are analysed separately for temporal anomalies (section 3).

Every pollutant is standardized with the **z-score** (StandardScaler, mean 0, standard deviation 1), as in the
exercise: CO is around 1 mg/m³ and PM10 up to 200 µg/m³, so without scaling the distance would be decided almost only
by PM10 and PM2.5. All pollutants are right-skewed (skewness {skewed}), which matters for the methods that assume a
Gaussian distribution.
{md(r.space)}

### 1.2 How the parameters are evaluated
The real data has no labels that say which days are anomalies (unsupervised setting). The approach of the exercises
is therefore used: to a copy of the data **known anomalies** are added, {r.n_injected} points
({O["injected_share"]:.0%}), each a real station-day changed in one of three ways that a faulty instrument produces:

| Injected type | Change | Why it is hard |
|---|---|---|
| gain error | one random pollutant × {lo:g}–{hi:g} (wrong calibration) | the new value is often still in the normal range |
| sensor at zero | one pollutant that was above its median is set to 0 | a low value is never extreme in skewed data |
| PM2.5 > PM10 | PM2.5 = 1.2–2 × PM10 (physically impossible) | both values stay in their normal range |

The other pollutants stay real, so the anomalies can only be found through the *combination* of the values. Every
setting is evaluated on {len(O["eval_seeds"])} different injections (seeds {", ".join(map(str, O["eval_seeds"]))});
the tables show the mean and the standard deviation of F1 between them.

- **Precision** = TP / (TP + FP): of the points marked as outliers, the share that are injected anomalies;
- **Recall** = TP / (TP + FN): of the injected anomalies, the share that was found;
- **F1** = 2·P·R / (P + R): the balance of the two, used to choose the best parameter.

Precision, recall and F1 depend on the threshold. Two threshold-free measures check the ranking of the score itself:
- **PR-AUC** (average precision): the mean precision over all recall levels, i.e. how well the anomalies are
  concentrated at the top of the ranking. A random score gets the share of anomalies (≈ {O["injected_share"]:.2f}), so
  it is the more informative measure for rare anomalies;
- **ROC-AUC**: the probability that a random anomaly gets a higher score than a random normal day (0.5 = random,
  1 = perfect). With 99% normal days it is easily high, so it is reported but not used for decisions.

The precision is a lower bound: the real data also contains real outliers (pollution episodes), which are counted
as false positives although they are genuinely unusual.

### 1.3 KNN distance (proximity-based): k, metric and score
The exercise uses the mean distance to the k nearest neighbours, the lecture the distance to the k-th neighbour; both
are compared, with the Euclidean and the Manhattan distance (the exercise task). k + 1 neighbours are searched,
because the nearest one is the point itself; k = 1 is not used, because two similar anomalies would hide each other
(lecture figure 10.5). F1 for every k and variant (precision and recall in `tables/o1_knn_evaluation.csv`):
{md(knn)}
{figs["knn"]}
F1 falls as k grows: with many neighbours the score becomes the distance to the whole region, so an anomaly just
outside a dense group looks as normal as a point on the edge of the group (the score is sensitive to k, as the lecture
warns). **Euclidean versus Manhattan:** {"Euclidean is better for every k" if euclid_wins else
"neither metric is better for every k"}: the Manhattan distance adds the differences of all six pollutants, so the
small natural differences of the five unchanged pollutants weigh as much as the one changed pollutant, while the
Euclidean distance squares the differences and lets the one large difference dominate. **Mean versus k-th
neighbour:** {"the mean is equal or better for every k" if mean_wins else "neither is better for every k"}; the mean of
k distances is less noisy than one single distance.
**Choice: k = {b["KNN"]["param"]:.0f}, {b["KNN"]["variant"]}** (F1 = {b["KNN"]["f1"]:.3f}).

### 1.4 Local Outlier Factor (density-based): n_neighbors
LOF compares the density of a point (reciprocal of the mean distance to its k neighbours) with the density of its
neighbours: LOF ≈ 1 normal, LOF ≫ 1 the point is much sparser than its neighbours. Unlike the KNN distance it is
relative, which matters here because the stations have very different levels (variable density): a day that is normal
for an industrial station in Obiliq can be unusual for a background station, and LOF judges every point by its own
neighbourhood.
{md(with_k(r.lof, "n_neighbors"))}
{figs["lof"]}
F1 is highest at n_neighbors = {b["LOF"]["param"]:.0f}. With fewer neighbours the density is estimated from too few
points (F1 {lof["f1"][0]:.3f} at {lof["param"][0]:.0f}), with more the neighbourhood mixes different groups of
stations and seasons and the local view is lost (F1 {lof["f1"][-1]:.3f} at {lof["param"][-1]:.0f}).
**Choice: n_neighbors = {b["LOF"]["param"]:.0f}** (F1 = {b["LOF"]["f1"]:.3f}).

### 1.5 K-Means (clustering-based): K
Every point is assigned to the nearest of K centroids; the score is the distance to it. The inertia (sum of squared
distances to the own centroid, real data) is used for the elbow method.
{md(r.kmeans.select(pl.col("param").cast(pl.Int64).alias("K"), "inertia", *EVAL_COLS))}
{figs["kmeans"]}
The inertia falls steadily without a clear elbow: the station-days do not form a few separate groups but a continuous
cloud (from clean summer days to polluted winter days). With few clusters a centroid stands in the middle of a large,
spread group, so the most distant points are simply the most polluted days and the anomalies are missed (the same
effect as K = 2 in the exercise; lecture: sensitive to the number of clusters). More clusters describe the cloud
better and F1 grows slowly, but even the best K stays far below the other methods, because a centroid describes a
group by a single point and cannot follow its shape. **Choice: K = {b["K-Means"]["param"]:.0f}** (F1 =
{b["K-Means"]["f1"]:.3f}).

### 1.6 Mahalanobis distance (statistical): covariance and threshold
The multivariate statistical approach of the lecture: the squared distance (x − x̄) S⁻¹ (x − x̄)ᵀ uses the covariance
matrix S, so a day that goes *against* the correlation of the pollutants is far even if every value is normal. Three
variants are compared: **classic** (mean and covariance of all days), **robust (MCD)** (Minimum Covariance
Determinant: estimated from the most central days only, the "robust covariance estimator" of the lecture) and
**classic on log values** (log(1 + x) makes the skewed distributions closer to the Gaussian that the method assumes).
{md(r.mahalanobis.select("variant", *EVAL_COLS))}
The robust variant is not better (F1 {maha["robust (MCD)"]:.3f}): the central days form a narrow ellipse, and in
skewed data every ordinary polluted day lies far outside it. The log transformation helps most
({maha["classic"]:.3f} → {maha["classic, log values"]:.3f}), because the method works best when its Gaussian
assumption is closer to the truth.

**Static χ² threshold versus percentile.** For Gaussian data the squared Mahalanobis distance follows a χ²
distribution with {len(FEATURES)} degrees of freedom, so the threshold χ²(1 − α) should flag a share α of the days:
{md(r.chi2)}
At α = {O["chi2_alpha"][0]} the classic distance flags {chi2_pct["classic"]:.1f}% and the robust one
{chi2_pct["robust (MCD)"]:.1f}% instead of {100 * O["chi2_alpha"][0]:g}%. The Gaussian assumption does not hold, so the
static threshold is not reliable and the percentile is used, as the lecture recommends for non-normal data.
**Choice: classic covariance on log values** (F1 = {b["Mahalanobis"]["f1"]:.3f}).

### 1.7 One-class SVM: gamma and nu
The one-class SVM learns a boundary that encloses as much data as possible with minimum volume; with the RBF kernel the
boundary can follow any shape (lecture: it adapts to several clusters, where a single Gaussian envelope fails).
**gamma** sets how tightly the boundary follows the data (small: smooth, almost an ellipse; large: it wraps every small
group), **nu** is the upper limit of the share of points allowed outside the boundary. F1 for every combination:
{md(f1_pivot(r.ocsvm, "gamma"))}
{figs["ocsvm"]}
The two parameters must fit together. A larger gamma lets the boundary follow the groups of stations and seasons and
F1 rises, but with a large gamma every support vector only "sees" its close surroundings, and a small nu allows only
few support vectors: most days are then far from all of them, get almost the same score, and the ranking collapses
(gamma = {svm_worst["param"]:g}, {svm_worst["variant"]}: F1 = {svm_worst["f1"]:.3f}). It is the method that is most
sensitive to its parameters. **Choice: gamma = {b["One-class SVM"]["param"]:g}, {b["One-class SVM"]["variant"]}**
(F1 = {b["One-class SVM"]["f1"]:.3f}).

### 1.8 Isolation Forest (comparison method, not in the lectures)
Isolation Forest is the most common anomaly detector in practice. Many random trees split the data at random values of
random pollutants; a point that is isolated after few splits is different from the rest. **max_samples** is the number
of days every tree sees, **n_estimators** the number of trees. F1 for every combination (all measures in
`tables/o1_iforest_evaluation.csv`):
{md(forest_grid)}
The number of trees hardly matters; more samples per tree help, because the trees then see more of the structure of
the data. Even the best setting ({forest["variant"]}, max_samples = {forest["param"]:.0f}) reaches only F1 =
{forest["f1"]:.3f} and PR-AUC = {forest["pr_auc"]:.3f}, far below LOF ({b["LOF"]["f1"]:.3f} / {b["LOF"]["pr_auc"]:.3f})
and close to the univariate z-score ({b[BASELINE]["f1"]:.3f}). The reason is how it splits: every split uses one
pollutant at a time, so a day is isolated quickly only when one value is extreme. The injected anomalies are
*combinations* of normal values (a zero, an impossible PM2.5/PM10 ratio), which random one-pollutant splits do not
separate. On the real data its outliers overlap little with those of LOF (Jaccard {r.iforest_lof_jaccard:.2f}). The
most popular method is therefore not the best one for this data, which is why every method was evaluated rather than
chosen by reputation.

### 1.9 Lecture methods that are not used
| Method | Why it is not used for this data |
|---|---|
| Grubbs' test | univariate and assumes a normal distribution, but the daily means are right-skewed; it is made for a few suspects in a small sample (one outlier at a time). For N = {r.n_days:,} the critical value is G = {r.grubbs_crit:.2f}, i.e. just a fixed \\|z\\| limit; {r.n_above_grubbs} days have a pollutant above it. This is the z-score baseline with another threshold. |
| Likelihood approach | needs a probability model M of the normal data; with a Gaussian M it ranks the days like the Mahalanobis distance (1.6), and a better M would need a mixture model, which is beyond the lectures |
| Density in a fixed radius (DBSCAN) | one radius for all points; the lecture notes that it works poorly with variable density, and LOF (1.4) is its relative version |
| Removing objects / discarding small clusters | needs thresholds for "small" and "far" and repeated clustering; the clustering family is represented by the K-Means distance (1.5) |

### 1.10 Choice of the threshold
The exercises use the 95th percentile (5% outliers). For the best setting of every method:
{md(r.threshold)}
{figs["threshold"]}
A lower threshold finds more anomalies but marks many normal days (LOF at the 95th percentile: recall
{th95["recall"]:.2f}, precision {th95["precision"]:.2f}); a higher threshold marks fewer but more certain days.
**Choice: the {p:g}th percentile** (top {pct_out:g}%, about {round(r.n_days * pct_out / 100)} station-days per method),
where precision and recall are balanced (LOF: {th99["precision"]:.2f} / {th99["recall"]:.2f}) and the number of days
can still be checked one by one. The 95th percentile of the exercise would mark about {round(r.n_days * 0.05)} days,
most of them ordinary winter days.

**Is the 99th percentile only chosen because 1% was injected?** Two checks separate the method choice from the
threshold. (1) The evaluation is repeated with {", ".join(f"{100 * x:g}%" for x in O["sensitivity_shares"])} injected
anomalies:
{md(sens_auc)}
{md(sens_p)}
The ranking of the methods {"does not change: LOF is the best method for every share" if same_leader else
"changes with the share"}, while the percentile with the highest F1 follows the injected share (LOF:
{" → ".join(f"{x:g}" for x in lof_moves)}). So the *method* is independent of the injection, and the *threshold*
cannot be learned from injected data – it has to be decided from the real data. (2) An estimate from the real data
alone: the common LOF limit 1.5 (LOF ≈ 1 means as dense as the neighbours) is exceeded by {lof15["days"]} station-days
({lof15["pct"]:.2f}%):
{md(r.lof_share)}
The real data therefore contains about 0.5–1% clear outliers. The 99th percentile (LOF ≥ {r.lof_cut:.2f}) agrees with
this and is slightly inclusive, which suits a screening step whose alarms are checked afterwards.

### 1.11 Comparison and final decision
Best setting of every method at the {p:g}th percentile:
{md(r.best.select("method", "variant", pl.col("param").alias("parameter"), *EVAL_COLS))}
Recall of every method by type of injected anomaly and by changed pollutant (gain error and sensor at zero):
{md(r.by_type)}
{md(r.by_pollutant)}
The methods see different anomalies. **{best_pm[0]}** finds the impossible PM2.5 > PM10 best (recall
{best_pm[1]:.2f}): the two values stay high together, i.e. along the PM10–PM2.5 correlation, so the Mahalanobis
distance hardly grows (recall {by_type["PM2.5 > PM10"]["Mahalanobis"]:.2f}), while LOF sees that such a ratio is locally
rare. **{best_zero[0]}** finds the sensor at zero best ({best_zero[1]:.2f}): on log values a zero lies far below every
usual value, while the univariate z-score almost never sees it ({by_type["sensor at zero"][BASELINE]:.2f}), because a
low value is never extreme in right-skewed data. Errors in CO and SO2 are hard for every method: these pollutants
already vary strongly between stations (SO2 near the power plants, CO from heating).

**Threshold-free check.** Is the result an effect of the 99th-percentile threshold? For every method the setting
chosen by F1 is compared with the setting that has the highest PR-AUC:
{md(r.auc)}
{auc_text}

**How to combine the methods.** Because the methods are complementary, four combination rules are evaluated on the
same injections:
{md(r.rules)}
{md(r.rules_by_type)}
Majority voting is clearly worse ({majority["rule"]}: F1 {majority["f1"]:.3f}): the weaker methods outvote LOF on the
anomalies that only LOF finds (PM2.5 > PM10 recall {rules_type["≥ 3 of 5 methods"]["PM2.5 > PM10"]:.2f}). LOF or
Mahalanobis finds the most anomalies (recall {union["recall"]:.2f}, sensor at zero
{rules_type["LOF or Mahalanobis"]["sensor at zero"]:.2f}) but marks {union["flagged_pct"]:.1f}% of the days with a
lower precision ({union["precision"]:.2f}).

**Final decision.** Ranking of the single methods by F1:
{", ".join(f"{row['method']} {row['f1']:.3f}" for row in ranking.iter_rows(named=True))}.
- **Point anomaly = {r.rule}** (highest F1 of all rules: {rules[r.rule]["f1"]:.3f}): LOF is the best method for
  {len(lof_types)} of the {len(INJECTED)} types ({", ".join(lof_types)}), its relative density fits data in which
  every station has its own level, and it is stable between the injections.
- If finding as many faults as possible matters more than false alarms (e.g. a data-quality check before an
  analysis), **LOF or Mahalanobis** is the better rule, because Mahalanobis adds the zero readings.
- K-Means and the univariate z-score are the weakest and are not needed for the decision; the comparison method
  Isolation Forest is also clearly weaker than LOF (1.8).

### 1.12 Point anomalies in the real data
The five methods run with their chosen settings on the {r.n_days:,} real station-days (no injected points):
{figs["pca"]}
Number of methods that flag the same day, and the overlap of every pair of methods:
{md(r.n_by_methods)}
{md(r.agreement)}
The methods agree only partly, because they define "different" differently (far away, less dense, far from a
centroid, against the correlation, outside a boundary). The final decision ({r.rule}) gives **{r.n_anomalies} point
anomalies**; {r.n_multivariate_only} of them have no pollutant with \\|z\\| > 3, so they are anomalies only because of
the combination of the pollutants and a univariate rule would not find them.

Per station (share of its days) and the pollutant with the largest \\|z\\| on these days:
{md(r.by_station)}
{md(r.by_driver)}
Dates on which the point anomalies appear at several stations at once:
{md(r.episodes, 10)}
{episode_text}

The strongest point anomalies (most methods first, then by LOF; all in `tables/o1_top_outliers.csv`):
{md(top)}
The list contains both kinds of outliers: **real events** (regional dust and winter smog days on several stations at
once; SO2 episodes at a single station, e.g. Mitrovica in May 2020, while NO2 and CO stay normal, which points to one
local source) and **suspicious values** (e.g. very high daily CO at a station while the particles are moderate), which
need a check of the instrument. This is why the anomalies are only flagged and not removed.
"""
