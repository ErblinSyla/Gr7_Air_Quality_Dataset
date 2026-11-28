"""o1 – Point anomalies: station-days whose combination of six daily means differs from the rest of the data.

One method from every family of the lectures is applied and its parameters are analysed:
- proximity-based: KNN distance (exercise: mean distance to k neighbours; lecture: distance to the k-th neighbour);
- density-based: Local Outlier Factor (LOF);
- clustering-based: K-Means (distance to the nearest centroid);
- statistical: Mahalanobis distance (classic and robust covariance), with the univariate z-score as baseline;
- boundary-based: one-class SVM.

Evaluation: as in the exercises, known anomalies of three types (gain error, sensor at zero, PM2.5 > PM10) are added
to a copy of the data, so precision, recall and F1 can be measured for every setting. Detection: every method runs
with its best setting on the real data; scores and 0/1 flags are stored in data/gold/anomalies_point.parquet.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from steps.second_phase.common import (
    FEATURES,
    O,
    best_per,
    flag,
    load_station_days,
    mean_over_seeds,
    precision_recall_f1,
    ranking_metrics,
    standardize,
)
from utils import table, write

METHODS = ["KNN", "LOF", "K-Means", "Mahalanobis", "One-class SVM"]
BASELINE = "z-score"
MAHALANOBIS = ["classic", "robust (MCD)", "classic, log values"]
INJECTED = ["gain error", "sensor at zero", "PM2.5 > PM10"]
RULES = {  # how the flags of the five methods are combined into one decision (evaluated in evaluate_rules)
    "LOF only": lambda f: f["LOF"],
    "LOF or Mahalanobis": lambda f: f["LOF"] | f["Mahalanobis"],
    "≥ 2 of 5 methods": lambda f: sum(f.values()) >= 2,
    "≥ 3 of 5 methods": lambda f: sum(f.values()) >= 3,
}
KEY = ["method", "variant", "param"]


# ---------------------------------------------------------------- injected anomalies -------------------------------

def inject_anomalies(x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Add known point anomalies to a copy of the data (the idea of the 5 anomalous customers in the exercise).

    `injected_share` of the rows are copied from random real station-days and changed in one of three ways, each the
    trace of a typical instrument fault:
    - gain error: one random pollutant × a factor in `injected_factor` (wrong calibration);
    - sensor at zero: one random pollutant that was above its median is set to 0 (the analyzer reports nothing);
    - PM2.5 > PM10: PM2.5 is set to 1.2–2 × PM10, which is physically impossible (PM2.5 is a part of PM10).
    The other pollutants of the day stay real, so most anomalies are *contextual within the day*: the changed value
    alone may look normal, only its combination with the other pollutants is unusual.

    Returns the data with the anomalies appended at the end, the true labels and one row per anomaly (type,
    pollutant, factor).
    """
    rng = np.random.default_rng(seed)
    n = round(O["injected_share"] * len(x))
    medians = np.median(x, axis=0)
    pm10, pm25 = FEATURES.index("PM10"), FEATURES.index("PM2_5")
    fake, info = [], []
    for row in rng.choice(len(x), 3 * n, replace=False):
        kind = INJECTED[len(fake) % len(INJECTED)]
        day, col, factor = x[row].copy(), int(rng.integers(len(FEATURES))), float("nan")
        if kind == "gain error":
            factor = rng.uniform(*O["injected_factor"])
            day[col] *= factor
        elif kind == "sensor at zero":
            if day[col] <= medians[col]:
                continue
            day[col] = 0.0
        else:
            col, factor = pm25, rng.uniform(1.2, 2.0)
            day[pm25] = day[pm10] * factor
        fake.append(day)
        info.append({"type": kind, "pollutant": FEATURES[col], "factor": factor})
        if len(fake) == n:
            break
    labels = np.r_[np.zeros(len(x), bool), np.ones(n, bool)]
    return np.vstack([x, fake]), labels, pl.DataFrame(info)


# ---------------------------------------------------------------- methods ------------------------------------------

def knn_score(z: np.ndarray, k: int, metric: str = "euclidean", score: str = "mean") -> np.ndarray:
    """Proximity-based score from the k nearest neighbours.

    score="mean" (exercise): the mean distance to the k nearest neighbours, D_i = (1/k) · Σ_j d(i, NN_j);
    score="kth" (lecture): the distance to the k-th nearest neighbour only.
    `n_neighbors = k + 1` because the nearest "neighbour" of every point is the point itself (distance 0), which is
    dropped. A large score means that the point is far from the others → candidate outlier.
    `metric` is "euclidean" (straight line, √Σ(xᵢ − yᵢ)²) or "manhattan" (Σ|xᵢ − yᵢ|), as in the exercise.
    """
    distances, _ = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(z).kneighbors(z)
    return distances[:, k] if score == "kth" else distances[:, 1:].mean(axis=1)


def lof_score(z: np.ndarray, k: int) -> np.ndarray:
    """Local Outlier Factor: the local density of a point compared with the local density of its k neighbours.

    LOF ≈ 1: the point is as dense as its neighbours (normal); LOF > 1: the point lies in a sparser region than its
    neighbours (outlier). Unlike the KNN distance, LOF is relative, so a point next to a dense cluster can be an
    outlier even if it is not far away in absolute terms. scikit-learn stores −LOF, so the sign is changed.
    """
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(z)
    return -lof.negative_outlier_factor_


def kmeans_score(z: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """K-Means score: the distance of every point to the nearest of the K centroids.

    The points are grouped into K clusters; a point far from every centroid does not fit any group → candidate
    outlier. Also returns the inertia (sum of squared distances to the own centroid), used for the elbow method.
    """
    km = KMeans(n_clusters=k, random_state=O["seed"], n_init=10).fit(z)
    return cdist(z, km.cluster_centers_).min(axis=1), float(km.inertia_)


def mahalanobis_score(x: np.ndarray, variant: str = "classic") -> np.ndarray:
    """Statistical score: the (squared) Mahalanobis distance to the mean, (x − x̄) S⁻¹ (x − x̄)ᵀ.

    Unlike the Euclidean distance it uses the covariance matrix S, so a day that goes *against* the correlation of the
    pollutants (e.g. high PM2.5 with low PM10) is far even if every value is normal. It assumes one Gaussian cloud.
    variant: "classic" (mean and covariance of all days), "robust (MCD)" (Minimum Covariance Determinant: mean and
    covariance of the most central days, so the outliers do not pull the estimate) or "classic, log values" (classic on
    log(1 + x), which makes the skewed pollutant distributions closer to Gaussian).
    """
    z = standardize(np.log1p(x) if variant.endswith("log values") else x)
    estimator = MinCovDet(random_state=O["seed"]) if variant.startswith("robust") else EmpiricalCovariance()
    return estimator.fit(z).mahalanobis(z)


def ocsvm_score(z: np.ndarray, gamma: float, nu: float) -> np.ndarray:
    """One-class SVM score: how far a point lies outside the boundary that encloses the normal data.

    The RBF kernel lets the boundary follow any shape (several groups of stations and seasons). gamma sets how
    tightly it follows the data (small → smooth, almost an ellipse; large → wraps every small group); nu is the upper
    limit of the share of points allowed outside the boundary. The score is −decision_function (larger = more outside).
    """
    return -OneClassSVM(kernel="rbf", gamma=gamma, nu=nu).fit(z).decision_function(z)


def zscore_score(z: np.ndarray) -> np.ndarray:
    """Statistical univariate baseline: the largest |z-score| of the row over all pollutants.

    It finds a day only if at least one pollutant is extreme by itself; it cannot see unusual combinations.
    """
    return np.abs(z).max(axis=1)


def method_score(method: str, variant: str, param: float, x: np.ndarray) -> np.ndarray:
    """Score of one setting (method, variant, numeric parameter) on the raw data x; used for evaluation and detection.

    KNN: variant "metric, mean|kth", param k; LOF: param n_neighbors; K-Means: param K; Mahalanobis: variant = type
    of covariance; one-class SVM: variant "nu=…", param gamma; z-score: no parameter.
    """
    z = standardize(x)
    if method == "KNN":
        metric, score = variant.split(", ")
        return knn_score(z, int(param), metric, score)
    if method == "LOF":
        return lof_score(z, int(param))
    if method == "K-Means":
        return kmeans_score(z, int(param))[0]
    if method == "Mahalanobis":
        return mahalanobis_score(x, variant)
    if method == "One-class SVM":
        return ocsvm_score(z, param, float(variant.removeprefix("nu=")))
    return zscore_score(z)


def grubbs_critical(n: int, alpha: float = 0.05) -> float:
    """Critical value of Grubbs' test (lecture): G_crit = (N − 1)/√N · √(t² / (N − 2 + t²)), t = t_{α/(2N), N−2}.

    Used only to show what the test becomes for a large sample: for N = 10 G_crit = 2.29 (lecture example), for
    thousands of days it is a fixed |z| limit of about 4.5–5.
    """
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))


# ---------------------------------------------------------------- evaluation ---------------------------------------

def settings() -> list[tuple[str, str, float]]:
    """Every (method, variant, parameter) combination of the config that is evaluated."""
    grid = [("KNN", f"{m}, {s}", k) for m in O["knn_metrics"] for s in O["knn_scores"] for k in O["knn_k"]]
    grid += [("LOF", "-", k) for k in O["lof_k"]]
    grid += [("K-Means", "-", k) for k in O["kmeans_k"]]
    grid += [("Mahalanobis", v, 0) for v in MAHALANOBIS]
    grid += [("One-class SVM", f"nu={nu}", g) for nu in O["ocsvm_nu"] for g in O["ocsvm_gamma"]]
    grid.append((BASELINE, "-", 0))
    return [(m, v, float(p)) for m, v, p in grid]


def evaluate(x: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate every setting on `eval_seeds` different injections and every threshold percentile.

    Returns (1) one row per seed × setting × percentile with precision, recall and F1 at that threshold and the
    threshold-free PR-AUC and ROC-AUC of the score, and (2) one row per injected anomaly and setting with the flag at
    the main threshold, used to see which anomalies are found.
    """
    rows, found = [], []
    for seed in O["eval_seeds"]:
        x_eval, actual, info = inject_anomalies(x, seed)
        for method, variant, param in settings():
            score = method_score(method, variant, param, x_eval)
            setting = {"method": method, "variant": variant, "param": param}
            ranking = ranking_metrics(actual, score)
            rows += [{"seed": seed, **setting, "percentile": p, **precision_recall_f1(actual, flag(score, p)), **ranking}
                     for p in O["threshold_percentiles"]]
            found.append(info.with_columns(*[pl.lit(v).alias(c) for c, v in setting.items()],
                                           pl.Series("detected", flag(score, O["threshold_percentile"])[actual])))
    return pl.DataFrame(rows), pl.concat(found)


def best_settings(summary: pl.DataFrame) -> pl.DataFrame:
    """For every method the setting with the highest mean F1 at the main threshold (smaller parameter on a tie)."""
    main = summary.filter(pl.col("percentile") == O["threshold_percentile"])
    order = {m: i for i, m in enumerate([*METHODS, BASELINE])}
    return best_per(main, "method", "param").sort(pl.col("method").replace_strict(order))


def evaluate_rules(x: np.ndarray, best: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate the ways of combining the five methods (`RULES`) on the same injections as the single methods.

    Every method uses its best setting and the main threshold; returns precision, recall and F1 of every rule and
    its recall for every type of injected anomaly.
    """
    chosen = {r["method"]: r for r in best.iter_rows(named=True)}
    rows, by_type = [], []
    for seed in O["eval_seeds"]:
        x_eval, actual, info = inject_anomalies(x, seed)
        flags = {m: flag(method_score(m, chosen[m]["variant"], chosen[m]["param"], x_eval), O["threshold_percentile"])
                 for m in METHODS}
        for rule, combine in RULES.items():
            predicted = combine(flags)
            rows.append({"seed": seed, "rule": rule, "flagged_pct": 100 * predicted.mean(),
                         **precision_recall_f1(actual, predicted)})
            by_type.append(info.with_columns(pl.lit(rule).alias("rule"), pl.Series("detected", predicted[actual])))
    summary = (pl.DataFrame(rows).group_by("rule", maintain_order=True)
               .agg(pl.col("flagged_pct", "precision", "recall", "f1").mean(), pl.col("f1").std().alias("f1_std")))
    recall = (pl.concat(by_type).group_by("rule", "type", maintain_order=True)
              .agg(pl.col("detected").mean()).pivot(on="type", index="rule", values="detected"))
    return summary, recall


def recall_by(found: pl.DataFrame, best: pl.DataFrame, by: str) -> pl.DataFrame:
    """Recall of the best setting of every method, split by the type or the changed pollutant of the anomaly."""
    return (found.join(best.select(KEY), on=KEY)
            .group_by(by, "method").agg(pl.col("detected").mean().alias("recall"))
            .pivot(on="method", index=by, values="recall").select(by, *METHODS, BASELINE).sort(by))


def elbow(x: np.ndarray) -> pl.DataFrame:
    """Inertia of K-Means on the real data for every K of the config (elbow method)."""
    z = standardize(x)
    return pl.DataFrame([{"param": float(k), "inertia": kmeans_score(z, k)[1]} for k in O["kmeans_k"]])


def chi2_threshold(x: np.ndarray) -> pl.DataFrame:
    """Share of the real days above the χ² threshold of the Mahalanobis distance.

    If the data were Gaussian, the squared Mahalanobis distance would follow a χ² distribution with 6 degrees of
    freedom, and a threshold χ²(1 − α) would mark a share α of the days. A much larger share shows that the
    Gaussian assumption does not hold (the lecture's argument for percentiles instead of a static threshold).
    """
    rows = []
    for variant in MAHALANOBIS[:2]:
        d2 = mahalanobis_score(x, variant)
        for alpha in O["chi2_alpha"]:
            limit = stats.chi2.ppf(1 - alpha, df=x.shape[1])
            rows.append({"covariance": variant, "alpha": alpha, "chi2_threshold": limit,
                         "expected_pct": 100 * alpha, "flagged_pct": 100 * float((d2 > limit).mean())})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- detection on the real data -----------------------

def auc_check(main: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    """Compare the choice by F1 (at the main threshold) with the choice by PR-AUC (threshold-free), per method.

    If both measures choose the same setting and rank the methods in the same order, the decision does not depend on
    the chosen threshold."""
    by_auc = (main.sort(pl.col("pr_auc").round(6), "param", descending=[True, False])
              .group_by("method", maintain_order=True).first()
              .select("method", pl.col("variant").alias("pr_auc_variant"), pl.col("param").alias("pr_auc_param"),
                      pl.col("pr_auc").alias("best_pr_auc")))
    chosen = best.select("method", pl.col("variant").alias("f1_variant"), pl.col("param").alias("f1_param"), "f1",
                         "pr_auc", "roc_auc")
    same = (pl.col("f1_variant") == pl.col("pr_auc_variant")) & (pl.col("f1_param") == pl.col("pr_auc_param"))
    return chosen.join(by_auc, on="method", how="left").with_columns(same.alias("same_choice"))


def name(method: str) -> str:
    """Column-friendly name of a method: "K-Means" → "kmeans", "One-class SVM" → "ocsvm"."""
    return {"One-class SVM": "ocsvm"}.get(method, method.lower().replace("-", ""))


def detect(days: pl.DataFrame, best: pl.DataFrame, rule: str) -> pl.DataFrame:
    """Run every method with its best setting on the real station-days (no injected points).

    Adds the score and the 0/1 flag of every method, the number of methods that flag the day, the final decision of
    the chosen combination `rule` (point_anomaly) and the pollutant with the largest |z| (the one that pulls the day
    away from the rest). The original values are kept unchanged: this phase only detects.
    """
    x = days.select(FEATURES).to_numpy()
    z = standardize(x)
    chosen = {r["method"]: r for r in best.iter_rows(named=True)}
    scores = {m: method_score(m, chosen[m]["variant"], chosen[m]["param"], x) for m in METHODS}
    p = O["threshold_percentile"]
    strongest = np.abs(z).argmax(axis=1)
    pc = PCA(n_components=2).fit_transform(z)
    out = days.with_columns(
        *[pl.Series(f"score_{name(m)}", s) for m, s in scores.items()],
        *[pl.Series(f"outlier_{name(m)}", flag(s, p).astype(np.int8)) for m, s in scores.items()],
        pl.Series("driver", [FEATURES[i] for i in strongest]),
        pl.Series("driver_z", z[np.arange(len(z)), strongest]),
        pl.Series("pc1", pc[:, 0]), pl.Series("pc2", pc[:, 1]),
    )
    decision = RULES[rule]({m: flag(s, p) for m, s in scores.items()})
    return out.with_columns(pl.sum_horizontal(pl.selectors.starts_with("outlier_")).alias("n_methods"),
                            pl.Series("point_anomaly", decision))


def agreement(flags: pl.DataFrame) -> pl.DataFrame:
    """Overlap of the outliers of every pair of methods: common days and the Jaccard index |A ∩ B| / |A ∪ B|."""
    rows = []
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            fa, fb = flags[f"outlier_{name(a)}"] == 1, flags[f"outlier_{name(b)}"] == 1
            rows.append({"pair": f"{a} & {b}", "common": int((fa & fb).sum()),
                         "jaccard": (fa & fb).sum() / (fa | fb).sum()})
    return pl.DataFrame(rows)


def episodes(anomalies: pl.DataFrame) -> pl.DataFrame:
    """Dates on which the outliers appear at several stations at once.

    An outlier at many stations on the same day points to a regional event (weather, transported dust), an outlier
    at one station only to a local source or a sensor problem. The PM2.5/PM10 ratio helps to tell coarse dust
    (low ratio) from combustion smoke (high ratio).
    """
    return (anomalies.group_by("date")
            .agg(pl.len().alias("stations"), pl.col("PM10").mean(), pl.col("PM2_5").mean(),
                 (pl.col("PM2_5") / pl.col("PM10")).mean().alias("pm25_pm10_ratio"),
                 pl.col("driver").mode().first().alias("main_driver"))
            .filter(pl.col("stations") >= 2).sort("stations", "date", descending=[True, False]))


def run() -> SimpleNamespace:
    """o1 pipeline: feature space → evaluation of all settings → best settings → detection on the real data."""
    days = load_station_days()
    x = days.select(FEATURES).to_numpy()
    ev, found = evaluate(x)
    summary = mean_over_seeds(ev, [*KEY, "percentile"])
    main = summary.filter(pl.col("percentile") == O["threshold_percentile"]).drop("percentile")

    def of(method: str) -> pl.DataFrame:
        """Evaluation rows of one method at the main threshold."""
        return main.filter(pl.col("method") == method).drop("method")

    best = table(best_settings(summary), "o1_best_settings")
    rules, rules_by_type = evaluate_rules(x, best)
    rule = rules.sort(pl.col("f1").round(6), descending=True)["rule"][0]
    flags = detect(days, best, rule)
    write(flags, "gold", "anomalies_point")
    anomalies = flags.filter("point_anomaly")
    return SimpleNamespace(
        n_days=days.height, n_stations=days["station_id"].n_unique(),
        n_injected=round(O["injected_share"] * days.height),
        space=table(days.select(FEATURES).describe(), "o1_feature_space"),
        knn=table(of("KNN"), "o1_knn_evaluation"),
        lof=table(of("LOF"), "o1_lof_evaluation"),
        kmeans=table(of("K-Means").join(elbow(x), on="param"), "o1_kmeans_evaluation"),
        mahalanobis=table(of("Mahalanobis"), "o1_mahalanobis_evaluation"),
        chi2=table(chi2_threshold(x), "o1_mahalanobis_chi2"),
        ocsvm=table(of("One-class SVM"), "o1_ocsvm_evaluation"),
        threshold=table(summary.join(best.select(KEY), on=KEY).select("method", "percentile", "precision", "recall",
                                                                        "f1"), "o1_threshold_evaluation"),
        best=best, rule=rule,
        auc=table(auc_check(main, best), "o1_auc_check"),
        rules=table(rules, "o1_combination_rules"),
        rules_by_type=table(rules_by_type, "o1_combination_rules_by_type"),
        by_type=table(recall_by(found, best, "type"), "o1_recall_by_type"),
        by_pollutant=table(recall_by(found.filter(pl.col("type") != "PM2.5 > PM10"), best, "pollutant"),
                           "o1_recall_by_pollutant"),
        flags=flags,
        agreement=table(agreement(flags), "o1_agreement"),
        n_by_methods=table(flags.group_by("n_methods").len("days").sort("n_methods"), "o1_days_by_n_methods"),
        by_station=table(anomalies.group_by("station_id").len("outlier_days")
                         .join(flags.group_by("station_id").len("days"), on="station_id")
                         .with_columns((100 * pl.col("outlier_days") / pl.col("days")).alias("pct"))
                         .sort("station_id"), "o1_outliers_by_station"),
        by_driver=table(anomalies.group_by("driver").len("outlier_days").sort("outlier_days", descending=True),
                        "o1_outliers_by_driver"),
        top=table(anomalies.sort("n_methods", "score_lof", descending=True)
                  .select("station_id", "date", *FEATURES, "driver", "driver_z", "n_methods", "score_lof"),
                  "o1_top_outliers"),
        episodes=table(episodes(anomalies), "o1_outlier_episodes"),
        ratio_median=float((days["PM2_5"] / days["PM10"]).median()),
        grubbs_crit=grubbs_critical(days.height),
        n_above_grubbs=flags.filter(pl.col("driver_z").abs() > grubbs_critical(days.height)).height,
        skew={f: float(stats.skew(days[f].to_numpy())) for f in FEATURES},
        n_anomalies=anomalies.height,
        n_multivariate_only=anomalies.filter(pl.col("driver_z").abs() <= 3).height,
    )
