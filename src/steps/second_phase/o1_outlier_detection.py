"""o1 – Outlier detection with the three methods from the exercises: KNN distance, Local Outlier Factor (LOF) and
K-Means (distance to the nearest centroid).

Workflow (detection only, nothing is removed):
1. Feature space: one row per station-day, the daily means of six pollutants, standardized with z-score.
2. Evaluation: as in the exercises, known anomalies are added to a copy of the data, so precision, recall and F1 can
   be measured for every parameter value (k of KNN, n_neighbors of LOF, K of K-Means, distance metric, threshold).
3. Detection: every method runs with its best parameters on the real data; the scores and 0/1 flags are stored in
   data/gold/outlier_flags.parquet next to the original values.
"""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utils import CFG, SEASONS, scan, table, write

O = CFG["outliers"]
FEATURES = O["features"]
METHODS = ["KNN", "LOF", "K-Means"]


# ---------------------------------------------------------------- data ---------------------------------------------

def load_station_days() -> pl.DataFrame:
    """Build the feature space: one row per station-day with the daily means of the pollutants in `FEATURES`.

    Only measured (not imputed) and valid days (≥ 18 valid hours, step p4) are used, so the methods judge real
    measurements and not values created by imputation. Rows where one of the pollutants is missing are left out,
    because the distance between two points needs every coordinate.
    """
    daily = (scan("gold", "daily")
             .filter((pl.col("source") == "measured") & pl.col("valid_day"))
             .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "date", "mean")
             .collect())
    wide = daily.pivot(on="pollutant", index=["station_id", "date"], values="mean")
    return (wide.select("station_id", "date", *FEATURES)
            .drop_nulls(FEATURES)
            .with_columns(pl.col("date").dt.month().alias("month"))
            .with_columns(pl.col("month").replace_strict(SEASONS, return_dtype=pl.String).alias("season"))
            .sort("station_id", "date"))


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score standardization of every column: z = (x − mean) / std.

    Needed because the pollutants have very different scales (CO in mg/m³ around 1, PM10 in µg/m³ up to 200);
    without it the distance would be decided almost only by the pollutant with the largest numbers.
    """
    return StandardScaler().fit_transform(x)


def inject_anomalies(x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Add known anomalies to a copy of the data (the same idea as the 5 anomalous customers in the exercise).

    `injected_share` of the rows are copied from random real station-days and one random pollutant is multiplied by
    a factor between `injected_factor[0]` and `injected_factor[1]`, as a sensor with a wrong calibration would do.
    The other pollutants of that day stay real, so many anomalies are *contextual*: the value alone may look normal,
    only the combination with the other pollutants is unusual.

    Returns the data with the anomalies appended at the end, the true labels (True = injected) and a table that
    describes every injected anomaly (which pollutant, which factor).
    """
    rng = np.random.default_rng(seed)
    n = round(O["injected_share"] * len(x))
    rows = rng.choice(len(x), n, replace=False)
    cols = rng.integers(0, x.shape[1], n)
    factors = rng.uniform(*O["injected_factor"], n)
    fake = x[rows].copy()
    fake[np.arange(n), cols] *= factors
    labels = np.r_[np.zeros(len(x), bool), np.ones(n, bool)]
    info = pl.DataFrame({"pollutant": [FEATURES[c] for c in cols], "factor": factors})
    return np.vstack([x, fake]), labels, info


# ---------------------------------------------------------------- methods ------------------------------------------

def knn_score(z: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """KNN distance score: the mean distance of every point to its k nearest neighbours,
    D_i = (1/k) · Σ_j d(i, NN_j).

    `n_neighbors = k + 1` because the nearest "neighbour" of every point is the point itself (distance 0), which is
    dropped. A large D_i means that the point is far from all others (isolated) → candidate outlier.
    `metric` is "euclidean" (straight line, √Σ(xᵢ − yᵢ)²) or "manhattan" (Σ|xᵢ − yᵢ|), as in the exercise.
    """
    distances, _ = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(z).kneighbors(z)
    return distances[:, 1:].mean(axis=1)


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


def univariate_score(z: np.ndarray) -> np.ndarray:
    """Baseline without neighbours or clusters: the largest |z-score| of the row over all pollutants.

    It finds a day only if at least one pollutant is extreme by itself; it cannot see unusual combinations.
    """
    return np.abs(z).max(axis=1)


def flag(score: np.ndarray, percentile: float) -> np.ndarray:
    """Threshold rule of the exercises: a point is an outlier when its score is above the given percentile."""
    return score > np.percentile(score, percentile)


def precision_recall_f1(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Evaluation against the known anomalies.

    Precision = TP / (TP + FP): of the points marked as outliers, the share that really are anomalies.
    Recall    = TP / (TP + FN): of the real anomalies, the share that was found.
    F1        = 2 · P · R / (P + R): the harmonic mean, the balance of the two.
    """
    tp = int((actual & predicted).sum())
    precision = tp / predicted.sum() if predicted.sum() else 0.0
    recall = tp / actual.sum() if actual.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------- evaluation ---------------------------------------

def score_all_settings(z: np.ndarray) -> list[tuple[str, str, int, np.ndarray, float | None]]:
    """Compute the score of every method for every parameter value of the config.

    Returns (method, metric, k, score, inertia) tuples; `metric` is the distance metric for KNN and "euclidean"
    for LOF and K-Means; `inertia` is only defined for K-Means.
    """
    runs = [("KNN", m, k, knn_score(z, k, m), None) for m in O["knn_metrics"] for k in O["knn_k"]]
    runs += [("LOF", "euclidean", k, lof_score(z, k), None) for k in O["lof_k"]]
    runs += [("K-Means", "euclidean", k, *kmeans_score(z, k)) for k in O["kmeans_k"]]
    runs.append(("univariate", "-", 0, univariate_score(z), None))
    return runs


def evaluate(x: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate every setting on `eval_seeds` different injections and every threshold percentile.

    Returns (1) one row per seed × method × metric × k × percentile with precision, recall and F1 and (2) one row
    per injected anomaly with the flags of every setting (at the main threshold), used to see which anomalies are
    found.
    """
    rows, found = [], []
    for seed in O["eval_seeds"]:
        x_eval, actual, info = inject_anomalies(x, seed)
        for method, metric, k, score, inertia in score_all_settings(standardize(x_eval)):
            for p in O["threshold_percentiles"]:
                rows.append({"seed": seed, "method": method, "metric": metric, "k": k, "percentile": p,
                             "inertia": inertia, **precision_recall_f1(actual, flag(score, p))})
            detected = flag(score, O["threshold_percentile"])[actual]
            found.append(info.with_columns(pl.lit(seed).alias("seed"), pl.lit(method).alias("method"),
                                           pl.lit(metric).alias("metric"), pl.lit(k).alias("k"),
                                           pl.Series("detected", detected)))
    return pl.DataFrame(rows, infer_schema_length=None), pl.concat(found)


def mean_over_seeds(ev: pl.DataFrame) -> pl.DataFrame:
    """Average precision, recall, F1 (and inertia) over the injections of the different seeds."""
    return (ev.group_by("method", "metric", "k", "percentile", maintain_order=True)
            .agg(pl.col("precision", "recall", "f1", "inertia").mean(), pl.col("f1").std().alias("f1_std")))


def best_settings(summary: pl.DataFrame) -> pl.DataFrame:
    """For every method the parameter values with the highest mean F1 at the main threshold percentile.

    On a tie (equal F1) the smaller k is chosen, because it is the simpler model.
    """
    main = summary.filter(pl.col("percentile") == O["threshold_percentile"])
    return (main.sort(pl.col("f1").round(6), "k", descending=[True, False])
            .group_by("method", maintain_order=True).first()
            .sort(pl.col("method").replace_strict({m: i for i, m in enumerate([*METHODS, "univariate"])})))


def recall_by(found: pl.DataFrame, best: pl.DataFrame, by: str) -> pl.DataFrame:
    """Recall of the best setting of every method, split by a property of the injected anomaly (factor or pollutant).

    Shows which anomalies are easy (large factor) and which are hard to find (small factor, pollutant with a large
    natural spread).
    """
    chosen = found.join(best.select("method", "metric", "k"), on=["method", "metric", "k"])
    if by == "factor":
        edges = np.linspace(*O["injected_factor"], 4)
        labels = [f"×{a:.0f}–{b:.0f}" for a, b in pairwise(edges)]
        chosen = chosen.with_columns(pl.col("factor").cut(edges[1:-1].tolist(), labels=labels).cast(pl.String))
    return (chosen.group_by(by, "method").agg(pl.col("detected").mean().alias("recall"))
            .pivot(on="method", index=by, values="recall").select(by, *[*METHODS, "univariate"]).sort(by))


# ---------------------------------------------------------------- detection on the real data -----------------------

def name(method: str) -> str:
    """Column-friendly name of a method: "K-Means" → "kmeans"."""
    return method.lower().replace("-", "")


def detect(days: pl.DataFrame, best: pl.DataFrame) -> pl.DataFrame:
    """Run every method with its best parameters on the real station-days (no injected points).

    Adds the score and the 0/1 flag of every method, the number of methods that flag the day and the pollutant with
    the largest |z| (the one that pulls the day away from the rest). The original values are kept unchanged:
    this phase only detects.
    """
    z = standardize(days.select(FEATURES).to_numpy())
    cfg = {r["method"]: r for r in best.iter_rows(named=True)}
    scores = {"KNN": knn_score(z, cfg["KNN"]["k"], cfg["KNN"]["metric"]),
              "LOF": lof_score(z, cfg["LOF"]["k"]),
              "K-Means": kmeans_score(z, cfg["K-Means"]["k"])[0]}
    p = O["threshold_percentile"]
    out = days.with_columns(
        *[pl.Series(f"score_{name(m)}", s) for m, s in scores.items()],
        *[pl.Series(f"outlier_{name(m)}", flag(s, p).astype(np.int8)) for m, s in scores.items()],
        pl.Series("driver", [FEATURES[i] for i in np.abs(z).argmax(axis=1)]),
        pl.Series("driver_z", z[np.arange(len(z)), np.abs(z).argmax(axis=1)]),
    )
    pc = PCA(n_components=2).fit_transform(z)
    return out.with_columns(pl.sum_horizontal(pl.selectors.starts_with("outlier_")).alias("n_methods"),
                            pl.Series("pc1", pc[:, 0]), pl.Series("pc2", pc[:, 1]))


def agreement(flags: pl.DataFrame) -> pl.DataFrame:
    """Overlap of the outliers of every pair of methods: common days and the Jaccard index |A ∩ B| / |A ∪ B|."""
    rows = []
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            fa, fb = flags[f"outlier_{name(a)}"] == 1, flags[f"outlier_{name(b)}"] == 1
            rows.append({"pair": f"{a} & {b}", "common": int((fa & fb).sum()), "jaccard": (fa & fb).sum() / (fa | fb).sum()})
    return pl.DataFrame(rows)


def episodes(consensus: pl.DataFrame) -> pl.DataFrame:
    """Dates on which the outliers appear at several stations at once.

    An outlier at many stations on the same day points to a regional event (weather, transported dust), an outlier
    at one station only to a local source or a sensor problem. The PM2.5/PM10 ratio helps to tell coarse dust
    (low ratio) from combustion smoke (high ratio).
    """
    return (consensus.group_by("date")
            .agg(pl.len().alias("stations"), pl.col("PM10").mean(), pl.col("PM2_5").mean(),
                 (pl.col("PM2_5") / pl.col("PM10")).mean().alias("pm25_pm10_ratio"),
                 pl.col("driver").mode().first().alias("main_driver"))
            .filter(pl.col("stations") >= 2).sort("stations", "date", descending=[True, False]))


def run() -> SimpleNamespace:
    """Phase II pipeline: feature space → evaluation of all settings → best settings → detection on the real data."""
    days = load_station_days()
    x = days.select(FEATURES).to_numpy()
    ev, found = evaluate(x)
    summary = mean_over_seeds(ev)
    main = summary.filter(pl.col("percentile") == O["threshold_percentile"])
    best = table(best_settings(summary), "o1_best_settings")
    flags = detect(days, best)
    write(flags, "gold", "outlier_flags")

    consensus = flags.filter(pl.col("n_methods") >= 2)
    univariate_extreme = (pl.col("driver_z").abs() > 3)
    return SimpleNamespace(
        n_days=days.height, n_stations=days["station_id"].n_unique(),
        n_injected=round(O["injected_share"] * days.height),
        space=table(days.select(FEATURES).describe(), "o1_feature_space"),
        knn=table(main.filter(pl.col("method") == "KNN").drop("inertia"), "o1_knn_evaluation"),
        lof=table(main.filter(pl.col("method") == "LOF").drop("inertia"), "o1_lof_evaluation"),
        kmeans=table(main.filter(pl.col("method") == "K-Means"), "o1_kmeans_evaluation"),
        threshold=table(summary.join(best.select("method", "metric", "k"), on=["method", "metric", "k"])
                        .select("method", "percentile", "precision", "recall", "f1"), "o1_threshold_evaluation"),
        best=best,
        by_factor=table(recall_by(found, best, "factor"), "o1_recall_by_factor"),
        by_pollutant=table(recall_by(found, best, "pollutant"), "o1_recall_by_pollutant"),
        flags=flags,
        agreement=table(agreement(flags), "o1_agreement"),
        n_by_methods=table(flags.group_by("n_methods").len("days").sort("n_methods"), "o1_days_by_n_methods"),
        by_station=table(consensus.group_by("station_id").len("outlier_days")
                         .join(flags.group_by("station_id").len("days"), on="station_id")
                         .with_columns((100 * pl.col("outlier_days") / pl.col("days")).alias("pct"))
                         .sort("station_id"), "o1_outliers_by_station"),
        by_month=table(consensus.group_by("month").len("outlier_days").sort("month"), "o1_outliers_by_month"),
        by_driver=table(consensus.group_by("driver").len("outlier_days").sort("outlier_days", descending=True),
                        "o1_outliers_by_driver"),
        top=table(flags.filter(pl.col("n_methods") == len(METHODS)).sort("score_lof", descending=True)
                  .select("station_id", "date", *FEATURES, "driver", "driver_z", "score_knn", "score_lof",
                          "score_kmeans"), "o1_top_outliers"),
        episodes=table(episodes(consensus), "o1_outlier_episodes"),
        ratio_median=float((days["PM2_5"] / days["PM10"]).median()),
        n_consensus=consensus.height,
        n_multivariate_only=consensus.filter(~univariate_extreme).height,
    )
