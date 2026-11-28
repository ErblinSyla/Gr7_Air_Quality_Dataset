"""Shared building blocks of phase II (anomaly detection): the station-day feature space, scaling, the percentile
threshold, robust z-scores and the evaluation measures used by o1–o3."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import CFG, SEASONS, scan

O = CFG["outliers"]
FEATURES = O["features"]


def load_station_days() -> pl.DataFrame:
    """One row per station-day with the daily means of the pollutants in `FEATURES`.

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


def robust_z(col: pl.Expr, over: list[str]) -> pl.Expr:
    """Robust z-score within groups: (x − median) / (1.4826 · MAD).

    The median and the MAD (median absolute deviation) are not pulled by the outliers themselves, unlike the mean and
    the standard deviation; 1.4826 · MAD equals the standard deviation for Gaussian data. A tiny constant avoids a
    division by zero in groups without variation.
    """
    dev = col - col.median().over(over)
    return dev / (1.4826 * dev.abs().median().over(over) + 1e-9)


def flag(score: np.ndarray, percentile: float) -> np.ndarray:
    """Threshold rule of the exercises: a point is an outlier when its score is above the given percentile."""
    return score > np.percentile(score, percentile)


def precision_recall_f1(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Evaluation against the known (injected) anomalies.

    Precision = TP / (TP + FP): of the points marked as anomalies, the share that really are anomalies.
    Recall    = TP / (TP + FN): of the real anomalies, the share that was found.
    F1        = 2 · P · R / (P + R): the harmonic mean, the balance of the two.
    """
    tp = int((actual & predicted).sum())
    precision = tp / predicted.sum() if predicted.sum() else 0.0
    recall = tp / actual.sum() if actual.sum() else 0.0
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall)}


def ranking_metrics(actual: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Threshold-free evaluation of the score: how well it ranks the known anomalies above the normal points.

    ROC-AUC: the probability that a random anomaly gets a higher score than a random normal point (0.5 = random,
    1 = perfect). PR-AUC (average precision): the mean precision over all recall levels, i.e. how well the anomalies
    are concentrated at the top of the ranking; a random score gets the share of anomalies (here ≈ 0.01), so PR-AUC is
    the more informative of the two when anomalies are rare.
    """
    return {"pr_auc": float(average_precision_score(actual, score)), "roc_auc": float(roc_auc_score(actual, score))}


def event_metrics(event: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Evaluation for anomalies that last several points (a stuck sensor, a drift).

    `event` holds the id of the injected event of every point (0 = normal). Precision is counted on points (the share
    of flagged points that lie inside an injected event), recall on events (the share of events with at least one
    flagged point), because one alarm is enough to find a stuck or drifting sensor.
    """
    inside = event > 0
    precision = (predicted & inside).sum() / predicted.sum() if predicted.sum() else 0.0
    recall = len(np.unique(event[predicted & inside])) / len(np.unique(event[inside])) if inside.any() else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": f1(precision, recall)}


def f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0 when both are 0)."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean_over_seeds(ev: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    """Average every evaluation measure over the injections of the different seeds, with the spread of F1."""
    measures = [c for c in ("precision", "recall", "f1", "pr_auc", "roc_auc") if c in ev.columns]
    return (ev.group_by(key, maintain_order=True)
            .agg(pl.col(measures).mean(), pl.col("f1").std().alias("f1_std")))


def best_per(summary: pl.DataFrame, by: str, tie: str) -> pl.DataFrame:
    """The row with the highest mean F1 for every value of `by`; on a tie the smallest `tie` (the simpler model)."""
    return (summary.sort(pl.col("f1").round(6), tie, descending=[True, False])
            .group_by(by, maintain_order=True).first())
