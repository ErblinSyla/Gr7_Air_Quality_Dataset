"""o2 – Contextual anomalies: station-days that are normal for the data set as a whole, but not for their context.

Lecture (week VI): contextual attributes decide whether the value of a behavioural attribute is normal (20 °C is
normal in May, not in January). Here the contextual attributes are the station, the month, the day type (weekday or
weekend) and the regional level of the day (the other stations, a proxy for the weather); the behavioural attributes
are the six daily means. The context is removed first – every value becomes a robust z-score within its context group
(residual) – and the anomaly methods then work on the residuals:
- contextual robust z-score: the largest |residual| of the day (statistical, univariate);
- LOF on the residual vectors (density-based, multivariate; the best point method of o1).
The context (from none to station × month + regional level) and n_neighbors are analysed with injected contextual
anomalies. Flags are stored in data/gold/anomalies_contextual.parquet.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from steps.second_phase.common import (
    FEATURES,
    O,
    best_per,
    flag,
    load_station_days,
    mean_over_seeds,
    precision_recall_f1,
    ranking_metrics,
    robust_z,
)
from steps.second_phase.o1_point_anomalies import lof_score
from utils import scan, table, write

CONTEXTS = {  # context name → grouping columns (from no context to the richest one)
    "none": [],
    "station": ["station_id"],
    "station × season": ["station_id", "season"],
    "station × month": ["station_id", "month"],
    "station × month × day type": ["station_id", "month", "day_type"],
    "station × month + regional level": ["station_id", "month"],
}
REGIONAL = "station × month + regional level"  # the values are first compared with the other stations of the day
INJECTED = ["wrong season", "wrong station"]
KEY = ["context", "method", "param"]


def inject_contextual(days: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame]:
    """Add known contextual anomalies: real daily values placed in the wrong context.

    - wrong season: the values of the same station from a day 150–215 days away (summer values in winter);
    - wrong station: the values of another station on the same date (e.g. an industrial station's day at a
      background station).
    Every injected day is a real, globally normal day, so it can only be found through its context. The copy keeps
    the station and date (context) of the day it replaces.
    """
    rng = np.random.default_rng(seed)
    n = round(O["injected_share"] * days.height)
    station, date = days["station_id"].to_numpy(), days["date"].to_numpy()
    x = days.select(FEATURES).to_numpy()
    rows, values, kinds = [], [], []
    for row in rng.choice(days.height, 3 * n, replace=False):
        kind = INJECTED[len(rows) % len(INJECTED)]
        if kind == "wrong season":
            gap = np.abs((date - date[row]).astype("timedelta64[D]").astype(int)) % 365
            candidates = np.flatnonzero((station == station[row]) & (gap >= 150) & (gap <= 215))
        else:
            candidates = np.flatnonzero((date == date[row]) & (station != station[row]))
        if len(candidates):
            rows.append(row)
            values.append(x[rng.choice(candidates)])
            kinds.append(kind)
        if len(rows) == n:
            break
    fake = days[rows].with_columns(*[pl.Series(f, np.array(values)[:, j]) for j, f in enumerate(FEATURES)])
    labels = np.r_[np.zeros(days.height, bool), np.ones(len(rows), bool)]
    return pl.concat([days, fake]), labels, pl.DataFrame({"type": kinds})


def grouped(days: pl.DataFrame, context: str) -> tuple[pl.DataFrame, list[str]]:
    """The data and the grouping columns of a context; "none" is one group with all days."""
    keys = CONTEXTS[context]
    return (days, keys) if keys else (days.with_columns(pl.lit(0).alias("_all")), ["_all"])


def with_day_type(days: pl.DataFrame) -> pl.DataFrame:
    """Add the day type (weekday / weekend): traffic, and with it NO2 and CO, is lower at the weekend."""
    weekend = pl.col("date").dt.weekday() >= 6
    return days.with_columns(pl.when(weekend).then(pl.lit("weekend")).otherwise(pl.lit("weekday")).alias("day_type"))


def regional_level(f: str) -> pl.Expr:
    """Mean log(1 + value) of the *other* stations on the same date (at least 3 of them), else null.

    The weather (wind, inversion, rain, transported dust) moves all stations together; subtracting this regional level
    keeps only what is special about the station on that day. The data set has no meteorological data, so the other
    stations are the best available proxy for the weather of the day."""
    lv = pl.col(f).log1p()
    total, count = lv.sum().over("date"), lv.count().over("date")
    return pl.when(count >= 4).then((total - lv) / (count - 1))


def log_values(days: pl.DataFrame, context: str) -> pl.DataFrame:
    """log(1 + value) of every pollutant; for the regional context relative to the other stations of the day
    (null where fewer than three other stations measured the pollutant)."""
    if context != REGIONAL:
        return days.with_columns(pl.col(f).log1p() for f in FEATURES)
    return days.with_columns((pl.col(f).log1p() - regional_level(f)).alias(f) for f in FEATURES)


def residuals(days: pl.DataFrame, context: str) -> np.ndarray:
    """Remove the context: robust z-score of the log values of every pollutant within its context group.

    The log makes the skewed values more symmetric; the median and the MAD describe what is usual for the group
    without being pulled by its outliers. With context "none" the result is the global robust z-score.
    """
    data, keys = grouped(log_values(days, context), context)
    r = data.select(robust_z(pl.col(f), keys).alias(f) for f in FEATURES).to_numpy()
    if context == REGIONAL:  # days without a regional reference: the residual of station × month alone
        r = np.where(np.isnan(r), residuals(days, "station × month"), r)
    return r


def usual_log(days: pl.DataFrame, context: str) -> np.ndarray:
    """The usual log value of every pollutant for the context of every day (the median of its group).

    For the regional context it is the usual level relative to the other stations plus today's regional level, i.e.
    what would be expected at this station on this day; without a regional reference the station × month median."""
    data, keys = grouped(log_values(days, context), context)
    typical = data.select(pl.col(f).median().over(keys) for f in FEATURES).to_numpy()
    if context != REGIONAL:
        return typical
    typical = typical + days.select(regional_level(f) for f in FEATURES).to_numpy()
    return np.where(np.isnan(typical), usual_log(days, "station × month"), typical)


def contextual_score(days: pl.DataFrame, context: str, method: str, k: int = 0) -> np.ndarray:
    """Score of one setting: the largest |residual| ("robust z") or LOF on the residual vectors ("LOF")."""
    r = residuals(days, context)
    return np.abs(r).max(axis=1) if method == "robust z" else lof_score(r, k)


def settings() -> list[tuple[str, str, int]]:
    """Every (context, method, n_neighbors) combination that is evaluated."""
    grid = [(c, "robust z", 0) for c in CONTEXTS]
    return grid + [(c, "LOF", k) for c in CONTEXTS for k in O["contextual_lof_k"]]


def evaluate(days: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evaluate every setting on `eval_seeds` injections; returns the measures and the flags of the injected days."""
    rows, found = [], []
    for seed in O["eval_seeds"]:
        data, actual, info = inject_contextual(days, seed)
        for context, method, k in settings():
            score = contextual_score(data, context, method, k)
            predicted = flag(score, O["threshold_percentile"])
            setting = {"context": context, "method": method, "param": k}
            rows.append({"seed": seed, **setting, **precision_recall_f1(actual, predicted),
                         **ranking_metrics(actual, score)})
            found.append(info.with_columns(*[pl.lit(v).alias(c) for c, v in setting.items()],
                                           pl.Series("detected", predicted[actual])))
    return pl.DataFrame(rows), pl.concat(found)


def detect(days: pl.DataFrame, best: dict) -> pl.DataFrame:
    """Run the best setting on the real station-days.

    Besides the score and the flag, every day gets the pollutant that is most unusual for its context, its residual
    and the usual value of the context (back-transformed median), so an alarm can be read as "PM10 was 95 µg/m³ where
    25 µg/m³ is usual for this station in this month".
    """
    r = residuals(days, best["context"])
    score = contextual_score(days, best["context"], best["method"], int(best["param"]))
    strongest = np.abs(r).argmax(axis=1)
    usual = np.expm1(usual_log(days, best["context"]))
    rows = np.arange(len(r))
    return days.select("station_id", "date", "month", *FEATURES).with_columns(
        pl.Series("score_contextual", score),
        pl.Series("contextual_anomaly", flag(score, O["threshold_percentile"])),
        pl.Series("context_driver", [FEATURES[i] for i in strongest]),
        pl.Series("context_residual", r[rows, strongest]),
        pl.Series("value", days.select(FEATURES).to_numpy()[rows, strongest]),
        pl.Series("usual_value", usual[rows, strongest]),
    )


def run() -> SimpleNamespace:
    """o2 pipeline: injected contextual anomalies → best context and method → detection on the real data."""
    days = with_day_type(load_station_days())
    ev, found = evaluate(days)
    summary = mean_over_seeds(ev, KEY)
    best_by_method = best_per(summary, "method", "param").sort("method")
    best = summary.sort(pl.col("f1").round(6), "param", descending=[True, False]).row(0, named=True)
    flags = detect(days, best)
    write(flags, "gold", "anomalies_contextual")
    point = scan("gold", "anomalies_point").select("station_id", "date", "point_anomaly").collect()
    both = flags.join(point, on=["station_id", "date"])
    alarms = both.filter("contextual_anomaly")
    return SimpleNamespace(
        evaluation=table(summary, "o2_evaluation"),
        robust=table(summary.filter(pl.col("method") == "robust z").select("context", "precision", "recall", "f1",
                                                                              "f1_std", "pr_auc", "roc_auc"),
                     "o2_robust_z_evaluation"),
        lof_auc=table(summary.filter(pl.col("method") == "LOF").pivot(on="context", index="param", values="pr_auc"),
                      "o2_lof_pr_auc_by_context"),
        best_auc=summary.sort("pr_auc", descending=True).row(0, named=True),
        lof=table(summary.filter(pl.col("method") == "LOF").pivot(on="context", index="param", values="f1"),
                  "o2_lof_f1_by_context"),
        by_type=table(found.group_by("context", "method", "param", "type").agg(pl.col("detected").mean().alias("recall"))
                      .join(best_by_method.select(KEY), on=KEY)
                      .pivot(on="type", index=["context", "method", "param"], values="recall"),
                      "o2_recall_by_type"),
        global_by_type=table(found.filter(pl.col("context") == "none").group_by("method", "param", "type")
                             .agg(pl.col("detected").mean().alias("recall"))
                             .pivot(on="type", index=["method", "param"], values="recall").sort("method", "param"),
                             "o2_global_recall_by_type"),
        best=best, best_by_method=best_by_method, flags=flags,
        repeated=table(alarms.group_by("station_id", "context_driver", pl.col("date").dt.strftime("%Y-%m").alias("month"))
                       .agg(pl.len().alias("days"), (pl.col("value") < pl.col("usual_value")).mean().alias("share_low"))
                       .filter(pl.col("days") >= 3).sort("days", descending=True), "o2_repeated_anomalies"),
        n_alarms=alarms.height,
        n_only_contextual=alarms.filter(~pl.col("point_anomaly")).height,
        by_driver=table(alarms.group_by("context_driver").len("days").sort("days", descending=True),
                        "o2_anomalies_by_pollutant"),
        by_month=table(alarms.group_by("month").len("days").sort("month"), "o2_anomalies_by_month"),
        top=table(alarms.filter(~pl.col("point_anomaly")).sort("score_contextual", descending=True)
                  .select("station_id", "date", "context_driver", "value", "usual_value", "context_residual",
                          "score_contextual"), "o2_top_contextual"),
    )
