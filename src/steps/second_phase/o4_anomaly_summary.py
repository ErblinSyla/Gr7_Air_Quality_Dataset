"""o4 – Summary of the anomaly detection: one catalogue of all station-days with an anomaly of any type.

It brings together the point anomalies (o1), the contextual anomalies (o2) and the collective anomalies of the
hourly series (o3: spikes and stuck sensors), shows how the types overlap and where and when they occur. The
catalogue is stored in data/gold/anomaly_catalogue.parquet; no value is changed.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import TABLES, scan, table, write

TYPES = ["point", "contextual", "spike", "stuck sensor"]
DAY = ["station_id", "date"]


def catalogue() -> pl.DataFrame:
    """One row per station-day with at least one anomaly: a True/False column per type and the list of types.

    The hourly anomalies of o3 are counted per local day (station-day of the spike or the frozen run)."""
    point = scan("gold", "anomalies_point").filter("point_anomaly").select(*DAY, pl.lit(True).alias("point"))
    context = (scan("gold", "anomalies_contextual").filter("contextual_anomaly")
               .select(*DAY, pl.lit(True).alias("contextual")))
    hourly = (scan("gold", "anomalies_temporal")
              .with_columns(pl.col("ts_utc").dt.convert_time_zone("Europe/Belgrade").dt.date().alias("date"))
              .group_by(DAY).agg(pl.col("spike").any(), pl.col("stuck").any().alias("stuck sensor")))
    days = point.join(context, on=DAY, how="full", coalesce=True).join(hourly, on=DAY, how="full", coalesce=True)
    return (days.with_columns(pl.col(TYPES).fill_null(False))
            .with_columns(pl.concat_list(pl.when(pl.col(t)).then(pl.lit(t)) for t in TYPES).list.drop_nulls()
                          .list.join(" + ").alias("types"),
                          pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month"))
            .sort(*DAY).collect())


def methods_overview() -> pl.DataFrame:
    """The chosen method of every anomaly type with its evaluation on injected anomalies (from o1–o3)."""
    o1 = pl.read_csv(TABLES / "o1_combination_rules.csv").sort("f1", descending=True).row(0, named=True)
    o2 = pl.read_csv(TABLES / "o2_evaluation.csv").sort("f1", descending=True).row(0, named=True)
    o3 = {r["fault"]: r for r in pl.read_csv(TABLES / "o3_best_settings.csv").iter_rows(named=True)}
    rows = [("point", "station-day (6 daily means)", f"LOF + Mahalanobis + KNN + K-Means + SVM, rule: {o1['rule']}", o1),
            ("contextual", "station-day minus the station × month profile",
             f"{o2['method']} on residuals, context {o2['context']}" + (f", k = {o2['param']}" if o2["param"] else ""),
             o2),
            ("spike", "hourly series", f"isolated spike, {o3['spike']['param']}", o3["spike"]),
            ("stuck sensor", "hourly series", f"constant non-zero run, {o3['stuck sensor']['param']}",
             o3["stuck sensor"]),
            ("drift", "daily series vs other stations", f"shift of the recent days, {o3['drift']['param']}",
             o3["drift"])]
    return pl.DataFrame([{"type": t, "representation": rep, "method": m, "precision": r["precision"],
                          "recall": r["recall"], "f1": r["f1"]} for t, rep, m, r in rows])


def run() -> SimpleNamespace:
    """o4 pipeline: catalogue of all anomalies → overview tables."""
    cat = catalogue()
    write(cat, "gold", "anomaly_catalogue")
    counts = cat.select(pl.col(TYPES).sum()).unpivot(variable_name="type", value_name="station_days")
    return SimpleNamespace(
        catalogue=cat,
        overview=table(methods_overview(), "o4_methods_overview"),
        counts=table(counts, "o4_station_days_by_type"),
        combinations=table(cat.group_by("types").len("station_days").sort("station_days", descending=True),
                           "o4_type_combinations"),
        by_station=table(cat.group_by("station_id").agg(pl.col(TYPES).sum()).sort("station_id"),
                         "o4_types_by_station"),
        by_year=table(cat.group_by("year").agg(pl.col(TYPES).sum()).sort("year"), "o4_types_by_year"),
        n_days=cat.height,
        n_multi=cat.filter(pl.sum_horizontal(TYPES) >= 2).height,
    )
