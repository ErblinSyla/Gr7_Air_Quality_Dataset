"""p4 – Aggregation (hour → day → month → year, station → zone → Kosovo) and sampling (SRSWOR, SRSWR,
stratified)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import polars as pl

from utils import CFG, POLLUTANT_TYPE, SEASONS, SERIES, lineage, scan, table, write

AG, S, T = CFG["aggregation"], CFG["sampling"], CFG["time"]
STRUCTURAL = ["not_measured", "outside_period"]
METHODS = ["SRSWOR", "SRSWR", "stratified"]


def zone(town: str) -> str:
    m = re.search(r"\(([^)]+)\)", town)
    return m.group(1) if m else town


ZONES = pl.DataFrame([{"station_id": k, "zone": zone(v["town"])}
                      for k, v in CFG["stations"]["town_centroids"].items()])


def daily(hourly: pl.DataFrame, col: str) -> pl.DataFrame:
    v = pl.col(col)
    return (hourly.sort(*SERIES, "ts_local")
            .group_by_dynamic("ts_local", every="1d", group_by=SERIES)
            .agg(v.count().alias("valid_hours"), v.mean().alias("_mean"), v.max().alias("max"))
            .with_columns((pl.col("valid_hours") >= AG["min_hours_per_day"]).alias("valid_day"),
                          pl.col("ts_local").dt.date().alias("date"))
            .with_columns(pl.when("valid_day").then(pl.col("_mean")).alias("mean"))
            .drop("_mean", "ts_local"))


def monthly(day: pl.DataFrame) -> pl.DataFrame:
    return (day.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by(*SERIES, "month")
            .agg(pl.col("valid_day").sum().alias("valid_days"), pl.col("mean").mean().alias("_mean"))
            .with_columns(pl.col("month").dt.month_end().dt.day().alias("days_in_month"))
            .with_columns(pl.when(pl.col("valid_days") >= AG["min_days_per_month_frac"] * pl.col("days_in_month"))
                          .then(pl.col("_mean")).alias("mean"))
            .drop("_mean").sort(*SERIES, "month"))


def annual(hourly: pl.DataFrame, day: pl.DataFrame, col: str) -> pl.DataFrame:
    hours = (hourly.group_by(*SERIES, pl.col("ts_local").dt.year().alias("year"))
             .agg(pl.col(col).count().alias("valid_hours")))
    means = day.group_by(*SERIES, pl.col("date").dt.year().alias("year")).agg(pl.col("mean").mean())
    year_hours = pl.when(pl.col("year") % 4 == 0).then(8784).otherwise(8760)
    return (hours.join(means, on=[*SERIES, "year"], how="left")
            .with_columns((pl.col("valid_hours") / year_hours).round(3).alias("coverage"))
            .with_columns((pl.col("coverage") >= AG["min_annual_coverage"]).alias("meets_90pct"))
            .with_columns(pl.when("meets_90pct").then(pl.col("mean")).alias("mean_official"))
            .sort(*SERIES, "year"))


def rollup(day: pl.DataFrame) -> pl.DataFrame:
    d = (day.filter("valid_day").with_columns(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String),
                                              pl.col("date").dt.year().alias("year"))
            .join(ZONES, on="station_id", how="left"))
    if d["zone"].null_count():
        raise SystemExit("STOP: station without a zone in config.yaml")
    levels = [("station", pl.col("station_id")), ("zone", pl.col("zone")), ("Kosovo", pl.lit("Kosovo"))]
    return (pl.concat([d.group_by(name.alias("name"), "pollutant", "year").agg(pl.col("mean").mean())
                       .with_columns(pl.lit(level).alias("level")) for level, name in levels])
              .select("level", "name", "pollutant", "year", "mean").sort("level", "name", "pollutant", "year"))


def population() -> pl.DataFrame:
    long = (scan("gold", "hourly_long").filter(~pl.col("status").is_in(STRUCTURAL))
            .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_utc", "ts_local",
                    "value_imputed").collect())
    return (long.pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc", "ts_local"],
                       values="value_imputed")
                .with_columns(pl.col("ts_local").dt.month().replace_strict(SEASONS, return_dtype=pl.String)
                              .alias("season"))
                .sort("station_id", "ts_utc"))


def draw(pop: pl.DataFrame, method: str) -> pl.DataFrame:
    if method == "SRSWOR":
        return pop.sample(fraction=S["fraction"], with_replacement=False, seed=S["seed"])
    if method == "SRSWR":
        return pop.sample(fraction=S["fraction"], with_replacement=True, seed=S["seed"])
    u = pl.Series("_u", np.random.default_rng(S["seed"]).random(pop.height))
    strata = S["strata"]
    rank = pl.col("_u").rank("ordinal").over(strata)
    return pop.with_columns(u).filter(rank <= (pl.len() * S["fraction"]).round().over(strata)).drop("_u")


def compare(pop: pl.DataFrame, sample: pl.DataFrame, method: str) -> tuple[list[dict], dict]:
    rows = []
    for p in S["compare_pollutants"]:
        a, b = pop[p].drop_nulls(), sample[p].drop_nulls()
        rows.append({"method": method, "pollutant": p, "mean_population": a.mean(), "mean_sample": b.mean(),
                     "mean_deviation_pct": 100 * (b.mean() / a.mean() - 1),
                     "std_population": a.std(), "std_sample": b.std(), "p95_population": a.quantile(0.95),
                     "p95_sample": b.quantile(0.95)})

    def max_dev(col: str) -> float:
        share = lambda d: d.group_by(col).len().with_columns(pl.col("len") / pl.col("len").sum())  # noqa: E731
        j = share(pop).join(share(sample), on=col, how="left", suffix="_s").fill_null(0)
        return round(100 * float((j["len"] - j["len_s"]).abs().max()), 3)

    unique = sample.select("station_id", "ts_utc").n_unique()
    return rows, {"method": method, "n": sample.height, "repeated_rows": sample.height - unique,
                  "max_station_deviation_pp": max_dev("station_id"), "max_season_deviation_pp": max_dev("season")}


def aggregate() -> SimpleNamespace:
    hourly = (scan("gold", "hourly_long").filter(~pl.col("status").is_in(STRUCTURAL))
              .with_columns(pl.col("pollutant").cast(pl.String).cast(POLLUTANT_TYPE))
              .select(*SERIES, "ts_utc", "ts_local", "value", "value_imputed").collect())
    days, months, years = [], [], []
    for source, col in (("measured", "value"), ("imputed", "value_imputed")):
        d = daily(hourly, col)
        days.append(d.with_columns(pl.lit(source).alias("source")))
        months.append(monthly(d).with_columns(pl.lit(source).alias("source")))
        years.append(annual(hourly, d, col).with_columns(pl.lit(source).alias("source")))
    day, month, year = pl.concat(days), pl.concat(months), pl.concat(years)
    write(day, "gold", "daily")
    write(month, "gold", "monthly")
    write(year, "gold", "annual")
    measured = lambda d: d.filter(pl.col("source") == "measured")  # noqa: E731
    lineage("p4", "aggregation hourly → daily", hourly.height, measured(day).height,
            hourly["value"].null_count(), measured(day)["mean"].null_count())

    gained = table(day.group_by("pollutant", "source").agg(pl.col("valid_day").sum())
                   .pivot(sort_columns=True, on="source", index="pollutant", values="valid_day")
                   .select("pollutant", "measured", "imputed")
                   .with_columns((pl.col("imputed") - pl.col("measured")).alias("days_gained"),
                                 (100 * (pl.col("imputed") / pl.col("measured") - 1)).round(1).alias("pct"))
                   .sort("pollutant"), "p4_days_gained")
    roll = rollup(measured(day))
    write(roll, "gold", "rollup_annual")
    table(roll, "p4_rollup_station_zone_kosovo")
    sizes = table(pl.DataFrame({"level": ["hourly", "daily", "monthly", "annual (station)", "annual (zone)",
                                          "annual (Kosovo)"],
                                "rows": [hourly.height, measured(day).height, measured(month).height,
                                         measured(year).height,
                                         roll.filter(pl.col("level") == "zone").height,
                                         roll.filter(pl.col("level") == "Kosovo").height]})
                  .with_columns((pl.col("rows").first() / pl.col("rows")).round(0).alias("reduction_x")),
                  "p4_size_reduction")
    cov = measured(year)
    coverage = table(cov.with_columns(pl.col("station_id").cast(pl.String))
                     .pivot(on="year", index=["station_id", "pollutant"], values="coverage", sort_columns=True)
                     .sort("station_id", "pollutant"), "p4_annual_coverage")
    heating = table(measured(day).filter("valid_day")
                    .with_columns(pl.col("date").dt.month().is_in(T["heating_months"]).alias("heating"))
                    .group_by("pollutant").agg(pl.col("mean").filter("heating").mean().alias("heating_season"),
                                               pl.col("mean").filter(~pl.col("heating")).mean().alias("rest_of_year"))
                    .with_columns((pl.col("heating_season") / pl.col("rest_of_year")).round(2).alias("ratio"))
                    .sort("pollutant"), "p4_heating_season")
    return SimpleNamespace(
        day=day, gained=gained, sizes=sizes, coverage=coverage, heating=heating, n_days=measured(day).height,
        n_years=cov.height, n_low=cov.filter(~pl.col("meets_90pct")).height,
        pm_roll=(roll.filter((pl.col("pollutant") == "PM2_5") & (pl.col("level") != "station"))
                     .pivot(sort_columns=True, on="year", index=["level", "name"], values="mean")
                     .sort(pl.col("level") == "Kosovo", "name")),
    )


def sample() -> SimpleNamespace:
    pop = population()
    rows, reps = [], []
    samples = {method: draw(pop, method) for method in METHODS}
    for method, s in samples.items():
        r, rep = compare(pop, s, method)
        rows += r
        reps.append(rep)
    write(samples["stratified"], "gold", "sample_10pct")
    lineage("p4", "stratified sample 10%", pop.height, samples["stratified"].height, 0, 0)
    dist = table(pl.DataFrame(rows), "p4_sampling_distributions")
    return SimpleNamespace(
        n_pop=pop.height, n_strat=samples["stratified"].height, dist=dist,
        rep=table(pl.DataFrame(reps), "p4_sampling_representativeness"), reps={r["method"]: r for r in reps},
        dev=dict(dist.group_by("method", maintain_order=True).agg(pl.col("mean_deviation_pct").abs().max()).rows()),
    )


def run() -> SimpleNamespace:
    return SimpleNamespace(agg=aggregate(), smp=sample())
