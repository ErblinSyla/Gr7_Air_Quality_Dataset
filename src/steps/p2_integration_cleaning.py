"""p2 – Data integration (station metadata, unit harmonisation) and data cleaning (no row is ever deleted)."""

from __future__ import annotations

import html
from types import SimpleNamespace

import numpy as np
import polars as pl

from steps.p1_collection_types_quality import VALID, add_quality_flags
from utils import CFG, NAMES, by_name, lineage, scan, table, write

Q = CFG["quality"]
NOX_FACTOR = by_name("ppb_to_ugm3", 1.0)["NOx"]
ACTIONS = ["keep", "zero_negative", "null_invalid", "null_sentinel", "null_absurd", "null_calibration", "null_physical",
           "null_negative", "null_stuck", "null_spike", "null_inconsistent"]


def record_sets(raw: pl.DataFrame) -> pl.DataFrame:

    def text(col: str) -> pl.Expr:
        e = pl.col(col).str.strip_chars().str.strip_chars('"').str.strip_chars()
        e = e.map_elements(html.unescape, return_dtype=pl.String)
        return pl.when(e == "").then(None).otherwise(e)

    year = pl.when(pl.col("Year") == "").then(None).otherwise(pl.col("Year"))
    group = (pl.when(year.is_null()).then(pl.lit("legacy"))
             .when((year == "2024") & (pl.col("Municipality") != "")).then(pl.lit("2024_municipality"))
             .otherwise(year))
    return raw.select(
        pl.col("Air Quality Station EoI Code").alias("station_id"),
        group.alias("record_set"),
        text("Air Quality Station Name").alias("station_name"),
        text("Air Quality Station Nat Code").alias("national_code"),
        pl.col("Latitude").cast(pl.Float64).alias("lat"),
        pl.col("Longitude").cast(pl.Float64).alias("lon"),
        pl.col("Altitude").cast(pl.Float64).replace(0.0, None).alias("altitude_m"),
        text("Air Quality Station Type").alias("station_type"),
        text("Air Quality Station Area").alias("station_area"),
        text("Municipality").alias("municipality"),
    )


def record_set_profile(m: pl.DataFrame) -> pl.DataFrame:
    return (m.group_by("record_set")
             .agg(pl.len().alias("rows"), pl.col("station_id").n_unique().alias("stations"),
                  pl.col("station_type").unique().sort().str.join(", ").alias("types"),
                  pl.col("municipality").drop_nulls().unique().sort().str.join(", ").alias("municipalities"))
             .sort("record_set"))


def distance_km(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    dy = (lat2 - lat1) * 111.0
    dx = (lon2 - lon1) * 111.0 * ((lat1 + lat2) / 2 * np.pi / 180).cos()
    return (dx ** 2 + dy ** 2).sqrt()


def clean_stations(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    m = record_sets(raw)
    ref = pl.DataFrame([{"station_id": k, **v} for k, v in CFG["stations"]["town_centroids"].items()])

    source_rank = pl.col("record_set").replace_strict({"2024": 0, "legacy": 1, "2017": 2, "2024_municipality": 3},
                                                      return_dtype=pl.Int8)
    coords = (m.select("station_id", "record_set", "lat", "lon", "altitude_m").unique()
               .join(ref.rename({"lat": "ref_lat", "lon": "ref_lon"}), on="station_id")
               .with_columns(distance_km(pl.col("lat"), pl.col("lon"), pl.col("ref_lat"), pl.col("ref_lon"))
                             .round(1).alias("distance_km"))
               .sort("station_id", "distance_km", pl.col("altitude_m").is_null(), source_rank, "lat", "lon"))
    chosen_coords = coords.group_by("station_id", maintain_order=True).first().select(
        "station_id", "town", "lat", "lon", "altitude_m", "distance_km", pl.col("record_set").alias("coords_from"))

    trusted = m.filter(pl.col("record_set") != "2024_municipality")
    priority = pl.col("record_set").replace_strict({"2024": 0, "2017": 1, "legacy": 2}, return_dtype=pl.Int8)

    def majority(attr: str) -> pl.DataFrame:
        votes = (trusted.drop_nulls(attr).group_by("station_id", attr)
                 .agg(pl.col("record_set").n_unique().alias("_n"), priority.min().alias("_p")))
        return (votes.sort("station_id", "_n", "_p", descending=[False, True, False])
                .group_by("station_id", maintain_order=True).first().select("station_id", attr))

    latest = (m.filter(pl.col("record_set") == "2024").group_by("station_id")
               .agg(pl.col("station_name").drop_nulls().first(), pl.col("national_code").drop_nulls().first()))
    variants = m.group_by("station_id").agg(
        pl.col("station_name").drop_nulls().unique().sort().str.join(" | ").alias("name_variants"),
        pl.col("national_code").drop_nulls().unique().sort().str.join(" | ").alias("national_code_variants"))
    stations = (latest.join(chosen_coords, on="station_id").join(majority("station_type"), on="station_id")
                      .join(majority("station_area"), on="station_id").join(variants, on="station_id")
                      .sort("station_id"))

    attrs = ["station_name", "national_code", "lat", "lon", "altitude_m", "station_type", "station_area"]
    conflicts = pl.DataFrame([{
        "attribute": a,
        "stations_with_conflict": m.group_by("station_id").agg(pl.col(a).drop_nulls().n_unique())
                                   .filter(pl.col(a) > 1).height,
        "example_XK0002A": str(m.filter(pl.col("station_id") == "XK0002A")[a].drop_nulls().unique().sort().to_list()),
    } for a in attrs])
    return stations, conflicts, coords.drop("ref_lat", "ref_lon"), record_set_profile(m)


def nox_triplets(df: pl.DataFrame, value: str) -> pl.DataFrame:
    return (df.filter(pl.col("pollutant").is_in(["NOx", "NO2", "NO"]))
              .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), value)
              .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values=value)
              .drop_nulls(["NOx", "NO2", "NO"]))


def ratio() -> pl.Expr:
    return pl.col("NOx") / (pl.col("NO2") + Q["no_to_no2"] * pl.col("NO"))


def nox_consistent_from(df: pl.DataFrame) -> pl.DataFrame:
    usable = df.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd")
                       & (pl.col("value_raw") > 0))
    lo, hi = Q["nox_ppb_ratio"]
    weekly = (nox_triplets(usable, "value_raw").with_columns(ratio().alias("r")).sort("ts_utc")
              .group_by_dynamic("ts_utc", every="1w", group_by="station_id").agg(pl.col("r").median())
              .with_columns(pl.col("r").is_between(lo, hi).alias("ok")))
    last_bad = weekly.filter(~pl.col("ok")).group_by("station_id").agg(pl.col("ts_utc").max().alias("last_bad"))
    return (weekly.join(last_bad, on="station_id", how="left")
                  .filter(pl.col("last_bad").is_null() | (pl.col("ts_utc") > pl.col("last_bad")))
                  .group_by("station_id").agg(pl.col("ts_utc").min().alias("consistent_from"),
                                              pl.col("last_bad").first())
                  .sort("station_id"))


def nox_inconsistent_flag(df: pl.DataFrame, periods: pl.DataFrame) -> pl.DataFrame:
    usable = df.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd"))
    hourly = (nox_triplets(usable, "value").filter((pl.col("NO2") + Q["no_to_no2"] * pl.col("NO")) > 0)
              .select("station_id", "ts_utc", ((ratio() - 1).abs() > Q["nox_tolerance"]).alias("_h")))
    df = (df.join(periods.select("station_id", "consistent_from"), on="station_id", how="left")
            .join(hourly.with_columns(pl.lit("NOx").cast(df.schema["pollutant"]).alias("pollutant")),
                  on=["station_id", "pollutant", "ts_utc"], how="left"))
    is_nox = pl.col("pollutant") == "NOx"
    before = (pl.col("ts_utc") < pl.col("consistent_from")).fill_null(False)
    period = is_nox & before
    return (df.with_columns(period.alias("f_nox_period"),
                            (is_nox & (period | pl.col("_h").fill_null(False))).alias("f_nox_inconsistent"),
                            (before & pl.col("pollutant").is_in(Q["analyzer_change"]["to_null"]))
                            .alias("f_calibration"),
                            (before & pl.col("pollutant").is_in(Q["analyzer_change"]["flag_only"]))
                            .alias("f_gas_period_flag"))
              .drop("consistent_from", "_h"))


def analyzer_change_evidence(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    good = (df.filter(pl.col("clean_action").is_in(["keep", "zero_negative"]))
              .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_local",
                      pl.col("value_clean").alias("v"))
              .with_columns(pl.col("ts_local").dt.week().alias("w"), pl.col("ts_local").dt.year().alias("y")))
    base = (good.filter(pl.col("y") >= 2020).group_by("station_id", "pollutant", "w")
                .agg(pl.col("v").median().alias("b")))
    y19 = good.filter(pl.col("y") == 2019).join(base, on=["station_id", "pollutant", "w"]).filter(pl.col("b") > 0)
    switch = pl.lit(Q["analyzer_change"]["approx_date"]).str.to_datetime().dt.replace_time_zone(CFG["time"]["local_tz"])
    summary = (y19.with_columns(pl.when(pl.col("ts_local") < switch).then(pl.lit("before_change"))
                                .otherwise(pl.lit("after_change")).alias("period"))
               .group_by("pollutant", "period").agg((pl.col("v") / pl.col("b")).median().round(2).alias("r"))
               .pivot(sort_columns=True, on="period", index="pollutant", values="r")
               .select("pollutant", "before_change", "after_change").sort("pollutant"))
    weekly = (y19.sort("ts_local").group_by_dynamic("ts_local", every="2w", group_by="pollutant")
                 .agg((pl.col("v") / pl.col("b")).median().alias("r")).with_columns(pl.col("ts_local").dt.date()))
    return summary, weekly


def clean_action(with_calibration: bool = True) -> pl.Expr:
    f, v, pol = pl.col, pl.col("value"), pl.col("pollutant").cast(pl.String)
    physical_max = pol.replace_strict(by_name("physical_max"), return_dtype=pl.Float64)
    neg_zero = pol.replace_strict(by_name("neg_zero"), return_dtype=pl.Float64)
    stuck_kept = f("f_stuck_zero") & pol.is_in(Q["stuck_zero_plausible"])
    return (pl.when(f("f_invalid")).then(pl.lit("null_invalid"))
            .when(f("f_sentinel")).then(pl.lit("null_sentinel"))
            .when(f("f_absurd")).then(pl.lit("null_absurd"))
            .when(f("f_calibration") & pl.lit(with_calibration)).then(pl.lit("null_calibration"))
            .when(v > physical_max).then(pl.lit("null_physical"))
            .when(f("f_negative") & (v >= neg_zero)).then(pl.lit("zero_negative"))
            .when(f("f_negative")).then(pl.lit("null_negative"))
            .when(f("f_stuck") & ~stuck_kept).then(pl.lit("null_stuck"))
            .when(f("f_spike")).then(pl.lit("null_spike"))
            .when(f("f_pm_out_tol") | f("f_nox_inconsistent")).then(pl.lit("null_inconsistent"))
            .otherwise(pl.lit("keep")).cast(pl.Enum(ACTIONS)).alias("clean_action"))


def cleaned(d: pl.DataFrame, calibration: bool) -> pl.DataFrame:
    d = d.with_columns(clean_action(calibration))
    return d.with_columns(pl.when(pl.col("clean_action") == "keep").then(pl.col("value"))
                          .when(pl.col("clean_action") == "zero_negative").then(0.0)
                          .otherwise(None).alias("value_clean"))


def describe(col: str) -> list[pl.Expr]:
    c = pl.col(col)
    return [c.count().alias("n"), c.mean().alias("mean"), c.std().alias("std"), c.min().alias("min"),
            c.median().alias("median"), c.quantile(0.99).alias("p99"), c.max().alias("max")]


def integrate_stations() -> tuple:
    stations_raw = scan("bronze", "stations_raw").collect()
    stations, conflicts, coords, sets = clean_stations(stations_raw)
    far = stations.filter(pl.col("distance_km") > CFG["stations"]["max_distance_km"])
    if far.height:
        raise SystemExit(f"STOP: coordinates > {CFG['stations']['max_distance_km']} km from the town: {far.rows()}")
    write(stations, "silver", "stations")
    table(conflicts, "p2_metadata_conflicts")
    table(sets, "p2_metadata_record_sets")
    table(coords, "p2_metadata_coordinate_candidates")
    table(stations, "p2_stations")
    lineage("p2", "metadata: deduplication", stations_raw.height, stations.height, 0, 0)
    m = record_sets(stations_raw)
    podgorica = m.filter(pl.col("municipality") == "Podgorica")["station_id"].n_unique()
    pristina = coords.filter(pl.col("station_id").is_in(["XK0001A", "XK0002A"]))["distance_km"].max()
    return stations_raw, stations, conflicts, coords, sets, podgorica, pristina


def run() -> SimpleNamespace:
    stations_raw, stations, conflicts, coords, sets, podgorica, pristina_far = integrate_stations()

    df = (scan("bronze", "measurements").with_columns(pl.col("pollutant").cast(pl.String).cast(pl.Enum(NAMES)))
          .collect().sort("station_id", "pollutant", "ts_utc"))
    df = add_quality_flags(df)
    numeric = ~pl.col("f_sentinel") & ~pl.col("f_absurd")
    df = df.with_columns(pl.when(numeric & (pl.col("pollutant") == "NOx")).then(pl.col("value_raw") * NOX_FACTOR)
                         .otherwise(pl.col("value_raw")).alias("value"))
    periods = nox_consistent_from(df)
    late = periods.filter(pl.col("consistent_from").dt.year() != Q["analyzer_change"]["expected_year"])
    if late.height:
        raise SystemExit(f"STOP: analyzer change date outside the expected year: {late.rows()}")
    df = nox_inconsistent_flag(df, periods)
    table(periods, "p2_nox_periods")

    provisional = cleaned(df, calibration=False)
    gas_summary, gas_weekly = analyzer_change_evidence(provisional)
    w = (provisional.filter(pl.col("pollutant").is_in(["NO", "NO2"]) & pl.col("f_calibration"))
         .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value_clean")
         .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value_clean").drop_nulls())
    nox_recomputed = w.select((pl.col("NO2") + Q["no_to_no2"] * pl.col("NO")).mean()).item()
    nox_measured = (provisional.filter((pl.col("pollutant") == "NOx") & ~pl.col("f_nox_inconsistent"))
                    ["value_clean"].mean())
    table(gas_summary, "p2_analyzer_change")
    table(gas_weekly.pivot(sort_columns=True, on="pollutant", index="ts_local", values="r").sort("ts_local"),
          "p2_analyzer_change_weekly")
    df = cleaned(df, calibration=True)

    after = (nox_triplets(df.drop_nulls("value_clean"), "value_clean").with_columns(ratio().alias("r"))
             .filter(pl.col("r").is_finite()))
    before = nox_triplets(df.filter(pl.col("validity").is_in(VALID) & numeric & (pl.col("value_raw") > 0)),
                          "value_raw").with_columns(ratio().alias("r"))
    check = (after.group_by("station_id", pl.col("ts_utc").dt.convert_time_zone(CFG["time"]["local_tz"]).dt.year()
                            .alias("year"))
             .agg(pl.len().alias("hours"), pl.col("r").median().round(4).alias("median"))
             .with_columns(((pl.col("median") - 1).abs() <= Q["unit_check_tolerance"]).alias("ok"))
             .sort("station_id", "year"))
    table(check, "p2_nox_ratio_after_conversion")
    if not check["ok"].all():
        raise SystemExit(f"STOP: NOx ratio after conversion is not ≈ 1: {check.filter(~pl.col('ok')).rows()}")

    n_before = df.height
    df = df.join(stations.select("station_id", "station_type", "station_area", "altitude_m")
                 .with_columns(pl.col("station_id").cast(df.schema["station_id"])), on="station_id", how="left")
    pmax = by_name("physical_max")
    n_neg = df.filter(pl.col("value_clean") < 0).height
    n_over = df.filter(pl.col("value_clean") > pl.col("pollutant").cast(pl.String)
                       .replace_strict(pmax, return_dtype=pl.Float64)).height
    no_reason = df.filter(pl.col("value_clean").is_null() & (pl.col("clean_action") == "keep")).height
    if n_neg or n_over or df.height != n_before or no_reason or df["station_type"].null_count():
        raise SystemExit(f"STOP: check failed (neg={n_neg}, above limit={n_over}, rows {n_before}→{df.height})")

    clean = df.select("station_id", "pollutant", "ts_utc", "ts_local", "value_raw", "value_clean", "clean_action",
                      "validity", "f_below_dl", "f_outlier_stat", "f_pm25_gt_pm10", "f_gas_period_flag",
                      "station_type", "station_area", "altitude_m").rename({"value_clean": "value"})
    write(clean, "silver", "measurements_clean", partition_by="pollutant")
    lineage("p2", "cleaning: bronze → silver", n_before, clean.height, 0, clean["value"].null_count())

    actions = table(df.group_by("clean_action", "pollutant").len()
                    .pivot(on="pollutant", index="clean_action", values="len", sort_columns=True).fill_null(0)
                    .with_columns(pl.sum_horizontal(pl.exclude("clean_action")).alias("TOTAL"))
                    .sort("clean_action"), "p2_clean_action_by_pollutant")
    valid_before = df.filter(pl.col("validity").is_in(VALID) & numeric)
    stats = table(pl.concat([
        valid_before.group_by("pollutant").agg(pl.lit("before").alias("stage"), *describe("value")),
        df.group_by("pollutant").agg(pl.lit("after").alias("stage"), *describe("value_clean")),
    ]).sort("pollutant", "stage", descending=[False, True]), "p2_stats_before_after")
    counts = df.select(
        pl.col("f_below_dl").sum().alias("below_dl"),
        (pl.col("f_stuck_zero") & (pl.col("clean_action") == "keep")).sum().alias("stuck_zero_kept"),
        (pl.col("f_pm25_gt_pm10") & ~pl.col("f_pm_out_tol")).sum().alias("pm_within_tol"),
        pl.col("f_outlier_stat").sum().alias("outliers"),
        pl.col("f_nox_period").sum().alias("nox_period"),
        (pl.col("clean_action") == "null_calibration").sum().alias("calibration"),
        pl.col("f_gas_period_flag").sum().alias("gas_flag"),
        (pl.col("f_nox_inconsistent") & ~pl.col("f_nox_period")).sum().alias("nox_hourly_only"),
    ).row(0, named=True)
    table(pl.DataFrame([counts]), "p2_kept_on_purpose")
    neg = table(df.group_by("pollutant").agg(
        pl.col("value_clean").mean().alias("mean_negatives_to_0"),
        pl.col("value_clean").filter(pl.col("clean_action") != "zero_negative").mean().alias("mean_negatives_null"),
        (pl.col("clean_action") == "zero_negative").sum().alias("values_affected"))
        .with_columns((100 * (pl.col("mean_negatives_null") / pl.col("mean_negatives_to_0") - 1)).round(2)
                      .alias("change_pct"))
        .sort("pollutant"), "p2_negative_sensitivity")

    hist = {p: (valid_before.filter(pl.col("pollutant") == p)["value"].clip(lower_bound=0).to_numpy(),
                df.filter(pl.col("pollutant") == p)["value_clean"].drop_nulls().to_numpy()) for p in NAMES}
    return SimpleNamespace(
        stations_raw=stations_raw, stations=stations, conflicts=conflicts, coords=coords, sets=sets,
        podgorica=podgorica, pristina_far=pristina_far,
        periods=periods, gas_summary=gas_summary, gas_weekly=gas_weekly, nox_recomputed=nox_recomputed,
        nox_measured=nox_measured, check=check, before=before, after=after, n_before=n_before, n_clean=clean.height,
        actions=actions, stats=stats, counts=counts, neg=neg, hist=hist,
        pm_null_hours=df.filter(pl.col("f_pm_out_tol") & (pl.col("pollutant") == "PM2_5")).height,
        pm_hours=df.filter(pl.col("pollutant") == "PM2_5").height,
    )
