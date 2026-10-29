"""p3 – Missing values: identification on the full hourly grid, analysis (reasons, gaps, χ² test), evaluation of
the lecture's filling methods on hidden values, and the treatment strategy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy import stats

from utils import CFG, NAMES, POLLUTANT_TYPE, SERIES, by_name, lineage, scan, table, write

M, T = CFG["missing"], CFG["time"]
ALPHA = CFG["chi2"]["alpha"]
KEY = [*SERIES, "ts_utc"]
STRUCTURAL = ["not_measured", "outside_period"]
CLASS = [*SERIES, "month", "hour"]
Q_STUCK_OK = CFG["quality"]["stuck_zero_plausible"]


def build_grid(clean: pl.DataFrame) -> pl.DataFrame:
    station_range = clean.group_by("station_id").agg(pl.col("ts_utc").min().alias("a"),
                                                     pl.col("ts_utc").max().alias("b"))
    grid = (station_range.select("station_id", pl.datetime_ranges("a", "b", "1h", time_unit="ns").alias("ts_utc"))
            .explode("ts_utc")
            .join(pl.DataFrame({"pollutant": NAMES}, schema={"pollutant": POLLUTANT_TYPE}), how="cross"))
    series = clean.group_by(SERIES).agg(pl.col("ts_utc").min().alias("first"), pl.col("ts_utc").max().alias("last"))
    g = (grid.join(series, on=SERIES, how="left")
             .join(clean.select(*KEY, "value", "value_raw", "clean_action", "f_gas_period_flag"), on=KEY, how="left")
             .with_columns(pl.col("f_gas_period_flag").fill_null(False)))
    status = (pl.when(pl.col("first").is_null()).then(pl.lit("not_measured"))
              .when((pl.col("ts_utc") < pl.col("first")) | (pl.col("ts_utc") > pl.col("last")))
              .then(pl.lit("outside_period"))
              .when(pl.col("clean_action").is_null()).then(pl.lit("missing_row"))
              .when(pl.col("value").is_not_null()).then(pl.lit("ok"))
              .otherwise(pl.col("clean_action").cast(pl.String)))
    return (g.with_columns(status.alias("status")).drop("first", "last").sort(KEY)
             .with_columns(pl.col("ts_utc").dt.convert_time_zone(T["local_tz"]).alias("ts_local"))
             .with_columns(pl.col("ts_local").dt.month().alias("month"), pl.col("ts_local").dt.hour().alias("hour"),
                           pl.col("ts_local").dt.year().alias("year")))


def add_gaps(df: pl.DataFrame) -> pl.DataFrame:
    real = ~pl.col("status").is_in(STRUCTURAL)
    gap = pl.col("_gap")
    start = gap & ~gap.shift(1).over(SERIES).fill_null(False)
    gid, ts = pl.col("gap_id"), pl.col("ts_utc")
    return (df.with_columns((real & pl.col("value").is_null()).alias("_gap"), real.alias("_real"))
              .with_columns(pl.when(gap).then(start.cum_sum()).alias("gap_id"),
                            ts.filter("_real").min().over(SERIES).alias("_first"),
                            ts.filter("_real").max().over(SERIES).alias("_last"))
              .with_columns(pl.when(gap).then(pl.len().over(gid)).cast(pl.Int32).alias("gap_len_h"),
                            pl.when(gap).then(pl.when((ts.min().over(gid) == pl.col("_first"))
                                                      | (ts.max().over(gid) == pl.col("_last")))
                                              .then(pl.lit("edge")).otherwise(pl.lit("inside")))
                            .alias("gap_position"))
              .drop("_gap", "_real", "_first", "_last"))


def chi2_test(sub: pl.DataFrame, dim: str) -> dict:
    observed = (sub.group_by(dim, "missing").len()
                   .pivot(sort_columns=True, on="missing", index=dim, values="len").fill_null(0)
                   .sort(dim).drop(dim).to_numpy().astype(float))
    expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    critical = float(stats.chi2.ppf(1 - ALPHA, dof))
    return {"chi2": round(chi2, 1), "df": dof, "critical_value": round(critical, 3),
            "result": "dependent" if chi2 > critical else "independent"}


def gap_length_table(df: pl.DataFrame) -> pl.DataFrame:
    gaps = df.filter(pl.col("gap_id").is_not_null()).group_by("gap_id").agg(pl.col("gap_len_h").first())
    bins = [(1, 1, "1 hour"), (2, 3, "2–3 hours"), (4, 24, "4–24 hours"), (25, 168, "1–7 days"),
            (169, None, "> 7 days")]
    label = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(bins):
        label = pl.when(pl.col("gap_len_h") >= lo if hi is None else pl.col("gap_len_h").is_between(lo, hi)) \
                  .then(pl.lit(name)).otherwise(label)
    order = pl.DataFrame({"length": [b[2] for b in bins], "_o": range(len(bins))})
    return (order.join(gaps.with_columns(label.alias("length")).group_by("length")
                       .agg(pl.len().alias("gaps"), pl.col("gap_len_h").sum().alias("hours")),
                       on="length", how="left").fill_null(0).sort("_o").drop("_o")
            .with_columns((100 * pl.col("gaps") / pl.col("gaps").sum()).round(1).alias("pct_gaps"),
                          (100 * pl.col("hours") / pl.col("hours").sum()).round(1).alias("pct_hours")))


def analysis(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    real = df.filter(~pl.col("status").is_in(STRUCTURAL)).with_columns(pl.col("value").is_null().alias("missing"))
    pct_missing = (100 * pl.col("missing").mean()).round(2).alias("pct")
    out: dict[str, pl.DataFrame] = {}
    out["status_by_pollutant"] = (df.group_by("pollutant", "status").len()
                                  .pivot(on="status", index="pollutant", values="len", sort_columns=True).fill_null(0)
                                  .sort("pollutant"))
    out["missing_by_station"] = (real.group_by("station_id", "pollutant")
                                 .agg((100 * pl.col("missing").mean()).round(1).alias("pct"))
                                 .with_columns(pl.col("station_id").cast(pl.String))
                                 .pivot(sort_columns=True, on="pollutant", index="station_id", values="pct")
                                 .sort("station_id"))
    out["gaps"] = gap_length_table(df)
    out["longest_gap"] = (df.filter(pl.col("gap_id").is_not_null())
                          .group_by("gap_id").agg(pl.col("pollutant").first(), pl.col("station_id").first(),
                                                  pl.col("ts_local").min().dt.date().alias("start"),
                                                  pl.col("gap_len_h").first())
                          .sort("gap_len_h", descending=True).group_by("pollutant", maintain_order=True).first()
                          .with_columns((pl.col("gap_len_h") / 24).round(1).alias("days")).drop("gap_id")
                          .sort("pollutant"))
    out["missing_by_month"] = real.group_by("month").agg(pct_missing).sort("month")
    out["missing_by_hour"] = real.group_by("hour").agg(pct_missing).sort("hour")
    out["missing_by_year"] = real.group_by("year").agg(pct_missing).sort("year")
    ind = (real.select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), pl.col("missing").cast(pl.Int8))
               .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="missing"))
    out["missing_correlation"] = pl.DataFrame([
        {"pollutant": a, **{b: 1.0 if a == b else round(ind.select(a, b).drop_nulls().select(pl.corr(a, b)).item(), 2)
                            for b in NAMES}} for a in NAMES])
    out["chi2"] = pl.DataFrame([{"pollutant": p, "attribute": dim,
                                 **chi2_test(real.filter(pl.col("pollutant") == p), dim)}
                                for p in NAMES for dim in ("month", "year", "hour")])
    for name, t in out.items():
        table(t, f"p3_{name}")
    return out


def nox_formula_check(df: pl.DataFrame) -> pl.DataFrame:
    w = (df.filter(pl.col("pollutant").is_in(["NOx", "NO2", "NO"]) & pl.col("value").is_not_null())
           .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value")
           .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value").drop_nulls())
    e = pl.col("NO2") + CFG["quality"]["no_to_no2"] * pl.col("NO") - pl.col("NOx")
    return w.select(pl.len().alias("hours"), e.abs().mean().alias("mean_absolute_error"),
                    pl.col("NOx").mean().alias("mean_NOx"))


def evaluate_methods(df: pl.DataFrame, class_mean: pl.DataFrame,
                     attr_mean: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(M["seed"])
    real_null = (pl.col("value").is_null() & ~pl.col("status").is_in(STRUCTURAL)).cast(pl.Int8)
    window = 2 * M["near_real_gap_h"] + 1
    h = (df.select(*SERIES, "month", "hour", "value", "status", "f_gas_period_flag")
           .with_columns(pl.int_range(pl.len()).over(SERIES).alias("_i"),
                         real_null.rolling_max(window, center=True, min_samples=1).over(SERIES).alias("_near"))
           .join(class_mean, on=CLASS, how="left").join(attr_mean, on=SERIES, how="left"))
    rows, ctx = [], []
    for length in M["eval_gap_lengths_h"]:
        span = length + 2
        complete = (pl.col("value").is_not_null() & ~pl.col("f_gas_period_flag")).cast(pl.Int32)
        cand = (h.with_columns(complete.rolling_sum(span).over(SERIES).alias("_ok"))
                 .filter(pl.col("_ok") == span).select(*SERIES, (pl.col("_i") - span + 1).alias("start")))
        cand = (cand.with_columns(pl.Series("_u", rng.random(cand.height))).sort("_u")
                    .filter(pl.int_range(pl.len()).over(SERIES) < M["eval_gaps_per_series"]).drop("_u"))
        seg = (cand.with_columns(pl.int_range(pl.len()).alias("seg"), pl.int_ranges(0, span).alias("k"))
                   .explode("k").with_columns((pl.col("start") + pl.col("k")).alias("_i"))
                   .join(h, on=[*SERIES, "_i"], how="left").sort("seg", "k")
                   .with_columns(pl.col("value").first().over("seg").alias("_a"),
                                 pl.col("value").last().over("seg").alias("_b"),
                                 (pl.col("_near").max().over("seg") == 1).alias("near_real_gap")))
        hidden = (seg.filter(pl.col("k").is_between(1, length))
                     .with_columns((pl.col("_a") + (pl.col("_b") - pl.col("_a")) * pl.col("k") / (length + 1))
                                   .alias("linear")))
        for method in ("attribute_mean", "class_mean", "linear"):
            err = (pl.col(method) - pl.col("value")).abs()
            rows += [{"gap_h": length, "pollutant": p, "method": method, "MAE": mae, "RMSE": rmse}
                     for p, mae, rmse in hidden.group_by("pollutant")
                     .agg(err.mean().alias("mae"), (err ** 2).mean().sqrt().alias("rmse")).rows()]
            ctx += [{"gap_h": length, "near_real_gap": near, "method": method, "MAE": mae, "hours": n}
                    for near, mae, n in hidden.group_by("near_real_gap").agg(err.mean(), pl.len()).rows()]
    evaluation = pl.DataFrame(rows, schema={"gap_h": pl.Int64, "pollutant": pl.String, "method": pl.String,
                                            "MAE": pl.Float64, "RMSE": pl.Float64}).sort("gap_h", "method", "pollutant")
    context = (pl.DataFrame(ctx).with_columns(pl.when("near_real_gap").then(pl.lit("near a real gap"))
                                              .otherwise(pl.lit("quiet period")).alias("context"))
                 .pivot(on="method", index=["gap_h", "context", "hours"], values="MAE", sort_columns=True)
                 .select("gap_h", "context", "hours", "attribute_mean", "class_mean", "linear")
                 .sort("gap_h", "context"))
    return evaluation, context


def impute(df: pl.DataFrame) -> pl.DataFrame:
    real_null = ~pl.col("status").is_in(["ok", *STRUCTURAL])
    df = df.with_columns(pl.col("value").alias("value_imputed"),
                         pl.when(real_null).then(pl.lit("left_null")).otherwise(pl.lit("none")).alias("impute_method"))

    def fill(df: pl.DataFrame, cond: pl.Expr, pred: pl.Expr, name: str) -> pl.DataFrame:
        use = cond & pl.col("value_imputed").is_null() & pred.is_not_null()
        return df.with_columns(
            pl.when(use).then(pl.lit(name)).otherwise(pl.col("impute_method")).alias("impute_method"),
            pl.when(use).then(pred.clip(lower_bound=0)).otherwise(pl.col("value_imputed")).alias("value_imputed"))

    w = (df.filter(pl.col("pollutant").is_in(["NO2", "NO"]))
           .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value")
           .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value"))
    df = df.join(w, on=["station_id", "ts_utc"], how="left").sort(KEY)
    nox = pl.col("NO2") + CFG["quality"]["no_to_no2"] * pl.col("NO")
    df = fill(df, real_null & (pl.col("pollutant") == "NOx"), nox, "deterministic").drop("NO2", "NO")

    inside = real_null & (pl.col("gap_position") == "inside") & (pl.col("gap_len_h") <= M["max_gap_h"])
    return fill(df, inside, pl.col("value_imputed").interpolate().over(SERIES), "linear")


def check_imputation(out: pl.DataFrame) -> None:
    pmax = by_name("physical_max")
    bad = out.filter((pl.col("status").is_in(STRUCTURAL) & pl.col("value_imputed").is_not_null())
                     | (pl.col("value_imputed") < 0)
                     | (pl.col("value_imputed") > pl.col("pollutant").cast(pl.String)
                        .replace_strict(pmax, return_dtype=pl.Float64))
                     | (pl.col("value").is_not_null() & (pl.col("value_imputed") != pl.col("value"))))
    if bad.height:
        raise SystemExit(f"STOP: {bad.height} invalid imputed values")


def decision_table(out: pl.DataFrame, clean: pl.DataFrame) -> pl.DataFrame:
    act = dict(clean["clean_action"].cast(pl.String).value_counts().rows())
    st = dict(out["status"].value_counts().rows())
    by_length = f"By gap length (≤ {M['max_gap_h']}h inside the series)"
    rows = [
        ("Pollutant not measured by the station", "attribute not applicable", "not_measured",
         st.get("not_measured", 0), "Never"),
        ("Before the start / after the end of the series", "not collected", "outside_period",
         st.get("outside_period", 0), "Never"),
        ("Hour missing completely", "equipment malfunction / not recorded", "missing_row",
         st.get("missing_row", 0), by_length),
        ("Invalid / maintenance (Validity −1/−99)", "deleted by the institution as incorrect", "null_invalid",
         act.get("null_invalid", 0), by_length),
        ("Sentinel / absurd (all also have Validity −1/−99, so they are counted as null_invalid)",
         "disguised missing value", "null_sentinel / null_absurd",
         act.get("null_sentinel", 0) + act.get("null_absurd", 0), by_length),
        ("NO/NO2 before the analyzer change, November 2019", "deleted: inconsistent with other recorded data",
         "null_calibration", act.get("null_calibration", 0), f"{by_length}; in practice longer → stays null"),
        ("Above the physical limit", "noise / instrument error", "null_physical", act.get("null_physical", 0),
         by_length),
        ("Large negative", "noise / instrument error", "null_negative", act.get("null_negative", 0), by_length),
        (f"Stuck value (non-zero, or zero for {', '.join(p for p in NAMES if p not in Q_STUCK_OK)})",
         "equipment malfunction", "null_stuck", act.get("null_stuck", 0), by_length),
        ("Isolated spike", "noise", "null_spike", act.get("null_spike", 0), by_length),
        ("PM2.5 > PM10 outside tolerance, inconsistent NOx", "deleted: inconsistent with other recorded data",
         "null_inconsistent", act.get("null_inconsistent", 0),
         f"{by_length}; NOx by formula when NO and NO2 are valid"),
    ]
    return table(pl.DataFrame(rows, schema=["category", "reason (lecture)", "action", "count", "imputed?"],
                              orient="row"), "p3_missing_decisions")


def why_still_null(out: pl.DataFrame) -> pl.DataFrame:
    long_gap = pl.col("gap_len_h") > M["max_gap_h"]
    period_2019 = (pl.col("clean_action") == "null_calibration") | (
        (pl.col("pollutant") == "NOx") & (pl.col("clean_action") == "null_inconsistent") & long_gap)
    reason = (pl.when(period_2019).then(pl.lit("2019 period before the analyzer change"))
                .when(pl.col("gap_position") == "edge").then(pl.lit("at the edge of the series (no value on one side)"))
                .when(long_gap).then(pl.lit(f"gap > {M['max_gap_h']}h"))
                .otherwise(pl.lit("other")))
    return table(out.filter(pl.col("impute_method") == "left_null").with_columns(reason.alias("reason"))
                 .group_by("reason").len().sort("len", descending=True), "p3_why_still_null")


def run() -> SimpleNamespace:
    clean = (scan("silver", "measurements_clean")
             .with_columns(pl.col("pollutant").cast(pl.String).cast(POLLUTANT_TYPE))
             .select(*KEY, "value", "value_raw", "clean_action", "f_gas_period_flag").collect())
    df = add_gaps(build_grid(clean))
    a = analysis(df)

    known = df.filter(pl.col("value").is_not_null() & ~pl.col("f_gas_period_flag"))
    class_mean = known.group_by(CLASS).agg(pl.col("value").mean().alias("class_mean"))
    attr_mean = known.group_by(SERIES).agg(pl.col("value").mean().alias("attribute_mean"))
    det = table(nox_formula_check(df), "p3_nox_formula_check")
    ex = known.filter((pl.col("pollutant") == "PM2_5") & (pl.col("station_id") == "XK0001A"))
    example = table(ex.filter(pl.col("month").is_in([1, 7]) & pl.col("hour").is_in([5, 20]))
                    .group_by("month", "hour").agg(pl.col("value").mean().round(1).alias("class_mean"))
                    .with_columns(pl.lit(round(ex["value"].mean(), 1)).alias("attribute_mean"))
                    .sort("month", "hour"), "p3_class_mean_example")
    evaluation, context = evaluate_methods(df, class_mean, attr_mean)
    context = table(context, "p3_method_evaluation_by_context")
    table(evaluation, "p3_method_evaluation")
    eval_wide = table(evaluation.group_by("gap_h", "method").agg(pl.col("MAE").mean())
                      .pivot(sort_columns=True, on="method", index="gap_h", values="MAE")
                      .select("gap_h", "attribute_mean", "class_mean", "linear").sort("gap_h"),
                      "p3_method_evaluation_summary")
    best = evaluation.sort("MAE").group_by("gap_h", "pollutant", maintain_order=True).first()
    linear_mae = evaluation.filter(pl.col("method") == "linear").select("gap_h", "pollutant",
                                                                         pl.col("MAE").alias("linear_MAE"))
    exceptions = (best.filter(pl.col("method") != "linear").join(linear_mae, on=["gap_h", "pollutant"])
                      .with_columns((100 * (pl.col("linear_MAE") / pl.col("MAE") - 1)).alias("linear_worse_pct"))
                      .sort("gap_h", "pollutant"))

    out = impute(df)
    check_imputation(out)
    write(out.select(*KEY, "ts_local", "value_raw", "value", "clean_action", "status", "gap_id", "gap_len_h",
                     "gap_position", "value_imputed", "impute_method", "f_gas_period_flag"),
          "gold", "hourly_long", partition_by="pollutant")
    measured = out.filter(~pl.col("status").is_in(STRUCTURAL))
    lineage("p3", "hourly grid + imputation → gold", clean.height, out.height, measured["value"].null_count(),
            measured["value_imputed"].null_count())

    sens = []
    for p in NAMES:
        s = out.filter(pl.col("pollutant") == p)
        x, y = s["value"].drop_nulls(), s["value_imputed"].drop_nulls()
        sens.append({"pollutant": p, "n_before": x.len(), "n_after": y.len(), "mean_before": x.mean(),
                     "mean_after": y.mean(), "std_before": x.std(), "std_after": y.std(),
                     "mean_change_pct": 100 * (y.mean() / x.mean() - 1),
                     "std_change_pct": 100 * (y.std() / x.std() - 1)})
    methods = table(out.group_by("pollutant", "impute_method").len()
                    .pivot(on="impute_method", index="pollutant", values="len", sort_columns=True)
                    .fill_null(0).sort("pollutant"), "p3_methods_by_pollutant")
    complete = measured.group_by("station_id", "ts_utc").agg(
        pl.col("value").is_not_null().all().alias("before"), pl.col("value_imputed").is_not_null().all().alias("after"))
    missing_map = (measured.with_columns(pl.col("ts_local").dt.strftime("%Y-%m").alias("ym"),
                                         pl.format("{} {}", pl.col("station_id"), pl.col("pollutant")).alias("series"))
                   .group_by("series", "ym").agg((100 * pl.col("value").is_null().mean()).alias("pct"))
                   .pivot(on="ym", index="series", values="pct", sort_columns=True).sort("series"))
    st = a["status_by_pollutant"]
    total = {c: int(st[c].sum()) for c in st.columns if c != "pollutant"}
    real_hours = sum(v for k, v in total.items() if k not in STRUCTURAL)
    return SimpleNamespace(
        a=a, det=det, example=example, attr_mean=ex["value"].mean(), methods=methods, why_left=why_still_null(out),
        decisions=decision_table(out, clean), sens=table(pl.DataFrame(sens), "p3_before_after_imputation"),
        eval_wide=eval_wide, linear_best=best.filter(pl.col("method") == "linear").height, n_best=best.height,
        exceptions=exceptions, context=context,
        complete_pct={k: round(100 * complete[k].mean(), 1) for k in ("before", "after")},
        n_hours=out.height, total=total, real_hours=real_hours, real_missing=real_hours - total.get("ok", 0),
        n_null_after=measured["value_imputed"].null_count(), missing_map=missing_map,
    )
