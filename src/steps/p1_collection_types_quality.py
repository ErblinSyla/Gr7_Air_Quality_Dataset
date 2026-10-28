"""p1 – Data collection, data type definition and data quality assessment (no value is changed here)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, DATA, POLLUTANT_TYPE, POLLUTANTS, ROOT, SERIES, by_name, lineage, table, write

Q = CFG["quality"]
VALID = [1, 2]

DICTIONARY = {
    "Samplingpoint": ("station_id, pollutant, instrument", "nominal", "discrete",
                      "sampling point ID: station_pollutant_instrument"),
    "Pollutant": ("pollutant_code", "nominal", "discrete", "EEA pollutant code"),
    "Start": ("ts_utc, ts_local", "interval", "continuous", "start of the hour (raw: fixed UTC+1)"),
    "End": ("ts_end", "interval", "continuous", "end of the hour"),
    "Value": ("value_raw", "ratio", "continuous", "concentration (true zero, ratios are meaningful)"),
    "Unit": ("unit", "nominal", "discrete", "declared unit"),
    "AggType": ("agg_type", "nominal", "discrete", "aggregation type"),
    "Validity": ("validity", "nominal (code)", "discrete",
                 "1 valid, 2 below detection limit, -1 invalid, -99 maintenance"),
    "Verification": ("verification", "nominal (code)", "discrete", "verification status"),
    "ResultTime": ("result_time", "interval", "continuous", "publication time of the result"),
    "DataCapture": ("data_capture", "ratio", "continuous", "data capture percentage"),
    "FkObservationLog": ("fk_observation_log", "nominal (identifier)", "discrete", "EEA administrative reference"),
}

FLAGS = ["f_invalid", "f_below_dl", "f_sentinel", "f_absurd", "f_negative", "f_stuck", "f_stuck_zero", "f_spike",
         "f_pm25_gt_pm10", "f_pm_out_tol", "f_outlier_stat"]


def collect() -> tuple[pl.DataFrame, dict]:
    files = sorted((ROOT / CFG["paths"]["raw"]).glob("*.parquet"))
    raw = pl.scan_parquet(files, include_file_paths="source_file")
    raw_schema = dict(raw.collect_schema())
    names = {code: p["name"] for code, p in POLLUTANTS.items()}
    t = CFG["time"]
    sp = pl.col("Samplingpoint")
    bronze = raw.select(
        sp.str.extract(r"(XK\d+[A-Z]?)", 1).cast(pl.Categorical).alias("station_id"),
        pl.col("Pollutant").cast(pl.Int16).alias("pollutant_code"),
        pl.col("Pollutant").replace_strict(names, return_dtype=pl.String).cast(POLLUTANT_TYPE).alias("pollutant"),
        sp.str.extract(r"_(\d+)_\d+$", 1).cast(pl.Int16).alias("code_in_id"),
        sp.str.extract(r"_(\d+)$", 1).cast(pl.Categorical).alias("instrument"),
        sp.str.split("/").list.last().cast(pl.Categorical).alias("sampling_point_id"),
        pl.col("Start").dt.replace_time_zone(t["raw_tz"]).dt.convert_time_zone("UTC").alias("ts_utc"),
        pl.col("End").dt.replace_time_zone(t["raw_tz"]).dt.convert_time_zone("UTC").alias("ts_end"),
        pl.col("Value").cast(pl.Float64).alias("value_raw"),
        pl.col("Unit").cast(pl.Categorical).alias("unit"),
        pl.col("AggType").cast(pl.Categorical).alias("agg_type"),
        pl.col("Validity").cast(pl.Int8).alias("validity"),
        pl.col("Verification").cast(pl.Int8).alias("verification"),
        pl.col("ResultTime").alias("result_time"),
        pl.col("DataCapture").cast(pl.Float64).alias("data_capture"),
        pl.col("FkObservationLog").alias("fk_observation_log"),
        pl.col("source_file").str.extract(r"(SPO-[^\\/]+)\.parquet$", 1).alias("source_file"),
    ).with_columns(pl.col("ts_utc").dt.convert_time_zone(t["local_tz"]).alias("ts_local")).collect()

    checks = {
        "rows": (bronze.height, CFG["expected"]["rows"]),
        "sampling points": (bronze["sampling_point_id"].n_unique(), CFG["expected"]["sampling_points"]),
        "stations": (bronze["station_id"].n_unique(), CFG["expected"]["stations"]),
        "code in ID != Pollutant": (bronze.filter(pl.col("code_in_id") != pl.col("pollutant_code")).height, 0),
        "file name != point ID": (bronze.filter(pl.col("source_file") != pl.col("sampling_point_id")
                                                .cast(pl.String)).height, 0),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"STOP: the data does not match its description: {bad}")
    return bronze.drop("code_in_id", "source_file").sort("station_id", "pollutant", "ts_utc"), raw_schema


def data_dictionary(bronze: pl.DataFrame, raw_schema: dict) -> pl.DataFrame:
    units = ", ".join(f"{p['name']}: {p['unit']}" for p in POLLUTANTS.values())
    rows = []
    for raw_col, (new_cols, stat, kind, desc) in DICTIONARY.items():
        if raw_col == "Value":
            desc = f"{desc} — units: {units}"
        first = new_cols.split(",")[0].strip()
        s = bronze[first] if first in bronze.columns else bronze["station_id"]
        rows.append({"raw_column": raw_col, "raw_type": str(raw_schema[raw_col]), "new_column": new_cols,
                     "new_type": str(s.dtype), "attribute_type": stat, "discrete_continuous": kind,
                     "description": desc, "pct_empty": round(100 * s.null_count() / len(s), 2),
                     "example": str(s.drop_nulls().head(1).to_list()[0]) if s.null_count() < len(s) else "–"})
    return table(pl.DataFrame(rows), "p1_data_dictionary")


def add_quality_flags(df: pl.DataFrame) -> pl.DataFrame:
    v, validity = pl.col("value_raw"), pl.col("validity")
    df = df.with_columns(
        validity.is_in([-1, -99]).alias("f_invalid"),
        (validity == 2).alias("f_below_dl"),
        v.is_in([float(s) for s in Q["sentinels"]]).alias("f_sentinel"),
        (v >= Q["absurd_threshold"]).alias("f_absurd"),
    ).with_columns(
        (validity.is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd")).alias("_usable"),
    ).with_columns((pl.col("_usable") & (v < 0)).alias("f_negative"))

    u = df.filter("_usable").select(*SERIES, "ts_utc", "value_raw")
    one_hour = pl.duration(hours=1, time_unit="ns")
    prev_v, prev_t = v.shift(1).over(SERIES), pl.col("ts_utc").shift(1).over(SERIES)
    new_run = ((v != prev_v) | (pl.col("ts_utc") - prev_t != one_hour)).fill_null(True)
    stuck = (u.with_columns(new_run.cum_sum().alias("_run"))
              .with_columns((pl.len().over("_run") >= Q["stuck_min_hours"]).alias("f_stuck"))
              .select(*SERIES, "ts_utc", "f_stuck", (pl.col("f_stuck") & (v == 0)).alias("f_stuck_zero")))

    d = u.with_columns(
        (v - prev_v).alias("_up"), (v - v.shift(-1).over(SERIES)).alias("_down"),
        (pl.col("ts_utc") - prev_t == one_hour).alias("_prev_ok"),
        (pl.col("ts_utc").shift(-1).over(SERIES) - pl.col("ts_utc") == one_hour).alias("_next_ok"))
    thr = (d.filter("_prev_ok").group_by("pollutant")
            .agg(pl.col("_up").abs().quantile(Q["spike_quantile"]).alias("spike_threshold")))
    table(thr.sort("pollutant"), "p1_spike_thresholds")
    spikes = (d.join(thr, on="pollutant").select(
        *SERIES, "ts_utc", (pl.col("_prev_ok") & pl.col("_next_ok") & (pl.col("_up") > pl.col("spike_threshold"))
                            & (pl.col("_down") > pl.col("spike_threshold"))).fill_null(False).alias("f_spike")))

    def part(name: str) -> pl.DataFrame:
        return u.filter(pl.col("pollutant") == name).select("station_id", "ts_utc", pl.col("value_raw").alias(name))

    tol = Q["pm_tolerance"]
    pm = (part("PM2_5").join(part("PM10"), on=["station_id", "ts_utc"]).filter(pl.col("PM2_5") > pl.col("PM10"))
          .select("station_id", "ts_utc", pl.lit(True).alias("f_pm25_gt_pm10"),
                  (pl.col("PM2_5") - pl.col("PM10")
                   > pl.max_horizontal(pl.lit(tol["abs"]), tol["rel"] * pl.col("PM10"))).alias("f_pm_out_tol")))
    pm = pl.concat([pm.with_columns(pl.lit(p).cast(POLLUTANT_TYPE).alias("pollutant")) for p in ("PM2_5", "PM10")])

    q1, q3, k = v.quantile(0.25).over("pollutant"), v.quantile(0.75).over("pollutant"), Q["outlier_iqr_k"]
    outl = u.select(*SERIES, "ts_utc", ((v < q1 - k * (q3 - q1)) | (v > q3 + k * (q3 - q1))).alias("f_outlier_stat"))

    key = [*SERIES, "ts_utc"]
    for f in (stuck, spikes, pm, outl):
        cols = [c for c in f.columns if c.startswith("f_")]
        df = df.join(f, on=key, how="left").with_columns(pl.col(cols).fill_null(False))
    return df.drop("_usable")


def nox_ratio_evidence(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    good = df.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd")
                     & (pl.col("value_raw") > 0))
    w = (good.filter(pl.col("pollutant").is_in(["NOx", "NO2", "NO"]))
             .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value_raw")
             .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value_raw").drop_nulls()
             .with_columns((pl.col("NOx") / (pl.col("NO2") + Q["no_to_no2"] * pl.col("NO"))).alias("ratio"),
                           pl.col("ts_utc").dt.convert_time_zone(CFG["time"]["local_tz"]).dt.year().alias("year")))
    lo, hi = Q["nox_ppb_ratio"]
    by_year = (w.group_by("station_id", "year")
                .agg(pl.len().alias("hours"), pl.col("ratio").median().round(4).alias("median"),
                     (100 * pl.col("ratio").is_between(lo, hi).mean()).round(1).alias("pct_in_ppb_range"))
                .sort("station_id", "year"))
    weekly = (w.sort("ts_utc").group_by_dynamic("ts_utc", every="1w", group_by="station_id")
               .agg(pl.col("ratio").median().alias("median")).sort("station_id", "ts_utc"))
    return by_year, weekly


def quality_report(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    n = df.height
    t: dict[str, pl.DataFrame] = {}
    t["completeness_columns"] = pl.DataFrame([{"column": c, "nulls": df[c].null_count(),
                                               "pct_null": round(100 * df[c].null_count() / n, 2)}
                                              for c in df.columns if not c.startswith("f_")])
    t["completeness_series"] = (df.group_by(SERIES)
                                .agg(pl.len().alias("rows"),
                                     ((pl.col("ts_utc").max() - pl.col("ts_utc").min()).dt.total_hours() + 1)
                                     .alias("expected_hours"))
                                .with_columns((pl.col("expected_hours") - pl.col("rows")).alias("hours_without_row"))
                                .with_columns((100 * pl.col("hours_without_row") / pl.col("expected_hours")).round(2)
                                              .alias("pct_without_row"))
                                .sort(SERIES))
    t["validity"] = (df.group_by("pollutant", "validity").len()
                     .with_columns((100 * pl.col("len") / pl.col("len").sum().over("pollutant")).round(2).alias("pct"))
                     .with_columns(pl.format("validity {}", pl.col("validity")).alias("validity"))
                     .pivot(on="validity", index="pollutant", values="pct", sort_columns=True)
                     .fill_null(0).sort("pollutant"))
    t["flags"] = (df.group_by("pollutant").agg(pl.len().alias("n"), *[pl.col(f).sum() for f in FLAGS])
                  .sort("pollutant"))
    pol = pl.col("pollutant").cast(pl.String)
    canonical = pl.col("value_raw") * pol.replace_strict(by_name("ppb_to_ugm3", 1.0), return_dtype=pl.Float64)
    threshold = pol.replace_strict(by_name("neg_zero"), return_dtype=pl.Float64)
    t["negatives"] = (df.filter("f_negative").group_by("pollutant")
                      .agg(pl.len().alias("n"), canonical.min().round(2).alias("min_canonical"),
                           threshold.first().alias("neg_zero_threshold"),
                           (canonical >= threshold).sum().alias("small_to_0"),
                           (canonical < threshold).sum().alias("large_to_null"))
                      .with_columns((100 * pl.col("small_to_0") / pl.col("n")).round(1).alias("pct_small"))
                      .sort("pollutant"))
    t["stuck"] = (df.filter("f_stuck").group_by("pollutant")
                  .agg(pl.len().alias("stuck_hours"), pl.col("f_stuck_zero").sum().alias("of_which_zero"))
                  .sort("pollutant"))
    t["units"] = (df.group_by("pollutant").agg(pl.col("unit").cast(pl.String).unique().str.join(", ")
                                               .alias("declared_unit")).sort("pollutant"))
    dup = n - df.select("station_id", "pollutant", "ts_utc").n_unique()
    end_ok = df.filter(pl.col("ts_end") - pl.col("ts_utc") != pl.duration(hours=1, time_unit="ns")).height
    t["uniqueness"] = pl.DataFrame({"rows": [n], "duplicate_keys": [dup]})
    t["uninformative_columns"] = pl.DataFrame([
        {"column": c, "unique_values": df[c].n_unique(), "pct_null": round(100 * df[c].null_count() / n, 2),
         "verdict": "100% empty" if df[c].null_count() == n else ("constant" if df[c].n_unique() == 1 else why)}
        for c, why in (("data_capture", ""), ("agg_type", ""), ("verification", ""),
                       ("fk_observation_log", "administrative identifier, no information about pollution"),
                       ("unit", "fully determined by the pollutant (CO mg/m³, others µg/m³): redundant"),
                       ("result_time", "administrative: publication time, unrelated to the measurement"))
    ] + [{"column": "ts_end", "unique_values": None, "pct_null": 0.0,
          "verdict": f"redundant: End = Start + 1h (exceptions: {end_ok})"}])
    t["invalid_station_year"] = (df.with_columns(pl.col("ts_local").dt.year().alias("year"))
                                 .group_by("station_id", "year").agg((100 * pl.col("f_invalid").mean()).round(1)
                                                                     .alias("pct"))
                                 .with_columns(pl.col("station_id").cast(pl.String))
                                 .pivot(on="year", index="station_id", values="pct", sort_columns=True)
                                 .sort("station_id"))
    delay = (pl.col("result_time") - pl.col("ts_end").dt.replace_time_zone(None)).dt.total_days()
    t["timeliness"] = (df.group_by(pl.col("ts_local").dt.year().alias("year"))
                       .agg(delay.min().alias("min_delay_days"), delay.median().alias("median_delay_days"),
                            delay.max().alias("max_delay_days"))
                       .sort("year"))
    for name, tab in t.items():
        table(tab, f"p1_quality_{name}")
    return t


def quality_dimensions(q: dict, flags: pl.DataFrame, bronze: pl.DataFrame) -> pl.DataFrame:
    tim = q["timeliness"]
    return table(pl.DataFrame([
        ("Accuracy", "Are the values correct?",
         f"{int(flags['f_invalid'][0]):,} invalid (Validity −1/−99), "
         f"{int(flags['f_sentinel'][0]):,} sentinel −9900, "
         f"{int(flags['f_negative'][0]):,} negative, {int(flags['f_stuck'][0]):,} stuck hours, "
         f"{int(flags['f_spike'][0]):,} spikes"),
        ("Completeness", "Is anything missing?",
         f"{int(q['completeness_series']['hours_without_row'].sum()):,} hours without a row; DataCapture 100% empty"),
        ("Consistency", "Do related values agree?",
         f"PM2.5 > PM10 in {int(flags['f_pm25_gt_pm10'][0]) // 2:,} hours; NOx is in ppb although labelled µg/m³"),
        ("Timeliness", "Is the data updated in time?",
         f"results are published {tim['median_delay_days'].min():.0f}–{tim['median_delay_days'].max():.0f} days "
         "(median per year) after measurement: E1a is a verified series, not real-time data"),
        ("Believability", "How much can the data be trusted?",
         f"official source (EEA, IHMK monitoring network); Verification = {bronze['verification'].unique().to_list()} "
         "(verified) for every row"),
        ("Interpretability", "How easily can the data be understood?",
         "codes (Pollutant 6001, Validity −99) are meaningless without a description → data dictionary (2.2)"),
    ], schema=["dimension", "key_question", "finding"], orient="row"), "p1_quality_dimensions")


def run() -> SimpleNamespace:
    bronze, raw_schema = collect()
    write(bronze, "bronze", "measurements", partition_by="pollutant")

    meta = pl.read_csv(next((ROOT / CFG["paths"]["metadata"]).glob("*.csv")), infer_schema=False)
    stations_raw = meta.filter(pl.col("Air Quality Station EoI Code").str.starts_with("XK"))
    write(stations_raw, "bronze", "stations_raw")
    lineage("p1", "collection: raw → bronze", bronze.height, bronze.height, 0, bronze["value_raw"].null_count())
    lineage("p1", "metadata: Europe → Kosovo", meta.height, stations_raw.height, 0, 0)

    dictionary = data_dictionary(bronze, raw_schema)
    flagged = add_quality_flags(bronze)
    q = quality_report(flagged)
    nox_year, nox_week = nox_ratio_evidence(flagged)
    table(nox_year, "p1_nox_ratio_station_year")

    flags = q["flags"].select(pl.exclude("pollutant", "n")).sum()
    inv = q["invalid_station_year"]
    usable = flagged.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd"))["value_raw"]
    return SimpleNamespace(
        bronze=bronze, meta=meta, stations_raw=stations_raw, dictionary=dictionary, q=q, flags=flags,
        nox_year=nox_year, nox_week=nox_week, usable_n=usable.len(), usable_mean=usable.mean(),
        mismatch=nox_year.filter(~pl.col("median").is_between(*Q["nox_ppb_ratio"])),
        worst=(inv.select("station_id", pl.mean_horizontal(pl.exclude("station_id")).alias("m"))
                  .sort("m", descending=True).row(0)),
        inv_years=list(zip(inv.columns[1:], inv.select(pl.exclude("station_id")).mean().row(0), strict=True)),
        dims=quality_dimensions(q, flags, bronze),
        size_raw=sum(f.stat().st_size for f in (ROOT / CFG["paths"]["raw"]).glob("*.parquet")) / 2**20,
        size_bronze=sum(f.stat().st_size for f in (DATA / "bronze" / "measurements").rglob("*.parquet")) / 2**20,
    )
