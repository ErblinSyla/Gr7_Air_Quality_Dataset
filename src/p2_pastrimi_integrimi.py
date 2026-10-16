from __future__ import annotations

import html
import time

import numpy as np
import polars as pl

from p1_mbledhja_tipet_kualiteti import VALID, add_quality_flags
from utils import (
    CFG, COLORS, NAMES, by_name, lineage, md, plt, save_figure, scan, table, write, write_section,
)

Q = CFG["quality"]
NOX_FACTOR = by_name("ppb_to_ugm3", 1.0)["NOx"]
ACTIONS = ["keep", "zero_negative", "null_invalid", "null_sentinel", "null_absurd", "null_calibration", "null_physical",
           "null_negative", "null_stuck", "null_spike", "null_inconsistent"]


def record_sets(raw: pl.DataFrame) -> pl.DataFrame:

    def text(col: str) -> pl.Expr:
        e = pl.col(col).str.strip_chars().str.strip_chars('"').str.strip_chars()
        e = e.map_elements(html.unescape, return_dtype=pl.String)   # p.sh. "Dardhisht&#235;" → "Dardhishtë"
        return pl.when(e == "").then(None).otherwise(e)

    year = pl.when(pl.col("Year") == "").then(None).otherwise(pl.col("Year"))
    group = (pl.when(year.is_null()).then(pl.lit("legacy"))
             .when((year == "2024") & (pl.col("Municipality") != "")).then(pl.lit("2024_komuna"))
             .otherwise(year))
    return raw.select(
        pl.col("Air Quality Station EoI Code").alias("station_id"),
        group.alias("record_set"),
        text("Air Quality Station Name").alias("station_name"),
        text("Air Quality Station Nat Code").alias("national_code"),
        pl.col("Latitude").cast(pl.Float64).alias("lat"),
        pl.col("Longitude").cast(pl.Float64).alias("lon"),
        pl.col("Altitude").cast(pl.Float64).replace(0.0, None).alias("altitude_m"),   # 0 m e pamundur në Kosovë
        text("Air Quality Station Type").alias("station_type"),
        text("Air Quality Station Area").alias("station_area"),
        text("Municipality").alias("municipality"),
    )


def record_set_profile(m: pl.DataFrame) -> pl.DataFrame:

    return (m.group_by("record_set")
             .agg(pl.len().alias("rreshta"), pl.col("station_id").n_unique().alias("stacione"),
                  pl.col("station_type").unique().sort().str.join(", ").alias("tipet"),
                  pl.col("municipality").drop_nulls().unique().sort().str.join(", ").alias("komunat"))
             .sort("record_set"))


def haversine_km(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    rad = np.pi / 180
    a = (((lat2 - lat1) * rad / 2).sin() ** 2
         + (lat1 * rad).cos() * (lat2 * rad).cos() * (((lon2 - lon1) * rad / 2).sin() ** 2))
    return 2 * 6371 * a.sqrt().arcsin()


def clean_stations(raw: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    sc = CFG["stations"]
    m = record_sets(raw)
    ref = pl.DataFrame([{"station_id": k, **v} for k, v in sc["town_centroids"].items()])


    source_rank = pl.col("record_set").replace_strict({"2024": 0, "legacy": 1, "2017": 2, "2024_komuna": 3},
                                                      return_dtype=pl.Int8)
    coords = (m.select("station_id", "record_set", "lat", "lon", "altitude_m").unique()
               .join(ref.rename({"lat": "ref_lat", "lon": "ref_lon"}), on="station_id")
               .with_columns(haversine_km(pl.col("lat"), pl.col("lon"), pl.col("ref_lat"), pl.col("ref_lon"))
                             .round(1).alias("distance_km"))
               .sort("station_id", "distance_km", pl.col("altitude_m").is_null(), source_rank, "lat", "lon"))
    chosen_coords = coords.group_by("station_id", maintain_order=True).first().select(
        "station_id", "town", "lat", "lon", "altitude_m", "distance_km", pl.col("record_set").alias("coords_from"))

    trusted = m.filter(pl.col("record_set") != "2024_komuna")
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
        "atributi": a,
        "stacione_me_konflikt": m.group_by("station_id").agg(pl.col(a).drop_nulls().n_unique())
                                 .filter(pl.col(a) > 1).height,
        "shembull_XK0002A": str(m.filter(pl.col("station_id") == "XK0002A")[a].drop_nulls().unique().sort().to_list()),
    } for a in attrs])
    return stations, conflicts, coords.drop("ref_lat", "ref_lon"), record_set_profile(m)


def nox_triplets(df: pl.DataFrame, value: str) -> pl.DataFrame:
    """NOx, NO2 dhe NO në të njëjtin stacion dhe orë (format i gjerë)."""
    return (df.filter(pl.col("pollutant").is_in(["NOx", "NO2", "NO"]))
              .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), value)
              .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values=value)
              .drop_nulls(["NOx", "NO2", "NO"]))


def ratio() -> pl.Expr:
    return pl.col("NOx") / (pl.col("NO2") + Q["no_to_no2"] * pl.col("NO"))


def nox_consistent_from(df: pl.DataFrame) -> pl.DataFrame:
    """Për çdo stacion: java nga e cila mediana javore e raportit (raw) mbetet pa ndërprerje në rangun e ppb."""
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
    """f_nox_inconsistent: (a) çdo orë NOx para `consistent_from` të stacionit; (b) pas konvertimit,
    |NOx/(NO2 + 1.53·NO) − 1| > tolerancës (vetëm vlera të vlefshme)."""
    usable = df.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd"))
    hourly = (nox_triplets(usable, "value").filter((pl.col("NO2") + Q["no_to_no2"] * pl.col("NO")) > 0)
              .select("station_id", "ts_utc", ((ratio() - 1).abs() > Q["nox_tolerance"]).alias("_h")))
    df = (df.join(periods.select("station_id", "consistent_from"), on="station_id", how="left")
            .join(hourly.with_columns(pl.lit("NOx").cast(df.schema["pollutant"]).alias("pollutant")),
                  on=["station_id", "pollutant", "ts_utc"], how="left"))
    is_nox = pl.col("pollutant") == "NOx"
    before = (pl.col("ts_utc") < pl.col("consistent_from")).fill_null(False)
    period = is_nox & before
    # periudha para ndërrimit të analizuesve të gazeve (D33): NO dhe NO2 (i njëjti analizues me NOx) → null;
    # CO dhe SO2 vetëm shënohen (rritje e vogël, evidencë jo bindëse)
    return (df.with_columns(period.alias("f_nox_period"),
                            (is_nox & (period | pl.col("_h").fill_null(False))).alias("f_nox_inconsistent"),
                            (before & pl.col("pollutant").is_in(Q["analyzer_change"]["to_null"])).alias("f_calibration"),
                            (before & pl.col("pollutant").is_in(Q["analyzer_change"]["flag_only"]))
                            .alias("f_gas_period_flag"))
              .drop("consistent_from", "_h"))


def analyzer_change_evidence(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Niveli i 2019 ndaj normales sezonale të 2020–2024 (mediana e stacionit sipas javës së vitit).

    Nëse një ndotës është shumë mbi normalen para ndërrimit dhe normal pas tij, ndërsa grimcat (të njëjtat burime
    djegieje, instrumente të tjera) janë normale, ndryshimi vjen nga instrumenti, jo nga ndotja.
    """
    good = (df.filter(pl.col("clean_action").is_in(["keep", "zero_negative"]))
              .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_local",
                      pl.col("value_clean").alias("v"))
              .with_columns(pl.col("ts_local").dt.week().alias("w"), pl.col("ts_local").dt.year().alias("y")))
    base = good.filter(pl.col("y") >= 2020).group_by("station_id", "pollutant", "w").agg(pl.col("v").median().alias("b"))
    y19 = good.filter(pl.col("y") == 2019).join(base, on=["station_id", "pollutant", "w"]).filter(pl.col("b") > 0)
    switch = pl.lit(Q["analyzer_change"]["approx_date"]).str.to_datetime().dt.replace_time_zone(CFG["time"]["local_tz"])
    summary = (y19.with_columns(pl.when(pl.col("ts_local") < switch).then(pl.lit("para ndërrimit"))
                                .otherwise(pl.lit("pas ndërrimit")).alias("periudha"))
               .group_by("pollutant", "periudha").agg((pl.col("v") / pl.col("b")).median().round(2).alias("r"))
               .pivot(sort_columns=True, on="periudha", index="pollutant", values="r").sort("pollutant"))
    weekly = (y19.sort("ts_local").group_by_dynamic("ts_local", every="2w", group_by="pollutant")
                 .agg((pl.col("v") / pl.col("b")).median().alias("r")).with_columns(pl.col("ts_local").dt.date()))
    return summary, weekly


# ------------------------------------------------------------------------------------------ 3. pastrimi
def clean_action(with_calibration: bool = True) -> pl.Expr:
    """Tabela e seksionit 3, në rendin: invalid → sentinel/absurde → kalibrim → fizik → negative → ngecje → maja →
    jokonsistenca. Çdo vlerë merr vetëm arsyen e parë që e prek. `with_calibration=False` përdoret vetëm për të matur
    evidencën e ndërrimit të analizuesve (D33) para se rregulli të aplikohet."""
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


def describe(col: str) -> list[pl.Expr]:
    c = pl.col(col)
    return [c.count().alias("n"), c.mean().alias("mean"), c.std().alias("std"), c.min().alias("min"),
            c.median().alias("median"), c.quantile(0.99).alias("p99"), c.max().alias("max")]


# ------------------------------------------------------------------------------------------ figurat
def figure_gas(weekly: pl.DataFrame) -> str:
    """Niveli i 2019 ndaj normales sezonale: gazet kundrejt grimcave (kontrolli)."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, p in enumerate(["NO", "NO2", "CO", "SO2", "PM2_5"]):
        s = weekly.filter(pl.col("pollutant") == p).sort("ts_local")
        ax.plot(s["ts_local"], s["r"], marker="o", ms=3, lw=2 if p in ("NO", "PM2_5") else 1.2, color=COLORS[i],
                label=p if p != "PM2_5" else "PM2.5 (kontrolli)")
    ax.axhline(1, color="#52514e", lw=0.8, ls="--")
    ax.set_ylabel("vlera / normalja sezonale (2020–24)")
    ax.set_title("2019: NO dhe NO2 shumë mbi normalen deri në nëntor, ndërsa PM2.5 është normal → ndërrim analizuesi")
    ax.legend(ncol=5, fontsize=8)
    return save_figure(fig, "p2_nderrimi_analizuesve")


def figure_nox(before: pl.DataFrame, after: pl.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.linspace(0, 1.6, 161)
    ax.hist(before["r"].clip(0, 1.6).to_numpy(), bins=bins, color=COLORS[1], alpha=0.75,
            label=f"raw: mediana {before['r'].median():.3f} (NOx në ppb)")
    ax.hist(after["r"].clip(0, 1.6).to_numpy(), bins=bins, color=COLORS[0], alpha=0.75,
            label=f"pas ×{NOX_FACTOR} dhe pastrimit: mediana {after['r'].median():.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("NOx / (NO2 + 1.53·NO)")
    ax.set_ylabel("orë")
    ax.set_title("Harmonizimi i njësive: NOx bëhet konsistent me NO2 dhe NO")
    ax.legend()
    return save_figure(fig, "p2_nox_raporti_para_pas")


def figure_hist(df: pl.DataFrame) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    before = df.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd"))
    for ax, p in zip(axes.flat, NAMES, strict=True):
        b = before.filter(pl.col("pollutant") == p)["value"].clip(lower_bound=0).log1p().to_numpy()
        a = df.filter(pl.col("pollutant") == p)["value_clean"].drop_nulls().log1p().to_numpy()
        ax.hist(b, bins=60, histtype="step", lw=1.5, color=COLORS[1], label="para (Validity 1/2)")
        ax.hist(a, bins=60, histtype="step", lw=1.5, color=COLORS[0], label="pas pastrimit")
        ax.set_yscale("log")
        ax.set_title(p, fontsize=10)
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("log1p(vlera)")
    fig.suptitle("Shpërndarja para dhe pas pastrimit (shkallë log)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "p2_histogramet_para_pas")


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.perf_counter()

    # 1. stacionet
    stations_raw = scan("bronze", "stations_raw").collect()
    stations, conflicts, coords, sets = clean_stations(stations_raw)
    far = stations.filter(pl.col("distance_km") > CFG["stations"]["max_distance_km"])
    if far.height:
        raise SystemExit(f"STOP: koordinata > {CFG['stations']['max_distance_km']} km nga qyteti: {far.rows()}")
    write(stations, "silver", "stations")
    table(conflicts, "p2_metadata_konfliktet")
    table(sets, "p2_metadata_grupet")
    table(coords, "p2_metadata_koordinatat_kandidate")
    table(stations, "p2_stacionet")
    lineage("p2", "metadata: deduplikim", stations_raw.height, stations.height, 0, 0)

    # 2. njësitë: NOx × 1.912 (vetëm vlera numerike reale; sentinel/absurde mbeten siç janë)
    df = (scan("bronze", "measurements").with_columns(pl.col("pollutant").cast(pl.String).cast(pl.Enum(NAMES)))
          .collect().sort("station_id", "pollutant", "ts_utc"))
    df = add_quality_flags(df)
    numeric = ~pl.col("f_sentinel") & ~pl.col("f_absurd")
    df = df.with_columns(pl.when(numeric & (pl.col("pollutant") == "NOx")).then(pl.col("value_raw") * NOX_FACTOR)
                         .otherwise(pl.col("value_raw")).alias("value"))
    periods = nox_consistent_from(df)
    # mbrojtje: ndërrimi duhet të bjerë në vitin e pritur; një javë jashtë rangut më vonë do ta shtynte datën dhe
    # do të fshinte padrejtësisht vite të tëra NO/NO2
    late = periods.filter(pl.col("consistent_from").dt.year() != Q["analyzer_change"]["expected_year"])
    if late.height:
        raise SystemExit(f"STOP: data e ndërrimit të analizuesit jashtë vitit të pritur: {late.rows()}")
    df = nox_inconsistent_flag(df, periods)
    table(periods, "p2_nox_periudhat")

    # 3. pastrimi. Evidenca e ndërrimit të analizuesve (D33) matet para se rregulli i kalibrimit të aplikohet.
    def cleaned(d: pl.DataFrame, calibration: bool) -> pl.DataFrame:
        d = d.with_columns(clean_action(calibration))
        return d.with_columns(pl.when(pl.col("clean_action") == "keep").then(pl.col("value"))
                              .when(pl.col("clean_action") == "zero_negative").then(0.0)
                              .otherwise(None).alias("value_clean"))

    provisional = cleaned(df, calibration=False)
    gas_summary, gas_weekly = analyzer_change_evidence(provisional)
    # sa do të ishte NOx i rillogaritur nga NO dhe NO2 të periudhës së dyshimtë, kundrejt NOx të matur më vonë
    w = (provisional.filter(pl.col("pollutant").is_in(["NO", "NO2"]) & pl.col("f_calibration"))
         .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value_clean")
         .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value_clean").drop_nulls())
    nox_recomputed = w.select((pl.col("NO2") + Q["no_to_no2"] * pl.col("NO")).mean()).item()
    nox_measured = provisional.filter((pl.col("pollutant") == "NOx") & ~pl.col("f_nox_inconsistent"))["value_clean"].mean()
    table(gas_summary, "p2_nderrimi_analizuesve")
    table(gas_weekly.pivot(sort_columns=True, on="pollutant", index="ts_local", values="r").sort("ts_local"),
          "p2_nderrimi_analizuesve_javor")
    fig_gas = figure_gas(gas_weekly)
    df = cleaned(df, calibration=True)

    # verifikimi i njësive pas konvertimit dhe pastrimit: raporti ≈ 1 për çdo stacion × vit
    after = (nox_triplets(df.drop_nulls("value_clean"), "value_clean").with_columns(ratio().alias("r"))
             .filter(pl.col("r").is_finite()))
    before = nox_triplets(df.filter(pl.col("validity").is_in(VALID) & numeric & (pl.col("value_raw") > 0)),
                          "value_raw").with_columns(ratio().alias("r"))
    check = (after.group_by("station_id", pl.col("ts_utc").dt.convert_time_zone(CFG["time"]["local_tz"]).dt.year().alias("year"))
             .agg(pl.len().alias("ore"), pl.col("r").median().round(4).alias("mediana"))
             .with_columns(((pl.col("mediana") - 1).abs() <= Q["unit_check_tolerance"]).alias("ok"))
             .sort("station_id", "year"))
    table(check, "p2_nox_raporti_pas_konvertimit")
    if not check["ok"].all():
        raise SystemExit(f"STOP: raporti i NOx pas konvertimit s'është ≈ 1: {check.filter(~pl.col('ok')).rows()}")
    fig_nox = figure_nox(before, after)

    # 4. bashkimi me stacionet
    n_before = df.height
    df = df.join(stations.select("station_id", "station_type", "station_area", "altitude_m")
                 .with_columns(pl.col("station_id").cast(df.schema["station_id"])), on="station_id", how="left")

    # kontrollet
    pmax = by_name("physical_max")
    n_neg = df.filter(pl.col("value_clean") < 0).height
    n_over = df.filter(pl.col("value_clean") > pl.col("pollutant").cast(pl.String)
                       .replace_strict(pmax, return_dtype=pl.Float64)).height
    no_reason = df.filter(pl.col("value_clean").is_null() & (pl.col("clean_action") == "keep")).height
    if n_neg or n_over or df.height != n_before or no_reason or df["station_type"].null_count():
        raise SystemExit(f"STOP: kontrolli dështoi (neg={n_neg}, mbi kufi={n_over}, rreshta {n_before}→{df.height})")

    clean = df.select("station_id", "pollutant", "ts_utc", "ts_local", "value_raw", "value_clean", "clean_action",
                      "validity", "f_below_dl", "f_outlier_stat", "f_pm25_gt_pm10", "f_gas_period_flag", "station_type", "station_area",
                      "altitude_m").rename({"value_clean": "value"})
    write(clean, "silver", "measurements_clean", partition_by="pollutant")
    lineage("p2", "pastrimi: bronze → silver", n_before, clean.height, 0, clean["value"].null_count())

    # tabelat e raportit
    actions = table(df.group_by("clean_action", "pollutant").len()
                    .pivot(on="pollutant", index="clean_action", values="len", sort_columns=True).fill_null(0)
                    .with_columns(pl.sum_horizontal(pl.exclude("clean_action")).alias("TOTAL"))
                    .sort("clean_action"), "p2_clean_action_ndotes")
    valid_before = df.filter(pl.col("validity").is_in(VALID) & numeric)
    stats = table(pl.concat([
        valid_before.group_by("pollutant").agg(pl.lit("para").alias("faza"), *describe("value")),
        df.group_by("pollutant").agg(pl.lit("pas").alias("faza"), *describe("value_clean")),
    ]).sort("pollutant", "faza", descending=[False, True]), "p2_statistikat_para_pas")
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
    table(pl.DataFrame([counts]), "p2_te_mbajtura")
    # ndjeshmëria e politikës së negativeve: "→ 0" kundrejt "të gjitha null"
    neg = table(df.group_by("pollutant").agg(
        pl.col("value_clean").mean().alias("mesatarja_negative_ne_0"),
        pl.col("value_clean").filter(pl.col("clean_action") != "zero_negative").mean().alias("mesatarja_negative_null"),
        (pl.col("clean_action") == "zero_negative").sum().alias("vlera_te_prekura"))
        .with_columns((100 * (pl.col("mesatarja_negative_null") / pl.col("mesatarja_negative_ne_0") - 1)).round(2)
                      .alias("ndryshimi_pct"))
        .sort("pollutant"), "p2_ndjeshmeria_negativeve")
    pm_null_hours = df.filter(pl.col("f_pm_out_tol") & (pl.col("pollutant") == "PM2_5")).height
    pm_hours = df.filter(pl.col("pollutant") == "PM2_5").height
    fig_hist = figure_hist(df)
    elapsed = time.perf_counter() - t0

    write_section("p2", f"""
## 3. Integrimi dhe pastrimi

### 3.1 Integrimi i metadata-ve të stacioneve
Metadata e EEA ka {stations_raw.height} rreshta për {stations.height} stacione, nga disa grupe regjistrimesh me vlera
kontradiktore:
{md(conflicts)}
**Analiza e grupeve të regjistrimeve:**
{md(sets)}
Grupi `2024_komuna` (importi i 2025, me kolonën Municipality të plotësuar) u jep **"background" të gjitha stacioneve**,
edhe XK0002A (Prishtinë, Rilindja) që dy burimet e tjera e quajnë "traffic", dhe ka komunën **"Podgorica"** (kryeqyteti
i Malit të Zi) për tre stacione në Kosovë: janë shenja të një shablloni të paplotësuar. Megjithatë ky grup ka
koordinatat e sakta të Prishtinës, ndërsa grupet e tjera i vendosin dy stacionet e Prishtinës ~30 km në jug.
Pra **asnjë grup s'është i besueshëm për të gjitha atributet**, dhe rregulli "viti më i ri" s'funksionon (në 2024
ka dy grupe që kundërshtojnë njëri-tjetrin).

**Vendimi (D09), rregull i veçantë për çdo atribut:**
- **Koordinatat:** nga të gjithë kandidatët merret ai më afër qendrës së qytetit/fshatit të stacionit (gazeter në
  `config.yaml`, kontroll i pavarur gjeografik); largësia > {CFG['stations']['max_distance_km']} km ndalon pipeline-in.
- **Lartësia:** nga i njëjti rregjistrim si koordinatat (që të përshkruajnë të njëjtin vend); 0 m → null.
- **Tipi dhe zona:** shumica mes burimeve të besueshme (`legacy`, `2024`, `2017`), pa `2024_komuna`; në barazim
  fiton raportimi më i ri.
- **Emri dhe kodi kombëtar:** nga raportimi 2024; emrat pastrohen (thonjëza, hapësira, kode HTML:
  `Dardhisht&#235;` → Dardhishtë) dhe të gjitha variantet ruhen.

Kandidatët e koordinatave dhe largësia nga qyteti (`tables/p2_metadata_koordinatat_kandidate.csv`), p.sh. Prishtina:
{md(coords.filter(pl.col("station_id").is_in(["XK0001A", "XK0002A"])))}
Rezultati, {stations.height} stacione (çdo stacion brenda {stations['distance_km'].max()} km nga qyteti i vet):
{md(stations.drop("name_variants", "national_code_variants"))}

### 3.2 Harmonizimi i njësive (para pastrimit, që pragjet të aplikohen në njësinë e duhur)
Te p1 u vërtetua që NOx është në ppb (raporti ≈ {1 / NOX_FACTOR:.3f} = 1/{NOX_FACTOR}). **Vendimi (D10):** NOx × {NOX_FACTOR} → µg/m³
(si NO2), kudo. CO mbetet në mg/m³ (praktikë standarde). Para mesit të nëntorit 2019 raporti s'përputhet me asnjë
njësi, prandaj NOx i asaj periudhe shënohet **jokonsistent** me dy burime evidence: (a) periudha: çdo orë para javës
nga e cila mediana javore e stacionit mbetet në rangun e ppb (`consistent_from`); (b) ora: pas konvertimit raporti
NOx/(NO2 + 1.53·NO) largohet nga 1 më shumë se {int(Q['nox_tolerance'] * 100)}%.
{md(periods)}
Orë NOx të shënuara: {counts['nox_period']:,} nga periudha + {counts['nox_hourly_only']:,} vetëm nga rregulli orar.
**Verifikimi:** pas konvertimit dhe pastrimit, mediana e raportit për çdo stacion × vit është në 1 ± 5%
({check['ok'].sum()}/{check.height} stacion-vite; mediana e përgjithshme {after['r'].median():.4f}).
{fig_nox}

### 3.2.1 Ndërrimi i analizuesve të gazeve në nëntor 2019 (D33)
Pyetja: a janë të sakta NO dhe NO2 të asaj periudhe (që do të përdoreshin për ta rillogaritur NOx)? Krahasohet niveli
i 2019 me normalen sezonale të të njëjtit stacion në të njëjtën javë të vitit (mediana 2020–2024); grimcat (PM2.5),
që vijnë nga të njëjtat burime djegieje por maten me instrumente të tjera, shërbejnë si kontroll:
{md(gas_summary)}
{fig_gas}
Para nëntorit **NO është ~{gas_summary.filter(pl.col('pollutant') == 'NO')['para ndërrimit'][0]}× dhe NO2
~{gas_summary.filter(pl.col('pollutant') == 'NO2')['para ndërrimit'][0]}× mbi normalen**, ndërsa PM2.5 dhe O3 janë
normale; kalimi te normalja ndodh brenda dy javësh në fund të tetorit / nëntor 2019, njëkohësisht me momentin kur
NOx bëhet konsistent. Një ndryshim i menjëhershëm, vetëm te gazet e analizuesit të NOx dhe jo te grimcat, s'është
ndryshim real i ndotjes por **ndërrim ose rikalibrim i analizuesit**.
**Vendimi:** NO dhe NO2 para `consistent_from` të stacionit → `null_calibration` ({counts['calibration']:,} orë).
Pasojë: NOx i 2019 **nuk rillogaritet** nga NO2 + 1.53·NO, sepse do të trashëgonte të njëjtin gabim (rillogaritja
do të jepte mesatare {nox_recomputed:.0f} µg/m³ kundrejt {nox_measured:.0f} të NOx të matur pas ndërrimit). Kjo e korrigjon opsionin A të D10/D15, që supozonte NO dhe
NO2 të sakta.

**Kriteri pse NO/NO2 fshihen, por CO/SO2 jo:** një ndryshim niveli vetëm nuk provon gabim instrumenti, sepse ndotja
mund të ndryshojë realisht. Prandaj kërkohen dy evidenca të pavarura: (1) **provë fizike** që instrumenti s'punonte
saktë, dhe (2) ndryshim niveli në të njëjtin moment. NO dhe NO2 maten nga i njëjti analizues me NOx, dhe për këtë treshe
ka provë fizike (NOx ≠ NO2 + 1.53·NO para ndërrimit) plus ndryshim niveli. CO dhe SO2 kanë gjithashtu ndryshim niveli
(SO2 {gas_summary.filter(pl.col('pollutant') == 'SO2')['para ndërrimit'][0]}× → {gas_summary.filter(pl.col('pollutant') == 'SO2')['pas ndërrimit'][0]}×,
CO {gas_summary.filter(pl.col('pollutant') == 'CO')['para ndërrimit'][0]}× → {gas_summary.filter(pl.col('pollutant') == 'CO')['pas ndërrimit'][0]}×),
por maten nga analizues të tjerë dhe s'kanë asnjë kontroll fizik; SO2 varet edhe nga puna e termocentraleve, që
ndryshon realisht. Fshirja e vlerave ndoshta të sakta do të ishte gabim tjetër, ndaj **mbahen** me flamurin
`f_gas_period_flag` ({counts['gas_flag']:,} orë) dhe trajtohen si kufizim i dokumentuar: analizat e CO dhe SO2 mund
ta përjashtojnë këtë periudhë me anë të flamurit.

### 3.3 Pastrimi
Rregullat e tabelës së vendimeve aplikohen me rend; çdo vlerë merr vetëm arsyen e parë. **Asnjë rresht s'fshihet**:
vlera bëhet null, arsyeja ruhet te `clean_action` dhe origjinali te `value_raw`.
{md(actions)}
Çka **mbahet qëllimisht**: Validity 2 nën pragun e detektimit ({counts['below_dl']:,}; heqja do ta anonte mesataren
lart, MNAR), zerot e gjata të besueshme për CO/O3/SO2/NO ({counts['stuck_zero_kept']:,}), PM2.5 > PM10 brenda
tolerancës ({counts['pm_within_tol']:,} rreshta, pasiguri matjeje), outliers statistikë ({counts['outliers']:,}; episode
reale ndotjeje, trajtohen me transformim te p5).

**PM2.5 > PM10 jashtë tolerancës (D25):** {pm_null_hours} orë ({100 * pm_null_hours / pm_hours:.2f}% e orëve PM2.5); bëhen null **të dyja** vlerat, sepse
s'ka bazë objektive për të ditur cili instrument gaboi, dhe çdo zgjedhje do të ishte hamendje. Humbja është e
papërfillshme dhe këto orë mbushen me interpolim te p3 nëse boshllëku është i shkurtër.

**Ndjeshmëria e politikës së negativeve (D12):** sa do të ndryshonte mesatarja nëse negativet e vogla do të bëheshin
null në vend të 0:
{md(neg)}
Heqja e tyre do ta rriste mesataren (deri {neg['ndryshimi_pct'].max()}% te {neg.sort('ndryshimi_pct')['pollutant'][-1]}), sepse heq pikërisht vlerat e vogla
(anim drejt vlerave të larta, si te MNAR). Politika "→ 0" e ruan pjesën e ulët të shpërndarjes.

**Kontrollet:** 0 vlera negative, 0 mbi kufirin fizik, rreshtat {n_before:,} → {clean.height:,} (asnjë i humbur
nga bashkimi me stacionet), çdo null ka arsye. Koha e p2: {elapsed:.1f} s.

**Statistikat para/pas** ("para" = vlerat me Validity 1/2 në njësinë kanonike, pa sentinel/absurde):
{md(stats, 16)}
{fig_hist}
""")
    print(f"p2 gati: {clean.height:,} rreshta, {clean['value'].null_count():,} null, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
