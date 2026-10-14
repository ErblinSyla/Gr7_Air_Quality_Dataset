"""p1 – Mbledhja e të dhënave, definimi i tipeve dhe kualiteti i të dhënave.

Lexon 93 skedarët raw, i kthen në tipe të sakta (bronze) dhe mat kualitetin sipas dimensioneve
(plotësia, vlefshmëria, saktësia, konsistenca, unikaliteti, kolonat pa informacion), PA ndryshuar asgjë.
Rregullat e kualitetit (`add_quality_flags`) ripërdoren te p2 për pastrimin.
"""

from __future__ import annotations

import time

import polars as pl

from utils import (
    CFG, COLORS, DATA, INK_2, POLLUTANT_TYPE, POLLUTANTS, ROOT, SERIES, by_name, heatmap, lineage, md, plt,
    save_figure, table, write, write_section,
)

Q = CFG["quality"]
VALID = [1, 2]

# Tipi statistikor dhe përshkrimi i çdo kolone (për fjalorin e të dhënave)
DICTIONARY = {
    # kolona raw: (kolona e re, tipi statistikor, përshkrimi)
    "Samplingpoint": ("station_id, pollutant, instrument", "nominale", "ID-ja e pikës: stacioni_ndotësi_instrumenti"),
    "Pollutant": ("pollutant_code", "nominale", "kodi EEA i ndotësit"),
    "Start": ("ts_utc, ts_local", "kohore", "fillimi i orës (raw: UTC+1 fiks)"),
    "End": ("ts_end", "kohore", "fundi i orës"),
    "Value": ("value_raw", "raport", "përqendrimi (zero e vërtetë, raportet kanë kuptim)"),
    "Unit": ("unit", "nominale", "njësia e deklaruar"),
    "AggType": ("agg_type", "nominale", "lloji i agregimit"),
    "Validity": ("validity", "nominale (kod)", "1 e vlefshme, 2 nën pragun e detektimit, -1 e pavlefshme, -99 mirëmbajtje"),
    "Verification": ("verification", "nominale (kod)", "statusi i verifikimit"),
    "ResultTime": ("result_time", "kohore", "koha e publikimit të rezultatit"),
    "DataCapture": ("data_capture", "raport", "përqindja e mbulimit"),
    "FkObservationLog": ("fk_observation_log", "nominale (identifikues)", "referencë administrative e EEA"),
}


# ------------------------------------------------------------------------------------------ mbledhja dhe tipet
def collect() -> tuple[pl.DataFrame, dict]:
    """Lexon 93 skedarët njëherësh dhe kthen tabelën bronze me tipe të sakta."""
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
        "rreshta": (bronze.height, CFG["expected"]["rows"]),
        "pika matjeje": (bronze["sampling_point_id"].n_unique(), CFG["expected"]["sampling_points"]),
        "stacione": (bronze["station_id"].n_unique(), CFG["expected"]["stations"]),
        "kodi në ID ≠ Pollutant": (bronze.filter(pl.col("code_in_id") != pl.col("pollutant_code")).height, 0),
        "skedari ≠ ID-ja e pikës": (bronze.filter(pl.col("source_file") != pl.col("sampling_point_id")
                                                  .cast(pl.String)).height, 0),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"STOP: të dhënat s'përputhen me përshkrimin: {bad}")
    return bronze.drop("code_in_id", "source_file").sort("station_id", "pollutant", "ts_utc"), raw_schema


def data_dictionary(bronze: pl.DataFrame, raw_schema: dict) -> pl.DataFrame:
    rows = []
    for raw_col, (new_cols, stat, desc) in DICTIONARY.items():
        first = new_cols.split(",")[0].strip()
        s = bronze[first] if first in bronze.columns else bronze["station_id"]
        rows.append({"kolona_raw": raw_col, "tipi_origjinal": str(raw_schema[raw_col]), "kolona_e_re": new_cols,
                     "tipi_i_ri": str(s.dtype), "tipi_statistikor": stat, "pershkrimi": desc,
                     "pct_bosh": round(100 * s.null_count() / len(s), 2),
                     "shembull": str(s.drop_nulls().head(1).to_list()[0]) if s.null_count() < len(s) else "–"})
    units = ", ".join(f"{p['name']}: {p['unit']}" for p in POLLUTANTS.values())
    rows[4]["pershkrimi"] += f" — njësitë: {units}"
    return table(pl.DataFrame(rows), "p1_data_dictionary")


# ------------------------------------------------------------------------------------------ rregullat e kualitetit
def add_quality_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Shton flamujt e kualitetit (kolona f_*). Vetëm shënon; s'ndryshon asnjë vlerë.

    Ngecjet dhe majat llogariten VETËM mbi vlerat e vlefshme (validity 1/2, pa sentinel, pa absurde), që
    seritë e -9900 të mos dalin si "sensor i ngecur". `df` duhet të jetë i renditur sipas serisë dhe kohës.
    """
    v, validity = pl.col("value_raw"), pl.col("validity")
    df = df.with_columns(
        validity.is_in([-1, -99]).alias("f_invalid"),
        (validity == 2).alias("f_below_dl"),
        v.is_in([float(s) for s in Q["sentinels"]]).alias("f_sentinel"),
        (v >= Q["absurd_threshold"]).alias("f_absurd"),
    ).with_columns(
        (validity.is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd")).alias("_usable"),
    ).with_columns((pl.col("_usable") & (v < 0)).alias("f_negative"))

    # --- ngecja: ≥ N orë të njëpasnjëshme me vlerë identike (vetëm vlerat e përdorshme)
    u = df.filter("_usable").select(*SERIES, "ts_utc", "value_raw")
    one_hour = pl.duration(hours=1, time_unit="ns")
    prev_v, prev_t = v.shift(1).over(SERIES), pl.col("ts_utc").shift(1).over(SERIES)
    new_run = ((v != prev_v) | (pl.col("ts_utc") - prev_t != one_hour)).fill_null(True)
    stuck = (u.with_columns(new_run.cum_sum().alias("_run"))
              .with_columns((pl.len().over("_run") >= Q["stuck_min_hours"]).alias("f_stuck"))
              .select(*SERIES, "ts_utc", "f_stuck", (pl.col("f_stuck") & (v == 0)).alias("f_stuck_zero")))

    # --- maja e izoluar: rritje > P99.9 e |Δ| të ndotësit dhe rënie > P99.9 në orën pasardhëse
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

    # --- PM2.5 > PM10 në të njëjtin stacion dhe orë (shënohen të dy rreshtat)
    def part(name: str) -> pl.DataFrame:
        return u.filter(pl.col("pollutant") == name).select("station_id", "ts_utc", pl.col("value_raw").alias(name))

    tol = Q["pm_tolerance"]
    pm = (part("PM2_5").join(part("PM10"), on=["station_id", "ts_utc"]).filter(pl.col("PM2_5") > pl.col("PM10"))
          .select("station_id", "ts_utc", pl.lit(True).alias("f_pm25_gt_pm10"),
                  (pl.col("PM2_5") - pl.col("PM10") > pl.max_horizontal(pl.lit(tol["abs"]), tol["rel"] * pl.col("PM10")))
                  .alias("f_pm_out_tol")))
    pm = pl.concat([pm.with_columns(pl.lit(p).cast(POLLUTANT_TYPE).alias("pollutant")) for p in ("PM2_5", "PM10")])

    # --- outlier statistikë (vetëm informativ): jashtë [Q1 − k·IQR, Q3 + k·IQR] të log1p sipas ndotësit
    lv = v.log1p()
    q1, q3, k = lv.quantile(0.25).over("pollutant"), lv.quantile(0.75).over("pollutant"), Q["outlier_iqr_k"]
    outl = u.filter(v >= 0).select(*SERIES, "ts_utc", ((lv < q1 - k * (q3 - q1)) | (lv > q3 + k * (q3 - q1)))
                                   .alias("f_outlier_stat"))

    key = [*SERIES, "ts_utc"]
    for f in (stuck, spikes, pm, outl):
        cols = [c for c in f.columns if c.startswith("f_")]
        df = df.join(f, on=key, how="left").with_columns(pl.col(cols).fill_null(False))
    return df.drop("_usable")


FLAGS = ["f_invalid", "f_below_dl", "f_sentinel", "f_absurd", "f_negative", "f_stuck", "f_stuck_zero", "f_spike",
         "f_pm25_gt_pm10", "f_pm_out_tol", "f_outlier_stat"]


def nox_ratio_evidence(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Raporti NOx / (NO2 + 1.53·NO) mbi vlerat raw të vlefshme: sipas stacionit × vitit dhe sipas javës."""
    good = df.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd")
                     & (pl.col("value_raw") > 0))
    w = (good.filter(pl.col("pollutant").is_in(["NOx", "NO2", "NO"]))
             .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value_raw")
             .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value_raw").drop_nulls()
             .with_columns((pl.col("NOx") / (pl.col("NO2") + Q["no_to_no2"] * pl.col("NO"))).alias("ratio"),
                           pl.col("ts_utc").dt.convert_time_zone(CFG["time"]["local_tz"]).dt.year().alias("year")))
    lo, hi = Q["nox_ppb_ratio"]
    by_year = (w.group_by("station_id", "year")
                .agg(pl.len().alias("ore"), pl.col("ratio").median().round(4).alias("mediana"),
                     (100 * pl.col("ratio").is_between(lo, hi).mean()).round(1).alias("pct_ne_rangun_ppb"))
                .sort("station_id", "year"))
    weekly = (w.sort("ts_utc").group_by_dynamic("ts_utc", every="1w", group_by="station_id")
               .agg(pl.col("ratio").median().alias("mediana")).sort("station_id", "ts_utc"))
    return by_year, weekly


# ------------------------------------------------------------------------------------------ raporti i kualitetit
def quality_report(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    n = df.height
    t: dict[str, pl.DataFrame] = {}

    # Plotësia: null eksplicite, kolona bosh dhe orë që mungojnë fare (pa rresht) brenda periudhës së serisë
    t["plotesia_kolonat"] = pl.DataFrame([{"kolona": c, "null": df[c].null_count(),
                                           "pct_null": round(100 * df[c].null_count() / n, 2)}
                                          for c in df.columns if not c.startswith("f_")])
    t["plotesia_seri"] = (df.group_by(SERIES)
                          .agg(pl.len().alias("rreshta"),
                               ((pl.col("ts_utc").max() - pl.col("ts_utc").min()).dt.total_hours() + 1)
                               .alias("ore_te_pritura"))
                          .with_columns((pl.col("ore_te_pritura") - pl.col("rreshta")).alias("ore_pa_rresht"))
                          .with_columns((100 * pl.col("ore_pa_rresht") / pl.col("ore_te_pritura")).round(2)
                                        .alias("pct_pa_rresht"))
                          .sort(SERIES))

    # Vlefshmëria: Validity sipas ndotësit (%)
    t["vlefshmeria"] = (df.group_by("pollutant", "validity").len()
                        .with_columns((100 * pl.col("len") / pl.col("len").sum().over("pollutant")).round(2)
                                      .alias("pct"))
                        .with_columns(pl.format("validity {}", pl.col("validity")).alias("validity"))
                        .pivot(on="validity", index="pollutant", values="pct", sort_columns=True)
                        .fill_null(0).sort("pollutant"))

    # Saktësia dhe konsistenca: numri i çdo flamuri sipas ndotësit
    t["flamujt"] = (df.group_by("pollutant").agg(pl.len().alias("n"), *[pl.col(f).sum() for f in FLAGS])
                    .sort("pollutant"))
    # negativet krahasohen me pragun në njësinë kanonike (NOx raw është në ppb → × 1.912, si te p2)
    pol = pl.col("pollutant").cast(pl.String)
    canonical = pl.col("value_raw") * pol.replace_strict(by_name("ppb_to_ugm3", 1.0), return_dtype=pl.Float64)
    threshold = pol.replace_strict(by_name("neg_zero"), return_dtype=pl.Float64)
    t["negative"] = (df.filter("f_negative").group_by("pollutant")
                     .agg(pl.len().alias("n"), canonical.min().round(2).alias("min_kanonike"),
                          threshold.first().alias("pragu_neg_zero"),
                          (canonical >= threshold).sum().alias("te_vogla_ne_0"),
                          (canonical < threshold).sum().alias("te_medha_null"))
                     .with_columns((100 * pl.col("te_vogla_ne_0") / pl.col("n")).round(1).alias("pct_te_vogla"))
                     .sort("pollutant"))
    t["ngecjet"] = (df.filter("f_stuck").group_by("pollutant")
                    .agg(pl.len().alias("ore_te_ngecura"), pl.col("f_stuck_zero").sum().alias("prej_tyre_zero"))
                    .sort("pollutant"))
    t["njesite"] = (df.group_by("pollutant").agg(pl.col("unit").cast(pl.String).unique().str.join(", ")
                                                 .alias("njesia_e_deklaruar")).sort("pollutant"))

    # Unikaliteti dhe kolonat pa informacion
    dup = n - df.select("station_id", "pollutant", "ts_utc").n_unique()
    end_ok = df.filter(pl.col("ts_end") - pl.col("ts_utc") != pl.duration(hours=1, time_unit="ns")).height
    t["unikaliteti"] = pl.DataFrame({"rreshta": [n], "duplikate_te_celesit": [dup]})
    t["kolonat_pa_informacion"] = pl.DataFrame([
        {"kolona": c, "vlera_unike": df[c].n_unique(), "pct_null": round(100 * df[c].null_count() / n, 2),
         "verdikti": "100% bosh" if df[c].null_count() == n else ("konstante" if df[c].n_unique() == 1 else why)}
        for c, why in (("data_capture", ""), ("agg_type", ""), ("verification", ""),
                       ("fk_observation_log", "identifikues administrativ, pa informacion për ndotjen"),
                       ("unit", "e përcaktuar plotësisht nga ndotësi (CO mg/m³, të tjerët µg/m³): e tepërt"),
                       ("result_time", "administrative: koha e publikimit, s'lidhet me matjen"))
    ] + [{"kolona": "ts_end", "vlera_unike": None, "pct_null": 0.0,
          "verdikti": f"e tepërt: End = Start + 1h (përjashtime: {end_ok})"}])

    # % e pavlefshme sipas stacionit × vitit
    t["pavlefshme_stacion_vit"] = (df.with_columns(pl.col("ts_local").dt.year().alias("viti"))
                                   .group_by("station_id", "viti").agg((100 * pl.col("f_invalid").mean()).round(1)
                                                                       .alias("pct"))
                                   .with_columns(pl.col("station_id").cast(pl.String))
                                   .pivot(on="viti", index="station_id", values="pct", sort_columns=True)
                                   .sort("station_id"))
    for name, tab in t.items():
        table(tab, f"p1_kualiteti_{name}")
    return t


def nox_figure(weekly: pl.DataFrame) -> str:
    lo, hi = Q["nox_ppb_ratio"]
    f = by_name("ppb_to_ugm3")["NOx"]   # faktori ppb → µg/m³ nga config
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axhspan(lo, hi, color=COLORS[2], alpha=0.15, lw=0)
    for (_,), g in weekly.group_by(["station_id"]):
        ax.plot(g["ts_utc"], g["mediana"], color="#b9b8b2", lw=0.8)
    net = weekly.group_by("ts_utc").agg(pl.col("mediana").median()).sort("ts_utc")
    ax.plot(net["ts_utc"], net["mediana"], color=COLORS[0], lw=2, label="mediana e rrjetit")
    ax.text(weekly["ts_utc"].min(), hi + 0.01, f"rangu i ppb: 1/{f} ≈ {1 / f:.3f}", color=INK_2, fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("NOx / (NO2 + 1.53·NO)")
    ax.set_title("Raporti javor i NOx: konsistent me ppb vetëm nga mesi i nëntorit 2019 (gri: stacionet)")
    ax.legend(loc="lower right")
    return save_figure(fig, "p1_nox_raporti_javor")


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.perf_counter()
    bronze, raw_schema = collect()
    write(bronze, "bronze", "measurements", partition_by="pollutant")

    meta = pl.read_csv(next((ROOT / CFG["paths"]["metadata"]).glob("*.csv")), infer_schema=False)
    stations_raw = meta.filter(pl.col("Air Quality Station EoI Code").str.starts_with("XK"))
    write(stations_raw, "bronze", "stations_raw")
    lineage("p1", "mbledhja: raw → bronze", bronze.height, bronze.height, 0, bronze["value_raw"].null_count())
    lineage("p1", "metadata: Evropa → Kosova", meta.height, stations_raw.height, 0, 0)

    dictionary = data_dictionary(bronze, raw_schema)
    flagged = add_quality_flags(bronze)
    q = quality_report(flagged)
    nox_year, nox_week = nox_ratio_evidence(flagged)
    table(nox_year, "p1_nox_raporti_stacion_vit")
    fig_nox = nox_figure(nox_week)
    fig_invalid = save_figure(heatmap(q["pavlefshme_stacion_vit"],
                                      "% e matjeve të pavlefshme (Validity −1/−99) sipas stacionit dhe vitit",
                                      "% e pavlefshme"), "p1_pavlefshme_stacion_vit")

    size_raw = sum(f.stat().st_size for f in (ROOT / CFG["paths"]["raw"]).glob("*.parquet")) / 2**20
    size_bronze = sum(f.stat().st_size for f in (DATA / "bronze" / "measurements").rglob("*.parquet")) / 2**20
    flags = q["flamujt"].select(pl.exclude("pollutant", "n")).sum()
    usable = flagged.filter(pl.col("validity").is_in(VALID) & ~pl.col("f_sentinel") & ~pl.col("f_absurd"))
    mismatch = nox_year.filter(~pl.col("mediana").is_between(*Q["nox_ppb_ratio"]))
    inv = q["pavlefshme_stacion_vit"]
    worst = (inv.select("station_id", pl.mean_horizontal(pl.exclude("station_id")).alias("m"))
                .sort("m", descending=True).row(0))
    inv_years = list(zip(inv.columns[1:], inv.select(pl.exclude("station_id")).mean().row(0), strict=True))
    neg = q["negative"]
    small_majority = neg.filter(pl.col("pct_te_vogla") >= 50)["pollutant"].cast(pl.String).to_list()
    small_minority = neg.filter(pl.col("pct_te_vogla") < 50)["pollutant"].cast(pl.String).to_list()
    neg_text = (f"Te {', '.join(small_majority)} shumica janë negative të vogla afër zeros (zhurmë instrumenti → 0)"
                + (f"; te {', '.join(small_minority)} shumica janë negative të mëdha (gabime → null)" if small_minority
                   else ""))
    st = q["ngecjet"].with_columns((pl.col("prej_tyre_zero") / pl.col("ore_te_ngecura")).alias("z"))
    stuck_text = (f"{int(st['prej_tyre_zero'].sum()):,} nga {int(st['ore_te_ngecura'].sum()):,} orë të ngecura "
                  f"({100 * st['prej_tyre_zero'].sum() / st['ore_te_ngecura'].sum():.0f}%) janë zero; më shumë te "
                  + ", ".join(f"{r[0]} ({r[1]:,})" for r in st.sort("ore_te_ngecura", descending=True)
                              .select("pollutant", "ore_te_ngecura").head(3).rows()))
    elapsed = time.perf_counter() - t0

    write_section("p1", f"""
## 2. Mbledhja e të dhënave, definimi i tipeve, kualiteti i të dhënave

### 2.1 Mbledhja
{CFG['expected']['sampling_points']} skedarë Parquet (një për çdo stacion × ndotës) lexohen njëherësh me `pl.scan_parquet`. Verifikimet:
{bronze.height:,} rreshta, {bronze['sampling_point_id'].n_unique()} pika matjeje, {bronze['station_id'].n_unique()}
stacione; kodi i ndotësit në ID = kolona `Pollutant`; emri i skedarit = ID-ja e pikës. Metadata e EEA përmban
{meta.height:,} rreshta për gjithë Evropën; mbahen {stations_raw.height} rreshtat e Kosovës.
Madhësia: {size_raw:.1f} MB raw → {size_bronze:.1f} MB bronze (Parquet zstd, i particionuar sipas ndotësit).
Koha e p1: {elapsed:.1f} s.

### 2.2 Definimi i tipeve
`Value` vjen si Decimal(38,18) → Float64 (0 dështime). Nga `Samplingpoint` nxirren stacioni, ndotësi dhe
instrumenti. Koha raw është UTC+1 fikse pa orë verore → `ts_utc` (çelësi kohor) dhe `ts_local`
(Europe/Belgrade, me orë verore, për veçoritë ditore). Kolonat tekstuale → Categorical, ndotësi → Enum.
Tipi statistikor tregon çfarë veprimesh kanë kuptim (p.sh. mesatarja ka kuptim për `raport`, jo për `nominale`).
{md(dictionary)}

### 2.3 Kualiteti i të dhënave (asgjë s'është ndryshuar ende)
**Plotësia.** Null eksplicite në `Value`: {bronze['value_raw'].null_count()}. Por mungojnë
{int(q['plotesia_seri']['ore_pa_rresht'].sum()):,} orë **pa rresht fare** brenda periudhës së serive
(mungesa implicite, nuk duken pa krijuar rrjetin orar te p3). `DataCapture` është 100% bosh.
{md(q['plotesia_seri'].sort('pct_pa_rresht', descending=True), 12)}

**Vlefshmëria** (% e rreshtave sipas Validity):
{md(q['vlefshmeria'])}
{fig_invalid}
Mesatarja e të pavlefshmeve sipas vitit: {', '.join(f"{c}: {v:.1f}%" for c, v in zip(
    q['pavlefshme_stacion_vit'].columns[1:], q['pavlefshme_stacion_vit'].select(pl.exclude('station_id')).mean().row(0),
    strict=True))}. Trendi është rritës: nga {inv_years[0][1]:.1f}% në {inv_years[0][0]} në {inv_years[-1][1]:.1f}% në
{inv_years[-1][0]} (viti më i keq: {max(inv_years, key=lambda r: r[1])[0]}). Stacioni më i keq (mesatarja e viteve):
{worst[0]} ({worst[1]:.1f}%).

**Saktësia.** Flamujt janë llogaritur, jo aplikuar (numri gjithsej):
{md(flags)}
- **Sentinel −9900:** kod për "pa vlerë", jo matje; mesatarja raw e të gjitha vlerave është
  {bronze['value_raw'].mean():,.0f}, kurse mbi vlerat e përdorshme (raw) {usable['value_raw'].mean():.2f}.
- **Negative me Validity 1/2** (në njësinë kanonike; pragu "e vogël → 0" është {CFG['pollutants'][1]['neg_zero']} µg/m³,
  {CFG['pollutants'][10]['neg_zero']} mg/m³ për CO):
{md(q['negative'])}
  {neg_text}. PM10 dhe PM2.5 s'kanë negative të vlefshme.
- **Sensor i ngecur** (≥ {Q['stuck_min_hours']}h e njëjta vlerë, vetëm mbi vlerat e vlefshme):
{md(q['ngecjet'])}
  {stuck_text} → zerot e gjata vlerësohen sipas ndotësit te p2 (D13).
- **Maja të izoluara:** rritje > P99.9 e ndryshimeve orare dhe rënie menjëherë pas (`tables/p1_spike_thresholds.csv`).
- **Outliers statistikë** (IQR mbi log1p): {int(flags['f_outlier_stat'][0]):,}. Janë informative, jo gabime.

**Konsistenca.**
- PM2.5 > PM10 në të njëjtën orë: {int(flags['f_pm25_gt_pm10'][0]) // 2:,} orë; jashtë tolerancës
  max({Q['pm_tolerance']['abs']} µg/m³, {int(Q['pm_tolerance']['rel'] * 100)}%): {int(flags['f_pm_out_tol'][0]) // 2:,}.
- Njësitë e deklaruara: {md(q['njesite'])}
- **NOx.** Kimikisht NOx = NO2 + NO. Në µg/m³ si NO2: NOx ≈ NO2 + 1.53·NO, pra raporti duhet ≈ 1. Raporti
  del ≈ {nox_year['mediana'].median():.4f} = 1/{by_name('ppb_to_ugm3')['NOx']} → **NOx është në ppb, edhe pse etiketohet µg/m³**.
  Por {mismatch.height} stacion-vite (vitet: {', '.join(str(y) for y in mismatch['year'].unique().sort())}) kanë mediana jashtë rangut të ppb:
{md(mismatch)}
{fig_nox}
  Para mesit të nëntorit 2019 raporti s'është konstant (asnjë njësi e vetme s'i shpjegon vlerat). Trajtimi: D10 te p2.

**Unikaliteti.** {q['unikaliteti']['duplikate_te_celesit'][0]} duplikate të çelësit (stacion, ndotës, orë).

**Kolonat pa informacion** (kandidate për heqje te p5):
{md(q['kolonat_pa_informacion'])}
""")
    print(f"p1 gati: {bronze.height:,} rreshta, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
