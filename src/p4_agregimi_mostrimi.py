"""p4 – Agregimi (orar → ditor → mujor → vjetor, me rregullat e plotësisë) dhe mostrimi
(i rastësishëm kundrejt i stratifikuar sipas stacion × sezon)."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy import stats

from utils import (
    CFG, COLORS, INK_2, POLLUTANT_TYPE, SEASONS, SERIES, lineage, md, plt, save_figure, scan, table, write,
    write_section,
)

AG, S, T = CFG["aggregation"], CFG["sampling"], CFG["time"]
STRUCTURAL = ["not_measured", "outside_period"]
COMPARE = S["compare_pollutants"]


# ------------------------------------------------------------------------------------------ agregimi
def daily(hourly: pl.DataFrame, col: str) -> pl.DataFrame:
    """Ditor sipas datës lokale: mesatarja (null kur < 18 orë të vlefshme), maksimumi, numri i orëve të vlefshme."""
    v = pl.col(col)
    return (hourly.sort(*SERIES, "ts_local")
            .group_by_dynamic("ts_local", every="1d", group_by=SERIES)
            .agg(v.count().alias("valid_hours"), v.mean().alias("_mean"), v.max().alias("max"))
            .with_columns((pl.col("valid_hours") >= AG["min_hours_per_day"]).alias("valid_day"),
                          pl.col("ts_local").dt.date().alias("date"))
            .with_columns(pl.when("valid_day").then(pl.col("_mean")).alias("mean"))
            .drop("_mean", "ts_local"))


def monthly(day: pl.DataFrame) -> pl.DataFrame:
    """Mesatarja e ditëve të vlefshme; null kur ditët e vlefshme < 75% e ditëve të muajit."""
    return (day.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
            .group_by(*SERIES, "month")
            .agg(pl.col("valid_day").sum().alias("valid_days"), pl.col("mean").mean().alias("_mean"))
            .with_columns(pl.col("month").dt.month_end().dt.day().alias("days_in_month"))
            .with_columns(pl.when(pl.col("valid_days") >= AG["min_days_per_month_frac"] * pl.col("days_in_month"))
                          .then(pl.col("_mean")).alias("mean"))
            .drop("_mean").sort(*SERIES, "month"))


def annual(hourly: pl.DataFrame, day: pl.DataFrame, col: str) -> pl.DataFrame:
    """Mesatarja vjetore nga ditët e vlefshme + mbulimi (orë të vlefshme / orë të vitit)."""
    hours = (hourly.group_by(*SERIES, pl.col("ts_local").dt.year().alias("year"))
             .agg(pl.col(col).count().alias("valid_hours")))
    means = day.group_by(*SERIES, pl.col("date").dt.year().alias("year")).agg(pl.col("mean").mean())
    year_hours = pl.when(pl.col("year") % 4 == 0).then(8784).otherwise(8760)
    return (hours.join(means, on=[*SERIES, "year"], how="left")
            .with_columns((pl.col("valid_hours") / year_hours).round(3).alias("coverage"))
            .with_columns((pl.col("coverage") >= AG["min_annual_coverage"]).alias("meets_90pct"))
            .with_columns(pl.when("meets_90pct").then(pl.col("mean")).alias("mean_official"))
            .sort(*SERIES, "year"))


# ------------------------------------------------------------------------------------------ mostrimi
def population() -> pl.DataFrame:
    """Stacion × orë, një kolonë për ndotës (vlerat pas imputimit), me sezonin."""
    long = (scan("gold", "hourly_long").filter(~pl.col("status").is_in(STRUCTURAL))
            .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_utc", "ts_local",
                    "value_imputed").collect())
    return (long.pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc", "ts_local"], values="value_imputed")
                .with_columns(pl.col("ts_local").dt.month().replace_strict(SEASONS, return_dtype=pl.String)
                              .alias("season"))
                .sort("station_id", "ts_utc"))


def draw(pop: pl.DataFrame, method: str) -> pl.DataFrame:
    if method == "rastësor":
        return pop.sample(fraction=S["fraction"], seed=S["seed"])
    # i stratifikuar: e njëjta përqindje nga çdo shtresë stacion × sezon (çelës i rastësishëm i pavarur për rresht)
    u = pl.Series("_u", np.random.default_rng(S["seed"]).random(pop.height))
    strata = S["strata"]
    rank = pl.col("_u").rank("ordinal").over(strata)
    return pop.with_columns(u).filter(rank <= (pl.len() * S["fraction"]).round().over(strata)).drop("_u")


def compare(pop: pl.DataFrame, sample: pl.DataFrame, method: str) -> tuple[list[dict], dict]:
    rows = []
    for p in COMPARE:
        a, b = pop[p].drop_nulls(), sample[p].drop_nulls()
        ks = stats.ks_2samp(a.to_numpy(), b.to_numpy())
        rows.append({"metoda": method, "ndotesi": p, "mesatarja_pop": a.mean(), "mesatarja_mostra": b.mean(),
                     "std_pop": a.std(), "std_mostra": b.std(), "p95_pop": a.quantile(0.95),
                     "p95_mostra": b.quantile(0.95), "KS": ks.statistic, "KS_p": ks.pvalue})

    def max_dev(col: str) -> float:
        share = lambda d: d.group_by(col).len().with_columns(pl.col("len") / pl.col("len").sum())  # noqa: E731
        j = share(pop).join(share(sample), on=col, how="left", suffix="_s").fill_null(0)
        return round(100 * float((j["len"] - j["len_s"]).abs().max()), 3)

    return rows, {"metoda": method, "n": sample.height, "devijimi_max_stacion_pp": max_dev("station_id"),
                  "devijimi_max_sezon_pp": max_dev("season")}


# ------------------------------------------------------------------------------------------ figura
def figure_daily(day: pl.DataFrame) -> str:
    pm = day.filter((pl.col("pollutant") == "PM2_5") & (pl.col("source") == "matje"))
    stations = sorted(pm["station_id"].cast(pl.String).unique().to_list())
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, stations, strict=False):
        d = pm.filter(pl.col("station_id") == s).sort("date")
        ax.plot(d["date"], d["mean"], lw=0.7, color=COLORS[0])
        ax.axhline(CFG["binarization"]["pm25_daily_who"], color=INK_2, lw=0.8, ls="--")
        ax.set_title(s, fontsize=9)
    fig.supylabel("PM2.5 mesatarja ditore (µg/m³)")
    fig.suptitle("PM2.5 ditor sipas stacionit (vija: udhëzimi ditor i OBSH-së 15 µg/m³) – kulme çdo dimër",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "p4_pm25_ditor_stacionet")


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.perf_counter()
    hourly = (scan("gold", "hourly_long").filter(~pl.col("status").is_in(STRUCTURAL))
              .with_columns(pl.col("pollutant").cast(pl.String).cast(POLLUTANT_TYPE))
              .select(*SERIES, "ts_utc", "ts_local", "value", "value_imputed").collect())

    days, months, years = [], [], []
    for source, col in (("matje", "value"), ("me_imputim", "value_imputed")):
        d = daily(hourly, col)
        days.append(d.with_columns(pl.lit(source).alias("source")))
        months.append(monthly(d).with_columns(pl.lit(source).alias("source")))
        years.append(annual(hourly, d, col).with_columns(pl.lit(source).alias("source")))
    day, month, year = pl.concat(days), pl.concat(months), pl.concat(years)
    write(day, "gold", "daily")
    write(month, "gold", "monthly")
    write(year, "gold", "annual")
    lineage("p4", "agregimi orar → ditor", hourly.height, day.filter(pl.col("source") == "matje").height,
            hourly["value"].null_count(), day.filter(pl.col("source") == "matje")["mean"].null_count())

    gained = table(day.group_by("pollutant", "source").agg(pl.col("valid_day").sum())
                   .pivot(sort_columns=True, on="source", index="pollutant", values="valid_day")
                   .with_columns((pl.col("me_imputim") - pl.col("matje")).alias("dite_te_fituara"),
                                 (100 * (pl.col("me_imputim") / pl.col("matje") - 1)).round(1).alias("pct"))
                   .sort("pollutant"), "p4_ditet_e_fituara")
    measured = lambda d: d.filter(pl.col("source") == "matje")  # noqa: E731
    sizes = table(pl.DataFrame({"niveli": ["orar", "ditor", "mujor", "vjetor"],
                                "rreshta": [hourly.height, measured(day).height, measured(month).height,
                                            measured(year).height]})
                  .with_columns((pl.col("rreshta").first() / pl.col("rreshta")).round(0).alias("zvogelimi_x")),
                  "p4_zvogelimi")
    cov = measured(year)
    coverage = table(cov.with_columns(pl.col("station_id").cast(pl.String))
                     .pivot(on="year", index=["station_id", "pollutant"], values="coverage", sort_columns=True)
                     .sort("station_id", "pollutant"), "p4_mbulimi_vjetor")
    heating = table(measured(day).filter("valid_day")
                    .with_columns(pl.col("date").dt.month().is_in(T["heating_months"]).alias("ngrohje"))
                    .group_by("pollutant").agg(pl.col("mean").filter("ngrohje").mean().alias("sezoni_ngrohjes"),
                                               pl.col("mean").filter(~pl.col("ngrohje")).mean().alias("pjesa_tjeter"))
                    .with_columns((pl.col("sezoni_ngrohjes") / pl.col("pjesa_tjeter")).round(2).alias("raporti"))
                    .sort("pollutant"), "p4_sezoni_ngrohjes")
    # robustësia: raporti brenda çdo stacioni (përbërja e stacioneve ndryshon pak mes sezoneve, sepse mungesa varet
    # nga muaji); pastaj mediana mes stacioneve dhe stacionet që sillen ndryshe nga rrjeti
    per_station = table(measured(day).filter("valid_day")
                        .with_columns(pl.col("date").dt.month().is_in(T["heating_months"]).alias("ngrohje"))
                        .group_by("pollutant", "station_id")
                        .agg((pl.col("mean").filter("ngrohje").mean() / pl.col("mean").filter(~pl.col("ngrohje")).mean())
                             .round(2).alias("raporti"))
                        .with_columns(pl.col("station_id").cast(pl.String))
                        .sort("pollutant", "station_id"), "p4_sezoni_ngrohjes_stacionet")
    heating = heating.join(per_station.group_by("pollutant").agg(pl.col("raporti").median().alias("mediana_stacioneve")),
                           on="pollutant")
    inverted_pm = per_station.filter(pl.col("pollutant").is_in(["PM10", "PM2_5"]) & (pl.col("raporti") < 1))
    fig_daily = figure_daily(day)

    pop = population()
    rows, reps = [], []
    for method in ("rastësor", "i_stratifikuar"):
        s = draw(pop, method)
        r, rep = compare(pop, s, method)
        rows += r
        reps.append(rep)
        if method == "i_stratifikuar":
            write(s, "gold", "sample_10pct")
    dist = table(pl.DataFrame(rows), "p4_mostrimi_shperndarjet")
    rep = table(pl.DataFrame(reps), "p4_mostrimi_perfaqesimi")
    lineage("p4", "mostrimi i stratifikuar 10%", pop.height, reps[1]["n"], 0, 0)
    elapsed = time.perf_counter() - t0

    n_low = cov.filter(~pl.col("meets_90pct")).height
    write_section("p4", f"""
## 5. Agregimi dhe mostrimi

### 5.1 Agregimi
**Ditor** (sipas datës lokale, sepse dita e njerëzve dhe e trafikut ndjek orën lokale): mesatarja, maksimumi dhe numri
i orëve të vlefshme. **Rregulli i plotësisë (D19):** mesatarja ditore llogaritet vetëm kur ka ≥ {AG['min_hours_per_day']}
orë të vlefshme (75%); përndryshe null, sepse një "mesatare" nga pak orë (p.sh. vetëm natën) do të ishte e anuar.
**Mujor:** mesatarja e ditëve të vlefshme, vetëm kur ≥ {int(AG['min_days_per_month_frac'] * 100)}% e ditëve janë të vlefshme.
**Vjetor:** mesatarja dhe mbulimi (orë të vlefshme / orë të vitit).

Agregimi bëhet dy herë: nga **matjet** (zyrtare) dhe nga vlerat **me imputim**. Ditët e vlefshme të fituara falë imputimit:
{md(gained)}
Zvogëlimi i madhësisë:
{md(sizes)}
**Mbulimi vjetor:** {n_low} nga {cov.height} stacion-ndotës-vite janë nën {int(AG['min_annual_coverage'] * 100)}%
(D26): për to `mean_official` = null dhe ruhet vetëm `mean` si informative, sepse një mesatare vjetore me më pak se
90% të orëve s'e përfaqëson vitin (p.sh. mungon gjithë një sezon). Tabela e plotë: `tables/p4_mbulimi_vjetor.csv`.
{md(coverage, 12)}
**Sezoni i ngrohjes (tetor–prill) kundrejt pjesës tjetër** (mesatarja e ditëve të vlefshme):
{md(heating)}
Grimcat (PM2.5, PM10), CO dhe NO janë shumë më të larta në sezonin e ngrohjes (djegia e druve, qymyrit dhe inversioni
termik në dimër); O3 është më i lartë jashtë tij (formohet nga drita e diellit). SO2 është pothuajse i njëjtë
gjatë gjithë vitit, sepse burimi i tij kryesor është industria dhe termocentralet, jo ngrohja e shtëpive.
Mediana e raportit të llogaritur brenda çdo stacioni (`mediana_stacioneve`) përputhet me raportin e përbashkët, pra
përfundimi s'vjen nga përbërja e ndryshme e stacioneve mes sezoneve (`tables/p4_sezoni_ngrohjes_stacionet.csv`).

**Përjashtim i vërtetë, jo gabim:** te {', '.join(sorted(set(inverted_pm['station_id'])))} grimcat janë **më të ulëta** në sezonin e
ngrohjes (raporti {', '.join(f"{r[0]} {r[2]}" for r in inverted_pm.sort('pollutant').rows())}). XK0007A (Brezovica) është stacion
sfondi rural-remote në 1,803 m: në dimër ndodhet mbi shtresën e inversionit termik që e mban ndotjen nga ngrohja poshtë
në lugina, dhe toka është e mbuluar me borë; në verë arrin ndotja rajonale dhe pluhuri. Modeli përsëritet çdo vit, ndaj
vlerat mbahen të pandryshuara (D34).
{fig_daily}

### 5.2 Mostrimi
Popullata: {pop.height:,} stacion-orë (formati i gjerë, pas imputimit). Krahasohen dy metoda me
{int(S['fraction'] * 100)}%: **e rastësishme e thjeshtë** dhe **e stratifikuar** sipas stacion × sezon
(e njëjta përqindje nga çdo shtresë).

Shpërndarja e mostrës kundrejt popullatës (testi Kolmogorov–Smirnov: KS i vogël dhe p i madh = e njëjta shpërndarje):
{md(dist)}
Përfaqësimi i shtresave (devijimi maksimal i përqindjes së një stacioni/sezoni nga popullata, në pikë përqindjeje):
{md(rep)}
**Vendimi (D20):** ruhet mostra **e stratifikuar** (`data/gold/sample_10pct.parquet`). Të dyja metodat i ruajnë
shpërndarjet (KS p më i vogël: {dist['KS_p'].min():.2f} {'> 0.05' if dist['KS_p'].min() > 0.05 else '≤ 0.05!'}), por mostra e stratifikuar i garanton saktë proporcionet e stacioneve dhe sezoneve
(devijim ≈ 0), ndërsa e rastësishmja mund t'i shmanget rastësisht. Kjo ka rëndësi sepse ndotja ndryshon shumë sipas
stacionit dhe sezonit. Koha e p4: {elapsed:.1f} s.
""")
    print(f"p4 gati: {measured(day).height:,} ditë, mostra {reps[1]['n']:,}, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
