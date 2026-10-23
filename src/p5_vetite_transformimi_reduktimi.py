"""p5 – Zgjedhja e nënbashkësisë së vetive, krijimi i vetive, diskretizimi, binarizimi, transformimi
dhe reduktimi i dimensionit. Output: `data/gold/features_hourly.parquet` (gati për fazat e ardhshme).

Gjithçka që "mësohet" nga të dhënat (kufijtë e frekuencës, transformuesi, PCA) mësohet vetëm nga 2019–2023.
"""

from __future__ import annotations

import time
from math import pi

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import PowerTransformer

from utils import (
    CFG, COLORS, INK, NAMES, SEASONS, heatmap, lineage, md, plt, save_figure, scan, table, write, write_section,
)

F, R, E, B, T = CFG["features"], CFG["reduction"], CFG["eaqi"], CFG["binarization"], CFG["time"]
STRUCTURAL = ["not_measured", "outside_period"]
TRAIN_END = pl.lit(T["train_end"]).str.to_datetime().dt.replace_time_zone("UTC")

# Kolonat e burimit që NUK kalojnë te tabela e vetive, me arsyen (evidenca te p1_kualiteti_kolonat_pa_informacion)
DROPPED = [
    ("DataCapture", "100% bosh"),
    ("AggType", "konstante (gjithmonë 'hour'), variancë zero"),
    ("Verification", "konstante (gjithmonë 1), variancë zero"),
    ("End", "e tepërt: End = Start + 1h (0 përjashtime)"),
    ("ResultTime", "administrative: koha e publikimit, s'lidhet me ndotjen"),
    ("FkObservationLog", "identifikues administrativ"),
    ("Unit", "e tepërt pas harmonizimit (të gjitha në njësi kanonike)"),
    ("Samplingpoint", "e zbërthyer në station_id dhe pollutant"),
    ("Validity", "e përthithur te clean_action/status (p2–p3)"),
]


# ------------------------------------------------------------------------------------------ formati i gjerë + vetitë
def wide() -> pl.DataFrame:
    """Një rresht = stacion × orë. `<P>` = vlera pas imputimit, `<P>_obs` = vlera e matur (për lag pa leakage)."""
    long = (scan("gold", "hourly_long")
            .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_utc", "ts_local",
                    "value", "value_imputed", "f_gas_period_flag").collect())
    # orët e periudhës 2019 para ndërrimit të analizuesve, kur CO/SO2 mbahen por janë të dyshimta (D33, D35)
    flagged = (long.filter("f_gas_period_flag").select("station_id", "ts_utc").unique()
                   .with_columns(pl.lit(True).alias("gas_period_2019")))
    w = long.pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc", "ts_local"], values=["value_imputed", "value"])
    w = w.rename({**{f"value_imputed_{p}": p for p in NAMES}, **{f"value_{p}": f"{p}_obs" for p in NAMES}})
    return (w.join(flagged, on=["station_id", "ts_utc"], how="left")
             .with_columns(pl.col("gas_period_2019").fill_null(False)).sort("station_id", "ts_utc"))


def create_features(w: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    loc = pl.col("ts_local")
    cyc = lambda col, period: [(2 * pi * pl.col(col) / period).sin().alias(f"{col}_sin"),  # noqa: E731
                               (2 * pi * pl.col(col) / period).cos().alias(f"{col}_cos")]
    lags = []
    for p in F["lag_pollutants"]:
        x = pl.col(f"{p}_obs")    # nga matjet (jo nga vlerat e imputuara), brenda stacionit, pa orën aktuale
        lags += [x.shift(k).over("station_id").alias(f"{p}_lag_{k}h") for k in F["lags_h"]]
        lags.append(x.rolling_mean_by("ts_utc", f"{F['rolling_h']}h", closed="left").over("station_id")
                    .alias(f"{p}_roll_mean_{F['rolling_h']}h"))
    pm25, pm10 = pl.col("PM2_5_obs"), pl.col("PM10_obs")   # vetitë fizike nga matjet, jo nga vlerat e imputuara
    imputed = [(pl.col(f"{p}_obs").is_null() & pl.col(p).is_not_null()).alias(f"{p}_is_imputed") for p in NAMES]
    return (w.with_columns(*imputed, loc.dt.hour().alias("hour"), loc.dt.weekday().alias("dow"),
                           loc.dt.month().alias("month"),
                           loc.dt.year().alias("year"), (loc.dt.weekday() >= 6).alias("is_weekend"))
             .with_columns(pl.col("month").replace_strict(SEASONS, return_dtype=pl.String).alias("season"),
                           pl.col("month").is_in(T["heating_months"]).alias("is_heating_season"),
                           *cyc("hour", 24), *cyc("month", 12),
                           pl.when((pm10 > 0) & (pm25 <= pm10)).then(pm25 / pm10).alias("pm25_pm10_ratio"),
                           (pm10 - pm25).alias("pm_coarse"),
                           *lags)
             .join(stations, on="station_id", how="left")
             .sort("station_id", "ts_utc"))   # join s'e garanton renditjen; target-i (shift) e kërkon


NEW_FEATURES = [
    ("hour, dow, month, year", "nga ora lokale", "ndotja ndjek ritmin ditor (trafiku), javor dhe sezonal"),
    ("is_weekend", "dita e javës ≥ 6", "më pak trafik në fundjavë"),
    ("season, is_heating_season", "muaji → stina; tetor–prill", "ngrohja me dru/qymyr: PM2.5 2.5× më i lartë"),
    ("hour_sin, hour_cos, month_sin, month_cos", "sin/cos(2π·x/periudha)",
     "ora 23 dhe 0 janë fqinje; numri 23 vs 0 s'e tregon këtë"),
    ("<P>_is_imputed", "vlera e orës u imputua te p3", "transparenca: modelet mund t'i trajtojnë ndryshe"),
    ("gas_period_2019", "ora bie para ndërrimit të analizuesve të gazeve (nëntor 2019)",
     "CO/SO2 të dyshimta (D33): analizat mund ta përjashtojnë këtë periudhë"),
    ("pm25_pm10_ratio", "PM2.5 / PM10 të matura (kur PM10 > 0, PM2.5 ≤ PM10)", "lart = djegie (grimca të imta); poshtë = pluhur"),
    ("pm_coarse", "PM10 − PM2.5 të matura", "grimcat e trasha (pluhur, rrugë); më pak e koreluar me PM2.5 se PM10"),
    ("<P>_lag_1h, <P>_lag_24h", "vlera e matur 1h / 24h më parë, brenda stacionit", "ndotja ka persistencë"),
    ("<P>_roll_mean_24h", "mesatarja e 24h të kaluara (pa orën aktuale)", "niveli i fundit i ndotjes"),
    ("station_type, altitude_m", "nga metadata (p2)", "lloji i burimit (trafik/industri/sfond) dhe lartësia"),
]


# ------------------------------------------------------------------------------------------ zgjedhja e vetive
def feature_selection(f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Filtri i variancës, i korrelacionit dhe mutual information ndaj PM2.5 të (matur) të orës pasardhëse.

    - Ndotësit e orës aktuale merren nga vlerat e **matura** (`<P>_obs`), jo nga të imputuarat: interpolimi linear
      përdor vlerën e orës pasardhëse, pra një veti e imputuar do ta "shihte" objektivin (leakage, D27).
    - Çdo veti vlerësohet mbi rreshtat ku ajo dhe objektivi kanë vlerë (jo vetëm rreshta të plotë në të gjitha
      kolonat), që XK0007A, e cila s'mat NO, NOx, O3, të mos përjashtohet.
    """
    seed = CFG["sampling"]["seed"]
    data = (f.with_columns(pl.col("PM2_5_obs").shift(-1).over("station_id").alias("target"))
              .filter((pl.col("ts_utc") <= TRAIN_END) & pl.col("target").is_not_null()))
    skip = {"year", "target", "gas_period_2019", *NAMES} | {c for c in f.columns if c.endswith("_is_imputed")}
    features = [c for c, d in f.schema.items() if (d.is_numeric() or d == pl.Boolean) and c not in skip]
    rows = []
    for c in features:
        sub = data.select(pl.col(c).cast(pl.Float64), "target").drop_nulls()
        sub = sub.sample(n=min(F["mi_sample"], sub.height), seed=seed)
        x, y = sub[c].to_numpy().reshape(-1, 1), sub["target"].to_numpy()
        rows.append({"veti": c, "rreshta": sub.height, "varianca": float(x.var()),
                     "kalon_variancen": bool(VarianceThreshold(0.0).fit(x).get_support()[0]) if x.var() > 0 else False,
                     "mutual_information": float(mutual_info_regression(x, y, random_state=seed)[0]),
                     "spearman_me_targetin": float(stats.spearmanr(x.ravel(), y).statistic)})
    ranking = pl.DataFrame(rows).sort("mutual_information", descending=True)

    obs = [f"{p}_obs" for p in NAMES]
    rho = np.array([[1.0 if a == b else data.select(a, b).drop_nulls()
                     .select(pl.corr(a, b, method="spearman")).item() for b in obs] for a in obs])
    pairs = pl.DataFrame([{"a": NAMES[i], "b": NAMES[j], "spearman": round(float(rho[i, j]), 3)}
                          for i in range(len(NAMES)) for j in range(i + 1, len(NAMES))
                          if abs(rho[i, j]) > R["corr_threshold"]])
    corr = pl.DataFrame({"ndotesi": NAMES, **{p: np.round(rho[:, i], 2) for i, p in enumerate(NAMES)}})
    return ranking, pairs, corr


# ------------------------------------------------------------------------------------------ diskretizimi, binarizimi
def bins(col: str, edges: list[float], alias: str) -> pl.Expr:
    """Kodi i klasës = sa kufij të brendshëm i kalon vlera (intervale (a, b]); null mbetet null."""
    x = pl.col(col)
    return pl.when(x.is_not_null()).then(pl.sum_horizontal([(x > e).cast(pl.Int8) for e in edges])).alias(alias)


def discretize(f: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    train = f.filter(pl.col("ts_utc") <= TRAIN_END)["PM2_5"].drop_nulls().to_numpy()
    n = F["n_bins"]
    edges = {"gjeresi_e_barabarte": list(np.linspace(train.min(), train.max(), n + 1)[1:-1]),
             "frekuence_e_barabarte": list(np.quantile(train, np.linspace(0, 1, n + 1)[1:-1])),
             "eaqi": E["bins"]["PM2_5"]}
    f = f.with_columns(*[bins("PM2_5", e, f"pm25_bin_{k}") for k, e in edges.items()])
    return f, {k: [round(float(v), 2) for v in e] for k, e in edges.items()}


def daily_exceedance(f: pl.DataFrame) -> pl.DataFrame:
    """Tejkalimet ditore 0/1 nga mesataret ditore të vlefshme (p4), të bashkuara me çdo orë të asaj dite."""
    day = (scan("gold", "daily").filter((pl.col("source") == "matje") & pl.col("valid_day")
                                        & pl.col("pollutant").cast(pl.String).is_in(["PM10", "PM2_5"])).collect()
           .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "date", "mean")
           .pivot(sort_columns=True, on="pollutant", index=["station_id", "date"], values="mean")
           .select("station_id", "date", (pl.col("PM10") > B["pm10_daily_limit"]).cast(pl.Int8).alias("pm10_exceed_eu"),
                   (pl.col("PM2_5") > B["pm25_daily_who"]).cast(pl.Int8).alias("pm25_exceed_who")))
    return (f.with_columns(pl.col("ts_local").dt.date().alias("date")).join(day, on=["station_id", "date"], how="left")
             .drop("date").sort("station_id", "ts_utc"))


# ------------------------------------------------------------------------------------------ transformimi dhe PCA
def transforms(f: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    train = f.filter((pl.col("ts_utc") <= TRAIN_END) & ~pl.col("gas_period_2019"))   # D35
    rows, samples = [], {}
    for p in NAMES:
        x = train[p].drop_nulls().to_numpy()
        yj = PowerTransformer(method="yeo-johnson").fit_transform(x.reshape(-1, 1)).ravel()
        samples[p] = {"origjinale": x, "log1p": np.log1p(x), "Yeo-Johnson": yj}
        rows.append({"ndotesi": p, "anshmeria_origjinale": stats.skew(x), "anshmeria_log1p": stats.skew(np.log1p(x)),
                     "anshmeria_yeo_johnson": stats.skew(yj), "kurtoza_origjinale": stats.kurtosis(x),
                     "kurtoza_yeo_johnson": stats.kurtosis(yj)})
    return pl.DataFrame(rows), samples


def pca_variants(f: pl.DataFrame) -> pl.DataFrame:
    """PCA kryesore (7 ndotës) mbulon 11 stacione, sepse XK0007A s'mat NO dhe O3. Varianti me 5 ndotësit që i
    masin të 12 stacionet kontrollon nëse përfundimet ndryshojnë kur përfshihet edhe XK0007A."""
    rows = []
    for name, cols in (("7 ndotës (pa NOx)", [p for p in NAMES if p != "NOx"]),
                       ("5 ndotës të përbashkët (SO2, PM10, PM2.5, NO2, CO)", ["SO2", "PM10", "PM2_5", "NO2", "CO"])):
        train = f.drop_nulls(cols).filter((pl.col("ts_utc") <= TRAIN_END) & ~pl.col("gas_period_2019"))   # D35
        z = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(train.select(cols).to_numpy())
        ev = PCA().fit(z).explained_variance_ratio_
        rows.append({"varianti": name, "stacione": train["station_id"].n_unique(), "ore_train": train.height,
                     "PC1": round(float(ev[0]), 3), "PC2": round(float(ev[1]), 3),
                     "komponente_per_90pct": int(np.searchsorted(np.cumsum(ev), R["pca_variance"]) + 1)})
    return pl.DataFrame(rows)


def pca(f: pl.DataFrame, sample: pl.DataFrame) -> dict:
    cols = [p for p in NAMES if p != "NOx"]
    complete = f.drop_nulls(cols)
    train = complete.filter((pl.col("ts_utc") <= TRAIN_END) & ~pl.col("gas_period_2019"))   # D35
    pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(train.select(cols).to_numpy())
    model = PCA().fit(pt.transform(train.select(cols).to_numpy()))
    ev = model.explained_variance_ratio_
    s = sample.drop_nulls(cols).with_columns(pl.col("ts_local").dt.month().is_in(T["heating_months"]).alias("heat"))
    return {"cols": cols, "model": model, "ev": ev, "n_train": train.height, "n_complete": complete.height,
            "n_90": int(np.searchsorted(np.cumsum(ev), R["pca_variance"]) + 1),
            "scores": model.transform(pt.transform(s.select(cols).to_numpy()))[:, :2], "heat": s["heat"].to_numpy()}


# ------------------------------------------------------------------------------------------ figurat
def figures(f: pl.DataFrame, samples: dict, p: dict, corr: pl.DataFrame) -> dict[str, str]:
    figs = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    names = {"gjeresi_e_barabarte": "gjerësi e barabartë", "frekuence_e_barabarte": "frekuencë e barabartë",
             "eaqi": "EAQI (domeni)"}
    for ax, (k, title) in zip(axes, names.items(), strict=True):
        c = f[f"pm25_bin_{k}"].drop_nulls().value_counts().sort(f"pm25_bin_{k}")
        ax.bar(c[f"pm25_bin_{k}"], 100 * c["count"] / c["count"].sum(), color=COLORS[0])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("klasa")
    axes[2].set_xticks(range(6), E["labels"], rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("% e orëve")
    fig.suptitle("PM2.5: shpërndarja e klasave sipas metodës së diskretizimit", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["klasat"] = save_figure(fig, "p5_diskretizimi_pm25")

    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5))
    for i, pol in enumerate(["PM2_5", "NO2", "O3"]):
        for j, (name, v) in enumerate(samples[pol].items()):
            axes[i, j].hist(v, bins=70, color=COLORS[j])
            axes[i, j].set_title(f"{pol}: {name} (anshmëria {stats.skew(v):.2f})", fontsize=9)
    fig.suptitle("Histogramet para dhe pas transformimit", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs["transformimi"] = save_figure(fig, "p5_transformimi")

    figs["korrelacioni"] = save_figure(heatmap(corr.with_columns(pl.exclude("ndotesi").abs()),
                                               "|Spearman ρ| mes ndotësve", "|ρ|"), "p5_korrelacioni")

    ev = p["ev"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    n = np.arange(1, len(ev) + 1)
    ax.bar(n, ev, color=COLORS[0], label="varianca e komponentit")
    ax.plot(n, np.cumsum(ev), marker="o", color=COLORS[1], label="kumulative")
    ax.axhline(R["pca_variance"], color=INK, lw=0.8, ls="--")
    ax.set_xticks(n, [f"PC{i}" for i in n])
    ax.set_ylabel("pjesa e variancës")
    ax.set_title(f"Scree plot: {p['n_90']} komponentë për {int(R['pca_variance'] * 100)}% të variancës")
    ax.legend(loc="center right")
    figs["scree"] = save_figure(fig, "p5_pca_scree")

    sc, heat, model = p["scores"], p["heat"], p["model"]
    order = np.random.default_rng(0).permutation(len(sc))
    lim = np.quantile(np.abs(sc), 0.995, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(sc[order, 0], sc[order, 1], s=3, alpha=0.25, linewidths=0,
               c=np.where(heat[order], COLORS[1], COLORS[0]))
    for i, label in enumerate(["jashtë sezonit të ngrohjes", "sezoni i ngrohjes (tet–pri)"]):
        ax.scatter([], [], s=20, color=COLORS[i], label=label)
    scale = 0.85 * min(lim[0] / np.abs(model.components_[0]).max(), lim[1] / np.abs(model.components_[1]).max())
    for j, name in enumerate(p["cols"]):
        dx, dy = model.components_[0, j] * scale, model.components_[1, j] * scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2})
        ax.text(dx * 1.1, dy * 1.1, name, fontsize=9, color=INK)
    ax.set_xlim(-1.15 * lim[0], 1.15 * lim[0])
    ax.set_ylim(-1.15 * lim[1], 1.15 * lim[1])
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title("Biplot PC1–PC2 (mostra e stratifikuar nga p4)")
    ax.legend(loc="lower right", fontsize=8)
    figs["biplot"] = save_figure(fig, "p5_pca_biplot")
    return figs


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.perf_counter()
    stations = (scan("silver", "stations").select("station_id", "station_type", "altitude_m").collect())
    w = wide()
    f = create_features(w, stations)
    ranking, pairs, corr = feature_selection(f)
    f, edges = discretize(f)
    f = daily_exceedance(f)
    onehot = f.select("station_id", "season", "station_type").to_dummies()
    f = pl.concat([f, onehot], how="horizontal")
    skew, samples = transforms(f)
    sample = scan("gold", "sample_10pct").select("station_id", "ts_utc", "ts_local", *NAMES).collect()
    p = pca(f, sample)
    variants = table(pca_variants(f), "p5_pca_variantet")
    # kontrolli kundër leakage (D21): lag_1h(t) duhet të jetë saktë vlera e matur e orës t − 1h të të njëjtit stacion
    prev = f.select("station_id", (pl.col("ts_utc") + pl.duration(hours=1, time_unit="ns")).alias("ts_utc"),
                    pl.col("PM2_5_obs").alias("_prev"))
    leaks = (f.select("station_id", "ts_utc", "PM2_5_lag_1h").join(prev, on=["station_id", "ts_utc"], how="left")
              .filter(~pl.col("PM2_5_lag_1h").eq_missing(pl.col("_prev"))).height)
    if leaks or not f.select("station_id", "ts_utc").is_unique().all():
        raise SystemExit(f"STOP: {leaks} lag të gabuara ose çelës jo unik te features_hourly")
    write(f, "gold", "features_hourly")
    lineage("p5", "formati i gjerë + vetitë → gold", w.height, f.height, 0, 0)

    dropped = table(pl.DataFrame(DROPPED, schema=["kolona", "arsyeja"], orient="row"), "p5_kolonat_e_hequra")
    created = table(pl.DataFrame(NEW_FEATURES, schema=["veti", "formula", "arsyeja"], orient="row"), "p5_vetite_e_reja")
    table(ranking, "p5_zgjedhja_vetive")
    table(pairs, "p5_ciftet_korrelacion_i_larte")
    table(corr, "p5_korrelacioni_spearman")
    table(skew, "p5_anshmeria")
    edges_t = table(pl.DataFrame([{"metoda": k, "kufijte": ", ".join(map(str, v))} for k, v in edges.items()]),
                    "p5_kufijte_diskretizimit")
    eaqi_dist = table(f["pm25_bin_eaqi"].drop_nulls().value_counts().sort("pm25_bin_eaqi")
                      .with_columns(pl.col("pm25_bin_eaqi").replace_strict(dict(enumerate(E["labels"])),
                                                                          return_dtype=pl.String).alias("klasa"),
                                    (100 * pl.col("count") / pl.col("count").sum()).round(1).alias("pct")),
                      "p5_eaqi_pm25")
    exceed = table(f.with_columns(pl.col("ts_local").dt.date().alias("d"))
                   .unique(["station_id", "d"]).group_by("station_id", "year")
                   .agg(pl.col("pm10_exceed_eu").sum().alias("dite_PM10_>50"),
                        pl.col("pm25_exceed_who").sum().alias("dite_PM2.5_>15"))
                   .sort("station_id", "year"), "p5_tejkalimet")
    ev = p["ev"]
    loadings = table(pl.DataFrame({"ndotesi": p["cols"], **{f"PC{i + 1}": np.round(p["model"].components_[i], 3)
                                                            for i in range(3)}}), "p5_pca_ngarkesat")
    explained = table(pl.DataFrame({"komponenti": [f"PC{i + 1}" for i in range(len(ev))],
                                    "varianca": np.round(ev, 3), "kumulative": np.round(np.cumsum(ev), 3)}),
                      "p5_pca_varianca")
    nox_r2 = f.filter(pl.col("NOx_obs").is_not_null() & pl.col("NO2_obs").is_not_null() & pl.col("NO_obs").is_not_null()) \
              .select(pl.corr("NOx_obs", pl.col("NO2_obs") + CFG["quality"]["no_to_no2"] * pl.col("NO_obs")) ** 2).item()
    figs = figures(f, samples, p, corr)
    zero_var = ", ".join(ranking.filter(~pl.col("kalon_variancen"))["veti"]) or "asnjë"
    skewed = ", ".join(skew.filter(pl.col("anshmeria_origjinale") > 1)["ndotesi"])
    o3_skew = skew.filter(pl.col("ndotesi") == "O3")["anshmeria_origjinale"][0]
    elapsed = time.perf_counter() - t0

    write_section("p5", f"""
## 6. Zgjedhja dhe krijimi i vetive, diskretizimi, binarizimi, transformimi, reduktimi

### 6.1 Formati i gjerë
Nga tabela e gjatë (p3) ndërtohet tabela e gjerë: një rresht = stacion × orë ({w.height:,} rreshta), një kolonë për çdo
ndotës (vlera pas imputimit) dhe `<P>_obs` (vlera e matur, që vetitë e vonuara të mos përdorin vlera të imputuara).

### 6.2 Zgjedhja e nënbashkësisë së vetive
**Kolonat e burimit që hiqen** (evidenca te p1, "kolonat pa informacion"):
{md(dropped)}
**Filtrat mbi vetitë numerike**, me objektiv PM2.5 **e matur** e orës pasardhëse (2019–2023, deri në
{F['mi_sample']:,} rreshta për veti):
- *filtri i variancës:* hiqet çdo veti me variancë 0 (konstante);
- *mutual information:* sa informacion jep vetia për objektivin, edhe për lidhje jolineare;
- *korrelacioni Spearman* me objektivin.

Dy masa kundër rezultateve të rreme (D27): (1) ndotësit e orës aktuale merren nga vlerat **e matura** (`<P>_obs`),
sepse interpolimi linear i p3 e përdor vlerën e orës pasardhëse, pra një vlerë e imputuar do ta "shihte" objektivin;
(2) çdo veti vlerësohet mbi rreshtat ku ajo dhe objektivi kanë vlerë, që XK0007A (pa NO, NOx, O3) të mos përjashtohet.
{md(ranking.select('veti', 'kalon_variancen', 'mutual_information', 'spearman_me_targetin'), 20)}
Më informative janë PM2.5 e matur e orës aktuale dhe e orës paraprake, PM10, dhe mesataret e 24h të kaluara
(persistenca e ndotjes); pastaj `pm_coarse`, NOx/NO2 dhe muaji (sezonaliteti). Dita e javës dhe fundjava s'japin
pothuajse asnjë informacion për orën pasardhëse (MI ≈ 0). Vetitë me variancë 0: {zero_var}.
**Filtri i korrelacionit** (|ρ| > {R['corr_threshold']}, mbi vlerat e matura) gjen çiftet e tepërta:
{md(pairs)}
Vendimi për to merret te reduktimi i dimensionit (6.7).

### 6.3 Krijimi i vetive
{md(created)}
Vetitë e vonuara llogariten brenda të njëjtit stacion, mbi rrjetin e plotë orar (shift 1 = saktë 1 orë më parë),
nga vlerat e matura dhe pa orën aktuale (`closed="left"`): asnjë veti s'përdor të ardhmen (D21).

### 6.4 Diskretizimi i PM2.5
Kufijtë e gjerësisë dhe të frekuencës së barabartë mësohen nga 2019–2023; EAQI përdor pragjet zyrtare të EEA
({E['source']}, revizioni 2024).
{md(edges_t)}
{figs['klasat']}
- **Gjerësi e barabartë:** shpërndarja shumë e anuar fut pothuajse gjithçka në klasën e parë; e papërdorshme.
- **Frekuencë e barabartë:** klasa të balancuara (rreth 20% secila), por kufijtë s'kanë kuptim shëndetësor.
- **EAQI:** klasa të pabalancuara, por **të interpretueshme** dhe të krahasueshme me Evropën → zgjidhet për analizë (D22).
{md(eaqi_dist)}

### 6.5 Binarizimi dhe kodimi
- `pm10_exceed_eu` = 1 kur mesatarja ditore e PM10 > {B['pm10_daily_limit']} µg/m³ (kufiri ditor i BE-së);
  `pm25_exceed_who` = 1 kur PM2.5 ditor > {B['pm25_daily_who']} µg/m³ (udhëzimi i OBSH-së 2021).
  Llogariten nga mesataret ditore **të matura** dhe vetëm për ditët e vlefshme (≥ 18 orë, p4), sepse numri i
  tejkalimeve krahasohet me kufijtë ligjorë, të cilët vlerësohen mbi matje, jo mbi vlera të imputuara (D28).
{md(exceed, 12)}
- **One-hot** për stacionin, sezonin dhe llojin e stacionit ({onehot.width} kolona 0/1): kategoritë nominale s'kanë
  renditje, ndaj s'mund të kodohen si 1, 2, 3.

### 6.6 Transformimi
Ndotësit janë të anuar djathtas (shumë orë me pak ndotje, pak orë me shumë): anshmëria > 1 te {skewed};
O3 është afërsisht simetrik ({o3_skew:.2f}).
{md(skew)}
{figs['transformimi']}
- **log1p** e ul anshmërinë shumë, por e prish O3-in (që ishte pothuajse simetrik → anshmëri negative).
- **Yeo-Johnson** e çon anshmërinë afër 0 për të gjithë ndotësit dhe pranon zero → zgjidhet (D23). Box-Cox s'përdoret
  sepse kërkon vlera > 0, ndërsa zerot janë reale.
- **Shkallëzimi:** `StandardScaler` (mesatarja 0, std 1) për vlera afërsisht normale, si pas Yeo-Johnson;
  `RobustScaler` (mediana dhe IQR) kur mbeten outliers, sepse s'ndikohet prej tyre. Këtu përdoret standardizimi pas
  Yeo-Johnson; RobustScaler është alternativa nëse vlerat ekstreme mbahen pa transformim.

### 6.7 Reduktimi i dimensionit
{figs['korrelacioni']}
**NOx hiqet (D24):** NOx = NO + NO2 kimikisht; mbi matjet, NO2 + 1.53·NO shpjegon NOx me R² = {nox_r2:.4f}: pas nëntorit 2019 NOx **llogaritet** nga NO dhe NO2 nga vetë rrjeti.
Është i tepërt dhe do ta dyfishonte peshën e NO/NO2 në PCA. PM10 dhe PM2.5 (ρ = {pairs.filter(pl.col('a') == 'PM10')['spearman'][0]})
mbahen të dyja, sepse `pm_coarse` = PM10 − PM2.5 tregon që PM10 mbart edhe informacion tjetër (pluhurin).

**PCA** mbi 7 ndotës (Yeo-Johnson + standardizim, mësuar nga {p['n_train']:,} orë të plota të 2019–2023; vlerat pas
imputimit, që të mos humbasin orët me një boshllëk të vetëm). Transformuesi dhe PCA s'mësojnë nga periudha e 2019 para
ndërrimit të analizuesve (`gas_period_2019`), ku CO dhe SO2 janë të dyshimta (D35). PCA kërkon rreshta të plotë, ndaj **XK0007A s'përfshihet**
(s'mat NO dhe O3). Kontrolli me 5 ndotësit që i masin të 12 stacionet (D29):
{md(variants)}
Struktura mbetet e njëjtë: PC1 (ndotja nga djegia) shpjegon {variants['PC1'][0]:.0%} me 11 stacione dhe
{variants['PC1'][1]:.0%} me 12; me më pak ndotës mjaftojnë më pak komponentë për 90%. Përfshirja e XK0007A s'e ndryshon
interpretimin.
{md(explained)}
{figs['scree']}
Ngarkesat:
{md(loadings)}
{figs['biplot']}
**Interpretimi:**
- **PC1 ({ev[0]:.0%})** = ndotja e përgjithshme nga djegia: PM2.5, PM10, NO, NO2 dhe CO me ngarkesa pozitive, O3 me
  shenjë të kundërt (O3 bie aty ku NO e konsumon). Mesatarja e PC1 në sezonin e ngrohjes është
  {p['scores'][p['heat'], 0].mean():.2f}, jashtë tij {p['scores'][~p['heat'], 0].mean():.2f}: dimri = më shumë ndotje nga djegia.
- **PC2 ({ev[1]:.0%})** dominohet nga SO2: burim më vete (industria/termocentralet), pa sezonalitet (p4).
- **{p['n_90']} komponentë** mjaftojnë për {int(R['pca_variance'] * 100)}% të variancës, nga 7 ndotës.

Output: `data/gold/features_hourly.parquet` ({f.height:,} rreshta × {f.width} kolona). Koha e p5: {elapsed:.1f} s.
""")
    print(f"p5 gati: {f.height:,} × {f.width}, PC1 {ev[0]:.1%}, {p['n_90']} PC për 90%, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
