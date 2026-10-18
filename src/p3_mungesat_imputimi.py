"""p3 – Vlerat e zbrazëta: identifikimi, analiza, vlerësimi i metodave dhe strategjia e trajtimit.

1. Rrjeti i plotë orar (orët që mungojnë fare bëhen të dukshme).
2. Statusi i çdo ore; mungesat strukturore (`not_measured`, `outside_period`) ndahen nga ato reale.
3. Analiza: gjatësia e boshllëqeve, kur mungojnë, a mungojnë bashkë → MCAR / MAR / MNAR.
4. Vlerësimi i 4 metodave mbi boshllëqe artificiale në segmente të plota (2024).
5. Imputimi sipas rendit të vendimeve; kontrollet para/pas.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy import stats

from utils import (
    CFG, COLORS, NAMES, POLLUTANT_TYPE, REPORTS, SERIES, by_name, heatmap, lineage, md, plt, save_figure, scan, table,
    write, write_section,
)

M, T = CFG["missing"], CFG["time"]
KEY = [*SERIES, "ts_utc"]
HOUR = pl.duration(hours=1, time_unit="ns")
STRUCTURAL = ["not_measured", "outside_period"]
TRAIN_END = pl.lit(T["train_end"]).str.to_datetime().dt.replace_time_zone("UTC")
METHODS = ["ffill", "linear", "station_median", "profile"]
Q_STUCK_OK = CFG["quality"]["stuck_zero_plausible"]


# ------------------------------------------------------------------------------------------ 1–2. rrjeti dhe statusi
def build_grid(clean: pl.DataFrame) -> pl.DataFrame:
    """Stacion × ndotës × orë mbi periudhën e stacionit, me statusin e çdo ore."""
    station_range = clean.group_by("station_id").agg(pl.col("ts_utc").min().alias("a"), pl.col("ts_utc").max().alias("b"))
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
    """`gap_id`, `gap_len_h` dhe `gap_position` (inside / edge) për boshllëqet reale."""
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


# ------------------------------------------------------------------------------------------ 3. analiza
def analysis(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    real = df.filter(~pl.col("status").is_in(STRUCTURAL)).with_columns(pl.col("value").is_null().alias("missing"))
    out: dict[str, pl.DataFrame] = {}
    out["status_ndotes"] = (df.group_by("pollutant", "status").len()
                            .pivot(on="status", index="pollutant", values="len", sort_columns=True).fill_null(0)
                            .sort("pollutant"))
    out["mungesa_reale_stacion"] = (real.group_by("station_id", "pollutant")
                                    .agg((100 * pl.col("missing").mean()).round(1).alias("pct"))
                                    .with_columns(pl.col("station_id").cast(pl.String))
                                    .pivot(sort_columns=True, on="pollutant", index="station_id", values="pct").sort("station_id"))
    gaps = df.filter(pl.col("gap_id").is_not_null()).group_by("gap_id").agg(pl.col("gap_len_h").first())
    bins = [(1, 1, "1 orë"), (2, 3, "2–3 orë"), (4, 24, "4–24 orë"), (25, 168, "1–7 ditë"), (169, None, "> 7 ditë")]
    label = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(bins):
        label = pl.when(pl.col("gap_len_h") >= lo if hi is None else pl.col("gap_len_h").is_between(lo, hi)) \
                  .then(pl.lit(name)).otherwise(label)
    order = pl.DataFrame({"gjatesia": [b[2] for b in bins], "_o": range(len(bins))})
    out["boshlleqet"] = (order.join(gaps.with_columns(label.alias("gjatesia")).group_by("gjatesia")
                                    .agg(pl.len().alias("boshlleqe"), pl.col("gap_len_h").sum().alias("ore")),
                                    on="gjatesia", how="left").fill_null(0).sort("_o").drop("_o")
                         .with_columns((100 * pl.col("boshlleqe") / pl.col("boshlleqe").sum()).round(1).alias("pct_boshlleqe"),
                                       (100 * pl.col("ore") / pl.col("ore").sum()).round(1).alias("pct_ore")))
    # boshllëku më i gjatë për çdo ndotës (me stacionin dhe fillimin), për ta shpjeguar jo vetëm numrin
    longest = (df.filter(pl.col("gap_id").is_not_null())
                 .group_by("gap_id").agg(pl.col("pollutant").first(), pl.col("station_id").first(),
                                         pl.col("ts_local").min().dt.date().alias("fillimi"),
                                         pl.col("gap_len_h").first())
                 .sort("gap_len_h", descending=True).group_by("pollutant", maintain_order=True).first()
                 .with_columns((pl.col("gap_len_h") / 24).round(1).alias("dite")).drop("gap_id").sort("pollutant"))
    out["boshlleqi_max_dite"] = longest
    out["mungesa_muaj"] = real.group_by("month").agg((100 * pl.col("missing").mean()).round(2).alias("pct")).sort("month")
    out["mungesa_ore"] = real.group_by("hour").agg((100 * pl.col("missing").mean()).round(2).alias("pct")).sort("hour")
    out["mungesa_vit_pa_nox"] = (real.filter(~pl.col("pollutant").is_in(["NO", "NO2", "NOx"])).group_by("year")
                                 .agg((100 * pl.col("missing").mean()).round(2).alias("pct")).sort("year"))
    out["mungesa_vit"] = real.group_by("year").agg((100 * pl.col("missing").mean()).round(2).alias("pct")).sort("year")

    # a mungojnë ndotësit bashkë? korrelacioni i treguesve të mungesës në të njëjtin stacion dhe orë
    ind = (real.select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), pl.col("missing").cast(pl.Int8))
               .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="missing"))
    out["korrelacioni_mungesave"] = pl.DataFrame([
        {"ndotesi": a, **{b: 1.0 if a == b else round(ind.select(a, b).drop_nulls().select(pl.corr(a, b)).item(), 2)
                          for b in NAMES}} for a in NAMES])
    # testi chi-square: a varet mungesa nga muaji / viti / ora? Me ~600k orë për ndotës, p del gjithmonë ≈ 0,
    # prandaj raportohet edhe madhësia e efektit, Cramér's V (0 = pa lidhje, > 0.1 e dobët, > 0.3 e moderuar)
    rows = []
    for p in NAMES:
        sub = real.filter(pl.col("pollutant") == p)
        row = {"ndotesi": p}
        for dim in ("month", "year", "hour"):
            ct = (sub.group_by(dim, "missing").len().pivot(sort_columns=True, on="missing", index=dim, values="len").fill_null(0)
                  .drop(dim).to_numpy())
            chi2, pval, _, _ = stats.chi2_contingency(ct)
            row[f"V_{dim}"] = round(float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1)))), 3)
            row[f"p_{dim}"] = f"{pval:.0e}"
        rows.append(row)
    out["chi2_cramers_v"] = pl.DataFrame(rows)

    # MCAR për boshllëqet 1-orëshe: nëse janë krejt të rastësishme, (a) ora e tyre ndjek shpërndarjen e të gjitha
    # orëve dhe (b) vlerat përreth tyre s'dallojnë nga vlerat e zakonshme të serisë
    one = df.filter(pl.col("gap_len_h") == 1)
    hours_all = real.group_by("hour").len().sort("hour")["len"].to_numpy()
    hours_gap = one.group_by("hour").len().join(pl.DataFrame({"hour": range(24)}, schema={"hour": pl.Int8}),
                                                on="hour", how="right").fill_null(0).sort("hour")["len"].to_numpy()
    expected = hours_all / hours_all.sum() * hours_gap.sum()
    chi2, pval = stats.chisquare(hours_gap, expected)
    v_hour = float(np.sqrt(chi2 / (hours_gap.sum() * 23)))
    before = (df.select(*SERIES, (pl.col("ts_utc") + HOUR).alias("ts_utc"), pl.col("value").alias("v_before"))
                .join(one.select(*KEY), on=KEY, how="inner"))
    neigh = (real.select("pollutant", "value").drop_nulls()
                 .group_by("pollutant").agg(pl.col("value").mean().alias("m"), pl.col("value").std().alias("s"))
                 .join(before.drop_nulls("v_before").group_by("pollutant").agg(pl.col("v_before").mean().alias("mb")),
                       on="pollutant")
                 .with_columns(((pl.col("mb") - pl.col("m")) / pl.col("s")).round(3).alias("diferenca_ne_std"))
                 .sort("pollutant"))
    out["mcar_1h"] = pl.DataFrame([{"boshlleqe_1h": int(hours_gap.sum()), "chi2_ora": round(chi2, 1),
                                    "p": f"{pval:.0e}", "cramers_v_ora": round(v_hour, 3)}])
    out["mcar_1h_vlerat"] = neigh.select("pollutant", pl.col("m").round(2).alias("mesatarja_seria"),
                                         pl.col("mb").round(2).alias("mesatarja_ora_para_boshllekut"),
                                         "diferenca_ne_std")
    for name, t in out.items():
        table(t, f"p3_{name}")
    return out


# ------------------------------------------------------------------------------------------ 4. vlerësimi i metodave
def artificial_gaps(test: pl.DataFrame, length: int) -> pl.DataFrame:
    """Boshllëqe artificiale me gjatësi `length` në segmente të plota (L orë + ora para dhe pas të vëzhguara)."""
    span = length + 2
    ok = pl.col("value").is_not_null().cast(pl.Int32)
    cand = (test.with_columns(ok.rolling_sum(span).over(SERIES).alias("_ok"),
                              pl.col("ts_utc").shift(span - 1).over(SERIES).alias("_t0"))
                .filter((pl.col("_ok") == span) & (pl.col("ts_utc") - pl.col("_t0") == HOUR * (span - 1)))
                .select(*SERIES, (pl.col("_t0") + HOUR).alias("gap_start")))
    rng = np.random.default_rng(M["seed"] + length)
    cand = (cand.with_columns(pl.Series("_u", rng.random(cand.height))).sort("_u")
                .filter(pl.int_range(pl.len()).over(SERIES) < M["eval_gaps_per_series"]))
    return (cand.with_columns(pl.int_ranges(0, length).alias("offset")).explode("offset")
                .with_columns((pl.col("gap_start") + HOUR * pl.col("offset")).alias("ts_utc"), pl.lit(length).alias("L")))


def evaluate(df: pl.DataFrame, profile: pl.DataFrame, station_median: pl.DataFrame
             ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """MAE dhe RMSE e çdo metode mbi boshllëqe artificiale të periudhës test (2024).

    Kthen edhe gabimin e ndarë sipas afërsisë me një boshllëk real (±24h): segmentet e plota zgjidhen në periudha
    "të qeta", ndërsa boshllëqet reale ndodhin pranë prishjeve, ndaj kontrollohet a mbetet strategjia e vlefshme aty.
    """
    obs = (df.filter(~pl.col("status").is_in(STRUCTURAL)).select(*KEY, "value", "month", "hour")
             .with_columns(pl.col("value").is_null().cast(pl.Int8).rolling_max(49, center=True, min_samples=1)
                           .over(SERIES).cast(pl.Boolean).alias("near_gap")))
    test = obs.filter(pl.col("ts_utc") > TRAIN_END)
    anchors = obs.select(*SERIES, "ts_utc", "value")
    parts = []
    for L in M["eval_gap_lengths_h"]:
        p = (artificial_gaps(test, L)
             .join(obs.drop("month", "hour").rename({"value": "truth"}), on=KEY, how="left")
             .join(obs.select("ts_utc", "month", "hour").unique(), on="ts_utc", how="left")
             .with_columns((pl.col("gap_start") - HOUR).alias("_tp"), (pl.col("gap_start") + HOUR * L).alias("_tn"))
             .join(anchors.rename({"ts_utc": "_tp", "value": "v_prev"}), on=[*SERIES, "_tp"], how="left")
             .join(anchors.rename({"ts_utc": "_tn", "value": "v_next"}), on=[*SERIES, "_tn"], how="left")
             .join(profile, on=[*SERIES, "month", "hour"], how="left")
             .join(station_median, on=SERIES, how="left"))
        frac = (pl.col("offset") + 1) / (L + 1)
        parts.append(p.select(
            *SERIES, "L", "truth", "near_gap",
            pl.col("v_prev").alias("ffill"),
            (pl.col("v_prev") + (pl.col("v_next") - pl.col("v_prev")) * frac).alias("linear"),
            "station_median", "profile"))
    long = (pl.concat(parts).unpivot(index=[*SERIES, "L", "truth", "near_gap"], on=METHODS, variable_name="method",
                                     value_name="pred").drop_nulls("pred"))
    err = pl.col("pred") - pl.col("truth")
    metrics = [pl.len().alias("n"), err.abs().mean().alias("MAE"), (err ** 2).mean().sqrt().alias("RMSE")]
    ev = long.group_by("pollutant", "L", "method").agg(*metrics).sort("pollutant", "L", "method")
    near = (long.group_by("near_gap", "L", "method").agg(*metrics)
                .with_columns(pl.when("near_gap").then(pl.lit("pranë boshllëkut real (±24h)"))
                              .otherwise(pl.lit("periudhë e qetë")).alias("konteksti"))
                .drop("near_gap").sort("konteksti", "L", "method"))
    return ev, near


def deterministic_check(df: pl.DataFrame) -> pl.DataFrame:
    """Sa saktë e jep NO2 + 1.53·NO vlerën e NOx kur të tre janë të matur (validimi i metodës deterministike)."""
    w = (df.filter(pl.col("pollutant").is_in(["NOx", "NO2", "NO"]) & pl.col("value").is_not_null())
           .select("station_id", "ts_utc", pl.col("pollutant").cast(pl.String), "value")
           .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value").drop_nulls())
    e = pl.col("NO2") + CFG["quality"]["no_to_no2"] * pl.col("NO") - pl.col("NOx")
    return w.select(pl.len().alias("ore"), e.abs().mean().alias("MAE"), (e ** 2).mean().sqrt().alias("RMSE"),
                    pl.col("NOx").mean().alias("mesatarja_NOx"))


# ------------------------------------------------------------------------------------------ 5. imputimi
def impute(df: pl.DataFrame, profile: pl.DataFrame, station_median: pl.DataFrame, mid_method: dict) -> pl.DataFrame:
    real_null = ~pl.col("status").is_in(["ok", *STRUCTURAL])
    df = df.with_columns(pl.col("value").alias("value_imputed"),
                         pl.when(real_null).then(pl.lit("left_null")).otherwise(pl.lit("none")).alias("impute_method"))

    def fill(df: pl.DataFrame, cond: pl.Expr, pred: pl.Expr, name: str) -> pl.DataFrame:
        use = cond & pl.col("value_imputed").is_null() & pred.is_not_null()
        return df.with_columns(pl.when(use).then(pl.lit(name)).otherwise(pl.col("impute_method")).alias("impute_method"),
                               pl.when(use).then(pred.clip(lower_bound=0)).otherwise(pl.col("value_imputed"))
                               .alias("value_imputed"))

    # 1. deterministik: NOx = NO2 + 1.53·NO në të njëjtën orë, kur NO2 dhe NO janë të vlefshme (D15, D33)
    w = (df.filter(pl.col("pollutant").is_in(["NO2", "NO"])).select("station_id", "ts_utc",
                                                                       pl.col("pollutant").cast(pl.String), "value")
           .pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc"], values="value"))
    df = df.join(w, on=["station_id", "ts_utc"], how="left").sort(KEY)   # join s'e garanton renditjen
    df = fill(df, real_null & (pl.col("pollutant") == "NOx"), pl.col("NO2") + CFG["quality"]["no_to_no2"] * pl.col("NO"),
              "deterministic").drop("NO2", "NO")

    # 2–3. boshllëqe brenda serisë: ≤ 3h linear; 4–24h metoda më e mirë sipas vlerësimit
    inside = real_null & (pl.col("gap_position") == "inside")
    linear = pl.col("value_imputed").interpolate().over(SERIES)
    df = fill(df, inside & (pl.col("gap_len_h") <= M["interp_max_h"]), linear, "linear")
    df = (df.join(profile, on=[*SERIES, "month", "hour"], how="left").join(station_median, on=SERIES, how="left")
            .sort(KEY)   # interpolimi dhe forward-fill kërkojnë renditje sipas kohës brenda serisë
            .with_columns(pl.col("value_imputed").forward_fill().over(SERIES).alias("ffill")))
    mid = inside & pl.col("gap_len_h").is_between(M["interp_max_h"] + 1, M["mid_gap_max_h"])
    candidates = {"linear": pl.col("value_imputed").interpolate().over(SERIES), "profile": pl.col("profile"),
                  "station_median": pl.col("station_median"), "ffill": pl.col("ffill")}
    for method in set(mid_method.values()):
        pols = [p for p, m in mid_method.items() if m == method]
        df = fill(df, mid & pl.col("pollutant").is_in(pols), candidates[method], method)
    return df.drop("profile", "station_median", "ffill")


# ------------------------------------------------------------------------------------------ figurat
def figures(df: pl.DataFrame, a: dict, ev: pl.DataFrame, methods: pl.DataFrame) -> dict[str, str]:
    figs = {}
    real = df.filter(~pl.col("status").is_in(STRUCTURAL))
    m = (real.with_columns(pl.col("ts_local").dt.strftime("%Y-%m").alias("ym"),
                           pl.format("{} {}", pl.col("station_id"), pl.col("pollutant")).alias("seria"))
             .group_by("seria", "ym").agg((100 * pl.col("value").is_null().mean()).alias("pct"))
             .pivot(on="ym", index="seria", values="pct", sort_columns=True).sort("seria"))
    figs["harta"] = save_figure(heatmap(m, "Harta e mungesave: % e orëve që mungojnë (seria × muaji)", "% mungon",
                                        annotate=False), "p3_harta_mungesave")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].bar(a["mungesa_muaj"]["month"], a["mungesa_muaj"]["pct"], color=COLORS[0])
    axes[0].set_title("% mungesë sipas muajit")
    axes[1].plot(a["mungesa_ore"]["hour"], a["mungesa_ore"]["pct"], marker="o", ms=3, color=COLORS[0])
    axes[1].set_title("% mungesë sipas orës lokale")
    for ax in axes:
        ax.set_ylabel("% e orëve")
    figs["kur"] = save_figure(fig, "p3_mungesa_muaj_ore")

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for ax, p in zip(axes.flat, NAMES, strict=True):
        for i, meth in enumerate(METHODS):
            s = ev.filter((pl.col("pollutant") == p) & (pl.col("method") == meth)).sort("L")
            ax.plot(s["L"], s["RMSE"], marker="o", ms=4, color=COLORS[i], label=meth)
        ax.set_title(p, fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(M["eval_gap_lengths_h"], [str(x) for x in M["eval_gap_lengths_h"]])
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("gjatësia e boshllëkut artificial (orë)")
    fig.supylabel("RMSE")
    fig.suptitle("Vlerësimi i metodave të imputimit (boshllëqe artificiale në 2024)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    figs["rmse"] = save_figure(fig, "p3_rmse_metodat")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = [c for c in ["deterministic", "linear", "profile", "station_median", "ffill", "left_null"]
            if c in methods.columns]
    left = np.zeros(methods.height)
    for i, c in enumerate(cols):
        ax.barh(methods["pollutant"].cast(pl.String), methods[c], left=left, color=COLORS[i], label=c)
        left += methods[c].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("orë")
    ax.set_title("Vlerat që mungonin: si u imputuan ose pse mbetën bosh")
    ax.legend(ncol=len(cols), loc="lower center", bbox_to_anchor=(0.5, -0.35), fontsize=8)
    figs["metodat"] = save_figure(fig, "p3_metodat_imputimit")
    return figs


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    t0 = time.perf_counter()
    clean = (scan("silver", "measurements_clean").with_columns(pl.col("pollutant").cast(pl.String).cast(POLLUTANT_TYPE))
             .select(*KEY, "value", "value_raw", "clean_action", "f_gas_period_flag").collect())
    df = add_gaps(build_grid(clean))
    a = analysis(df)

    train = df.filter((pl.col("ts_utc") <= TRAIN_END) & pl.col("value").is_not_null() & ~pl.col("f_gas_period_flag"))  # D35
    profile = train.group_by(*SERIES, "month", "hour").agg(pl.col("value").median().alias("profile"))
    station_median = train.group_by(SERIES).agg(pl.col("value").median().alias("station_median"))
    ev, ev_near = evaluate(df, profile, station_median)
    table(ev, "p3_vleresimi_metodave")
    ev_near = table(ev_near, "p3_vleresimi_konteksti")
    mid = (ev.filter(pl.col("L").is_between(M["interp_max_h"] + 1, M["mid_gap_max_h"]))
             .group_by("pollutant", "method").agg(pl.col("RMSE").mean())
             .sort("pollutant", "RMSE").group_by("pollutant", maintain_order=True).first())
    mid_method = dict(zip(mid["pollutant"].cast(pl.String), mid["method"], strict=True))
    table(mid, "p3_metoda_4_24h")
    det = table(deterministic_check(df), "p3_validimi_deterministik")
    near_wide = ev_near.pivot(sort_columns=True, on="method", index=["konteksti", "L"], values="RMSE").sort("konteksti", "L")
    lin = near_wide.pivot(sort_columns=True, on="konteksti", index="L", values="linear").sort("L")
    near_ratio = lin["pranë boshllëkut real (±24h)"] / lin["periudhë e qetë"]
    linear_always_best = near_wide.select(
        (pl.col("linear") <= pl.min_horizontal(pl.exclude("konteksti", "L"))).all()).item()
    ranking_text = ("**renditja e metodave s'ndryshon**: interpolimi linear mbetet më i miri në çdo gjatësi dhe në të dy "
                    "kontekstet. Strategjia është e qëndrueshme (D30)." if linear_always_best else
                    "**renditja ndryshon** në disa gjatësi pranë boshllëqeve reale; shih tabelën (D30).")
    cross = table(ev.filter(pl.col("method").is_in(["linear", "profile"]))
                  .pivot(sort_columns=True, on="method", index=["pollutant", "L"], values="RMSE")
                  .with_columns((100 * (pl.col("profile") / pl.col("linear") - 1)).round(1).alias("profili_vs_linear_pct"))
                  .filter(pl.col("profili_vs_linear_pct") < 0).sort("pollutant", "L"), "p3_profili_kalon_linearin")
    short = (ev.filter(pl.col("L") <= M["interp_max_h"]).group_by("pollutant", "L")
               .agg(pl.col("method").sort_by("RMSE").first().alias("me_e_mira")).sort("pollutant", "L"))

    out = impute(df, profile, station_median, mid_method)
    write(out.select(*KEY, "ts_local", "value_raw", "value", "clean_action", "status", "gap_id", "gap_len_h",
                     "gap_position", "value_imputed", "impute_method", "f_gas_period_flag"),
          "gold", "hourly_long", partition_by="pollutant")
    n_null_before = out.filter(~pl.col("status").is_in(STRUCTURAL))["value"].null_count()
    n_null_after = out.filter(~pl.col("status").is_in(STRUCTURAL))["value_imputed"].null_count()
    lineage("p3", "rrjeti orar + imputimi → gold", clean.height, out.height, n_null_before, n_null_after)

    # kontrollet: strukturoret kurrë; asnjë negative/mbi kufi; shpërndarja para/pas
    pmax = by_name("physical_max")
    bad = out.filter((pl.col("status").is_in(STRUCTURAL) & pl.col("value_imputed").is_not_null())
                     | (pl.col("value_imputed") < 0)
                     | (pl.col("value_imputed") > pl.col("pollutant").cast(pl.String).replace_strict(pmax,
                                                                                                  return_dtype=pl.Float64)))
    if bad.height:
        raise SystemExit(f"STOP: {bad.height} vlera të imputuara të pavlefshme")
    sens = []
    for p in NAMES:
        s = out.filter(pl.col("pollutant") == p)
        x, y = s["value"].drop_nulls(), s["value_imputed"].drop_nulls()
        sens.append({"ndotesi": p, "n_para": x.len(), "n_pas": y.len(), "mesatarja_para": x.mean(),
                     "mesatarja_pas": y.mean(), "std_para": x.std(), "std_pas": y.std(),
                     "ndryshimi_std_pct": 100 * (y.std() / x.std() - 1),
                     "KS": stats.ks_2samp(x.to_numpy(), y.to_numpy()).statistic})
    sens = table(pl.DataFrame(sens), "p3_para_pas_imputimit")
    methods = table(out.group_by("pollutant", "impute_method").len()
                    .pivot(on="impute_method", index="pollutant", values="len", sort_columns=True)
                    .fill_null(0).sort("pollutant"), "p3_metodat_sipas_ndotesit")
    measured = out.filter(~pl.col("status").is_in(STRUCTURAL))
    complete = measured.group_by("station_id", "ts_utc").agg(pl.col("value").is_not_null().all().alias("para"),
                                                             pl.col("value_imputed").is_not_null().all().alias("pas"))
    complete_pct = {k: round(100 * complete[k].mean(), 1) for k in ("para", "pas")}
    left = out.filter(pl.col("impute_method") == "left_null")
    period_2019 = (pl.col("clean_action") == "null_calibration") | (
        (pl.col("pollutant") == "NOx") & (pl.col("clean_action") == "null_inconsistent")
        & (pl.col("gap_len_h") > M["mid_gap_max_h"]))
    why_left = table(left.with_columns(
        pl.when(period_2019).then(pl.lit("periudha 2019 para ndërrimit të analizuesit (D10, D33)"))
          .when(pl.col("gap_position") == "edge").then(pl.lit("në skaj të serisë (pa vlerë në njërën anë)"))
          .when(pl.col("gap_len_h") > M["mid_gap_max_h"]).then(pl.lit(f"boshllëk > {M['mid_gap_max_h']}h"))
          .otherwise(pl.lit("tjetër")).alias("arsyeja"))
        .group_by("arsyeja").len().sort("len", descending=True), "p3_pse_mbeten_null")
    write_5a(out, clean)
    figs = figures(out, a, ev, methods)
    elapsed = time.perf_counter() - t0

    cv = a["chi2_cramers_v"]
    cv_nox = cv.filter(pl.col("ndotesi").is_in(["NO", "NO2", "NOx"]))
    cv_other = cv.filter(~pl.col("ndotesi").is_in(["NO", "NO2", "NOx"]))
    mv = a["mcar_1h_vlerat"]
    corr_min = a["korrelacioni_mungesave"].select(pl.exclude("ndotesi")).min().min_horizontal().item()

    ymin = a["mungesa_vit_pa_nox"].sort("pct").row(0)
    ymax = a["mungesa_vit_pa_nox"].sort("pct").row(-1)

    def rng(t: pl.DataFrame, col: str) -> str:
        return f"{t[col].min():.2f}–{t[col].max():.2f}"

    gl = a["boshlleqet"].filter(pl.col("gjatesia").is_in(["1–7 ditë", "> 7 ditë"]))
    long_gaps_pct, long_hours_pct = gl["pct_boshlleqe"].sum(), gl["pct_ore"].sum()
    nox_longest = a["boshlleqi_max_dite"].filter(pl.col("pollutant").is_in(["NO", "NO2", "NOx"]))["dite"].max()
    short_linear_share = "për të gjithë" if (short["me_e_mira"] == "linear").all() else "për shumicën"
    st = a["status_ndotes"]
    total = {c: int(st[c].sum()) for c in st.columns if c != "pollutant"}
    real_hours = sum(v for k, v in total.items() if k not in STRUCTURAL)
    real_missing = real_hours - total.get("ok", 0)
    write_section("p3", f"""
## 4. Vlerat e zbrazëta: identifikimi dhe strategjia e trajtimit

### 4.1 Identifikimi
Krijohet rrjeti i plotë orar: çdo stacion × çdo ndotës × çdo orë mbi periudhën e stacionit ({out.height:,} orë).
Kështu bëhen të dukshme orët që **mungojnë fare** (pa rresht), të cilat në të dhënat raw nuk shihen.
Çdo orë merr një `status`:
{md(st)}
**Mungesat strukturore** (vlera "nuk duhej të ekzistonte"): `not_measured` ({total.get('not_measured', 0):,}; p.sh.
XK0007A s'mat NO, NOx, O3) dhe `outside_period` ({total.get('outside_period', 0):,}; seria s'kishte nisur ose kishte
mbaruar). Nuk numërohen si mungesë dhe **nuk imputohen kurrë**.
**Mungesat reale:** {real_missing:,} nga {real_hours:,} orë të pritura ({100 * real_missing / real_hours:.1f}%), prej tyre
{total.get('missing_row', 0):,} pa rresht fare dhe pjesa tjetër null me arsye nga p2.

% e mungesës reale sipas stacionit dhe ndotësit:
{md(a['mungesa_reale_stacion'])}

### 4.2 Analiza e mungesave
Gjatësia e boshllëqeve (një boshllëk = orë të njëpasnjëshme pa vlerë):
{md(a['boshlleqet'])}
{a['boshlleqet']['pct_boshlleqe'][0]}% e boshllëqeve janë 1 orë, por boshllëqet më të gjata se 1 ditë
({long_gaps_pct:.1f}% e boshllëqeve) mbajnë {long_hours_pct:.1f}% të orëve që mungojnë.
Boshllëku më i gjatë për çdo ndotës:
{md(a['boshlleqi_max_dite'])}
Te NO, NO2 dhe NOx boshllëqet më të gjata (deri {nox_longest:.0f} ditë) janë periudha e 2019 para ndërrimit të
analizuesit (D10, D33);
te ndotësit e tjerë boshllëku më i gjatë është
{a['boshlleqi_max_dite'].filter(~pl.col('pollutant').is_in(['NO', 'NO2', 'NOx']))['dite'].max()} ditë
(ndërprerje e gjatë e stacionit).
{figs['harta']}
{figs['kur']}
% mungesë sipas vitit: {', '.join(f"{y}: {p}%" for y, p in a['mungesa_vit'].rows())}.

Korrelacioni i treguesve të mungesës (1 = mungojnë gjithmonë bashkë):
{md(a['korrelacioni_mungesave'])}
Testi chi-square dhe Cramér's V: {md(a['chi2_cramers_v'])}

Me ~600k orë për ndotës, chi-square del gjithmonë "domethënës" (p ≈ 0), prandaj vendimi bazohet te **Cramér's V**
(madhësia e efektit: < 0.1 e papërfillshme, 0.1–0.3 e dobët deri mesatare, > 0.3 e fortë).

**Testi i MCAR për boshllëqet 1-orëshe** ({a['mcar_1h']['boshlleqe_1h'][0]:,} boshllëqe): (a) a varen nga ora e ditës?
{md(a['mcar_1h'])}
(b) a ndodhin në kushte të veçanta ndotjeje? Vlera e orës para boshllëkut kundrejt mesatares së serisë:
{md(a['mcar_1h_vlerat'])}

**Mekanizmi (MCAR / MAR / MNAR), sipas evidencës (D18):**
- **MAR (mekanizmi kryesor).** Mungesa varet nga **viti**: V = {rng(cv_other, 'V_year')} për SO2, PM, O3, CO
  (mungesa vjetore nga {ymin[1]}% në {ymin[0]} deri {ymax[1]}% në {ymax[0]}: vite me ndërprerje të gjata), dhe
  V = {rng(cv_nox, 'V_year')} për NO, NO2, NOx (periudha 2019 para ndërrimit të analizuesit, D10/D33). Nga **muaji**
  varet dobët (V = {rng(cv, 'V_month')}), nga **ora** pothuajse aspak (V = {rng(cv, 'V_hour')}). Ndotësit e të
  njëjtit stacion mungojnë bashkë (korrelacion {corr_min:.2f}–1.00): ndalime të gjithë stacionit (rryma, komunikimi,
  mirëmbajtja). Mungesa varet nga koha dhe stacioni, që janë të njohura → imputimi që përdor kohën (interpolimi,
  profili) është i përshtatshëm.
- **Boshllëqet 1-orëshe: afërsisht MCAR, jo plotësisht.** Ora e ditës s'ka rëndësi praktike
  (V = {a['mcar_1h']['cramers_v_ora'][0]}), por ora para boshllëkut ka ndotje pak më të ulët se mesatarja
  ({mv['diferenca_ne_std'].abs().min():.2f}–{mv['diferenca_ne_std'].abs().max():.2f} std më e ulët): ndodhin pak më shpesh në kushte me ndotje të ulët. Efekti është i vogël dhe,
  për një orë të vetme, interpolimi mes dy fqinjëve të matur e kompenson.
- **MNAR:** shënimi Validity 2 ("nën pragun e detektimit") varet nga vetë vlera (vetëm vlerat shumë të vogla e marrin).
  Nëse këto vlera do të hiqeshin, mungesa do të ishte MNAR dhe mesatarja do të anohej lart; prandaj **u mbajtën** te p2.

### 4.3 Vlerësimi i metodave (evidenca për strategjinë)
Mbi segmente të plota të vitit 2024 fshihen artificialisht boshllëqe {M['eval_gap_lengths_h']} orë
({M['eval_gaps_per_series']} për seri, seed {M['seed']}); vlera e vërtetë dihet, ndaj gabimi matet. Profili dhe mediana
mësohen vetëm nga 2019–2023 (pa periudhën e dyshimtë të CO/SO2 para ndërrimit të analizuesve, D35), që vlerësimi
që bëhet në 2024 të jetë i ndershëm.
{figs['rmse']}
Metoda më e mirë për boshllëqet ≤ {M['interp_max_h']}h: {md(short.pivot(sort_columns=True, on='L', index='pollutant', values='me_e_mira'))}
Metoda e zgjedhur për boshllëqet {M['interp_max_h'] + 1}–{M['mid_gap_max_h']}h (RMSE mesatare më e ulët për 6, 12 dhe 24 orë):
{md(mid)}
Ku e kalon profili linearin (RMSE, ndryshimi në %):
{md(cross)}
Profili e kalon linearin vetëm në {', '.join(f'{x}h' for x in cross['L'].unique().sort())} dhe vetëm për
{', '.join(cross['pollutant'].unique().sort())}, sepse e ruan ciklin ditor. Strategjia mbetet linear për 4–24h (D16):
një prag i veçantë mbi një pikë të vetme vlerësimi do të ishte përshtatje e tepërt.

**Kontrolli i qëndrueshmërisë:** segmentet e plota zgjidhen kryesisht në periudha të qeta, ndërsa boshllëqet reale
ndodhin pranë prishjeve. Prandaj gabimi ndahet sipas kontekstit (RMSE, të gjithë ndotësit bashkë):
{md(near_wide)}
Pranë boshllëqeve reale gabimi i interpolimit linear është {near_ratio.min():.1f}–{near_ratio.max():.1f} herë më i madh
(vlerësimi në periudha të qeta është optimist), por {ranking_text}

**Validimi i metodës deterministike** (orët ku NOx, NO2 dhe NO janë të matur): {md(det)}
Gabimi mesatar ({det['MAE'][0]:.2f} µg/m³) është i papërfillshëm krahasuar me mesataren e NOx
({det['mesatarja_NOx'][0]:.1f} µg/m³), pra NOx = NO2 + 1.53·NO është e besueshme kur NO dhe NO2 janë të sakta.

### 4.4 Strategjia dhe imputimi
Rendi (vetëm mungesat reale; strukturoret kurrë):
1. **Deterministik:** NOx = NO2 + 1.53·NO në të njëjtën orë, kur NO2 dhe NO janë të matura dhe të vlefshme
   (relacion fizik, jo parashikim në kohë, ndaj vlen edhe në skaj të serisë, D15). Për periudhën para nëntorit 2019
   s'aplikohet, sepse aty NO dhe NO2 janë null (ndërrimi i analizuesit, D33).
2. **Boshllëk ≤ {M['interp_max_h']}h brenda serisë:** interpolim linear (metoda më e mirë në vlerësim
   {short_linear_share}: {short.filter(pl.col('me_e_mira') == 'linear').height}/{short.height} kombinime ndotës × gjatësi).
3. **Boshllëk {M['interp_max_h'] + 1}–{M['mid_gap_max_h']}h brenda serisë:** metoda fituese për ndotësin (tabela më sipër).
4. **Boshllëk > {M['mid_gap_max_h']}h ose në skaj të serisë:** mbetet null. Imputimi i ditëve të tëra do të shpikte të
   dhëna dhe do ta ulte variancën artificialisht.
{md(methods)}
{figs['metodat']}
Pse mbeten null: {md(why_left)}

### 4.5 Kontrollet
0 vlera të imputuara negative ose mbi kufirin fizik; 0 mungesa strukturore të imputuara. Shpërndarja para/pas:
{md(sens)}
Ndryshimi më i madh i std-së është {sens['ndryshimi_std_pct'].abs().max():.1f}% dhe statistika KS më e madhe
{sens['KS'].max():.3f}: imputimi s'e shtrembëron shpërndarjen.
**Rreshta të plotë** në format të gjerë (stacion × orë ku çdo ndotës që stacioni mat ka vlerë): {complete_pct['para']}%
→ {complete_pct['pas']}%. Përkufizimi këtu përfshin edhe orët pa rresht në rrjetin e plotë orar, prandaj ndryshon nga
64.2% i analizës eksploruese.
Koha e p3: {elapsed:.1f} s. Output: `data/gold/hourly_long/`.
""")
    print(f"p3 gati: {out.height:,} orë, mungesa reale {real_missing:,} → {n_null_after:,} null, {elapsed:.1f} s")


def write_5a(out: pl.DataFrame, clean: pl.DataFrame) -> None:
    """Tabela e vendimeve (seksioni 3) me numrat realë, te decisions.md."""
    a = clean["clean_action"].cast(pl.String).value_counts()
    act = dict(a.rows())
    st = dict(out["status"].value_counts().rows())
    p2 = pl.read_csv(REPORTS / "tables" / "p2_te_mbajtura.csv").row(0, named=True)
    rows = [
        ("Ndotës që stacioni s'e mat", "not_measured", st.get("not_measured", 0), "Kurrë"),
        ("Para fillimit / pas mbarimit të serisë", "outside_period", st.get("outside_period", 0), "Kurrë"),
        ("Orë që mungon fare", "missing_row", st.get("missing_row", 0), "Sipas gjatësisë"),
        ("E pavlefshme / mirëmbajtje (Validity −1/−99)", "null_invalid", act.get("null_invalid", 0), "Sipas gjatësisë"),
        ("Sentinel / absurde (të gjitha kanë edhe Validity −1/−99, D11)", "null_sentinel / null_absurd",
         act.get("null_sentinel", 0) + act.get("null_absurd", 0), "Sipas gjatësisë"),
        ("NO/NO2 para ndërrimit të analizuesit, nëntor 2019 (D33)", "null_calibration",
         act.get("null_calibration", 0), "Sipas gjatësisë (në praktikë > 24h → mbetet null)"),
        ("Mbi kufirin fizik", "null_physical", act.get("null_physical", 0), "Sipas gjatësisë"),
        ("Nën pragun e detektimit (Validity 2)", "mbahet", p2["below_dl"], "–"),
        ("Negative e vogël", "→ 0 (zero_negative)", act.get("zero_negative", 0), "–"),
        ("Negative e madhe", "null_negative", act.get("null_negative", 0), "Sipas gjatësisë"),
        (f"Vlerë e ngecur (jo-zero, ose zero te {', '.join(p for p in NAMES if p not in Q_STUCK_OK)})", "null_stuck",
         act.get("null_stuck", 0),
         "Sipas gjatësisë"),
        (f"Zero e ngecur te {'/'.join(Q_STUCK_OK)}", "mbahet", p2["stuck_zero_kept"], "–"),
        ("Majë e izoluar", "null_spike", act.get("null_spike", 0), "Sipas gjatësisë"),
        ("PM2.5 > PM10 brenda tolerancës", "mbahen të dyja", p2["pm_within_tol"], "–"),
        ("PM2.5 > PM10 jashtë tolerancës + NOx jokonsistent (D10)", "null_inconsistent",
         act.get("null_inconsistent", 0), "Sipas gjatësisë; NOx deterministikisht kur NO, NO2 janë të vlefshme"),
        ("Outlier statistikë", "mbahet, vetëm flamur", p2["outliers"], "–"),
    ]
    tab = table(pl.DataFrame(rows, schema=["Kategoria", "Veprimi", "Numri", "Imputohet?"], orient="row"),
                "p3_tabela_vendimeve_numrat")
    path = REPORTS / "decisions.md"
    text = path.read_text(encoding="utf-8").split("\n## Tabela e vendimeve")[0].rstrip()
    path.write_text(f"{text}\n\n## Tabela e vendimeve për vlerat null dhe të pavlefshme (numrat realë)\n{md(tab)}",
                    encoding="utf-8")


if __name__ == "__main__":
    main()
