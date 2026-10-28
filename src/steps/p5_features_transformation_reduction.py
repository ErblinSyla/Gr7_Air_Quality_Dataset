"""p5 – Feature subset selection, feature creation, discretization and binarization, transformation (normalization)
and dimensionality reduction (PCA). Output: data/gold/features_hourly.parquet."""

from __future__ import annotations

from math import pi
from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from utils import CFG, NAMES, SEASONS, lineage, scan, table, write

F, R, E, B, T = CFG["features"], CFG["reduction"], CFG["eaqi"], CFG["binarization"], CFG["time"]
ALPHA = CFG["chi2"]["alpha"]
PCA_COLS = [p for p in NAMES if p != "NOx"]
FIT_YEAR = T["fit_until_year"]
FIT = pl.col("year") <= FIT_YEAR

DROPPED = [
    ("DataCapture", "irrelevant: 100% empty"),
    ("AggType", "irrelevant: constant (always 'hour'), zero variance"),
    ("Verification", "irrelevant: constant (always 1), zero variance"),
    ("End", "redundant: End = Start + 1h (0 exceptions)"),
    ("ResultTime", "irrelevant: publication time, unrelated to pollution"),
    ("FkObservationLog", "irrelevant: administrative identifier (like a student ID for predicting GPA)"),
    ("Unit", "redundant after harmonisation (all values in canonical units)"),
    ("Samplingpoint", "redundant: split into station_id and pollutant"),
    ("Validity", "redundant: absorbed into clean_action/status (p2–p3)"),
]

NEW_FEATURES = [
    ("hour, dow, month, year", "extracted from local time",
     "pollution follows daily (traffic), weekly and seasonal rhythms"),
    ("is_weekend", "day of week ≥ 6 (like \"weekend?\" from a timestamp in the lecture)", "less traffic at weekends"),
    ("season, is_heating_season", "month → season; October–April", "wood/coal heating: much higher PM2.5"),
    ("hour_sin, hour_cos, month_sin, month_cos", "sin/cos(2π·x/period)",
     "hour 23 and hour 0 are neighbours; the numbers 23 and 0 do not show this"),
    ("<P>_is_imputed", "the hour's value was imputed in p3",
     "transparency: analyses can treat these values differently"),
    ("gas_period_2019", "the hour falls before the gas analyzer change (November 2019)",
     "suspicious CO/SO2 (p2): analyses can exclude this period"),
    ("pm25_pm10_ratio", "measured PM2.5 / PM10 (when PM10 > 0 and PM2.5 ≤ PM10)",
     "attribute construction (like BMI): high = combustion (fine particles); low = dust"),
    ("pm_coarse", "measured PM10 − PM2.5", "coarse particles (dust, roads)"),
    ("<P>_lag_1h, <P>_lag_24h", "measured value 1h / 24h earlier, within the station", "pollution is persistent"),
    ("<P>_roll_mean_24h", "mean of the previous 24h (without the current hour)", "recent pollution level"),
    ("station_type, altitude_m", "from the metadata (p2)", "type of source (traffic/industry/background) and altitude"),
    ("pm25_bin_*, pm10_exceed_eu, pm25_exceed_who, one-hot", "discretization and binarization (6.4–6.5)",
     "classes and 0/1 indicators"),
    ("<P>_minmax, <P>_z, <P>_dec", "normalization (6.6)", "every pollutant on a comparable scale"),
]


def minmax(x, lo: float, hi: float, new_lo: float = 0.0, new_hi: float = 1.0):
    return (x - lo) / (hi - lo) * (new_hi - new_lo) + new_lo


def zscore(x, mean: float, std: float):
    return (x - mean) / std


def decimal_j(max_abs: float) -> int:
    j = 0
    while max_abs / 10 ** j >= 1:
        j += 1
    return j


def smooth_by_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    parts = np.array_split(np.sort(values), n_bins)
    means = np.concatenate([np.full(len(b), b.mean()) for b in parts])
    bounds = np.concatenate([np.where(b - b[0] <= b[-1] - b, b[0], b[-1]) for b in parts])
    return means, bounds, parts


def self_test() -> None:
    assert round(minmax(73600, 12000, 98000), 3) == 0.716
    assert round(zscore(73600, 54000, 16000), 3) == 1.225
    assert decimal_j(986) == 3 and (-986 / 10 ** 3, 917 / 10 ** 3) == (-0.986, 0.917)
    means, bounds, _ = smooth_by_bins(np.array([24, 9, 28, 4, 21, 34, 15, 26, 8, 29, 25, 21], float), 3)
    assert np.round(means).tolist() == [9] * 4 + [23] * 4 + [29] * 4
    assert bounds.tolist() == [4, 4, 4, 15, 21, 21, 25, 25, 26, 26, 26, 34]


def wide() -> pl.DataFrame:
    long = (scan("gold", "hourly_long")
            .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_utc", "ts_local",
                    "value", "value_imputed", "f_gas_period_flag").collect())
    flagged = (long.filter("f_gas_period_flag").select("station_id", "ts_utc").unique()
                   .with_columns(pl.lit(True).alias("gas_period_2019")))
    w = long.pivot(sort_columns=True, on="pollutant", index=["station_id", "ts_utc", "ts_local"],
                   values=["value_imputed", "value"])
    w = w.rename({**{f"value_imputed_{p}": p for p in NAMES}, **{f"value_{p}": f"{p}_obs" for p in NAMES}})
    return (w.join(flagged, on=["station_id", "ts_utc"], how="left")
             .with_columns(pl.col("gas_period_2019").fill_null(False)).sort("station_id", "ts_utc"))


def create_features(w: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    loc = pl.col("ts_local")
    cyc = lambda col, period: [(2 * pi * pl.col(col) / period).sin().alias(f"{col}_sin"),  # noqa: E731
                               (2 * pi * pl.col(col) / period).cos().alias(f"{col}_cos")]
    lags = []
    for p in F["lag_pollutants"]:
        x = pl.col(f"{p}_obs")
        lags += [x.shift(k).over("station_id").alias(f"{p}_lag_{k}h") for k in F["lags_h"]]
        lags.append(x.rolling_mean_by("ts_utc", f"{F['rolling_h']}h", closed="left").over("station_id")
                    .alias(f"{p}_roll_mean_{F['rolling_h']}h"))
    pm25, pm10 = pl.col("PM2_5_obs"), pl.col("PM10_obs")
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
             .sort("station_id", "ts_utc"))


def check_lags(f: pl.DataFrame) -> None:
    prev = f.select("station_id", (pl.col("ts_utc") + pl.duration(hours=1, time_unit="ns")).alias("ts_utc"),
                    pl.col("PM2_5_obs").alias("_prev"))
    wrong = (f.select("station_id", "ts_utc", "PM2_5_lag_1h").join(prev, on=["station_id", "ts_utc"], how="left")
              .filter(~pl.col("PM2_5_lag_1h").eq_missing(pl.col("_prev"))).height)
    if wrong or not f.select("station_id", "ts_utc").is_unique().all():
        raise SystemExit(f"STOP: {wrong} wrong lags or non-unique key in features_hourly")


def redundancy(f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    obs = [f"{p}_obs" for p in NAMES]
    rows, rho = [], np.eye(len(NAMES))
    for i, a in enumerate(obs):
        for j in range(i + 1, len(obs)):
            b = obs[j]
            x = f.select(a, b).drop_nulls()
            cov = float(np.cov(x[a].to_numpy(), x[b].to_numpy(), ddof=0)[0, 1])
            r = cov / (x[a].std(ddof=0) * x[b].std(ddof=0))
            rho[i, j] = rho[j, i] = r
            rows.append({"a": NAMES[i], "b": NAMES[j], "hours": x.height, "covariance": cov, "correlation": r})
    pairs = pl.DataFrame(rows)
    corr = pl.DataFrame({"pollutant": NAMES, **{p: np.round(rho[:, i], 2) for i, p in enumerate(NAMES)}})
    return pairs, pairs.filter(pl.col("correlation").abs() > R["corr_threshold"]), corr


def feature_selection(f: pl.DataFrame) -> SimpleNamespace:
    seed = F["seed"]
    skip = {"year", "target", "gas_period_2019", *NAMES} | {c for c in f.columns if c.endswith("_is_imputed")}
    candidates = [c for c, d in f.schema.items() if (d.is_numeric() or d == pl.Boolean) and c not in skip]
    variance = (f.select([pl.col(c).cast(pl.Float64).var().alias(c) for c in candidates])
                 .transpose(include_header=True, header_name="feature", column_names=["variance"])
                 .with_columns((pl.col("variance") > 0).fill_null(False).alias("passes_variance")))
    kept = variance.filter("passes_variance")["feature"].to_list()

    data = (f.with_columns(pl.col("PM2_5_obs").shift(-1).over("station_id").alias("target"))
              .select(*[pl.col(c).cast(pl.Float64) for c in kept], "target", "station_id", "year").drop_nulls())
    fit_pool, test_pool = data.filter(FIT), data.filter(~FIT)
    n = min(F["selection_sample"], fit_pool.height, test_pool.height)
    fit_rows, test_rows = fit_pool.sample(n, seed=seed), test_pool.sample(n, seed=seed)
    x_fit, y_fit = fit_rows.select(kept).to_numpy(), fit_rows["target"].to_numpy()
    x_test, y_test = test_rows.select(kept).to_numpy(), test_rows["target"].to_numpy()
    col = {c: i for i, c in enumerate(kept)}

    def r2(features: list[str]) -> float:
        if not features:
            return 0.0
        idx = [col[c] for c in features]
        return float(LinearRegression().fit(x_fit[:, idx], y_fit).score(x_test[:, idx], y_test))

    gain = F["selection_min_gain"]
    forward, chosen, current = [], [], 0.0
    while len(chosen) < len(kept):
        best = max((c for c in kept if c not in chosen), key=lambda c: r2([*chosen, c]))
        score = r2([*chosen, best])
        if score - current < gain:
            break
        chosen.append(best)
        forward.append({"step": len(chosen), "feature_added": best, "R2": score, "gain": score - current})
        current = score

    backward, remaining = [], list(kept)
    current = r2(remaining)
    backward.append({"step": 0, "feature_removed": "–", "R2": current, "loss": 0.0})
    while len(remaining) > 1:
        worst = max(remaining, key=lambda c: r2([x for x in remaining if x != c]))
        score = r2([x for x in remaining if x != worst])
        if current - score >= gain:
            break
        remaining.remove(worst)
        backward.append({"step": len(backward), "feature_removed": worst, "R2": score, "loss": current - score})
        current = score

    tree = DecisionTreeRegressor(max_depth=F["tree_max_depth"], random_state=seed).fit(x_fit, y_fit)
    importance = dict(zip(kept, tree.feature_importances_, strict=True))
    summary = (pl.DataFrame({"feature": kept})
                 .with_columns(pl.col("feature").is_in(chosen).alias("forward"),
                               pl.col("feature").is_in(remaining).alias("backward"),
                               pl.col("feature").replace_strict(importance, return_dtype=pl.Float64)
                               .alias("tree_importance"))
                 .with_columns((pl.col("tree_importance") > 0).alias("in_tree"))
                 .with_columns((pl.col("forward").cast(pl.Int8) + pl.col("backward").cast(pl.Int8)
                                + pl.col("in_tree").cast(pl.Int8)).alias("selected_by"))
                 .sort("selected_by", "tree_importance", descending=True))
    return SimpleNamespace(
        variance=table(variance, "p5_feature_variance"), forward=table(pl.DataFrame(forward), "p5_forward_selection"),
        backward=table(pl.DataFrame(backward), "p5_backward_elimination"),
        summary=table(summary, "p5_feature_selection"), n_candidates=len(candidates), n_kept=len(kept),
        n_rows=data.height, n_sample=n, stations=data["station_id"].n_unique(), r2_all=r2(kept),
        r2_forward=r2(chosen), r2_backward=r2(remaining), chosen=chosen, remaining=remaining,
    )


def bins(col: str, edges: list[float], alias: str) -> pl.Expr:
    x = pl.col(col)
    return pl.when(x.is_not_null()).then(pl.sum_horizontal([(x > e).cast(pl.Int8) for e in edges])).alias(alias)


def discretize(f: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    x = f.filter(FIT)["PM2_5"].drop_nulls().to_numpy()
    n = F["n_bins"]
    edges = {"equal_width": list(np.linspace(x.min(), x.max(), n + 1)[1:-1]),
             "equal_frequency": list(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1])),
             "eaqi": E["bins"]["PM2_5"]}
    f = f.with_columns(*[bins("PM2_5", e, f"pm25_bin_{k}") for k, e in edges.items()])
    return f, {k: [round(float(v), 2) for v in e] for k, e in edges.items()}


def smoothing_table(f: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    x = f["PM2_5"].drop_nulls().to_numpy()
    means, bounds, parts = smooth_by_bins(x, F["n_bins"])
    tab = pl.DataFrame([{"bin": i + 1, "values": len(b), "min": float(b[0]), "max": float(b[-1]),
                         "bin_mean": float(b.mean())} for i, b in enumerate(parts)])
    xs = np.sort(x)
    info = {"std": float(xs.std()), "std_means": float(means.std()), "std_bounds": float(bounds.std()),
            "mae_means": float(np.abs(xs - means).mean()), "mae_bounds": float(np.abs(xs - bounds).mean())}
    return table(tab, "p5_bin_smoothing"), info


def daily_exceedance(f: pl.DataFrame) -> pl.DataFrame:
    day = (scan("gold", "daily").filter((pl.col("source") == "measured") & pl.col("valid_day")
                                        & pl.col("pollutant").cast(pl.String).is_in(["PM10", "PM2_5"])).collect()
           .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "date", "mean")
           .pivot(sort_columns=True, on="pollutant", index=["station_id", "date"], values="mean")
           .select("station_id", "date",
                   (pl.col("PM10") > B["pm10_daily_limit"]).cast(pl.Int8).alias("pm10_exceed_eu"),
                   (pl.col("PM2_5") > B["pm25_daily_who"]).cast(pl.Int8).alias("pm25_exceed_who")))
    return (f.with_columns(pl.col("ts_local").dt.date().alias("date")).join(day, on=["station_id", "date"], how="left")
             .drop("date").sort("station_id", "ts_utc"))


def chi2_heating(f: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    d = (f.with_columns(pl.col("ts_local").dt.date().alias("d")).unique(["station_id", "d"])
          .drop_nulls("pm25_exceed_who").select("is_heating_season", "pm25_exceed_who"))
    obs = np.array([[d.filter((pl.col("is_heating_season") == h) & (pl.col("pm25_exceed_who") == e)).height
                     for e in (1, 0)] for h in (True, False)], dtype=float)
    exp = obs.sum(axis=1, keepdims=True) * obs.sum(axis=0, keepdims=True) / obs.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    critical = float(stats.chi2.ppf(1 - ALPHA, 1))
    labels = ["heating season", "rest of the year"]
    tab = pl.DataFrame([{"season": labels[i], "daily PM2.5 > 15": f"{obs[i, 0]:,.0f} ({exp[i, 0]:,.1f})",
                         "daily PM2.5 ≤ 15": f"{obs[i, 1]:,.0f} ({exp[i, 1]:,.1f})", "sum": f"{obs[i].sum():,.0f}"}
                        for i in range(2)]
                       + [{"season": "sum", "daily PM2.5 > 15": f"{obs[:, 0].sum():,.0f}",
                           "daily PM2.5 ≤ 15": f"{obs[:, 1].sum():,.0f}", "sum": f"{obs.sum():,.0f}"}])
    return table(tab, "p5_chi2_heating_exceedance"), {"chi2": chi2, "critical": critical,
                                                      "share": obs[:, 0] / obs.sum(axis=1)}


def normalize(f: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, cols = [], []
    for p in NAMES:
        x = f.filter(FIT)[p].drop_nulls()
        lo, hi, mean, std = x.min(), x.max(), x.mean(), x.std()
        j = decimal_j(x.abs().max())
        v = float(x.quantile(0.95))
        rows.append({"pollutant": p, "min": lo, "max": hi, "mean": mean, "std": std, "j": j, "example_v_P95": v,
                     "v_minmax": minmax(v, lo, hi), "v_zscore": zscore(v, mean, std), "v_decimal": v / 10 ** j})
        cols += [minmax(pl.col(p), lo, hi).alias(f"{p}_minmax"), zscore(pl.col(p), mean, std).alias(f"{p}_z"),
                 (pl.col(p) / 10 ** j).alias(f"{p}_dec")]
    f = f.with_columns(cols)
    fit = f.filter(FIT)
    for p in NAMES:
        mm, z, dec = fit[f"{p}_minmax"].drop_nulls(), fit[f"{p}_z"].drop_nulls(), fit[f"{p}_dec"].drop_nulls()
        if not (abs(mm.min()) < 1e-9 and abs(mm.max() - 1) < 1e-9 and abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9
                and dec.abs().max() < 1):
            raise SystemExit(f"STOP: normalization of {p} does not meet its conditions")
    return f, table(pl.DataFrame(rows), "p5_normalization")


def pca(f: pl.DataFrame, sample: pl.DataFrame) -> SimpleNamespace:
    complete = f.drop_nulls(PCA_COLS)
    fit = complete.filter(~pl.col("gas_period_2019") & FIT).select(PCA_COLS).to_numpy()
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    model = PCA().fit((fit - mean) / std)
    ev = model.explained_variance_ratio_
    s = sample.drop_nulls(PCA_COLS).with_columns(pl.col("ts_local").dt.month().is_in(T["heating_months"]).alias("heat"))
    loadings = table(pl.DataFrame({"pollutant": PCA_COLS, **{f"PC{i + 1}": np.round(model.components_[i], 3)
                                                             for i in range(3)}}), "p5_pca_loadings")
    explained = table(pl.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ev))], "variance": np.round(ev, 3),
                                    "cumulative": np.round(np.cumsum(ev), 3)}), "p5_pca_variance")
    return SimpleNamespace(
        model=model, loadings=loadings, explained=explained, ev=ev, n_fit=len(fit), n_complete=complete.height,
        stations=complete["station_id"].n_unique(),
        n_90=int(np.searchsorted(np.cumsum(ev), R["pca_variance"]) + 1),
        scores=model.transform((s.select(PCA_COLS).to_numpy() - mean) / std)[:, :2], heat=s["heat"].to_numpy(),
    )


def run() -> SimpleNamespace:
    self_test()
    stations = scan("silver", "stations").select("station_id", "station_type", "altitude_m").collect()
    w = wide()
    f = create_features(w, stations)
    check_lags(f)

    pairs_all, pairs, corr = redundancy(f)
    table(pairs_all, "p5_covariance_correlation")
    table(pairs, "p5_high_correlation_pairs")
    table(corr, "p5_pearson_correlation")
    sel = feature_selection(f)
    f, edges = discretize(f)
    smooth, smooth_info = smoothing_table(f)
    f = daily_exceedance(f)
    chi_tab, chi = chi2_heating(f)
    onehot = f.select("station_id", "season", "station_type").to_dummies()
    f = pl.concat([f, onehot], how="horizontal")
    f, norm = normalize(f)
    sample = scan("gold", "sample_10pct").select("station_id", "ts_utc", "ts_local", *NAMES).collect()
    p = pca(f, sample)
    write(f, "gold", "features_hourly")
    lineage("p5", "wide format + features → gold", w.height, f.height, 0, 0)

    nox_ok = pl.col("NOx_obs").is_not_null() & pl.col("NO2_obs").is_not_null() & pl.col("NO_obs").is_not_null()
    return SimpleNamespace(
        f=f, n_wide=w.height, pairs=pairs, corr=corr, sel=sel, smooth=smooth, smooth_info=smooth_info, chi_tab=chi_tab,
        chi=chi, onehot_width=onehot.width, norm=norm, p=p,
        dropped=table(pl.DataFrame(DROPPED, schema=["column", "reason"], orient="row"), "p5_dropped_columns"),
        created=table(pl.DataFrame(NEW_FEATURES, schema=["feature", "formula", "reason"], orient="row"),
                      "p5_new_features"),
        edges=table(pl.DataFrame([{"method": k, "edges": ", ".join(map(str, v))} for k, v in edges.items()]),
                    "p5_discretization_edges"),
        eaqi=table(f["pm25_bin_eaqi"].drop_nulls().value_counts().sort("pm25_bin_eaqi")
                   .with_columns(pl.col("pm25_bin_eaqi").replace_strict(dict(enumerate(E["labels"])),
                                                                       return_dtype=pl.String).alias("class"),
                                 (100 * pl.col("count") / pl.col("count").sum()).round(1).alias("pct")),
                   "p5_eaqi_pm25"),
        exceed=table(f.with_columns(pl.col("ts_local").dt.date().alias("d"))
                     .unique(["station_id", "d"]).group_by("station_id", "year")
                     .agg(pl.col("pm10_exceed_eu").sum().alias("days_PM10_>50"),
                          pl.col("pm25_exceed_who").sum().alias("days_PM2.5_>15"))
                     .sort("station_id", "year"), "p5_exceedances"),
        nox_r=f.filter(nox_ok).select(pl.corr("NOx_obs", pl.col("NO2_obs") + CFG["quality"]["no_to_no2"]
                                              * pl.col("NO_obs"))).item(),
        first_bin=100 * (f["pm25_bin_equal_width"] == 0).sum() / f["pm25_bin_equal_width"].count(),
    )
