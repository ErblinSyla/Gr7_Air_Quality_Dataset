"""Appendix – analyses beyond the lectures: MCAR check of 1-hour gaps, RMSE, Kolmogorov–Smirnov, Yeo-Johnson and
mutual information. They only check the main pipeline and change no data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PowerTransformer

from steps.p4_aggregation_sampling import draw, population
from utils import CFG, NAMES, SERIES, TABLES, scan, table

FIT_YEAR, SEED = CFG["time"]["fit_until_year"], CFG["features"]["seed"]
STRUCTURAL = ["not_measured", "outside_period"]
ALPHA = CFG["chi2"]["alpha"]


def mcar_1h() -> tuple[pl.DataFrame, pl.DataFrame]:
    h = (scan("gold", "hourly_long").filter(~pl.col("status").is_in(STRUCTURAL))
         .select(*SERIES, "ts_utc", pl.col("ts_local").dt.hour().alias("hour"), "value", "gap_len_h").collect()
         .sort(*SERIES, "ts_utc"))
    gap = pl.col("value").is_null() & (pl.col("gap_len_h") == 1)
    counts = h.group_by("hour").agg(gap.sum().alias("observed"), pl.len().alias("hours")).sort("hour")
    n = counts["observed"].sum()
    expected = counts["hours"] / counts["hours"].sum() * n
    chi2 = float(((counts["observed"] - expected) ** 2 / expected).sum())
    k = counts.height
    test = pl.DataFrame([{"gaps_1h": n, "chi2_hour": round(chi2, 1), "df": k - 1,
                          "critical_value": round(float(stats.chi2.ppf(1 - ALPHA, k - 1)), 1),
                          "cramers_v": round(float(np.sqrt(chi2 / (n * (k - 1)))), 3)}])
    before = (h.with_columns(gap.shift(-1).over(SERIES).fill_null(False).alias("_next_gap"))
               .group_by("pollutant")
               .agg(pl.col("value").mean().alias("series_mean"),
                    pl.col("value").filter("_next_gap").mean().alias("mean_hour_before_gap"))
               .with_columns((100 * (pl.col("mean_hour_before_gap") / pl.col("series_mean") - 1)).round(1)
                             .alias("difference_pct"))
               .with_columns(pl.col("pollutant").cast(pl.String)).sort("pollutant"))
    return table(test, "x_mcar_1h_hour"), table(before, "x_mcar_1h_values")


def rmse_summary() -> tuple[pl.DataFrame, int, int]:
    ev = pl.read_csv(TABLES / "p3_method_evaluation.csv")
    wide = (ev.group_by("gap_h", "method").agg(pl.col("RMSE").mean())
              .pivot(on="method", index="gap_h", values="RMSE", sort_columns=True)
              .select("gap_h", "attribute_mean", "class_mean", "linear").sort("gap_h"))
    rank = lambda m: ev.sort(m).group_by("gap_h", "pollutant", maintain_order=True).first()  # noqa: E731
    same = rank("MAE").join(rank("RMSE"), on=["gap_h", "pollutant"]).filter(pl.col("method") == pl.col("method_right"))
    return table(wide, "x_rmse_summary"), same.height, ev.select("gap_h", "pollutant").n_unique()


def ks_tests() -> tuple[pl.DataFrame, pl.DataFrame]:
    h = (scan("gold", "hourly_long").filter(~pl.col("status").is_in(STRUCTURAL))
         .select(pl.col("pollutant").cast(pl.String), "value", "value_imputed").collect())
    imputed = []
    for p in NAMES:
        s = h.filter(pl.col("pollutant") == p)
        x, y = s["value"].drop_nulls().to_numpy(), s["value_imputed"].drop_nulls().to_numpy()
        imputed.append({"pollutant": p, "ks_statistic": round(float(stats.ks_2samp(x, y).statistic), 4)})
    pop = population()
    sampling = []
    for method in ("SRSWOR", "SRSWR", "stratified"):
        s = draw(pop, method)
        for p in CFG["sampling"]["compare_pollutants"]:
            res = stats.ks_2samp(pop[p].drop_nulls().to_numpy(), s[p].drop_nulls().to_numpy())
            sampling.append({"method": method, "pollutant": p, "ks_statistic": round(float(res.statistic), 4),
                             "p_value": round(float(res.pvalue), 3)})
    ks_imp = table(pl.DataFrame(imputed).sort("pollutant"), "x_ks_imputation")
    return ks_imp, table(pl.DataFrame(sampling), "x_ks_sampling")


def yeo_johnson(f: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for p in NAMES:
        x = f.filter(pl.col("year") <= FIT_YEAR)[p].drop_nulls().to_numpy().reshape(-1, 1)
        pt = PowerTransformer(method="yeo-johnson").fit(x)
        rows.append({"pollutant": p, "lambda": round(float(pt.lambdas_[0]), 3),
                     "skew_original": round(float(stats.skew(x.ravel())), 2),
                     "skew_log1p": round(float(stats.skew(np.log1p(x.ravel()))), 2),
                     "skew_yeo_johnson": round(float(stats.skew(pt.transform(x).ravel())), 2)})
    return table(pl.DataFrame(rows), "x_yeo_johnson")


def mutual_information(f: pl.DataFrame) -> pl.DataFrame:
    summary = pl.read_csv(TABLES / "p5_feature_selection.csv")
    kept = summary["feature"].to_list()
    data = (f.filter(pl.col("year") <= FIT_YEAR)
              .with_columns(pl.col("PM2_5_obs").shift(-1).over("station_id").alias("target"))
              .select(*[pl.col(c).cast(pl.Float64) for c in kept], "target").drop_nulls())
    data = data.sample(min(CFG["features"]["selection_sample"], data.height), seed=SEED)
    mi = mutual_info_regression(data.select(kept).to_numpy(), data["target"].to_numpy(), random_state=SEED)
    return table(pl.DataFrame({"feature": kept, "mutual_information": np.round(mi, 3)})
                 .join(summary.select("feature", "selected_by"), on="feature")
                 .sort("mutual_information", descending=True), "x_mutual_information")


def run() -> SimpleNamespace:
    f = scan("gold", "features_hourly").collect()
    mcar, before = mcar_1h()
    ks_imp, ks_smp = ks_tests()
    rmse, same, total = rmse_summary()
    pm25 = f.filter(pl.col("year") <= FIT_YEAR)["PM2_5"].drop_nulls().to_numpy().reshape(-1, 1)
    pm25_yj = PowerTransformer(method="yeo-johnson").fit_transform(pm25).ravel()
    return SimpleNamespace(mcar=mcar, before=before, rmse=rmse, same=same, total=total, ks_imp=ks_imp, ks_smp=ks_smp,
                           yj=yeo_johnson(f), mi=mutual_information(f), pm25=pm25.ravel(), pm25_yj=pm25_yj)
