"""Integration tests: every assignment requirement is checked on the pipeline outputs
(run `python run_all.py` first)."""

import re

import numpy as np
import polars as pl
import pytest

from utils import CFG, DATA, NAMES, REPORTS, ROOT, TABLES, by_name, scan

pytestmark = pytest.mark.skipif(not (DATA / "gold" / "features_hourly.parquet").exists(),
                                reason="pipeline outputs not found: run `python run_all.py` first")
STRUCTURAL = pl.col("status").is_in(["not_measured", "outside_period"])
FIT = pl.col("year") <= CFG["time"]["fit_until_year"]


@pytest.fixture(scope="module")
def bronze():
    return scan("bronze", "measurements").collect()


@pytest.fixture(scope="module")
def silver():
    return scan("silver", "measurements_clean").collect()


@pytest.fixture(scope="module")
def hourly():
    return scan("gold", "hourly_long").collect()


@pytest.fixture(scope="module")
def features():
    return scan("gold", "features_hourly").collect()


@pytest.fixture(scope="module")
def daily():
    return scan("gold", "daily").collect().filter(pl.col("source") == "measured")


def test_collection_row_counts(bronze):
    raw = pl.scan_parquet(sorted((ROOT / CFG["paths"]["raw"]).glob("*.parquet"))).select(pl.len()).collect().item()
    assert raw == bronze.height == CFG["expected"]["rows"]
    assert bronze["station_id"].n_unique() == CFG["expected"]["stations"]
    assert bronze["pollutant"].n_unique() == len(NAMES)


def test_data_types(bronze):
    assert bronze.schema["value_raw"] == pl.Float64
    assert bronze.schema["ts_utc"] == pl.Datetime("ns", "UTC")
    dictionary = pl.read_csv(TABLES / "p1_data_dictionary.csv")
    assert dictionary.height == 12
    assert dictionary["attribute_type"].null_count() == dictionary["discrete_continuous"].null_count() == 0


def test_data_quality(bronze):
    assert bronze.select("station_id", "pollutant", "ts_utc").is_unique().all()
    assert pl.read_csv(TABLES / "p1_quality_dimensions.csv").height == 6


def test_integration():
    stations = scan("silver", "stations").collect()
    assert stations.height == CFG["expected"]["stations"] and stations["station_id"].is_unique().all()
    assert stations["distance_km"].max() <= CFG["stations"]["max_distance_km"]
    check = pl.read_csv(TABLES / "p2_nox_ratio_after_conversion.csv")
    assert check["ok"].all()


def test_cleaning(bronze, silver):
    pmax = pl.col("pollutant").cast(pl.String).replace_strict(by_name("physical_max"), return_dtype=pl.Float64)
    assert silver.height == bronze.height
    assert silver.filter(pl.col("value") < 0).height == 0
    assert silver.filter(pl.col("value") > pmax).height == 0
    assert silver.filter(pl.col("value").is_in([float(s) for s in CFG["quality"]["sentinels"]])).height == 0
    assert silver.filter(pl.col("value").is_null() & (pl.col("clean_action") == "keep")).height == 0


def test_missing_values_identification(hourly):
    assert hourly.filter(STRUCTURAL & pl.col("value_imputed").is_not_null()).height == 0
    missing = hourly.filter(~STRUCTURAL & pl.col("value").is_null())
    assert missing.filter(pl.col("impute_method") == "none").height == 0
    assert missing["gap_len_h"].null_count() == 0


def test_missing_values_strategy(hourly):
    max_gap = CFG["missing"]["max_gap_h"]
    assert hourly.filter(pl.col("value").is_not_null() & (pl.col("value_imputed") != pl.col("value"))).height == 0
    assert hourly.filter(pl.col("value_imputed") < 0).height == 0
    linear = hourly.filter(pl.col("impute_method") == "linear")
    assert linear.filter((pl.col("gap_len_h") > max_gap) | (pl.col("gap_position") != "inside")).height == 0
    deterministic = hourly.filter(pl.col("impute_method") == "deterministic")
    assert (deterministic["pollutant"].cast(pl.String) == "NOx").all()
    left = hourly.filter(pl.col("impute_method") == "left_null")
    assert left["value_imputed"].null_count() == left.height
    evaluation = pl.read_csv(TABLES / "p3_method_evaluation_summary.csv")
    assert (evaluation["linear"] < evaluation["class_mean"]).all()
    assert (evaluation["class_mean"] < evaluation["attribute_mean"]).all()
    context = pl.read_csv(TABLES / "p3_method_evaluation_by_context.csv")
    assert set(context["context"]) == {"near a real gap", "quiet period"}


def test_aggregation(daily):
    monthly = scan("gold", "monthly").collect().filter(pl.col("source") == "measured")
    annual = scan("gold", "annual").collect().filter(pl.col("source") == "measured")
    agg = CFG["aggregation"]
    assert daily.filter(pl.col("mean").is_not_null() & (pl.col("valid_hours") < agg["min_hours_per_day"])).height == 0
    too_few_days = pl.col("valid_days") < agg["min_days_per_month_frac"] * pl.col("days_in_month")
    assert monthly.filter(pl.col("mean").is_not_null() & too_few_days).height == 0
    assert annual.filter(pl.col("mean_official").is_not_null()
                         & (pl.col("coverage") < agg["min_annual_coverage"])).height == 0
    assert set(scan("gold", "rollup_annual").collect()["level"]) == {"station", "zone", "Kosovo"}


def test_sampling(features):
    sample = scan("gold", "sample_10pct").collect()
    strata = (features.group_by("station_id", "season").len()
              .join(sample.group_by("station_id", "season").len(), on=["station_id", "season"], suffix="_s")
              .with_columns((pl.col("len_s") / pl.col("len")).alias("share")))
    assert strata.height == features.select("station_id", "season").n_unique()
    assert strata["share"].is_between(0.099, 0.101).all()
    assert sample.select("station_id", "ts_utc").is_unique().all()


def test_feature_subset_selection(features):
    dropped = {"data_capture", "agg_type", "verification", "fk_observation_log", "unit", "result_time", "validity"}
    assert not dropped & set(features.columns)
    summary = pl.read_csv(TABLES / "p5_feature_selection.csv")
    assert {"forward", "backward", "in_tree"} <= set(summary.columns) and summary["forward"].any()


def test_feature_creation(features):
    assert features.select("station_id", "ts_utc").is_unique().all()
    prev = features.select("station_id", (pl.col("ts_utc") + pl.duration(hours=1, time_unit="ns")).alias("ts_utc"),
                           pl.col("PM2_5_obs").alias("prev"))
    lag = features.select("station_id", "ts_utc", "PM2_5_lag_1h").join(prev, on=["station_id", "ts_utc"], how="left")
    assert lag.filter(~pl.col("PM2_5_lag_1h").eq_missing(pl.col("prev"))).height == 0
    assert features["hour"].is_between(0, 23).all() and features["month"].is_between(1, 12).all()


def test_discretization(features):
    pm = features.drop_nulls("PM2_5")
    expected = np.searchsorted(CFG["eaqi"]["bins"]["PM2_5"], pm["PM2_5"].to_numpy(), side="left")
    assert (expected == pm["pm25_bin_eaqi"].to_numpy()).all()
    fit = pm.filter(FIT)
    shares = fit["pm25_bin_equal_frequency"].value_counts()["count"] / fit.height
    assert shares.is_between(0.19, 0.21).all()


def test_binarization(features, daily):
    days = features.select("station_id", pl.col("ts_local").dt.date().alias("date"), "pm25_exceed_who").unique()
    pm25 = daily.filter(pl.col("pollutant").cast(pl.String) == "PM2_5").select(pl.col("station_id").cast(pl.String),
                                                                                 "date", "mean")
    joined = days.join(pm25, on=["station_id", "date"], how="left").drop_nulls("pm25_exceed_who")
    limit = CFG["binarization"]["pm25_daily_who"]
    assert (joined["pm25_exceed_who"] == (joined["mean"] > limit).cast(pl.Int8)).all()
    for prefix in ("station_id_", "season_", "station_type_"):
        cols = [c for c in features.columns if c.startswith(prefix)]
        assert len(cols) > 1 and (features.select(pl.sum_horizontal(cols)).to_series() == 1).all()


@pytest.mark.parametrize("pollutant", NAMES)
def test_transformation(features, pollutant):
    fit = features.filter(FIT)
    mm, z, dec = (fit[f"{pollutant}_{s}"].drop_nulls() for s in ("minmax", "z", "dec"))
    assert mm.min() == pytest.approx(0, abs=1e-9) and mm.max() == pytest.approx(1, abs=1e-9)
    assert z.mean() == pytest.approx(0, abs=1e-9) and z.std() == pytest.approx(1, abs=1e-9)
    assert dec.abs().max() < 1


def test_dimensionality_reduction():
    variance = pl.read_csv(TABLES / "p5_pca_variance.csv")["variance"]
    assert variance.sum() == pytest.approx(1, abs=0.01)
    assert (variance.diff().drop_nulls() <= 0).all()
    assert "NOx" not in pl.read_csv(TABLES / "p5_pca_loadings.csv")["pollutant"].to_list()


def test_report():
    report = (REPORTS / "report_phase1.md").read_text(encoding="utf-8")
    figures = re.findall(r"\(figures/([^)]+)\)", report)
    assert figures and all((REPORTS / "figures" / f).exists() for f in figures)
    assert "{" not in report and "}" not in report and " None " not in report
    for title in ("## 2. Data collection", "## 3. Data integration", "## 4. Missing values",
                  "## 5. Aggregation and sampling", "## 6. Feature selection", "## 7. Decision log",
                  "## Appendix A"):
        assert title in report
