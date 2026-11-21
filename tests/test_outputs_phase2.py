"""Integration tests of phase II (outlier detection) on the pipeline outputs (run `python run_all.py phase2` first)."""

import re

import polars as pl
import pytest

from utils import CFG, DATA, REPORTS, TABLES, scan

pytestmark = pytest.mark.skipif(not (DATA / "gold" / "outlier_flags.parquet").exists(),
                                reason="phase II outputs not found: run `python run_all.py phase2` first")
O = CFG["outliers"]
METHODS = ["knn", "lof", "kmeans"]


@pytest.fixture(scope="module")
def flags():
    return scan("gold", "outlier_flags").collect()


def test_detection_only_keeps_every_station_day(flags):
    daily = (scan("gold", "daily").filter((pl.col("source") == "measured") & pl.col("valid_day")).collect()
             .with_columns(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String))
             .pivot(on="pollutant", index=["station_id", "date"], values="mean").drop_nulls(O["features"]))
    assert flags.height == daily.height
    assert flags.select("station_id", "date").is_unique().all()
    joined = flags.join(daily, on=["station_id", "date"], suffix="_daily")
    for f in O["features"]:
        assert (joined[f] == joined[f"{f}_daily"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_threshold_marks_top_share(flags, method):
    share = flags[f"outlier_{method}"].mean()
    assert share == pytest.approx(1 - O["threshold_percentile"] / 100, abs=1e-3)
    out, rest = flags.filter(pl.col(f"outlier_{method}") == 1), flags.filter(pl.col(f"outlier_{method}") == 0)
    assert out[f"score_{method}"].min() > rest[f"score_{method}"].max()


def test_n_methods(flags):
    assert (flags["n_methods"] == flags.select(pl.sum_horizontal(f"outlier_{m}" for m in METHODS)).to_series()).all()


def test_every_parameter_value_is_evaluated():
    for method, key in (("knn", "knn_k"), ("lof", "lof_k"), ("kmeans", "kmeans_k")):
        ev = pl.read_csv(TABLES / f"o1_{method}_evaluation.csv")
        assert sorted(set(ev["k"])) == sorted(O[key])
        assert ev["f1"].is_between(0, 1).all()
    assert set(pl.read_csv(TABLES / "o1_knn_evaluation.csv")["metric"]) == set(O["knn_metrics"])
    best = pl.read_csv(TABLES / "o1_best_settings.csv")
    assert best["method"].to_list() == ["KNN", "LOF", "K-Means", "univariate"]
    threshold = pl.read_csv(TABLES / "o1_threshold_evaluation.csv")
    assert sorted(set(threshold["percentile"])) == sorted(O["threshold_percentiles"])


def test_report_phase2():
    report = (REPORTS / "report_phase2.md").read_text(encoding="utf-8")
    figures = re.findall(r"\(figures/([^)]+)\)", report)
    assert len(figures) == 6 and all((REPORTS / "figures" / f).exists() for f in figures)
    assert "{" not in report and "}" not in report and " None " not in report
    for title in ("## 1. Outlier detection", "### 1.3 KNN", "### 1.4 Local Outlier Factor", "### 1.5 K-Means",
                  "### 1.6 Choice of the threshold", "### 1.7 Comparison", "### 1.8 Outliers in the real data"):
        assert title in report
