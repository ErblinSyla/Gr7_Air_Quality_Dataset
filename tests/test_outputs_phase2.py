"""Integration tests of phase II (anomaly detection) on the pipeline outputs (run `python run_all.py phase2` first)."""

import re

import polars as pl
import pytest

from utils import CFG, DATA, REPORTS, TABLES, scan

pytestmark = pytest.mark.skipif(not (DATA / "gold" / "anomaly_catalogue.parquet").exists(),
                                reason="phase II outputs not found: run `python run_all.py phase2` first")
O = CFG["outliers"]
METHODS = ["knn", "lof", "kmeans", "mahalanobis", "ocsvm", "iforest"]
TOP_SHARE = 1 - O["threshold_percentile"] / 100


@pytest.fixture(scope="module")
def point():
    return scan("gold", "anomalies_point").collect()


def test_detection_only_keeps_every_station_day(point):
    daily = (scan("gold", "daily").filter((pl.col("source") == "measured") & pl.col("valid_day")).collect()
             .with_columns(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String))
             .pivot(on="pollutant", index=["station_id", "date"], values="mean").drop_nulls(O["features"]))
    assert point.height == daily.height
    assert point.select("station_id", "date").is_unique().all()
    joined = point.join(daily, on=["station_id", "date"], suffix="_daily")
    for f in O["features"]:
        assert (joined[f] == joined[f"{f}_daily"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_threshold_marks_top_share(point, method):
    assert point[f"outlier_{method}"].mean() == pytest.approx(TOP_SHARE, abs=1e-3)
    out, rest = point.filter(pl.col(f"outlier_{method}") == 1), point.filter(pl.col(f"outlier_{method}") == 0)
    assert out[f"score_{method}"].min() > rest[f"score_{method}"].max()


def test_point_decision_follows_the_best_rule(point):
    rules = pl.read_csv(TABLES / "o1_combination_rules.csv").sort("f1", descending=True)
    assert rules.height == 4
    if rules["rule"][0] == "LOF only":
        assert (point["point_anomaly"] == (point["outlier_lof"] == 1)).all()


def test_every_parameter_value_is_evaluated():
    for method, key in (("knn", "knn_k"), ("lof", "lof_k"), ("kmeans", "kmeans_k"), ("ocsvm", "ocsvm_gamma")):
        ev = pl.read_csv(TABLES / f"o1_{method}_evaluation.csv")
        assert sorted(set(ev["param"])) == sorted(O[key])
        assert ev["f1"].is_between(0, 1).all()
    assert pl.read_csv(TABLES / "o1_knn_evaluation.csv")["variant"].n_unique() == 4
    assert pl.read_csv(TABLES / "o1_mahalanobis_evaluation.csv").height == 3
    assert set(pl.read_csv(TABLES / "o1_recall_by_type.csv")["type"]) == {"gain error", "sensor at zero",
                                                                         "PM2.5 > PM10"}
    assert pl.read_csv(TABLES / "o2_evaluation.csv").height == 6 * (1 + len(O["contextual_lof_k"]))
    assert sorted(set(pl.read_csv(TABLES / "o1_iforest_evaluation.csv")["param"])) == sorted(O["iforest_max_samples"])
    sens = pl.read_csv(TABLES / "o1_threshold_sensitivity.csv")
    assert sorted(set(sens["injected_pct"])) == sorted(100 * x for x in O["sensitivity_shares"])
    assert set(pl.read_csv(TABLES / "o3_best_settings.csv")["fault"]) == {"spike", "stuck sensor", "drift"}
    for name in ("o1_knn_evaluation", "o1_lof_evaluation", "o2_evaluation"):
        ev = pl.read_csv(TABLES / f"{name}.csv")
        assert ev["pr_auc"].is_between(0, 1).all() and ev["roc_auc"].is_between(0, 1).all()
    assert pl.read_csv(TABLES / "o1_auc_check.csv").height == 7  # 5 lecture methods, Isolation Forest, z-score


def test_contextual_flags():
    ctx = scan("gold", "anomalies_contextual").collect()
    assert ctx["contextual_anomaly"].mean() == pytest.approx(TOP_SHARE, abs=1e-3)
    robust = pl.read_csv(TABLES / "o2_robust_z_evaluation.csv")
    assert robust.filter(pl.col("context") == "none")["f1"].item() < robust["f1"].max()


def test_temporal_flags_match_their_definition():
    alarms = scan("gold", "anomalies_temporal").collect()
    best = dict(pl.read_csv(TABLES / "o3_best_settings.csv").select("fault", "param").iter_rows())
    min_len = int(best["stuck sensor"].split()[2])
    assert (alarms.filter("stuck")["run_length"] >= min_len).all()
    assert (alarms["spike"] | alarms["stuck"]).all()
    check = pl.read_csv(TABLES / "o3_phase1_check.csv")
    assert check.filter(pl.col("fault") == "stuck sensor")["share_found"].item() > 0.9


def test_catalogue_combines_all_types():
    cat = scan("gold", "anomaly_catalogue").collect()
    types = ["point", "contextual", "spike", "stuck sensor"]
    assert cat.select("station_id", "date").is_unique().all()
    assert cat.select(pl.any_horizontal(types)).to_series().all()
    assert cat["point"].sum() == scan("gold", "anomalies_point").filter("point_anomaly").collect().height


def test_report_phase2():
    report = (REPORTS / "report_phase2.md").read_text(encoding="utf-8")
    figures = re.findall(r"\(figures/([^)]+)\)", report)
    assert len(figures) == 12 and all((REPORTS / "figures" / f).exists() for f in figures)
    assert "{" not in report and "}" not in report and " None " not in report
    for title in ("## Summary", "Isolation Forest (comparison method", "Is the 99th percentile only chosen",
                  "Tositti et al.", "Regional level (weather proxy)", "## 1. Point anomalies", "## 2. Contextual anomalies", "## 3. Collective anomalies",
                  "## 4. Summary", "Lecture methods that are not used", "Comparison and final decision",
                  "Check against the rules of phase I"):
        assert title in report
