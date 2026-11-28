"""Unit tests: the outlier detection functions of phase II reproduce the results of the exercises (KNN customers, LOF
and K-Means with make_blobs) and the examples of the lectures (Grubbs' test, Mahalanobis vs Euclidean distance).
No project data needed."""

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, precision_score, recall_score

from steps.second_phase.common import (
    event_metrics,
    flag,
    precision_recall_f1,
    ranking_metrics,
    robust_z,
    standardize,
)
from steps.second_phase.o1_point_anomalies import (
    grubbs_critical,
    inject_anomalies,
    kmeans_score,
    knn_score,
    lof_score,
    mahalanobis_score,
    ocsvm_score,
)


@pytest.fixture
def customers():
    """The 95 normal and 5 anomalous customers of the KNN exercise (monthly purchases, average amount)."""
    np.random.seed(42)
    normal = np.column_stack([np.random.normal(8, 2.5, 95), np.random.normal(60, 15, 95)])
    normal[:, 0] = np.clip(normal[:, 0], 1, 16)
    normal[:, 1] = np.clip(normal[:, 1], 10, 110)
    return np.vstack([normal, [[30, 250], [1, 250], [35, 15], [40, 300], [25, 350]]])


@pytest.fixture
def blobs_kmeans():
    """The 300 points in 3 clusters and the 4 extreme points of the K-Means exercise."""
    x, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)
    return np.vstack([x, [[10, 10], [-10, -10], [12, -8], [-8, 10]]])


@pytest.fixture
def blobs_lof():
    """The 3 clusters of different density and the 10 anomalous points of the LOF exercise."""
    x, _ = make_blobs(n_samples=[120, 100, 70], centers=[[-5, -5], [0, 5], [5, -2]], cluster_std=[0.5, 1.0, 1.5],
                      random_state=42)
    anomalies = [[-9, 8], [-10, -7], [8, 8], [10, -8], [0, -10], [9, 5], [-8, 2], [7, -10], [2, 10], [-11, -3]]
    return np.vstack([x, anomalies])


def test_knn_exercise(customers):
    score = knn_score(standardize(customers), k=5)
    assert np.percentile(score, 95) == pytest.approx(0.582514, abs=1e-6)
    assert np.where(flag(score, 95))[0].tolist() == [95, 96, 97, 98, 99]


def test_knn_manhattan_is_at_least_euclidean(customers):
    z = standardize(customers)
    assert (knn_score(z, 5, "manhattan") >= knn_score(z, 5, "euclidean") - 1e-12).all()


def test_knn_kth_versus_mean_distance():
    points = np.array([[0.0], [1.0], [3.0], [6.0]])
    assert knn_score(points, 2, score="kth").tolist() == [3, 2, 3, 5]
    assert knn_score(points, 2, score="mean").tolist() == [2, 1.5, 2.5, 4]


@pytest.mark.parametrize(("k", "threshold", "inertia", "found"),
                         [(2, 6.129368, 5947.865, 3), (3, 1.703262, 569.826, 4), (4, 1.599881, 417.444, 3),
                          (5, 1.599881, 326.413, 2)])
def test_kmeans_exercise(blobs_kmeans, k, threshold, inertia, found):
    score, inert = kmeans_score(blobs_kmeans, k)
    detected = flag(score, 95)
    assert np.percentile(score, 95) == pytest.approx(threshold, abs=1e-5)
    assert inert == pytest.approx(inertia, abs=1e-3)
    assert detected.sum() == 16 and detected[300:].sum() == found


@pytest.mark.parametrize("k", [5, 10, 15, 20, 30, 50])
def test_lof_exercise(blobs_lof, k):
    detected = flag(lof_score(standardize(blobs_lof), k), 100 * (1 - 10 / 300))
    assert detected.sum() == 10 and detected[290:].all()


def test_precision_recall_f1_like_sklearn():
    rng = np.random.default_rng(0)
    actual, predicted = rng.random(500) < 0.1, rng.random(500) < 0.15
    m = precision_recall_f1(actual, predicted)
    assert m["precision"] == pytest.approx(precision_score(actual, predicted))
    assert m["recall"] == pytest.approx(recall_score(actual, predicted))
    assert m["f1"] == pytest.approx(f1_score(actual, predicted))


def test_injected_anomalies_change_one_value_only():
    x = np.random.default_rng(1).uniform(1, 10, size=(1000, 6))
    x_eval, actual, info = inject_anomalies(x, seed=42)
    assert actual.sum() == info.height == 10 and len(x_eval) == 1010
    assert np.array_equal(x_eval[:1000], x)
    assert [np.isin(row, x).sum() for row in x_eval[actual]] == [5] * 10
    assert set(info["type"]) == {"gain error", "sensor at zero", "PM2.5 > PM10"}
    pm = x_eval[actual][(info["type"] == "PM2.5 > PM10").to_numpy()]
    assert (pm[:, 1] > pm[:, 0]).all()  # FEATURES order: PM10, PM2_5, …


def test_robust_z_ignores_the_outlier():
    df = pl.DataFrame({"g": ["a"] * 6, "v": [10.0, 11, 9, 10, 12, 100]})
    z = df.select(robust_z(pl.col("v"), ["g"])).to_series()
    assert abs(z[0]) < 1 and z[5] > 30


def test_event_metrics_counts_events_not_points():
    event = np.array([0, 1, 1, 1, 0, 2, 2, 0])
    found = np.array([0, 1, 0, 0, 0, 0, 0, 1], bool)
    m = event_metrics(event, found)
    assert m["recall"] == 0.5 and m["precision"] == 0.5


def test_grubbs_lecture_example():
    times = np.array([12, 14, 15, 15, 16, 17, 18, 18, 19, 45], float)
    g = (times.max() - times.mean()) / times.std(ddof=1)
    assert g == pytest.approx(2.77, abs=0.01)
    assert grubbs_critical(10) == pytest.approx(2.29, abs=0.01)
    assert g > grubbs_critical(10)
    rest = times[:-1]
    assert (rest.mean() - rest.min()) / rest.std(ddof=1) == pytest.approx(1.79, abs=0.01)
    assert grubbs_critical(9) == pytest.approx(2.21, abs=0.01)


def test_mahalanobis_point_against_the_correlation_is_farther():
    rng = np.random.default_rng(0)
    data = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=2000)
    a, b = [-2.0, 2.0], [2.5, 2.5]
    d = mahalanobis_score(np.vstack([data, a, b]))
    assert np.linalg.norm(a) < np.linalg.norm(b)
    assert d[-2] > d[-1]


def test_ocsvm_follows_several_clusters_better_than_one_ellipse(blobs_lof):
    share = 100 * (1 - 10 / 300)
    svm = flag(ocsvm_score(standardize(blobs_lof), gamma=0.5, nu=0.05), share)
    ellipse = flag(mahalanobis_score(blobs_lof), share)
    assert svm[290:].sum() == 9 and ellipse[290:].sum() == 7


def test_ranking_metrics_perfect_and_random():
    actual = np.r_[np.zeros(990, bool), np.ones(10, bool)]
    perfect = ranking_metrics(actual, actual.astype(float))
    assert perfect["pr_auc"] == pytest.approx(1) and perfect["roc_auc"] == pytest.approx(1)
    rng = np.random.default_rng(0)
    random = [ranking_metrics(actual, rng.random(1000)) for _ in range(50)]
    assert np.mean([m["roc_auc"] for m in random]) == pytest.approx(0.5, abs=0.05)
    assert np.mean([m["pr_auc"] for m in random]) < 0.05  # ≈ the share of anomalies (0.01)
