"""Unit tests: the outlier detection functions of phase II reproduce the results of the exercises
(Anomaly_detection notebook: KNN customers, LOF with make_blobs, K-Means with make_blobs). No project data needed."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, precision_score, recall_score

from steps.second_phase.o1_outlier_detection import (
    flag,
    inject_anomalies,
    kmeans_score,
    knn_score,
    lof_score,
    precision_recall_f1,
    standardize,
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
    changed = [(np.isin(row, x).sum()) for row in x_eval[actual]]
    assert changed == [5] * 10
