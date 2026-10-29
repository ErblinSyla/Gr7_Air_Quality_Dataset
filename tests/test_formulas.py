"""Unit tests: the preprocessing formulas reproduce the numeric examples from the lecture slides (no data needed)."""

import numpy as np
import polars as pl
import pytest

from steps.p2_integration_cleaning import distance_km
from steps.p3_missing_values import chi2_test
from steps.p4_aggregation_sampling import zone
from steps.p5_features_transformation_reduction import decimal_j, minmax, smooth_by_bins, zscore


def test_min_max_normalization():
    assert minmax(73600, 12000, 98000) == pytest.approx(0.716, abs=5e-4)


def test_z_score_normalization():
    assert zscore(73600, 54000, 16000) == pytest.approx(1.225)


def test_decimal_scaling():
    assert decimal_j(986) == 3
    assert decimal_j(917) == 3
    assert decimal_j(0.5) == 0
    assert decimal_j(1000) == 4


def test_smoothing_by_bin_means_and_boundaries():
    prices = np.array([24, 9, 28, 4, 21, 34, 15, 26, 8, 29, 25, 21], float)
    means, bounds, parts = smooth_by_bins(prices, 3)
    assert [b.tolist() for b in parts] == [[4, 8, 9, 15], [21, 21, 24, 25], [26, 28, 29, 34]]
    assert np.round(means).tolist() == [9] * 4 + [23] * 4 + [29] * 4
    assert bounds.tolist() == [4, 4, 4, 15, 21, 21, 25, 25, 26, 26, 26, 34]


def test_chi_square_lecture_example():
    cells = [(True, False, 250), (True, True, 200), (False, False, 50), (False, True, 1000)]
    df = pl.DataFrame([{"science_fiction": s, "missing": m} for s, m, n in cells for _ in range(n)])
    result = chi2_test(df, "science_fiction")
    assert result["chi2"] == pytest.approx(507.9, abs=0.1)
    assert result["df"] == 1
    assert result["critical_value"] == pytest.approx(3.841, abs=1e-3)
    assert result["result"] == "dependent"


def test_distance_km():
    df = pl.DataFrame({"a": [42.663], "b": [21.166], "c": [42.663 + 1 / 111], "d": [21.166]})
    km = df.select(distance_km(pl.col("a"), pl.col("b"), pl.col("c"), pl.col("d"))).item()
    assert km == pytest.approx(1.0, abs=1e-6)


def test_zone_concept_hierarchy():
    assert zone("Palaj (Obiliq)") == "Obiliq"
    assert zone("Pristina") == "Pristina"
