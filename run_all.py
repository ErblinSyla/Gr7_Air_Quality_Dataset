"""Runs the pipeline: for every step (p1–p5 and appendix x), process the data (steps), draw the figures (figures)
and write the report section (sections); then builds reports/report_phase1.md. Usage: python run_all.py [p1 ...]"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl  # noqa: E402

from sections import decisions  # noqa: E402
from utils import REPORTS, SECTIONS, md, scan, write_section  # noqa: E402

STEPS = {
    "p1": "p1_collection_types_quality",
    "p2": "p2_integration_cleaning",
    "p3": "p3_missing_values",
    "p4": "p4_aggregation_sampling",
    "p5": "p5_features_transformation_reduction",
    "x": "x_beyond_lectures",
}
APPENDIX = ["x"]

MAP = [
    ("Data preprocessing for analysis", "major tasks: cleaning, integration, reduction, transformation and "
     "discretization (weeks 3–5)", "all", "data/bronze → silver → gold, lineage.csv", "–"),
    ("Data collection", "source, accessibility, reliability, CC BY 4.0 licence (week 2)", "p1", "data/bronze/", "–"),
    ("Data type definition", "nominal / ordinal / interval / ratio; discrete / continuous; type of data set (week 3)",
     "p1", "tables/p1_data_dictionary.csv", "–"),
    ("Data quality", "accuracy, completeness, consistency, timeliness, believability, interpretability; noise, "
     "outliers, missing values, duplicates (weeks 3–4)", "p1",
     "tables/p1_quality_dimensions.csv, p1_quality_*.csv", "p1_invalid_station_year, p1_nox_weekly_ratio"),
    ("Integration", "schema integration, entity identification, data value / unit conflicts (week 4)", "p2",
     "data/silver/stations.parquet, tables/p2_metadata_*.csv, p2_nox_*.csv",
     "p2_nox_ratio_before_after, p2_analyzer_change"),
    ("Aggregation", "hour → day → month → year; data cube and concept hierarchy station → zone → Kosovo "
     "(weeks 3, 5)", "p4", "data/gold/daily, monthly, annual, rollup_annual; tables/p4_*", "p4_pm25_daily_stations"),
    ("Sampling", "SRSWOR, SRSWR, stratified; representative sample (weeks 3, 5)", "p4",
     "data/gold/sample_10pct.parquet, tables/p4_sampling_*", "–"),
    ("Cleaning", "noise, outliers, inconsistencies; data cleaning as a process (week 4)", "p2",
     "data/silver/measurements_clean/, tables/p2_clean_action_by_pollutant.csv", "p2_histograms_before_after"),
    ("Identification and treatment strategy for missing values",
     "reasons for missing values, χ² test; attribute mean vs class mean vs most probable value evaluated on "
     "hidden values; formula, interpolation, ignoring (weeks 3–4)", "p3", "data/gold/hourly_long/, tables/p3_*",
     "p3_missing_map, p3_missing_month_hour, p3_imputation_methods"),
    ("Dimensionality reduction", "PCA with the 5 steps from the lecture (weeks 3, 5)", "p5", "tables/p5_pca_*.csv",
     "p5_pca_scree, p5_pca_biplot"),
    ("Feature subset selection", "irrelevant/redundant attributes, correlation, covariance, forward selection, "
     "backward elimination, decision tree (weeks 3–5)", "p5",
     "tables/p5_dropped_columns.csv, p5_covariance_correlation.csv, p5_forward_selection.csv, "
     "p5_backward_elimination.csv, p5_feature_selection.csv", "p5_correlation"),
    ("Feature creation", "attribute extraction and construction (week 5)", "p5",
     "data/gold/features_hourly.parquet, tables/p5_new_features.csv", "–"),
    ("Discretization and binarization", "equal width / equal frequency, concept hierarchy (EAQI), smoothing by "
     "binning, 0/1 indicators, one-hot, χ² test (weeks 4–5)", "p5",
     "tables/p5_discretization_edges.csv, p5_eaqi_pm25, p5_bin_smoothing, p5_exceedances, p5_chi2_*",
     "p5_discretization_pm25"),
    ("Transformation", "min-max, z-score and decimal scaling normalization (week 5)", "p5",
     "tables/p5_normalization.csv", "p5_normalization"),
]


def run_step(name: str) -> float:
    module = STEPS[name]
    step = importlib.import_module(f"steps.{module}")
    figures = importlib.import_module(f"figures.{module}")
    section = importlib.import_module(f"sections.{module}")
    t0 = time.perf_counter()
    result = step.run()
    figs = figures.make(result)
    elapsed = time.perf_counter() - t0
    write_section(name, section.render(result, figs))
    print(f"{name} done: {elapsed:.1f} s")
    return elapsed


def build_report() -> None:
    lineage = pl.read_csv(REPORTS / "lineage.csv")
    facts = (scan("bronze", "measurements")
             .select(pl.col("ts_local").min().dt.date().alias("start"), pl.col("ts_local").max().dt.date().alias("end"),
                     pl.col("station_id").n_unique().alias("stations"),
                     pl.col("pollutant").n_unique().alias("pollutants"),
                     pl.len().alias("rows"))
             .collect().row(0, named=True))
    read = lambda name: (SECTIONS / f"{name}.md").read_text(encoding="utf-8")  # noqa: E731
    sections = "\n\n".join(read(name) for name in STEPS if name not in APPENDIX)
    appendix = "\n\n".join(read(name) for name in APPENDIX)
    requirements = pl.DataFrame(MAP, schema=["Requirement", "Lecture methods", "Script", "Outputs / tables", "Figures"],
                                orient="row")
    report = f"""# Phase I report: preprocessing of the air quality data of Kosovo

**Data set:** EEA Air Quality, country XK, E1a (verified), hourly measurements, {facts['start']} → {facts['end']},
{facts['stations']} stations, {facts['pollutants']} pollutants, {facts['rows']:,} rows. Licence CC BY 4.0 (© European
Environment Agency); measurements: IHMK monitoring network.

## 1. Data preprocessing for analysis
The goal of preprocessing (lecture): smaller, cleaner and more informative data, so that the analysis is faster and
more accurate ("garbage in, garbage out"). The data passes through four layers: **raw** (the originals, read-only) →
**bronze** (correct types, no changes) → **silver** (integrated and cleaned) → **gold** (hourly grid with imputation,
aggregates, sample, features). Principles: no row is deleted (a problematic value becomes null and the reason is
stored); structural gaps are separated from real ones; "valid" according to the institution is checked again;
statistical outliers are not errors; imputation is transparent (`value` and `value_imputed` + `impute_method`); every
formula from the lecture (normalization, smoothing by binning) is checked with the lecture's numeric examples before
it is applied.

**Mapping of the assignment requirements to methods, scripts, outputs and figures:**
{md(requirements)}
**Lineage** (from {facts['rows']:,} raw rows to the gold tables):
{md(lineage)}
{sections}

{decisions.render()}

{appendix}
"""
    (REPORTS / "report_phase1.md").write_text(report.replace("../figures/", "figures/"), encoding="utf-8")


def main() -> None:
    selected = sys.argv[1:] or list(STEPS)
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        raise SystemExit(f"Unknown step(s): {unknown}. Choose from {list(STEPS)}.")
    total = sum(run_step(name) for name in selected)
    if selected == list(STEPS):
        build_report()
        print(f"Done: reports/report_phase1.md ({total:.0f} s in total)")
    else:
        print(f"Done: {', '.join(selected)} ({total:.0f} s); run all steps to rebuild the full report")


if __name__ == "__main__":
    main()
