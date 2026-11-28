"""Runs the pipeline: for every step, process the data (steps), draw the figures (figures) and write the report
section (sections); then builds the report of every phase that was run completely.

Phase I  (first_phase):  p1–p5 and appendix x → reports/report_phase1.md
Phase II (second_phase): anomaly detection – o1 point, o2 contextual, o3 collective (hourly), o4 summary
                         → reports/report_phase2.md

Usage: python run_all.py [phase1 | phase2 | p1 o1 ...]   (no argument: both phases)"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl  # noqa: E402

from sections.first_phase import decisions  # noqa: E402
from sections.second_phase import summary  # noqa: E402
from utils import REPORTS, SECTIONS, md, scan, write_section  # noqa: E402

PHASES = {
    "phase1": {
        "p1": "first_phase.p1_collection_types_quality",
        "p2": "first_phase.p2_integration_cleaning",
        "p3": "first_phase.p3_missing_values",
        "p4": "first_phase.p4_aggregation_sampling",
        "p5": "first_phase.p5_features_transformation_reduction",
        "x": "first_phase.x_beyond_lectures",
    },
    "phase2": {
        "o1": "second_phase.o1_point_anomalies",
        "o2": "second_phase.o2_contextual_anomalies",
        "o3": "second_phase.o3_collective_anomalies",
        "o4": "second_phase.o4_anomaly_summary",
    },
}
STEPS = {name: module for steps in PHASES.values() for name, module in steps.items()}
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
    """Run one step: data (steps) → figures → report section; returns the duration in seconds."""
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
    """Build reports/report_phase1.md from the sections p1–p5, the decision log and the appendix."""
    lineage = pl.read_csv(REPORTS / "lineage.csv")
    facts = (scan("bronze", "measurements")
             .select(pl.col("ts_local").min().dt.date().alias("start"), pl.col("ts_local").max().dt.date().alias("end"),
                     pl.col("station_id").n_unique().alias("stations"),
                     pl.col("pollutant").n_unique().alias("pollutants"),
                     pl.len().alias("rows"))
             .collect().row(0, named=True))
    read = lambda name: (SECTIONS / f"{name}.md").read_text(encoding="utf-8")  # noqa: E731
    sections = "\n\n".join(read(name) for name in PHASES["phase1"] if name not in APPENDIX)
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


def build_report_phase2() -> None:
    """Build reports/report_phase2.md from the sections of phase II."""
    sections = "\n\n".join((SECTIONS / f"{name}.md").read_text(encoding="utf-8") for name in PHASES["phase2"])
    report = f"""# Phase II report: anomaly detection in the air quality data of Kosovo

**Data set:** the cleaned measurements of phase I (EEA Air Quality, country XK, E1a, CC BY 4.0): station-day means
(`data/gold/daily.parquet`) and hourly series (`data/gold/hourly_long/`). This phase only **detects** anomalies; no
value is changed or removed.

**Approach.** An anomaly (outlier) is an object that differs from most other objects. Following the lecture it can be
caused by a measurement error, by natural variation or by a different mechanism (a real pollution episode), and it has
one of three structures. Every structure needs its own representation of the data and its own methods:

| Section | Type of anomaly (lecture) | Representation | Methods | Evaluated with injected |
|---|---|---|---|---|
| 1 | point | station-day: six daily means | KNN distance, LOF, K-Means, Mahalanobis, one-class SVM (z-score as baseline) | gain error, sensor at zero, PM2.5 > PM10 |
| 2 | contextual | station-day relative to its station × month | robust z-score and LOF on the context residuals | wrong season, wrong station |
| 3 | collective | hourly series of every station and pollutant | isolated-spike, stuck-sensor and drift detectors | spikes, frozen runs, drifts |
| 4 | summary | all flags together | catalogue of all anomalies | – |

The setting is **unsupervised** (no labels), so every method follows the three steps of the lecture – build a profile
of normal behaviour, measure how far every observation is from it (the anomaly score), flag the largest scores – and
every parameter is chosen with **known anomalies injected into a copy of the data**, as in the exercises
(precision, recall, F1).

{summary.render()}

{sections}
"""
    (REPORTS / "report_phase2.md").write_text(report.replace("../figures/", "figures/"), encoding="utf-8")


def main() -> None:
    """Run the selected steps (or whole phases) and rebuild the report of every completed phase."""
    args = sys.argv[1:] or list(PHASES)
    selected = [name for a in args for name in PHASES.get(a, [a])]
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        raise SystemExit(f"Unknown step(s): {unknown}. Choose from {list(PHASES)} or {list(STEPS)}.")
    total = sum(run_step(name) for name in selected)
    builders = {"phase1": (build_report, "reports/report_phase1.md"),
                "phase2": (build_report_phase2, "reports/report_phase2.md")}
    built = []
    for phase, (build, path) in builders.items():
        if all(name in selected for name in PHASES[phase]):
            build()
            built.append(path)
    print(f"Done: {', '.join(selected)} ({total:.0f} s)" + (f" → {', '.join(built)}" if built else
          "; run a whole phase to rebuild its report"))


if __name__ == "__main__":
    main()
