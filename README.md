# Gr7_Air_Quality_Dataset

Data preparation and visualization (FIEK, University of Prishtina, 2026/27) – Group 7.

- **Phase I:** data preprocessing → `reports/report_phase1.md`
- **Phase II:** anomaly detection – point, contextual and collective (temporal) anomalies → `reports/report_phase2.md`

## Data set
EEA Air Quality, country XK (Kosovo), E1a series (verified hourly measurements), 2019-01-01 → 2024-12-30:
12 stations, 8 pollutants (SO2, PM10, PM2.5, O3, NO2, NOx, CO, NO), 4,261,594 rows.
Source: EEA Air Quality Download Service (country XK, E1a) and the EEA station metadata (`DataExtract.csv`). Licence: CC BY 4.0 (© European Environment Agency).

The data is not stored in git. Place it as follows before running:

```
data/raw/E1a/*.parquet          # 93 files, one per station × pollutant
data/raw/metadata/*.csv         # EEA station metadata (DataExtract.csv)
```

## How to run
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python run_all.py
```
The full pipeline runs in about ten minutes and writes `reports/report_phase1.md` and `reports/report_phase2.md`.
One phase can be run with `.venv/Scripts/python run_all.py phase1` or `... phase2` (phase II reads the daily means of
phase I, so phase I must have run once). Single steps can be re-run with e.g. `.venv/Scripts/python run_all.py p3`
(each step reads the output of the previous one; the report of a phase is rebuilt only when all its steps run).

## Tests
```bash
.venv/Scripts/python -m pytest -q
```
`tests/test_formulas.py` checks the preprocessing formulas against the numeric examples of the lecture slides
(min-max, z-score, decimal scaling, smoothing by binning, χ² test). `tests/test_outputs.py` checks every assignment
requirement on the pipeline outputs (run `run_all.py` first). `tests/test_outlier_methods.py` checks that the outlier
detection functions reproduce the results of the exercises (KNN customers, LOF and K-Means with `make_blobs`) and
the lecture examples (Grubbs' test, Mahalanobis vs Euclidean distance);
`tests/test_outputs_phase2.py` checks the phase II outputs (flags, evaluations, catalogue, report).

## Structure
Every step has three modules with the same name, one per responsibility; each folder has a `first_phase/` and a
`second_phase/` subfolder:

| Folder | Responsibility |
|---|---|
| `src/steps/` | data processing only: read, clean, compute, save data and tables |
| `src/figures/` | figures only |
| `src/sections/` | report text only (Markdown) |

| Phase | Step | Requirement |
|---|---|---|
| I | `p1_collection_types_quality` | data collection, data types, data quality |
| I | `p2_integration_cleaning` | integration, cleaning |
| I | `p3_missing_values` | identification and treatment strategy for missing values |
| I | `p4_aggregation_sampling` | aggregation, sampling |
| I | `p5_features_transformation_reduction` | feature subset selection, feature creation, discretization and binarization, transformation, dimensionality reduction |
| I | `x_beyond_lectures` | appendix (optional, beyond the lectures): MCAR check, RMSE, Kolmogorov–Smirnov, Yeo-Johnson, mutual information |
| II | `common` | shared parts of phase II: station-day feature space, scaling, percentile threshold, robust z-score, precision / recall / F1, PR-AUC / ROC-AUC |
| II | `o1_point_anomalies` | point anomalies of the station-days: KNN distance (k, Euclidean vs Manhattan, mean vs k-th), LOF (n_neighbors), K-Means (K, elbow), Mahalanobis (classic vs robust covariance, log values, χ² threshold), one-class SVM (gamma, nu), Isolation Forest as comparison (max_samples, trees), threshold incl. sensitivity to the injected share, combination rules; evaluated with injected gain errors, zero readings and PM2.5 > PM10 |
| II | `o2_contextual_anomalies` | contextual anomalies: robust z-score and LOF on the residuals of the context profile; contexts from station to station × month × day type and the regional level of the day (weather proxy); evaluated with injected wrong-season and wrong-station days |
| II | `o3_collective_anomalies` | collective anomalies of the hourly series: isolated spike (threshold), stuck sensor (run length), drift (window, threshold); evaluated with injected faults and checked against the phase I rules |
| II | `o4_anomaly_summary` | catalogue of all anomalies, overlap of the types, where and when they occur; `sections/second_phase/summary.py` writes the 1–2 page summary at the start of the report |

Other files: `src/utils.py` (shared helpers), `run_all.py` (runs steps → figures → sections and builds the report),
`config.yaml` (all paths and parameters).

Data layers: `data/bronze` (correct types) → `data/silver` (integrated, cleaned) → `data/gold` (hourly grid with
imputation, aggregates, sample, features; phase II: `anomalies_point`, `anomalies_contextual`,
`anomalies_temporal` and `anomaly_catalogue` with scores and 0/1 flags; no value is changed).

Outputs: `reports/report_phase1.md` and `reports/report_phase2.md` (full reports), `reports/sections/` (one section per script),
`reports/tables/` (CSV tables; phase II: `o1_*`–`o4_*`), `reports/figures/` (PNG figures), `reports/lineage.csv` (rows and nulls per step).
