# Gr7_Air_Quality_Dataset

Data preparation and visualization (FIEK, University of Prishtina, 2026/27) – Group 7, Phase I: data preprocessing.

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
The full pipeline runs in about two minutes and writes `reports/report_phase1.md`.
Single steps can be re-run with e.g. `.venv/Scripts/python run_all.py p3` (each step reads the output of the previous
one; the full report is rebuilt only when all steps run).

## Tests
```bash
.venv/Scripts/python -m pytest -q
```
`tests/test_formulas.py` checks the preprocessing formulas against the numeric examples of the lecture slides
(min-max, z-score, decimal scaling, smoothing by binning, χ² test). `tests/test_outputs.py` checks every assignment
requirement on the pipeline outputs (run `run_all.py` first).

## Structure
Every step has three modules with the same name, one per responsibility:

| Folder | Responsibility |
|---|---|
| `src/steps/` | data processing only: read, clean, compute, save data and tables |
| `src/figures/` | figures only |
| `src/sections/` | report text only (Markdown) |

| Step | Requirement |
|---|---|
| `p1_collection_types_quality` | data collection, data types, data quality |
| `p2_integration_cleaning` | integration, cleaning |
| `p3_missing_values` | identification and treatment strategy for missing values |
| `p4_aggregation_sampling` | aggregation, sampling |
| `p5_features_transformation_reduction` | feature subset selection, feature creation, discretization and binarization, transformation, dimensionality reduction |
| `x_beyond_lectures` | appendix (optional, beyond the lectures): MCAR check, RMSE, Kolmogorov–Smirnov, Yeo-Johnson, mutual information |

Other files: `src/utils.py` (shared helpers), `run_all.py` (runs steps → figures → sections and builds the report),
`config.yaml` (all paths and parameters).

Data layers: `data/bronze` (correct types) → `data/silver` (integrated, cleaned) → `data/gold` (hourly grid with
imputation, aggregates, sample, features).

Outputs: `reports/report_phase1.md` (full report), `reports/sections/` (one section per script),
`reports/tables/` (CSV tables), `reports/figures/` (PNG figures), `reports/lineage.csv` (rows and nulls per step).
