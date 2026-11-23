"""Exploratory data analysis for the Phase I stratified hourly sample.

Run after Phase I has produced data/gold/sample_10pct.parquet. This script is
read-only with respect to the input and writes tables, figures, and a report
under reports/eda/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
POLLUTANTS = ["SO2", "PM10", "PM2_5", "O3", "NO2", "NOx", "CO", "NO"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/gold/sample_10pct.parquet",
                        help="Phase I p4 stratified sample parquet")
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}\nRun the Phase I pipeline first: python run_all.py")

    out = ROOT / "reports/eda"
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(source)
    missing = [c for c in POLLUTANTS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input schema is missing pollutant columns: {missing}; got {df.columns}")
    numeric = [c for c in POLLUTANTS if df.schema[c].is_numeric()]
    if len(numeric) != len(POLLUTANTS):
        raise SystemExit("Expected all pollutant columns to be numeric")

    # Pollutant concentrations are measured numerical variables. Station and
    # season are categorical; timestamps and keys are identifiers/time axes.
    summary_rows = []
    for col in numeric:
        s = df[col].drop_nulls()
        summary_rows.append({
            "pollutant": col, "unit": "mg/m³" if col == "CO" else "µg/m³",
            "count": s.len(), "missing": df[col].null_count(), "mean": s.mean(),
            "median": s.median(), "std": s.std(ddof=1), "variance": s.var(ddof=1),
            "min": s.min(), "p05": s.quantile(0.05), "q1": s.quantile(0.25),
            "q3": s.quantile(0.75), "p95": s.quantile(0.95), "max": s.max(),
        })
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(tables / "summary_by_pollutant.csv")