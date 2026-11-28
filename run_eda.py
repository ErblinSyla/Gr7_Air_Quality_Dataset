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

    categorical = []
    for col in ("station_id", "season"):
        if col in df.columns:
            categorical.extend({"variable": col, "category": str(r[0]), "count": r[1],
                                "share_pct": 100 * r[1] / df.height}
                               for r in df.group_by(col).len().sort(col).iter_rows())
    if categorical:
        pl.DataFrame(categorical).write_csv(tables / "categorical_counts.csv")

    # Pairwise-complete Pearson correlations preserve the available observations
    # for each pair and record their sample sizes to make missingness visible.
    corr = np.eye(len(numeric), dtype=float)
    pairs = []
    for i, left in enumerate(numeric):
        for j in range(i + 1, len(numeric)):
            right = numeric[j]
            pair = df.select(left, right).drop_nulls()
            value = pair.select(pl.corr(left, right)).item() if pair.height >= 2 else None
            corr[i, j] = corr[j, i] = value if value is not None else np.nan
            pairs.append({"pollutant_a": left, "pollutant_b": right,
                          "complete_pairs": pair.height, "pearson_r": value})
    pl.DataFrame({"pollutant": numeric, **{p: corr[:, i].tolist() for i, p in enumerate(numeric)}}).write_csv(
        tables / "correlation_matrix.csv")
    pl.DataFrame(pairs).with_columns(pl.col("pearson_r").abs().alias("abs_r")).sort(
        "abs_r", descending=True).drop("abs_r").write_csv(tables / "correlation_pairs.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), numeric, rotation=40, ha="right")
    ax.set_yticks(range(len(numeric)), numeric)
    ax.set_title("Pearson correlation between hourly pollutant concentrations")
    for i in range(len(numeric)):
        for j in range(len(numeric)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(figures / "correlation_heatmap.png", dpi=160)
    plt.close(fig)