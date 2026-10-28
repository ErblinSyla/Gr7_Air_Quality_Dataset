"""p3 figures: missing value map, missingness by month and hour, imputation methods."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from utils import COLORS, heatmap, plt, save_figure

METHOD_ORDER = ["deterministic", "linear", "left_null"]


def missing_month_hour(a: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    axes[0].bar(a["missing_by_month"]["month"], a["missing_by_month"]["pct"], color=COLORS[0])
    axes[0].set_title("% missing by month")
    axes[1].plot(a["missing_by_hour"]["hour"], a["missing_by_hour"]["pct"], marker="o", ms=3, color=COLORS[0])
    axes[1].set_title("% missing by local hour")
    for ax in axes:
        ax.set_ylabel("% of hours")
    return save_figure(fig, "p3_missing_month_hour")


def imputation_methods(methods: pl.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cols = [c for c in METHOD_ORDER if c in methods.columns]
    left = np.zeros(methods.height)
    for i, c in enumerate(cols):
        ax.barh(methods["pollutant"].cast(pl.String), methods[c], left=left, color=COLORS[i], label=c)
        left += methods[c].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("hours")
    ax.set_title("Missing values: how they were imputed or why they stayed empty")
    ax.legend(ncol=len(cols), loc="lower center", bbox_to_anchor=(0.5, -0.35), fontsize=8)
    return save_figure(fig, "p3_imputation_methods")


def make(r: SimpleNamespace) -> dict[str, str]:
    return {
        "map": save_figure(heatmap(r.missing_map, "Missing value map: % of missing hours (series × month)",
                                   "% missing", annotate=False), "p3_missing_map"),
        "when": missing_month_hour(r.a),
        "methods": imputation_methods(r.methods),
    }
