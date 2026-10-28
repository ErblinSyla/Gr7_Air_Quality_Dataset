"""p2 figures: gas analyzer change, NOx ratio before/after harmonisation, distributions before/after cleaning."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from steps.p2_integration_cleaning import NOX_FACTOR
from utils import COLORS, NAMES, plt, save_figure


def analyzer_change(weekly: pl.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, p in enumerate(["NO", "NO2", "CO", "SO2", "PM2_5"]):
        s = weekly.filter(pl.col("pollutant") == p).sort("ts_local")
        ax.plot(s["ts_local"], s["r"], marker="o", ms=3, lw=2 if p in ("NO", "PM2_5") else 1.2, color=COLORS[i],
                label=p if p != "PM2_5" else "PM2.5 (control)")
    ax.axhline(1, color="#52514e", lw=0.8, ls="--")
    ax.set_ylabel("value / seasonal normal (2020–24)")
    ax.set_title("2019: NO and NO2 far above normal until November while PM2.5 is normal → analyzer change")
    ax.legend(ncol=5, fontsize=8)
    return save_figure(fig, "p2_analyzer_change")


def nox_ratio(before: pl.DataFrame, after: pl.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.linspace(0, 1.6, 161)
    ax.hist(before["r"].clip(0, 1.6).to_numpy(), bins=bins, color=COLORS[1], alpha=0.75,
            label=f"raw: median {before['r'].median():.3f} (NOx in ppb)")
    ax.hist(after["r"].clip(0, 1.6).to_numpy(), bins=bins, color=COLORS[0], alpha=0.75,
            label=f"after ×{NOX_FACTOR} and cleaning: median {after['r'].median():.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("NOx / (NO2 + 1.53·NO)")
    ax.set_ylabel("hours")
    ax.set_title("Unit harmonisation: NOx becomes consistent with NO2 and NO")
    ax.legend()
    return save_figure(fig, "p2_nox_ratio_before_after")


def histograms(hist: dict[str, tuple[np.ndarray, np.ndarray]]) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    for ax, p in zip(axes.flat, NAMES, strict=True):
        b, a = hist[p]
        edges = np.linspace(0, max(b.max(), a.max()), 61)
        ax.hist(b, bins=edges, histtype="step", lw=1.5, color=COLORS[1], label="before (Validity 1/2)")
        ax.hist(a, bins=edges, histtype="step", lw=1.5, color=COLORS[0], label="after cleaning")
        ax.set_yscale("log")
        ax.set_title(p, fontsize=10)
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("value (canonical unit)")
    fig.suptitle("Distribution before and after cleaning (log count axis)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "p2_histograms_before_after")


def make(r: SimpleNamespace) -> dict[str, str]:
    return {"gas": analyzer_change(r.gas_weekly), "nox": nox_ratio(r.before, r.after), "hist": histograms(r.hist)}
