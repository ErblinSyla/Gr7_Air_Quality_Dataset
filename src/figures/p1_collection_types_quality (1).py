"""p1 figures: invalid share by station and year, weekly NOx ratio."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, COLORS, INK_2, by_name, heatmap, plt, save_figure


def nox_weekly_ratio(weekly: pl.DataFrame) -> str:
    lo, hi = CFG["quality"]["nox_ppb_ratio"]
    f = by_name("ppb_to_ugm3")["NOx"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axhspan(lo, hi, color=COLORS[2], alpha=0.15, lw=0)
    for g in weekly.sort("station_id", "ts_utc").partition_by("station_id", maintain_order=True):
        ax.plot(g["ts_utc"], g["median"], color="#b9b8b2", lw=0.8)
    net = weekly.group_by("ts_utc").agg(pl.col("median").median()).sort("ts_utc")
    ax.plot(net["ts_utc"], net["median"], color=COLORS[0], lw=2, label="network median")
    ax.text(weekly["ts_utc"].min(), hi + 0.01, f"ppb range: 1/{f} ≈ {1 / f:.3f}", color=INK_2, fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("NOx / (NO2 + 1.53·NO)")
    ax.set_title("Weekly NOx ratio: consistent with ppb only from mid-November 2019 (grey: stations)")
    ax.legend(loc="lower right")
    return save_figure(fig, "p1_nox_weekly_ratio")


def make(r: SimpleNamespace) -> dict[str, str]:
    return {
        "invalid": save_figure(heatmap(r.q["invalid_station_year"],
                                       "% of invalid measurements (Validity −1/−99) by station and year",
                                       "% invalid"), "p1_invalid_station_year"),
        "nox": nox_weekly_ratio(r.nox_week),
    }
