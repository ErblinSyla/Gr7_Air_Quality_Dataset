"""p4 figures: daily PM2.5 by station."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, COLORS, INK_2, plt, save_figure


def pm25_daily_stations(day: pl.DataFrame) -> str:
    pm = day.filter((pl.col("pollutant") == "PM2_5") & (pl.col("source") == "measured"))
    stations = sorted(pm["station_id"].cast(pl.String).unique().to_list())
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, stations, strict=False):
        d = pm.filter(pl.col("station_id") == s).sort("date")
        ax.plot(d["date"], d["mean"], lw=0.7, color=COLORS[0])
        ax.axhline(CFG["binarization"]["pm25_daily_who"], color=INK_2, lw=0.8, ls="--")
        ax.set_title(s, fontsize=9)
    fig.supylabel("PM2.5 daily mean (µg/m³)")
    fig.suptitle("Daily PM2.5 by station (line: WHO daily guideline 15 µg/m³) – peaks every winter",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "p4_pm25_daily_stations")


def make(r: SimpleNamespace) -> dict[str, str]:
    return {"daily": pm25_daily_stations(r.agg.day)}
