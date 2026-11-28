"""o4 figures: when the anomalies of every type occur (monthly timeline) and where (station × type)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o4_anomaly_summary import TYPES
from utils import COLORS, heatmap, plt, save_figure

COLOR = dict(zip(TYPES, [COLORS[1], COLORS[3], COLORS[0], COLORS[6]], strict=True))


def timeline(cat: pl.DataFrame) -> str:
    """Station-days with an anomaly per month, one small panel per type (the types have very different counts)."""
    monthly = (cat.with_columns(pl.col("date").dt.truncate("1mo").alias("month_start"))
               .group_by("month_start").agg(pl.col(TYPES).sum()).sort("month_start"))
    fig, axes = plt.subplots(len(TYPES), 1, figsize=(12, 7.5), sharex=True)
    for ax, t in zip(axes, TYPES, strict=True):
        ax.bar(monthly["month_start"], monthly[t], width=25, color=COLOR[t])
        ax.set_title(f"{t} ({int(monthly[t].sum())} station-days)", fontsize=9)
        ax.set_ylabel("days")
    fig.suptitle("When the anomalies occur: station-days with an anomaly per month", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o4_timeline")


def station_type(by_station: pl.DataFrame) -> str:
    """Station-days with an anomaly per station and type."""
    fig = heatmap(by_station.select("station_id", *TYPES), "Where the anomalies occur: station-days per station and type",
                  "station-days")
    return save_figure(fig, "o4_station_type")


def make(r: SimpleNamespace) -> dict[str, str]:
    """Draw every figure of o4 and return the Markdown image links for the report section."""
    return {"timeline": timeline(r.catalogue), "stations": station_type(r.by_station)}
