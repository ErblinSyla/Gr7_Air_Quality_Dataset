"""o3 figures: parameter analysis of the three temporal detectors and examples of a real spike and a stuck sensor."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import polars as pl

from utils import CFG, COLORS, INK_2, plt, save_figure

O = CFG["outliers"]
MEASURES = {"precision": COLORS[0], "recall": COLORS[1], "f1": COLORS[2]}


def measures(ax, d: pl.DataFrame, x: list, title: str, xlabel: str) -> None:
    """Precision, recall and F1 of one detector against its parameter."""
    for measure, color in MEASURES.items():
        ax.plot(x, d[measure], marker="o", ms=5, lw=2, color=color, label="F1" if measure == "f1" else measure)
    ax.set(title=title, xlabel=xlabel, ylim=(0, 1.02))


def evaluation(r: SimpleNamespace) -> str:
    """Three panels: spike threshold t, minimum run length L of the stuck detector, drift window W and threshold t."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    measures(axes[0], r.spike, O["spike_t"], "Isolated spike", "threshold t (robust std of hourly changes)")
    measures(axes[1], r.stuck, O["stuck_min_len"], "Stuck sensor", "minimum run length L (hours)")
    for i, w in enumerate(O["drift_window"]):
        d = r.drift.filter(pl.col("param").str.starts_with(f"W = {w} d"))
        axes[2].plot(O["drift_t"], d["f1"], marker="o", ms=5, lw=2, color=COLORS[[0, 1, 6][i]], label=f"W = {w} d")
    axes[2].set(title="Drift (F1)", xlabel="threshold t (robust z of the shift)", ylim=(0, 1.02))
    axes[0].legend()
    axes[2].legend()
    fig.suptitle("Temporal detectors: stuck sensors are found reliably, spikes partly, drift practically not",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o3_evaluation")


def examples(r: SimpleNamespace, hourly: pl.DataFrame) -> str:
    """The strongest real spike and the longest real stuck run, each with two days of context."""
    spike = r.top_spikes.row(0, named=True)
    stuck = r.top_stuck.row(0, named=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    for ax, case, when, title in ((axes[0], spike, spike["ts_utc"], "spike"), (axes[1], stuck, stuck["start"],
                                                                              "stuck sensor")):
        d = hourly.filter((pl.col("station_id") == case["station_id"]) & (pl.col("pollutant") == case["pollutant"])
                          & pl.col("ts_utc").is_between(when - timedelta(days=1), when + timedelta(days=1)))
        ax.plot(d["ts_utc"], d["v"], lw=1.5, marker="o", ms=3, color=COLORS[0])
        ax.axvline(when, color=INK_2, ls="--", lw=0.8)
        ax.set_title(f"{title}: {case['station_id']}, {case['pollutant']}, {when:%Y-%m-%d %H:%M} UTC")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return save_figure(fig, "o3_examples")


def make(r: SimpleNamespace) -> dict[str, str]:
    """Draw every figure of o3 and return the Markdown image links for the report section."""
    return {"evaluation": evaluation(r), "examples": examples(r, r.hourly)}
