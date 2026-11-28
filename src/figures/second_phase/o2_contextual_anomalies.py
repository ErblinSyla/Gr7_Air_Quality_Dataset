"""o2 figures: F1 for every context granularity and method, and the real contextual anomalies against their context."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o2_contextual_anomalies import CONTEXTS
from utils import COLORS, GRID, INK_2, plt, save_figure


def context_f1(evaluation: pl.DataFrame) -> str:
    """F1 of the contextual robust z-score and of LOF (every n_neighbors) for every context, from none to finest."""
    order = list(CONTEXTS)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    settings = evaluation.select("method", "param").unique().sort("method", "param")
    lof_colors = iter([COLORS[1], COLORS[3], COLORS[6]])
    for method, k in settings.iter_rows():
        d = evaluation.filter((pl.col("method") == method) & (pl.col("param") == k))
        d = d.sort(pl.col("context").replace_strict({c: j for j, c in enumerate(order)}))
        label = "robust z (largest |residual|)" if method == "robust z" else f"LOF on residuals, k = {k}"
        ax.errorbar(range(len(order)), d["f1"], yerr=d["f1_std"], marker="o", ms=5, lw=2, capsize=3,
                    color=INK_2 if method == "robust z" else next(lof_colors), label=label)
    ax.set_xticks(range(len(order)), order)
    ax.set(xlabel="context (contextual attributes)", ylabel="F1", ylim=(0, None))
    ax.set_title("Contextual anomalies: invisible without context (F1 ≈ 0), best with station × month")
    ax.legend()
    return save_figure(fig, "o2_context_f1")


def value_vs_usual(flags: pl.DataFrame) -> str:
    """Every real station-day: its value against the usual value of its context, for the most unusual pollutant.

    Contextual anomalies are far from the diagonal (much higher or much lower than usual for the station and month)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    normal, alarm = flags.filter(~pl.col("contextual_anomaly")), flags.filter("contextual_anomaly")
    ax.scatter(normal["usual_value"], normal["value"], s=4, color=GRID, label="normal")
    ax.scatter(alarm["usual_value"], alarm["value"], s=18, marker="x", color=COLORS[7], label="contextual anomaly")
    top = max(flags["value"].max(), flags["usual_value"].max())
    ax.plot([0.01, top], [0.01, top], color=INK_2, lw=0.8, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="usual value for the station and month (median)",
           ylabel="measured daily mean")
    ax.set_title("Value of the most unusual pollutant vs its usual value (log scales)")
    ax.legend(markerscale=2)
    return save_figure(fig, "o2_value_vs_usual")


def make(r: SimpleNamespace) -> dict[str, str]:
    """Draw every figure of o2 and return the Markdown image links for the report section."""
    return {"context": context_f1(r.evaluation), "usual": value_vs_usual(r.flags)}
