"""o1 figures: parameter analysis of every method (k, metric, K, threshold) and the outliers found in the real data."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o1_outlier_detection import METHODS, name
from utils import CFG, COLORS, GRID, INK_2, heatmap, plt, save_figure

O = CFG["outliers"]
COLOR = {"KNN": COLORS[0], "LOF": COLORS[1], "K-Means": COLORS[2], "univariate": INK_2}
OUTLIER = COLORS[7]


def f1_line(ax, d: pl.DataFrame, label: str, color: str) -> None:
    """Mean F1 over the injections as a line, with the standard deviation between them as error bars."""
    ax.errorbar(d["k"], d["f1"], yerr=d["f1_std"], marker="o", ms=5, lw=2, capsize=3, color=color, label=label)


def knn_k(knn: pl.DataFrame, best: dict) -> str:
    """F1 of the KNN distance for every k, Euclidean versus Manhattan distance."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, metric in enumerate(O["knn_metrics"]):
        f1_line(ax, knn.filter(pl.col("metric") == metric).sort("k"), metric, COLORS[i * 6])
    ax.axvline(best["KNN"]["k"], color=INK_2, ls="--", lw=0.8)
    ax.set(xlabel="k (number of neighbours)", ylabel="F1", ylim=(0, None))
    ax.set_title(f"KNN distance: F1 falls as k grows; Euclidean > Manhattan (best k = {best['KNN']['k']})")
    ax.legend(title="distance")
    return save_figure(fig, "o1_knn_k")


def lof_k(lof: pl.DataFrame, best: dict) -> str:
    """F1 of LOF for every n_neighbors."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    f1_line(ax, lof.sort("k"), "LOF", COLOR["LOF"])
    ax.axvline(best["LOF"]["k"], color=INK_2, ls="--", lw=0.8)
    ax.set(xlabel="n_neighbors", ylabel="F1", ylim=(0, None))
    ax.set_title(f"LOF: stable F1 for n_neighbors 10–50, too few or too many neighbours are worse "
                 f"(best = {best['LOF']['k']})")
    return save_figure(fig, "o1_lof_k")


def kmeans_k(km: pl.DataFrame, best: dict) -> str:
    """K-Means: inertia (elbow method) and F1 for every K, in two panels (two different scales, no dual axis)."""
    km = km.sort("k")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(km["k"], km["inertia"], marker="o", ms=5, lw=2, color=COLOR["K-Means"])
    a.set(xlabel="K (number of clusters)", ylabel="inertia (sum of squared distances)", title="Elbow method")
    f1_line(b, km, "K-Means", COLOR["K-Means"])
    b.axvline(best["K-Means"]["k"], color=INK_2, ls="--", lw=0.8)
    b.set(xlabel="K (number of clusters)", ylabel="F1", ylim=(0, None),
          title=f"F1 grows with K (best K = {best['K-Means']['k']})")
    fig.suptitle("K-Means: distance to the nearest centroid", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o1_kmeans_k")


def threshold(th: pl.DataFrame) -> str:
    """Precision, recall and F1 of the best setting of every method for every threshold percentile."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, measure in zip(axes, ["precision", "recall", "f1"], strict=True):
        for method in [*METHODS, "univariate"]:
            d = th.filter(pl.col("method") == method).sort("percentile")
            ax.plot(d["percentile"], d[measure], marker="o", ms=5, lw=2, color=COLOR[method], label=method)
        ax.axvline(O["threshold_percentile"], color=INK_2, ls="--", lw=0.8)
        ax.set(xlabel="threshold percentile", title=measure.capitalize() if measure != "f1" else "F1")
    axes[0].legend()
    fig.suptitle("A higher threshold: fewer, more certain outliers (precision ↑, recall ↓)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o1_threshold")


def detected_pca(flags: pl.DataFrame, best: dict) -> str:
    """The real station-days on the first two principal components, with the outliers of every method in red."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS, strict=True):
        out = flags[f"outlier_{name(method)}"] == 1
        ax.scatter(flags.filter(~out)["pc1"], flags.filter(~out)["pc2"], s=4, color=GRID, alpha=0.8, label="normal")
        ax.scatter(flags.filter(out)["pc1"], flags.filter(out)["pc2"], s=18, color=OUTLIER, marker="x",
                   label="outlier")
        k = best[method]["k"]
        ax.set_title(f"{method} ({'K' if method == 'K-Means' else 'k'} = {k}): {int(out.sum())} outliers")
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=2)
    fig.suptitle("Outliers of the real data (top 1% of every score) on the first two principal components",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o1_detected_pca")


def station_month(flags: pl.DataFrame) -> str:
    """Number of outlier days (flagged by ≥ 2 methods) per station and month."""
    months = [str(m) for m in range(1, 13)]
    wide = (flags.filter(pl.col("n_methods") >= 2)
            .with_columns(pl.col("month").cast(pl.String))
            .group_by("station_id", "month").len()
            .pivot(on="month", index="station_id", values="len")
            .join(flags.select("station_id").unique(), on="station_id", how="right"))
    wide = wide.with_columns([pl.lit(0).alias(m) for m in months if m not in wide.columns]).fill_null(0)
    fig = heatmap(wide.select("station_id", *months).sort("station_id"),
                  "Outlier days (≥ 2 methods) per station and month: winter and the end of March", "days")
    fig.axes[0].set_xlabel("month")
    return save_figure(fig, "o1_outliers_station_month")


def make(r: SimpleNamespace) -> dict[str, str]:
    """Draw every figure of o1 and return the Markdown image links for the report section."""
    best = {row["method"]: row for row in r.best.iter_rows(named=True)}
    return {"knn": knn_k(r.knn, best), "lof": lof_k(r.lof, best), "kmeans": kmeans_k(r.kmeans, best),
            "threshold": threshold(r.threshold), "pca": detected_pca(r.flags, best),
            "station_month": station_month(r.flags)}
