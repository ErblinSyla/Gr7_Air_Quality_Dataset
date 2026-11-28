"""o1 figures: parameter analysis of every point method (k, metric, K, gamma/nu, threshold) and the real outliers."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from steps.second_phase.o1_point_anomalies import BASELINE, METHODS, name
from utils import CFG, COLORS, GRID, INK_2, plt, save_figure

O = CFG["outliers"]
COLOR = {"KNN": COLORS[0], "LOF": COLORS[1], "K-Means": COLORS[2], "Mahalanobis": COLORS[3],
         "One-class SVM": COLORS[6], BASELINE: INK_2}
OUTLIER = COLORS[7]


def f1_line(ax, d: pl.DataFrame, label: str, color: str, ls: str = "-") -> None:
    """Mean F1 over the injections as a line, with the standard deviation between them as error bars."""
    d = d.sort("param")
    ax.errorbar(d["param"], d["f1"], yerr=d["f1_std"], marker="o", ms=5, lw=2, ls=ls, capsize=3, color=color,
                label=label)


def best_line(ax, best: dict, method: str) -> None:
    """Dashed vertical line at the chosen parameter value."""
    ax.axvline(best[method]["param"], color=INK_2, ls="--", lw=0.8)


def knn_k(knn: pl.DataFrame, best: dict) -> str:
    """F1 of the KNN distance for every k: colour = distance metric, line style = mean vs k-th neighbour distance."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, metric in enumerate(O["knn_metrics"]):
        for score, ls in (("mean", "-"), ("kth", "--")):
            label = f"{metric}, {'mean of k' if score == 'mean' else 'k-th neighbour'}"
            f1_line(ax, knn.filter(pl.col("variant") == f"{metric}, {score}"), label, COLORS[i * 6], ls)
    best_line(ax, best, "KNN")
    ax.set(xlabel="k (number of neighbours)", ylabel="F1", ylim=(0, None))
    ax.set_title(f"KNN distance: F1 falls as k grows; Euclidean > Manhattan, mean ≥ k-th "
                 f"(best k = {best['KNN']['param']:.0f})")
    ax.legend(title="distance, score")
    return save_figure(fig, "o1_knn_k")


def lof_k(lof: pl.DataFrame, best: dict) -> str:
    """F1 of LOF for every n_neighbors."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    f1_line(ax, lof, "LOF", COLOR["LOF"])
    best_line(ax, best, "LOF")
    ax.set(xlabel="n_neighbors", ylabel="F1", ylim=(0, None))
    ax.set_title(f"LOF: stable F1 for n_neighbors 10–50, too few or too many neighbours are worse "
                 f"(best = {best['LOF']['param']:.0f})")
    return save_figure(fig, "o1_lof_k")


def kmeans_k(km: pl.DataFrame, best: dict) -> str:
    """K-Means: inertia (elbow method) and F1 for every K, in two panels (two different scales, no dual axis)."""
    km = km.sort("param")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(km["param"], km["inertia"], marker="o", ms=5, lw=2, color=COLOR["K-Means"])
    a.set(xlabel="K (number of clusters)", ylabel="inertia (sum of squared distances)", title="Elbow method")
    f1_line(b, km, "K-Means", COLOR["K-Means"])
    best_line(b, best, "K-Means")
    b.set(xlabel="K (number of clusters)", ylabel="F1", ylim=(0, None),
          title=f"F1 grows with K (best K = {best['K-Means']['param']:.0f})")
    fig.suptitle("K-Means: distance to the nearest centroid", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o1_kmeans_k")


def ocsvm_gamma(svm: pl.DataFrame, best: dict) -> str:
    """F1 of the one-class SVM for every gamma (log scale), one line per nu."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, nu in enumerate(O["ocsvm_nu"]):
        f1_line(ax, svm.filter(pl.col("variant") == f"nu={nu}"), f"ν = {nu}", COLORS[6 - 5 * i])
    best_line(ax, best, "One-class SVM")
    ax.set_xscale("log")
    ax.set(xlabel="gamma (log scale)", ylabel="F1", ylim=(0, None))
    ax.set_title("One-class SVM: gamma and nu must fit together; a tight boundary needs a larger ν")
    ax.legend()
    return save_figure(fig, "o1_ocsvm_gamma")


def threshold(th: pl.DataFrame) -> str:
    """Precision, recall and F1 of the best setting of every method for every threshold percentile."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, measure in zip(axes, ["precision", "recall", "f1"], strict=True):
        for method in [*METHODS, BASELINE]:
            d = th.filter(pl.col("method") == method).sort("percentile")
            ax.plot(d["percentile"], d[measure], marker="o", ms=5, lw=2, color=COLOR[method], label=method)
        ax.axvline(O["threshold_percentile"], color=INK_2, ls="--", lw=0.8)
        ax.set(xlabel="threshold percentile", title="F1" if measure == "f1" else measure.capitalize())
    axes[0].legend()
    fig.suptitle("A higher threshold: fewer, more certain outliers (precision ↑, recall ↓)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o1_threshold")


def scatter(ax, flags: pl.DataFrame, out: pl.Series, title: str) -> None:
    """Station-days on PC1/PC2: normal points in grey, outliers as red crosses."""
    ax.scatter(flags.filter(~out)["pc1"], flags.filter(~out)["pc2"], s=4, color=GRID, alpha=0.8, label="normal")
    ax.scatter(flags.filter(out)["pc1"], flags.filter(out)["pc2"], s=18, color=OUTLIER, marker="x", label="outlier")
    ax.set_title(title)


def detected_pca(flags: pl.DataFrame, rule: str) -> str:
    """The real station-days on the first two principal components: the outliers of every method and the decision."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS, strict=False):
        out = flags[f"outlier_{name(method)}"] == 1
        scatter(ax, flags, out, f"{method}: {int(out.sum())} outliers")
    final = flags["point_anomaly"]
    scatter(axes.flat[-1], flags, final, f"Final decision ({rule}): {int(final.sum())} point anomalies")
    for ax in axes[-1]:
        ax.set_xlabel("PC1")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2")
    axes[0, 0].legend(markerscale=2)
    fig.suptitle("Outliers of the real data (top 1% of every score) on the first two principal components",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "o1_detected_pca")


def make(r: SimpleNamespace) -> dict[str, str]:
    """Draw every figure of o1 and return the Markdown image links for the report section."""
    best = {row["method"]: row for row in r.best.iter_rows(named=True)}
    return {"knn": knn_k(r.knn, best), "lof": lof_k(r.lof, best), "kmeans": kmeans_k(r.kmeans, best),
            "ocsvm": ocsvm_gamma(r.ocsvm, best), "threshold": threshold(r.threshold),
            "pca": detected_pca(r.flags, r.rule)}
