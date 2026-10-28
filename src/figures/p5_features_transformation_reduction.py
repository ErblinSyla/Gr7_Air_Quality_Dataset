"""p5 figures: discretization classes, normalization, correlation, PCA scree plot and biplot."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from steps.p5_features_transformation_reduction import PCA_COLS
from utils import CFG, COLORS, INK, NAMES, heatmap, plt, save_figure

E, R = CFG["eaqi"], CFG["reduction"]
BIN_METHODS = {"equal_width": "equal width", "equal_frequency": "equal frequency", "eaqi": "EAQI (concept hierarchy)"}


def discretization(f: pl.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, (k, title) in zip(axes, BIN_METHODS.items(), strict=True):
        c = f[f"pm25_bin_{k}"].drop_nulls().value_counts().sort(f"pm25_bin_{k}")
        ax.bar(c[f"pm25_bin_{k}"], 100 * c["count"] / c["count"].sum(), color=COLORS[0])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("class")
    axes[2].set_xticks(range(6), E["labels"], rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("% of hours")
    fig.suptitle("PM2.5: class distribution by discretization method", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "p5_discretization_pm25")


def normalization(f: pl.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (suffix, title) in zip(axes, [("", "original (different units)"), ("_minmax", "min-max [0, 1]"),
                                          ("_z", "z-score")], strict=True):
        ax.boxplot([f[f"{p}{suffix}"].drop_nulls().to_numpy() for p in NAMES], showfliers=False)
        ax.set_xticks(range(1, len(NAMES) + 1), NAMES, rotation=45, fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Normalization: afterwards every pollutant is on a comparable scale (boxplots without outliers)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "p5_normalization")


def scree(p: SimpleNamespace) -> str:
    ev = p.ev
    fig, ax = plt.subplots(figsize=(7, 3.4))
    n = np.arange(1, len(ev) + 1)
    ax.bar(n, ev, color=COLORS[0], label="variance of the component")
    ax.plot(n, np.cumsum(ev), marker="o", color=COLORS[1], label="cumulative")
    ax.axhline(R["pca_variance"], color=INK, lw=0.8, ls="--")
    ax.set_xticks(n, [f"PC{i}" for i in n])
    ax.set_ylabel("share of variance")
    ax.set_title(f"Scree plot: {p.n_90} components for {int(R['pca_variance'] * 100)}% of the variance")
    ax.legend(loc="center right")
    return save_figure(fig, "p5_pca_scree")


def biplot(p: SimpleNamespace) -> str:
    sc, heat, ev = p.scores, p.heat, p.ev
    order = np.random.default_rng(0).permutation(len(sc))
    lo, hi = np.quantile(sc, 0.002, axis=0), np.quantile(sc, 0.998, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(sc[order, 0], sc[order, 1], s=3, alpha=0.25, linewidths=0,
               c=np.where(heat[order], COLORS[1], COLORS[0]))
    for i, label in enumerate(["outside the heating season", "heating season (Oct–Apr)"]):
        ax.scatter([], [], s=20, color=COLORS[i], label=label)
    comp = p.model.components_[:2]
    room = [hi[k] / c if c > 0 else lo[k] / c for k in range(2) for c in comp[k] if c != 0]
    scale = 0.8 * min(room)
    tips = [(comp[0, j] * scale, comp[1, j] * scale, name) for j, name in enumerate(PCA_COLS)]
    groups: list[list] = []
    for dx, dy, name in sorted(tips, key=lambda t: -np.hypot(t[0], t[1])):
        ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2})
        angle = np.degrees(np.arctan2(dy, dx))
        for grp in groups:
            if abs(grp[0] - angle) < 6:
                grp[3].append(name)
                break
        else:
            groups.append([angle, dx, dy, [name]])
    for _, dx, dy, names in groups:
        ax.text(dx * 1.06, dy * 1.06, ", ".join(names), fontsize=9, color=INK, ha="left" if dx >= 0 else "right",
                va="center", bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 1})
    ax.set_xlim(lo[0] - 0.05 * (hi[0] - lo[0]), hi[0] + 0.05 * (hi[0] - lo[0]))
    ax.set_ylim(lo[1] - 0.05 * (hi[1] - lo[1]), hi[1] + 0.05 * (hi[1] - lo[1]))
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title("Biplot PC1–PC2 (stratified sample from p4)")
    ax.legend(loc="lower right", fontsize=8)
    return save_figure(fig, "p5_pca_biplot")


def make(r: SimpleNamespace) -> dict[str, str]:
    return {
        "classes": discretization(r.f),
        "normalization": normalization(r.f),
        "correlation": save_figure(heatmap(r.corr.with_columns(pl.exclude("pollutant").abs()),
                                           "|Pearson r| between measured pollutants", "|r|"), "p5_correlation"),
        "scree": scree(r.p),
        "biplot": biplot(r.p),
    }
