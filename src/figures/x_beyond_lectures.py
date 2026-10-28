"""Appendix figures: PM2.5 before and after Yeo-Johnson; RMSE of the imputation methods by gap length."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from utils import COLORS, plt, save_figure


def yeo_johnson(x: np.ndarray, y: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, v, title in ((axes[0], x, "original (µg/m³)"), (axes[1], y, "after Yeo-Johnson")):
        ax.hist(v, bins=60, color=COLORS[0])
        ax.set_title(title, fontsize=10)
    axes[0].set_xlim(0, np.quantile(x, 0.995))
    fig.suptitle("PM2.5 distribution, 2019–2023: the long right tail disappears", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "x_yeo_johnson_pm25")


def rmse_by_gap(rmse) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for color, method in zip(COLORS, ("attribute_mean", "class_mean", "linear"), strict=False):
        ax.plot(rmse["gap_h"], rmse[method], marker="o", color=color, label=method.replace("_", " "))
    ax.set_xlabel("gap length (hours)")
    ax.set_ylabel("RMSE (mean over pollutants)")
    ax.legend(frameon=False)
    fig.suptitle("Imputation error by gap length", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "x_rmse_by_gap")


def make(r: SimpleNamespace) -> dict[str, str]:
    return {"yj": yeo_johnson(r.pm25, r.pm25_yj), "rmse": rmse_by_gap(r.rmse)}
