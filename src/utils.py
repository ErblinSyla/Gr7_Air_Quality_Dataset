"""Shared helpers: configuration, Parquet I/O, report tables, figures, lineage and report sections."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

DATA = ROOT / CFG["paths"]["data"]
REPORTS = ROOT / CFG["paths"]["reports"]
TABLES, FIGURES, SECTIONS = REPORTS / "tables", REPORTS / "figures", REPORTS / "sections"
for folder in (DATA / "bronze", DATA / "silver", DATA / "gold", TABLES, FIGURES, SECTIONS):
    folder.mkdir(parents=True, exist_ok=True)

POLLUTANTS = {int(code): p for code, p in CFG["pollutants"].items()}
NAMES = [p["name"] for p in POLLUTANTS.values()]
POLLUTANT_TYPE = pl.Enum(NAMES)
SERIES = ["station_id", "pollutant"]
KEY = [*SERIES, "ts_utc"]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def by_name(field: str, default=None) -> dict[str, float]:
    return {p["name"]: p.get(field, default) for p in POLLUTANTS.values()}


def write(df: pl.DataFrame, layer: str, name: str, partition_by: str | None = None) -> Path:
    if partition_by:
        path = DATA / layer / name
        shutil.rmtree(path, ignore_errors=True)
        df.write_parquet(path, partition_by=partition_by, compression="zstd")
    else:
        path = DATA / layer / f"{name}.parquet"
        df.write_parquet(path, compression="zstd")
    return path


def scan(layer: str, name: str) -> pl.LazyFrame:
    folder = DATA / layer / name
    if folder.is_dir():
        return pl.scan_parquet(folder / "**" / "*.parquet", hive_partitioning=True)
    return pl.scan_parquet(DATA / layer / f"{name}.parquet")


def _round(decimals: int) -> pl.Expr:
    x = pl.selectors.float().round(decimals)
    return pl.when(x == 0).then(0.0).otherwise(x).name.keep()


def table(df: pl.DataFrame, name: str) -> pl.DataFrame:
    df.with_columns(_round(6)).write_csv(TABLES / f"{name}.csv")
    return df


def md(df: pl.DataFrame, max_rows: int = 40) -> str:
    df = df.with_columns(_round(3))
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * df.width
    rows = "\n".join("| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in r) + " |"
                     for r in df.head(max_rows).iter_rows())
    more = f"\n\n*… {df.height - max_rows} more rows in the CSV.*" if df.height > max_rows else ""
    return f"\n\n{head}\n{rows}{more}\n\n"


def write_section(part: str, text: str) -> None:
    (SECTIONS / f"{part}.md").write_text(text.strip() + "\n", encoding="utf-8")


def lineage(script: str, step: str, rows_before: int, rows_after: int, null_before: int, null_after: int) -> None:
    path = REPORTS / "lineage.csv"
    row = pl.DataFrame([{"script": script, "step": step, "rows_before": rows_before, "rows_after": rows_after,
                         "nulls_before": null_before, "nulls_after": null_after}])
    if path.exists():
        old = pl.read_csv(path)
        if old.columns == row.columns:
            row = pl.concat([old.filter((pl.col("script") != script) | (pl.col("step") != step)), row])
    row.sort("script", maintain_order=True).write_csv(path)


SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9, "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=COLORS),
})


def save_figure(fig, name: str) -> str:
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f"\n![{name}](../figures/{name}.png)\n"


def heatmap(wide: pl.DataFrame, title: str, label: str, annotate: bool = True):
    rows, cols = wide[wide.columns[0]].cast(pl.String).to_list(), wide.columns[1:]
    v = np.array([[np.nan if x is None else x for x in r] for r in wide.select(cols).rows()], dtype=float)
    size = (max(6, 0.9 * len(cols)), max(3, 0.4 * len(rows))) if annotate else (12, max(4, 0.12 * len(rows)))
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(v, cmap=SEQUENTIAL, aspect="auto", vmin=0)
    ax.set_yticks(range(len(rows)), rows, fontsize=8 if annotate else 6)
    ax.set_xticks(range(len(cols)), cols) if annotate else ax.set_xticks(
        range(0, len(cols), 6), cols[::6], rotation=45, ha="right")
    ax.grid(False)
    if annotate:
        top = np.nanmax(v) or 1
        for i, j in zip(*np.where(~np.isnan(v)), strict=True):
            ax.text(j, i, f"{v[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v[i, j] > 0.55 * top else INK)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    return fig
