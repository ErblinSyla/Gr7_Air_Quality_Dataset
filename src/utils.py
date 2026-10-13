"""Funksione të përbashkëta: konfigurimi, leximi/shkrimi i Parquet, tabelat, figurat, lineage dhe seksionet e raportit."""

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
POLLUTANT_TYPE = pl.Enum(NAMES)          # renditja e ndotësve si në config
SERIES = ["station_id", "pollutant"]     # një seri = një stacion × një ndotës
KEY = [*SERIES, "ts_utc"]                # çelësi unik i një matjeje
SEASONS = {12: "dimër", 1: "dimër", 2: "dimër", 3: "pranverë", 4: "pranverë", 5: "pranverë",
           6: "verë", 7: "verë", 8: "verë", 9: "vjeshtë", 10: "vjeshtë", 11: "vjeshtë"}


def by_name(field: str, default=None) -> dict[str, float]:
    """{emri i ndotësit: vlera e fushës} nga config, p.sh. by_name("physical_max")."""
    return {p["name"]: p.get(field, default) for p in POLLUTANTS.values()}


# ------------------------------------------------------------------------------------- Parquet
def write(df: pl.DataFrame, layer: str, name: str, partition_by: str | None = None) -> Path:
    """Shkruan te data/<layer>/<name>: një skedar, ose dosje e particionuar (p.sh. sipas `pollutant`)."""
    if partition_by:
        path = DATA / layer / name
        shutil.rmtree(path, ignore_errors=True)
        df.write_parquet(path, partition_by=partition_by, compression="zstd")
    else:
        path = DATA / layer / f"{name}.parquet"
        df.write_parquet(path, compression="zstd")
    return path


def scan(layer: str, name: str) -> pl.LazyFrame:
    """Lexim lazy i një tabele (skedar ose dosje e particionuar)."""
    folder = DATA / layer / name
    if folder.is_dir():
        return pl.scan_parquet(folder / "**" / "*.parquet", hive_partitioning=True)
    return pl.scan_parquet(DATA / layer / f"{name}.parquet")


# ------------------------------------------------------------------------------------- raportimi
def _round(decimals: int) -> pl.Expr:
    """Rrumbullakon kolonat float dhe e kthen -0.0 në 0.0."""
    x = pl.selectors.float().round(decimals)
    return pl.when(x == 0).then(0.0).otherwise(x).name.keep()


def table(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Ruan tabelën te reports/tables/<name>.csv dhe e kthen (për ta përdorur edhe në raport)."""
    # 6 shifra dhjetore dhe -0.0 → 0.0: mbledhja paralele e float-eve ndryshon vetëm shifrat e fundit (dhe shenjën
    # e zeros), kështu CSV-të mbeten identike
    df.with_columns(_round(6)).write_csv(TABLES / f"{name}.csv")
    return df


def md(df: pl.DataFrame, max_rows: int = 40) -> str:
    """DataFrame → tabelë Markdown."""
    df = df.with_columns(_round(3))
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * df.width
    rows = "\n".join("| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in r) + " |"
                     for r in df.head(max_rows).iter_rows())   # "|" brenda vlerës do ta prishte tabelën
    more = f"\n\n*… {df.height - max_rows} rreshta të tjerë te CSV.*" if df.height > max_rows else ""
    return f"\n\n{head}\n{rows}{more}\n\n"   # rreshta bosh rreth tabelës, që Markdown ta njohë si tabelë


def write_section(part: str, text: str) -> None:
    """Seksioni i raportit për një skriptë (run_all.py i bashkon te report_faza1.md)."""
    (SECTIONS / f"{part}.md").write_text(text.strip() + "\n", encoding="utf-8")


def lineage(script: str, step: str, rows_before: int, rows_after: int, null_before: int, null_after: int) -> None:
    """Shton/zëvendëson një rresht te reports/lineage.csv."""
    path = REPORTS / "lineage.csv"
    row = pl.DataFrame([{"skripta": script, "hapi": step, "rreshta_para": rows_before, "rreshta_pas": rows_after,
                         "null_para": null_before, "null_pas": null_after}])
    if path.exists():
        old = pl.read_csv(path)
        row = pl.concat([old.filter((pl.col("skripta") != script) | (pl.col("hapi") != step)), row])
    row.sort("skripta", maintain_order=True).write_csv(path)


# ------------------------------------------------------------------------------------- figurat
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
    """Ruan figurën dhe kthen lidhjen Markdown për raportin."""
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f"\n![{name}](../figures/{name}.png)\n"


def heatmap(wide: pl.DataFrame, title: str, label: str, annotate: bool = True):
    """Heatmap nga tabelë e gjerë: kolona e parë = etiketat e rreshtave, të tjerat = vlerat."""
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
