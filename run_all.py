"""Ekzekuton p1–p5 me radhë dhe krijon reports/report_faza1.md nga seksionet e tyre."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import polars as pl  # noqa: E402

import p1_mbledhja_tipet_kualiteti  # noqa: E402
import p2_pastrimi_integrimi  # noqa: E402
import p3_mungesat_imputimi  # noqa: E402
import p4_agregimi_mostrimi  # noqa: E402
import p5_vetite_transformimi_reduktimi  # noqa: E402
from utils import REPORTS, SECTIONS, md  # noqa: E402

SCRIPTS = [p1_mbledhja_tipet_kualiteti, p2_pastrimi_integrimi, p3_mungesat_imputimi, p4_agregimi_mostrimi,
           p5_vetite_transformimi_reduktimi]

MAP = [
    ("Para-procesimi për përgatitjen e të dhënave për analizë", "të gjitha", "data/bronze → silver → gold",
     "lineage.csv, decisions.md"),
    ("Mbledhja e të dhënave", "p1", "data/bronze/", "–"),
    ("Definimi i tipeve të dhënave", "p1", "tables/p1_data_dictionary.csv", "–"),
    ("Kualiteti i të dhënave", "p1", "tables/p1_kualiteti_*.csv", "p1_pavlefshme_stacion_vit, p1_nox_raporti_javor"),
    ("Integrimi", "p2", "data/silver/stations.parquet, tables/p2_metadata_konfliktet.csv, p2_nox_*",
     "p2_nox_raporti_para_pas, p2_nderrimi_analizuesve"),
    ("Pastrimi", "p2", "data/silver/measurements_clean/, tables/p2_clean_action_ndotes.csv",
     "p2_histogramet_para_pas"),
    ("Identifikimi dhe strategjia për vlerat e zbrazëta", "p3", "data/gold/hourly_long/, tables/p3_*",
     "p3_harta_mungesave, p3_mungesa_muaj_ore, p3_rmse_metodat, p3_metodat_imputimit"),
    ("Agregimi", "p4", "data/gold/daily, monthly, annual.parquet; tables/p4_*", "p4_pm25_ditor_stacionet"),
    ("Mostrimi", "p4", "data/gold/sample_10pct.parquet, tables/p4_mostrimi_*", "–"),
    ("Zgjedhja e nënbashkësisë së vetive", "p5", "tables/p5_kolonat_e_hequra.csv, p5_zgjedhja_vetive.csv", "–"),
    ("Krijimi i vetive", "p5", "data/gold/features_hourly.parquet, tables/p5_vetite_e_reja.csv", "–"),
    ("Diskretizimi dhe binarizimi", "p5", "tables/p5_kufijte_diskretizimit.csv, p5_eaqi_pm25, p5_tejkalimet",
     "p5_diskretizimi_pm25"),
    ("Transformimi", "p5", "tables/p5_anshmeria.csv", "p5_transformimi"),
    ("Reduktimi i dimensionit", "p5", "tables/p5_pca_*.csv, p5_korrelacioni_spearman.csv",
     "p5_korrelacioni, p5_pca_scree, p5_pca_biplot"),
]


def main() -> None:
    times = []
    for script in SCRIPTS:
        t0 = time.perf_counter()
        script.main()
        times.append({"skripta": script.__name__, "koha_s": round(time.perf_counter() - t0, 1)})

    lineage = pl.read_csv(REPORTS / "lineage.csv")
    sections = "\n\n".join((SECTIONS / f"p{i}.md").read_text(encoding="utf-8") for i in range(1, 6))
    report = f"""# Raporti i Fazës I: Para-procesimi i të dhënave të cilësisë së ajrit në Kosovë

**Dataseti:** EEA Air Quality, vendi XK, E1a (i verifikuar), matje orare, 2019-01-01 → 2024-12-30, 12 stacione,
8 ndotës, 4,261,594 rreshta. Licenca CC BY 4.0 (© European Environment Agency); matjet: rrjeti i IHMK-së.

## 1. Para-procesimi për përgatitjen e të dhënave për analizë
Të dhënat kalojnë nëpër katër shtresa: **raw** (origjinalet, read-only) → **bronze** (tipe të sakta, pa ndryshime)
→ **silver** (të integruara dhe të pastruara) → **gold** (rrjet orar me imputim, agregate, mostër, veti). Parimet:
asnjë rresht s'fshihet (vlera problematike bëhet null dhe arsyeja ruhet); mungesat strukturore ndahen nga ato reale;
"e vlefshme" sipas institucionit kontrollohet përsëri; outliers statistikë s'janë gabime; imputimi është transparent
(`value` dhe `value_imputed` + `impute_method`); gjithçka që mësohet nga të dhënat mësohet vetëm nga 2019–2023.
Çdo vendim, me arsyetim dhe evidencë, është te `decisions.md` (D01–D35, plus tabela e vlerave null me numrat realë).

**Lidhja e kërkesave të detyrës me skriptat, output-et dhe figurat:**
{md(pl.DataFrame(MAP, schema=["Kërkesa", "Skripta", "Output / tabelat", "Figurat"], orient="row"))}
**Lineage** (nga 4,261,594 rreshta raw deri te tabelat gold):
{md(lineage)}
**Koha e ekzekutimit:** {md(pl.DataFrame(times))}

{sections}
"""
    (REPORTS / "report_faza1.md").write_text(report.replace("../figures/", "figures/"), encoding="utf-8")
    print(f"Gati: reports/report_faza1.md ({sum(t['koha_s'] for t in times):.0f} s gjithsej)")


if __name__ == "__main__":
    main()
