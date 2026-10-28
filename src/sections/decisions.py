"""Decision log of the report: every important preprocessing choice, the step where it is made and the reason."""

from __future__ import annotations

import polars as pl

from utils import CFG, md

M, A, S, T = CFG["missing"], CFG["aggregation"], CFG["sampling"], CFG["time"]

DECISIONS = [
    ("p1", "No row is ever deleted; a wrong value becomes null and the reason is stored",
     "every change stays traceable and can be undone"),
    ("p1", "Every value is checked again, even if the institution marked it as valid",
     "believability: the validity flag alone does not catch every impossible value"),
    ("p2", "NOx is converted from ppb to µg/m³ where the NOx/NO2/NO ratio shows ppb",
     "value conflict between sources (different units for the same attribute)"),
    ("p2", "CO and SO2 before the analyzer change of November 2019 are flagged, not deleted",
     "suspicious, but there is no proof that they are wrong; analyses can exclude them"),
    ("p2", "Statistical outliers are kept",
     "high winter PM values are real pollution episodes, not errors (lecture: outliers may be legitimate)"),
    ("p3", "Structural gaps (not measured, outside the period) are separated from real missing values",
     "a station that never measures a pollutant has nothing to impute"),
    ("p3", f"Imputation order: physical formula (NOx), linear interpolation for inside gaps ≤ {M['max_gap_h']}h, "
           "otherwise null",
     "linear interpolation has the smallest error in the evaluation; longer gaps would be invented data"),
    ("p3", "Imputed values are stored next to the measured ones (`value`, `value_imputed`, `impute_method`)",
     "transparency: every analysis can choose measured or imputed values"),
    ("p4", f"A day needs ≥ {A['min_hours_per_day']} valid hours, a month ≥ {A['min_days_per_month_frac']:.0%} "
           f"valid days, a year ≥ {A['min_annual_coverage']:.0%} coverage",
     "a mean of a few hours does not represent a day (EU coverage rules)"),
    ("p4", f"Stratified sample {S['fraction']:.0%} by {' × '.join(S['strata'])}",
     "small strata are not left out (lecture: SRS performs poorly under skew)"),
    ("p5", "NOx is excluded from PCA", "derivable data: NOx = NO2 + 1.53·NO"),
    ("p5", "EAQI classes are chosen for PM2.5 discretization",
     "concept hierarchy with health meaning, comparable with the rest of Europe"),
    ("p5", "Z-score (not min-max) is used before PCA", "min-max is dominated by outliers"),
    ("p5", f"All parameters are learned from 2019–{T['fit_until_year']} only",
     f"{T['fit_until_year'] + 1} stays unseen for later analyses (no data leakage)"),
]


def render() -> str:
    log = pl.DataFrame(DECISIONS, schema=["Step", "Decision", "Reason"], orient="row")
    return f"""## 7. Decision log
The most important preprocessing decisions in one place (details in the sections above):
{md(log.with_row_index("No.", offset=1))}"""
