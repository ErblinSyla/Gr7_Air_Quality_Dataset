"""Summary at the start of the phase II report (1–2 pages): approach, chosen methods, key findings and limitations.

It is written after o1–o4 have run and reads only their result tables, so every number matches the detailed sections.
"""

from __future__ import annotations

import polars as pl

from utils import CFG, TABLES, md, scan

O = CFG["outliers"]
DUST_PAPER = "https://acp.copernicus.org/articles/22/4047/2022/"


def read(name: str) -> pl.DataFrame:
    """One result table of phase II."""
    return pl.read_csv(TABLES / f"{name}.csv")


def render() -> str:
    """Markdown text of the summary; every number comes from the tables of o1–o4."""
    best = {r["method"]: r for r in read("o1_best_settings").iter_rows(named=True)}
    rules = read("o1_combination_rules").sort("f1", descending=True)
    majority = rules.filter(pl.col("rule") == "≥ 3 of 5 methods").row(0, named=True)
    by_type = {r["type"]: r for r in read("o1_recall_by_type").iter_rows(named=True)}
    sens = read("o1_threshold_sensitivity")
    leaders = sens.sort("pr_auc", descending=True).group_by("injected_pct").first()["method"].unique().to_list()
    lof15 = read("o1_lof_share").filter(pl.col("lof_above") == 1.5).row(0, named=True)
    o2 = read("o2_evaluation")
    ctx = o2.sort("f1", descending=True).row(0, named=True)
    ctx_month = o2.filter((pl.col("method") == "LOF") & (pl.col("context") == "station × month"))["f1"].max()
    none = read("o2_robust_z_evaluation").filter(pl.col("context") == "none").row(0, named=True)
    o3 = {r["fault"]: r for r in read("o3_best_settings").iter_rows(named=True)}
    check = {r["fault"]: r for r in read("o3_phase1_check").iter_rows(named=True)}
    hours = read("o3_anomalies_by_pollutant").select(pl.col("spike_hours", "stuck_hours").sum()).row(0, named=True)
    days = dict(read("o4_station_days_by_type").iter_rows())
    alarms = scan("gold", "anomalies_contextual").filter("contextual_anomaly").collect()
    low_share = alarms.select((pl.col("value") < pl.col("usual_value")).mean()).item()
    repeated = read("o2_repeated_anomalies")
    rep = repeated.row(0, named=True) if repeated.height else None
    rep_text = "" if rep is None else (f" Repeated alarms point to instruments: e.g. {rep['days']} days of unusual"
                                        f" {rep['context_driver']} at {rep['station_id']} in {rep['month']}.")
    episodes = read("o1_outlier_episodes")
    dust = episodes.filter(pl.col("date").cast(pl.String).str.starts_with("2020-03-2"))
    lof = best["LOF"]
    overview = pl.DataFrame([
        {"type of anomaly": "point (station-day)", "chosen method": f"LOF, n_neighbors = {lof['param']:.0f}",
         "F1": lof["f1"], "PR-AUC": lof["pr_auc"], "found in the real data": f"{days['point']} station-days"},
        {"type of anomaly": "contextual (station, month, weather)",
         "chosen method": f"{ctx['method']} on residuals, {ctx['context']}", "F1": ctx["f1"], "PR-AUC": ctx["pr_auc"],
         "found in the real data": f"{days['contextual']} station-days"},
        {"type of anomaly": "collective: isolated spike", "chosen method": f"spike detector, {o3['spike']['param']}",
         "F1": o3["spike"]["f1"], "PR-AUC": None, "found in the real data": f"{hours['spike_hours']:,} hours"},
        {"type of anomaly": "collective: stuck sensor", "chosen method": f"run length, {o3['stuck sensor']['param']}",
         "F1": o3["stuck sensor"]["f1"], "PR-AUC": None, "found in the real data": f"{hours['stuck_hours']:,} hours"},
        {"type of anomaly": "collective: drift", "chosen method": "shift against the other stations",
         "F1": o3["drift"]["f1"], "PR-AUC": None, "found in the real data": "not detectable (not flagged)"},
    ])
    dust_text = "" if dust.is_empty() else (
        f" The strongest regional event, {dust['stations'].max()} stations on the same day at the end of March 2020 with"
        " a low PM2.5/PM10 ratio, is the Central Asian dust incursion documented in the literature"
        f" ([Tositti et al., 2022]({DUST_PAPER})).")
    return f"""
## Summary

**Goal and data.** Detect – not remove – anomalies in the cleaned air-quality measurements of Kosovo (12 stations,
2019–2024): station-day means for point and contextual anomalies, hourly series for collective anomalies.

**Approach.** The three types of anomalies of the lecture are detected separately, each with the representation of the
data in which it is visible. Because the data has no labels, every method and parameter was chosen with known anomalies
injected into a copy of the data (as in the exercises), measured with precision, recall, F1 and the threshold-free
PR-AUC; the chosen settings were then applied to the real data.
{md(overview)}
**Key findings**
1. **LOF is the best point method** (F1 {lof["f1"]:.2f}, PR-AUC {lof["pr_auc"]:.2f}; a random score has PR-AUC
   {O["injected_share"]:.2f}), ahead of Mahalanobis ({best["Mahalanobis"]["f1"]:.2f}), the one-class SVM
   ({best["One-class SVM"]["f1"]:.2f}), KNN ({best["KNN"]["f1"]:.2f}) and K-Means ({best["K-Means"]["f1"]:.2f}). Isolation
   Forest, the most common method in practice, reaches only {best["Isolation Forest"]["f1"]:.2f}, close to the univariate
   z-score ({best["z-score"]["f1"]:.2f}): it isolates single extreme values, while the anomalies here are unusual
   combinations of normal values.
2. **The methods are complementary, but majority voting is worse.** Mahalanobis finds a sensor reading zero best
   (recall {by_type["sensor at zero"]["Mahalanobis"]:.2f} vs LOF {by_type["sensor at zero"]["LOF"]:.2f}), LOF the impossible
   PM2.5 > PM10 ({by_type["PM2.5 > PM10"]["LOF"]:.2f}); a vote of ≥ 3 of 5 methods drops to F1 {majority["f1"]:.2f}
   because the weaker methods outvote LOF. Final rule: **{rules["rule"][0]}**.
3. **The choice does not depend on the threshold.** The method ranking is the same with
   {", ".join(f"{100 * x:g}%" for x in O["sensitivity_shares"])} injected anomalies (best method:
   {", ".join(leaders)}), and the real data alone has {lof15["pct"]:.1f}% of days above the usual LOF limit 1.5, so the
   99th-percentile threshold (top 1%) fits the data.
4. **Contextual anomalies need context.** Without it the score is random (ROC-AUC {none["roc_auc"]:.2f}); comparing every
   day with its station, month and the other stations of the same day (a weather proxy) raises F1 from
   {ctx_month:.2f} (station × month) to {ctx["f1"]:.2f}. {low_share:.0%} of the real contextual anomalies are values
   *below* the expected level – never visible as global outliers, but a typical sign of an analyzer problem.{rep_text}
5. **Temporal faults:** stuck sensors are found reliably – the detector finds
   {check["stuck sensor"]["share_found"]:.0%} of the stuck hours that phase I removed with fixed rules – spikes partly
   (F1 {o3["spike"]["f1"]:.2f}); a slow drift cannot be separated from natural differences without a reference
   instrument (F1 {o3["drift"]["f1"]:.2f}) and is reported as a limitation.
6. **The anomalies are both real events and faults.**{dust_text} Other examples: an SO2 episode at one station
   (Mitrovica, May 2020), single-hour SO2 and NO peaks near industry and traffic, and runs of NO2 = 0 that were created
   when phase I set small negative readings to zero. That is why nothing is removed: the reason for every anomaly
   decides whether it is an error or an important pollution episode.

**Limitations.** The injected anomalies are simplified models of real faults; there are no confirmed labels for the
real data; the data set has no meteorological data (the other stations are used as a proxy); drift needs a reference
instrument.
"""
