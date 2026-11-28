"""o4 report section: catalogue of all anomalies, overlap of the types, where and when they occur, decisions."""

from __future__ import annotations

from types import SimpleNamespace

from utils import CFG, md

O = CFG["outliers"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    """Markdown text of the summary section; every number comes from the results of the step."""
    counts = dict(r.counts.iter_rows())
    peak_year = r.by_year.sort("point", descending=True).row(0, named=True)
    return f"""
## 4. Summary: catalogue of all anomalies

Every anomaly type was detected with the representation and the method that fit it, and every choice was checked with
injected anomalies of that type. The overview of the chosen methods (`tables/o4_methods_overview.csv`):
{md(r.overview)}
The F1 values are not comparable between the types (different data and different injected faults), but they show how
reliably each type can be detected: stuck sensors very well, point anomalies well, spikes and contextual anomalies
partly, drift not.

### 4.1 One catalogue
All results are combined into one catalogue (`data/gold/anomaly_catalogue.parquet`): **{r.n_days:,} station-days**
have at least one anomaly, {r.n_multi} of them more than one type.
{md(r.counts)}
{md(r.combinations)}
The types overlap little: each representation finds its own anomalies, which is why all of them are needed to find
"any" anomaly. Spikes are the most frequent ({counts["spike"]:,} station-days) because a single hour is enough.

### 4.2 Where and when
{figs["timeline"]}
{figs["stations"]}
{md(r.by_year)}
Point anomalies are most frequent in {peak_year["year"]} ({peak_year["point"]} station-days, among them the dust
episode of March 2020 and the SO2 episode in Mitrovica in May 2020, section 1.12). The stuck-sensor alarms
are concentrated at Brezovica (XK0007A, NO2 at zero) and Palaj (XK0010A), which shows that a few instruments cause
most of the technical faults.

### 4.3 Decisions of the anomaly detection
| Decision | Reason |
|---|---|
| Three types of anomalies (point, contextual, collective) with their own representation | the lecture's structure of anomalies; one representation cannot show all types |
| Every choice evaluated with injected anomalies of its type ({len(O["eval_seeds"])} seeds) | the real data has no labels; injection gives precision, recall and F1 |
| Point anomalies: LOF, threshold {O["threshold_percentile"]:g}th percentile; Mahalanobis as an optional addition | best F1 of all methods and combination rules; Mahalanobis adds the zero readings |
| Majority voting is not used | the weaker methods outvote LOF (lower F1) |
| Isolation Forest only as a comparison | not in the lectures and clearly weaker than LOF on this data |
| Threshold: 99th percentile | the method ranking does not depend on the injected share; the real data has ≈ 0.5–1% clear outliers (LOF > 1.5) |
| Contextual anomalies: LOF on the residuals of the station × month profile after removing the regional level of the day | without context the F1 is ≈ 0; the regional level (weather proxy) gives the best context |
| Temporal anomalies: isolated-spike and stuck-sensor detectors; drift not flagged | drift cannot be separated from natural differences without a reference instrument |
| Static thresholds (\\|z\\| > 3, χ², Grubbs) are not used | the data is not Gaussian; the χ² threshold marks several times the expected share |
| Anomalies are flagged, not removed | many are real events (dust, smog, local emissions); removal needs the reason for every anomaly |
"""
