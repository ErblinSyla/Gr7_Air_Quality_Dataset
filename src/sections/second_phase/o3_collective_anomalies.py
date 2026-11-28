"""o3 report section: collective (temporal) anomalies in the hourly series – spikes, stuck sensors, drift."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, md

O = CFG["outliers"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    """Markdown text of the temporal anomaly section; every number comes from the results of the step."""
    best = {row["fault"]: row for row in r.best.iter_rows(named=True)}
    check = {row["fault"]: row for row in r.phase1.iter_rows(named=True)}
    spike, stuck, drift = best["spike"], best["stuck sensor"], best["drift"]
    spike_hours, stuck_hours = int(r.counts["spike_hours"].sum()), int(r.counts["stuck_hours"].sum())
    top_spikes = r.top_spikes.with_columns(pl.col("v", "spike_score").round(1))
    return f"""
## 3. Collective anomalies in the hourly series

A **collective anomaly** is a sequence of related values that is anomalous together, while the single values may look
normal (lecture: the flat segment in an ECG). It needs sequential data, so this section uses the **hourly series** of
every station and pollutant ({r.n_series} series, {r.n_hours:,} measured hours that passed the cleaning of phase I)
instead of the station-days. Three typical faults of an air-quality analyzer are searched, each with a detector that
fits its shape:

| Fault | Shape in the series | Detector (score) | Parameter |
|---|---|---|---|
| isolated spike | one value jumps up and falls straight back | min(v − previous, v − next) / robust std of the hourly changes | threshold t |
| stuck sensor | the same value repeated hour after hour (a flat line) | length of the run of identical values | minimum length L |
| drift | the station slowly moves away from the others | shift of the last W days against the previous {O["drift_baseline_days"]} days, relative to the other stations, as a robust z-score | window W, threshold t |

The spike score takes the *smaller* of the two jumps, so the start of a real episode (a rise that is not followed by a
fall) gets a low score; dividing by the robust spread of the hourly changes (1.4826 · MAD) makes quiet and busy series
comparable. For the stuck sensor, runs of 0 are ignored for {", ".join(CFG["quality"]["stuck_zero_plausible"])}, which
can really stay below the detection limit in clean air (the same rule as phase I). For the drift the other stations
are the reference: the regional weather moves all stations together and cancels out.

### 3.1 Evaluation with injected faults
Every detector is evaluated with injected faults of its own type ({len(O["eval_seeds"])} seeds): spikes in
{100 * O["spike_share"]:g}% of the hours (value × {O["spike_factor"][0]:g}–{O["spike_factor"][1]:g}), {O["stuck_runs"]}
frozen runs of {O["stuck_length"][0]}–{O["stuck_length"][1]} hours, and {O["drift_events"]} drifts in which a series
grows to {O["drift_factor"]:g}× its value over {O["drift_days"]} days. For the faults that last several points,
precision is counted on points and recall on events (one alarm is enough to find a frozen or drifting sensor).

Isolated spike:
{md(r.spike)}
Stuck sensor:
{md(r.stuck)}
Drift:
{md(r.drift)}
{figs["evaluation"]}
- **Stuck sensor** is found reliably: with L = {O["stuck_min_len"][0]} h every run is found but short natural repeats
  also count; longer L misses the short faults. **Choice: {stuck["param"]}** (F1 = {stuck["f1"]:.3f}).
- **Spike:** a low threshold finds most spikes but also many short natural peaks (a passing truck, a plume from a
  chimney); a high threshold is precise but misses the smaller spikes. **Choice: {spike["param"]}** (F1 =
  {spike["f1"]:.3f}); the injected spikes are only {100 * O["spike_share"]:g}% of the hours, so even a few hundred natural
  peaks lower the precision.
- **Drift is not detectable** with this data: the best setting ({drift["param"]}) finds {drift["recall"]:.0%} of the
  drifts, but {1 - drift["precision"]:.0%} of its alarms are false. A slow change up to ×{O["drift_factor"]:g} in a month
  is within the natural day-to-day differences between stations. In practice drift is found with a reference
  instrument at the same place or with regular calibration checks; the detector is therefore **not used** for flags,
  and drift remains a limitation of the analysis.

### 3.2 Check against the rules of phase I (real labels)
Phase I removed {check["spike"]["phase1_hours"]} hours as spikes and {check["stuck sensor"]["phase1_hours"]} hours as
stuck with fixed rules. Running the statistical detectors on the raw values of these hours shows whether both
approaches agree on real faults:
{md(r.phase1)}
The stuck detector finds **{check["stuck sensor"]["share_found"]:.0%}** of the stuck hours of phase I again, the spike
detector {check["spike"]["share_found"]:.0%} of the spikes: phase I used one threshold per pollutant (P99.9 of the
changes), the detector one per series, so small spikes in very variable series are judged differently. The agreement
on real faults supports both the rules of phase I and the detectors.

### 3.3 Temporal anomalies in the real (cleaned) data
After the cleaning of phase I the detectors still find **{spike_hours:,} spike hours** and **{stuck_hours:,} stuck
hours** (`data/gold/anomalies_temporal.parquet`):
{md(r.counts)}
{figs["examples"]}
The strongest spikes:
{md(top_spikes)}
Spikes are most frequent for NO, SO2 and CO, the pollutants of local sources: a plume from a chimney or the power plant
(SO2) or traffic near the analyzer (NO, CO) can raise a single hour; such hours may be real short events or faults and
need a look at the neighbouring stations. Of the stuck hours, {r.n_stuck_zero:,} are runs of 0, mostly NO2 at
Brezovica (XK0007A), a remote mountain station. {r.n_stuck_from_negative:,} of these zeros exist only because phase I
set small negative readings to 0 (`zero_negative`): the analyzer was measuring noise around zero, i.e. NO2 at the
detection limit, which after the conversion looks like a frozen line. They are therefore low-concentration periods
rather than frozen sensors, and the phase-I rule "0 is not plausible for NO2" is too strict for such a clean station.
The non-zero runs (e.g. CO at Palaj, XK0010A, in 2024) are more likely real frozen values. All of them are flagged,
none is removed.
"""
