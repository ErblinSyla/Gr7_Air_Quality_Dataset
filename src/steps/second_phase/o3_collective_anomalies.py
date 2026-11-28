"""o3 – Collective (temporal) anomalies in the hourly series of every station and pollutant.

Lecture (week VI): a collective anomaly is a sequence of related values that is anomalous together, while every
single value may look normal (the flat line in an ECG). It needs sequential data, so the hourly series are used
instead of the station-day points of o1/o2. Three typical sensor faults, one detector each:
- isolated spike: one value that jumps up and falls straight back (score = the smaller of the two jumps, in robust
  standard deviations of the hourly changes of the series);
- stuck sensor: the same non-zero value repeated for L or more consecutive hours (score = run length);
- drift: a slow, growing deviation from the other stations (score = shift of the recent W days against the previous
  weeks, as a robust z-score); tested, but not used for flags (see the report).
Every detector is evaluated with injected faults of its own type; the spike and stuck detectors are also checked
against the hours that the rules of phase I removed. Flags are stored in data/gold/anomalies_temporal.parquet.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl

from steps.second_phase.common import O, best_per, event_metrics, mean_over_seeds, robust_z
from utils import CFG, scan, table, write

SERIES = ["station_id", "pollutant"]
HOUR = timedelta(hours=1)
ZERO_PLAUSIBLE = CFG["quality"]["stuck_zero_plausible"]  # pollutants for which long runs of 0 are real (phase I)
LOG_OFFSET = pl.when(pl.col("pollutant") == "CO").then(0.05).otherwise(1.0)  # to take the log of zero values


# ---------------------------------------------------------------- data ---------------------------------------------

def load_hourly(column: str = "value", statuses: tuple[str, ...] = ("ok",)) -> pl.DataFrame:
    """The hourly series of all stations and pollutants, sorted by series and time; `v` is the value analysed.

    By default only the measured values that passed the cleaning of phase I ("ok"); with column="value_raw" and the
    statuses of the removed hours, the raw values before cleaning (used to check the detectors against phase I).
    """
    return (scan("gold", "hourly_long").filter(pl.col("status").is_in(statuses), pl.col(column).is_not_null())
            .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "ts_utc", "status",
                    pl.col("clean_action").cast(pl.String), pl.col(column).alias("v"))
            .sort(*SERIES, "ts_utc").collect())


def load_daily() -> pl.DataFrame:
    """Daily means of the valid measured days (step p4), sorted by series and date; `v` is the value analysed."""
    return (scan("gold", "daily").filter((pl.col("source") == "measured") & pl.col("valid_day"))
            .select(pl.col("station_id").cast(pl.String), pl.col("pollutant").cast(pl.String), "date",
                    pl.col("mean").alias("v"))
            .sort(*SERIES, "date").collect())


# ---------------------------------------------------------------- detectors ----------------------------------------

def spike_score(h: pl.DataFrame) -> np.ndarray:
    """Isolated spike score: min(v − previous, v − next) / robust std of the hourly changes of the series.

    Both neighbours must be the directly adjacent hours. Taking the smaller jump means that the value must rise *and*
    fall back: the start of a real pollution episode (rise without fall) gets a low score. Dividing by the robust
    spread of the hourly changes (1.4826 · MAD) makes the score comparable between quiet and busy series.
    """
    gap_prev = pl.col("ts_utc") - pl.col("ts_utc").shift(1)
    gap_next = pl.col("ts_utc").shift(-1) - pl.col("ts_utc")
    x = h.with_columns(
        pl.when(gap_prev == HOUR).then(pl.col("v").shift(1)).over(SERIES).alias("_prev"),
        pl.when(gap_next == HOUR).then(pl.col("v").shift(-1)).over(SERIES).alias("_next"),
        (pl.col("v") - pl.col("v").shift(1)).over(SERIES).alias("_change"))
    x = x.with_columns((1.4826 * (pl.col("_change") - pl.col("_change").median()).abs().median() + 1e-9)
                       .over(SERIES).alias("_scale"))
    rise = pl.min_horizontal(pl.col("v") - pl.col("_prev"), pl.col("v") - pl.col("_next"))
    return x.select((rise / pl.col("_scale")).fill_null(0)).to_series().to_numpy()


def run_length(h: pl.DataFrame) -> np.ndarray:
    """Length of the run of identical consecutive values that every hour belongs to (a new run starts when the value
    changes or an hour is missing). Runs of 0 get length 0 for the pollutants in `stuck_zero_plausible` (CO, SO2, O3,
    NO): in clean air they are really below the detection limit (same rule as phase I); for PM and NO2 a run of 0 is
    a fault like any other frozen value."""
    new_run = ((pl.col("v") != pl.col("v").shift(1)) | (pl.col("ts_utc") - pl.col("ts_utc").shift(1) != HOUR))
    run_id = new_run.fill_null(True).cum_sum().over(SERIES)
    lengths = h.with_columns(run_id.alias("_run")).select(pl.len().over(*SERIES, "_run"))
    plausible_zero = (h["v"].to_numpy() == 0) & h["pollutant"].is_in(ZERO_PLAUSIBLE).to_numpy()
    return np.where(plausible_zero, 0, lengths.to_series().to_numpy())


def drift_score(d: pl.DataFrame, window: int) -> np.ndarray:
    """Drift score: how much the station has moved away from the other stations in the last `window` days.

    1. Relative level r = log(v) − mean log(v) of the other stations on the same day and pollutant (the regional
       weather, which moves all stations together, cancels out);
    2. shift = mean r of the last `window` days − median r of the previous `drift_baseline_days` days;
    3. score = |robust z-score of the shift| within the series.
    Days with fewer than three stations for the pollutant have no reference and get score 0.
    """
    x = d.with_columns((pl.col("v") + LOG_OFFSET).log().alias("lv"))
    x = x.with_columns(pl.col("lv").sum().over("pollutant", "date").alias("_sum"),
                       pl.len().over("pollutant", "date").alias("_n"))
    relative = pl.col("lv") - (pl.col("_sum") - pl.col("lv")) / (pl.col("_n") - 1)
    x = x.with_columns(pl.when(pl.col("_n") >= 3).then(relative).alias("r"))
    base = O["drift_baseline_days"]
    shift = (pl.col("r").rolling_mean(window, min_samples=max(2, window // 2))
             - pl.col("r").shift(window).rolling_median(base, min_samples=base // 3)).over(SERIES)
    x = x.with_columns(shift.alias("shift"))
    return x.select(robust_z(pl.col("shift"), SERIES).abs().fill_null(0)).to_series().to_numpy()


# ---------------------------------------------------------------- injected faults ----------------------------------

def inject_spikes(h: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """`spike_share` of the hours (with a value > 0) are multiplied by a factor in `spike_factor`.

    Returns the changed data and the event id of every hour (0 = normal, 1, 2, … = injected spike)."""
    rng = np.random.default_rng(seed)
    v = h["v"].to_numpy().copy()
    rows = rng.choice(np.flatnonzero(v > 0), round(O["spike_share"] * len(v)), replace=False)
    v[rows] *= rng.uniform(*O["spike_factor"], len(rows))
    event = np.zeros(len(v), np.int64)
    event[rows] = np.arange(1, len(rows) + 1)
    return h.with_columns(pl.Series("v", v)), event


def inject_stuck(h: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """`stuck_runs` times, a value > 0 is repeated for a random number of hours in `stuck_length` (a frozen sensor).

    A run is placed only where the following hours are consecutive hours of the same series."""
    rng = np.random.default_rng(seed)
    v, ts = h["v"].to_numpy().copy(), h["ts_utc"].to_numpy()
    series = h.select(pl.concat_str(SERIES)).to_series().to_numpy()
    event = np.zeros(len(v), np.int64)
    lo, hi = O["stuck_length"]
    for start in rng.permutation(len(v) - hi):
        length = int(rng.integers(lo, hi + 1))
        end = start + length
        if (v[start] > 0 and series[start] == series[end - 1] and not event[start:end].any()
                and ts[end - 1] - ts[start] == np.timedelta64(length - 1, "h")):
            v[start:end] = v[start]
            event[start:end] = event.max() + 1
        if event.max() == O["stuck_runs"]:
            break
    return h.with_columns(pl.Series("v", v)), event


def inject_drift(d: pl.DataFrame, seed: int) -> tuple[pl.DataFrame, np.ndarray]:
    """`drift_events` times, the daily values of one series are multiplied by a ramp from 1 to `drift_factor` over
    `drift_days` days (a sensor that slowly loses its calibration)."""
    rng = np.random.default_rng(seed)
    v = d["v"].to_numpy().copy()
    event = np.zeros(len(v), np.int64)
    starts = d.with_row_index("i").group_by(SERIES).agg(pl.col("i").min().alias("lo"), pl.col("i").max().alias("hi"))
    days = O["drift_days"]
    ramp = np.linspace(1, O["drift_factor"], days)
    for e in range(1, O["drift_events"] + 1):
        series = starts.row(int(rng.integers(starts.height)), named=True)
        if series["hi"] - series["lo"] <= days:
            continue
        start = int(rng.integers(series["lo"], series["hi"] - days))
        if event[start:start + days].any():
            continue
        v[start:start + days] *= ramp
        event[start:start + days] = e
    return d.with_columns(pl.Series("v", v)), event


# ---------------------------------------------------------------- evaluation ---------------------------------------

def evaluate(hourly: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Every detector × parameter on `eval_seeds` injections of its own fault type."""
    rows = []
    for seed in O["eval_seeds"]:
        data, event = inject_spikes(hourly, seed)
        score = spike_score(data)
        rows += [{"seed": seed, "fault": "spike", "param": f"t = {t}", "value": t, **event_metrics(event, score > t)}
                 for t in O["spike_t"]]
        data, event = inject_stuck(hourly, seed)
        length = run_length(data)
        rows += [{"seed": seed, "fault": "stuck sensor", "param": f"L = {n} h", "value": n,
                  **event_metrics(event, length >= n)} for n in O["stuck_min_len"]]
        data, event = inject_drift(daily, seed)
        for w in O["drift_window"]:
            score = drift_score(data, w)
            rows += [{"seed": seed, "fault": "drift", "param": f"W = {w} d, t = {t}", "value": w * 100 + t,
                      **event_metrics(event, score > t)} for t in O["drift_t"]]
    return pl.DataFrame(rows)


def phase1_check(best: dict) -> pl.DataFrame:
    """Real labels: the share of the hours that phase I removed as spike / stuck that the detectors find again.

    The detectors run on the raw values of the hours that were kept or removed for these two reasons; a high share
    shows that the statistical detectors and the rules of phase I agree on real faults."""
    raw = load_hourly("value_raw", ("ok", "null_spike", "null_stuck"))
    spikes = spike_score(raw) > best["spike"]
    stuck = run_length(raw) >= best["stuck sensor"]
    status = raw["status"].to_numpy()
    rows = []
    for name, removed, found in (("spike", "null_spike", spikes), ("stuck sensor", "null_stuck", stuck)):
        target = status == removed
        rows.append({"fault": name, "phase1_hours": int(target.sum()), "found_again": int((found & target).sum()),
                     "share_found": float((found & target).sum() / target.sum()),
                     "flags_on_kept_hours": int((found & (status == "ok")).sum())})
    return pl.DataFrame(rows)


def run() -> SimpleNamespace:
    """o3 pipeline: evaluation of the three detectors → best parameters → check against phase I → flags."""
    hourly, daily = load_hourly(), load_daily()
    summary = mean_over_seeds(evaluate(hourly, daily), ["fault", "param", "value"])
    best_rows = best_per(summary, "fault", "value")
    best = dict(best_rows.select("fault", "value").iter_rows())
    alarms = hourly.with_columns(pl.Series("spike_score", spike_score(hourly)),
                                 pl.Series("run_length", run_length(hourly)))
    alarms = alarms.with_columns((pl.col("spike_score") > best["spike"]).alias("spike"),
                                 (pl.col("run_length") >= best["stuck sensor"]).alias("stuck"))
    alarms = alarms.filter(pl.col("spike") | pl.col("stuck")).drop("status")
    stuck_zero = alarms.filter(pl.col("stuck") & (pl.col("v") == 0))
    write(alarms, "gold", "anomalies_temporal")
    counts = (alarms.group_by("pollutant").agg(pl.col("spike").sum().alias("spike_hours"),
                                               pl.col("stuck").sum().alias("stuck_hours"))
              .join(hourly.group_by("pollutant").len("hours"), on="pollutant")
              .with_columns((1000 * pl.col("spike_hours") / pl.col("hours")).alias("spikes_per_1000h"))
              .sort("pollutant"))
    return SimpleNamespace(
        n_hours=hourly.height, n_series=hourly.select(SERIES).n_unique(),
        spike=table(summary.filter(pl.col("fault") == "spike").drop("fault", "value"), "o3_spike_evaluation"),
        stuck=table(summary.filter(pl.col("fault") == "stuck sensor").drop("fault", "value"), "o3_stuck_evaluation"),
        drift=table(summary.filter(pl.col("fault") == "drift").drop("fault", "value"), "o3_drift_evaluation"),
        best=table(best_rows.drop("value"), "o3_best_settings"), best_value=best,
        phase1=table(phase1_check(best), "o3_phase1_check"),
        counts=table(counts, "o3_anomalies_by_pollutant"),
        top_spikes=table(alarms.filter("spike").sort("spike_score", descending=True).head(10)
                         .select("station_id", "pollutant", "ts_utc", "v", "spike_score"), "o3_top_spikes"),
        top_stuck=table(alarms.filter("stuck").group_by("station_id", "pollutant", "v", "run_length")
                        .agg(pl.col("ts_utc").min().alias("start")).sort("run_length", descending=True).head(10),
                        "o3_top_stuck"),
        n_stuck_zero=stuck_zero.height,
        n_stuck_from_negative=stuck_zero.filter(pl.col("clean_action") == "zero_negative").height,
        alarms=alarms, hourly=hourly,
    )
