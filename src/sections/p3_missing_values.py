"""p3 report section: identification and treatment strategy for missing values."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, md

M = CFG["missing"]
ALPHA = CFG["chi2"]["alpha"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    a, total, sens, det = r.a, r.total, r.sens, r.det
    chi = a["chi2"]
    dependent = chi.filter(pl.col("result") == "dependent").height
    corr_min = a["missing_correlation"].select(pl.exclude("pollutant")).min().min_horizontal().item()
    gl = a["gaps"].filter(pl.col("length").is_in(["1–7 days", "> 7 days"]))
    nox_longest = a["longest_gap"].filter(pl.col("pollutant").is_in(["NO", "NO2", "NOx"]))["days"].max()
    exc = r.exceptions
    exceptions_text = ("" if exc.is_empty() else
                       "Exceptions: " + "; ".join(f"{p} at {g}h ({m.replace('_', ' ')} better, interpolation "
                                                  f"{w:.1f}% worse)" for g, p, m, w in
                                                  exc.select("gap_h", "pollutant", "method", "linear_worse_pct").rows())
                       + ". There the class mean keeps the daily cycle of a long gap better; the differences are small "
                       "and one rule for every pollutant avoids fitting the method to the evaluation sample.")
    lost = (r.context.with_columns(pl.min_horizontal("attribute_mean", "class_mean").alias("best_mean"))
             .filter(pl.col("linear") > pl.col("best_mean"))
             .with_columns((100 * (pl.col("linear") / pl.col("best_mean") - 1)).alias("pct")))
    won = r.context.height - lost.height
    context_text = f"Interpolation has the smallest error in {won} of {r.context.height} gap-length × context cases."
    if lost.height:
        cases = "; ".join(f"{g}h {c} (interpolation {lin:.2f} vs class or attribute mean {m:.2f}, {p:.1f}% worse)"
                          for g, c, lin, m, p in lost.select("gap_h", "context", "linear", "best_mean", "pct").rows())
        small = lost["pct"].max() < 5
        context_text += (f" Exception: {cases}. " + ("The difference is small, so one rule for all gaps up to "
                         f"{M['max_gap_h']}h is kept." if small else "This limits the rule for long gaps."))
    other_longest = a["longest_gap"].filter(~pl.col("pollutant").is_in(["NO", "NO2", "NOx"]))["days"].max()
    return f"""
## 4. Missing values: identification and treatment strategy

### 4.1 Identification
A full hourly grid is built: every station × every pollutant × every hour over the station's period
({r.n_hours:,} hours). This makes visible the hours that are **missing completely** (no row), which cannot be seen in
the raw data. Every hour receives a `status`:
{md(a['status_by_pollutant'])}
**Structural gaps** (a value "should not exist"): `not_measured` ({total.get('not_measured', 0):,}; e.g. XK0007A does
not measure NO, NOx, O3: attribute not applicable) and `outside_period` ({total.get('outside_period', 0):,}; the series
had not started yet or had already ended). They are not counted as missing and are **never imputed**.
**Real missing values:** {r.real_missing:,} of {r.real_hours:,} expected hours
({100 * r.real_missing / r.real_hours:.1f}%), of which {total.get('missing_row', 0):,} have no row at all and the rest
are nulls with a reason from p2.

**Reasons for missing values** (following the lecture: equipment malfunction, data deleted because it was
inconsistent with other recorded data, data not recorded, attributes not applicable) and the action for each:
{md(r.decisions)}
% of real missing values by station and pollutant:
{md(a['missing_by_station'])}

### 4.2 Analysis of the missing values
Gap length (a gap = consecutive hours without a value):
{md(a['gaps'])}
{a['gaps']['pct_gaps'][0]}% of the gaps last 1 hour, but gaps longer than 1 day ({gl['pct_gaps'].sum():.1f}% of the
gaps) hold {gl['pct_hours'].sum():.1f}% of the missing hours.
Longest gap per pollutant:
{md(a['longest_gap'])}
For NO, NO2 and NOx the longest gaps (up to {nox_longest:.0f} days) are the 2019 period before the analyzer change
(p2); for the other pollutants the longest gap is {other_longest} days (a long station outage).
{figs['map']}
{figs['when']}
% missing by year: {', '.join(f"{y}: {p}%" for y, p in a['missing_by_year'].rows())}.

**Are pollutants missing together?** Correlation (Pearson) of the missing indicators (1 = always missing together):
{md(a['missing_correlation'])}
The pollutants of the same station are missing together (correlation {corr_min:.2f}–1.00): outages of the whole
station (power, communication, maintenance), i.e. equipment malfunction, not deliberately hidden values.

**Does missingness depend on time? χ² test** (as in the lecture: contingency table missing/not missing × month, year
or hour; expected count = row total × column total / grand total; if χ² > the critical value at the {ALPHA} level, the
independence hypothesis is rejected; df = (rows − 1) × (columns − 1)):
{md(chi)}
Missingness is **dependent** on time in {dependent} of {chi.height} tests. The largest values are for the **year**
(years with long outages, and the 2019 period for NO/NO2/NOx); for the hour of the day the values are much smaller.
Consequence for the strategy: imputation must use time (the neighbouring hour, the month and the hour of the day), not
a single mean for the whole series. With this many hours even small differences come out "significant", so the sizes
of χ² are also compared between attributes.

### 4.3 Treatment strategy
The lecture lists these options: ignore the tuple, fill in manually, a global constant, the attribute mean, **the
attribute mean of the same class** and **the most probable value** (regression / formula). A global constant
("unknown") is meaningless for concentrations (it would create a fake value) and manual filling is impossible for
{r.real_missing:,} hours. Example (PM2.5, XK0001A): the attribute mean ({r.attr_mean:.1f} µg/m³) is far from the class
values (winter evening versus summer morning), so the mean of the same class is smarter than the attribute mean:
{md(r.example)}
**Which method is best? Evaluation on hidden values.** In complete segments of the data, known values are hidden on
purpose (gaps of {', '.join(str(x) for x in M['eval_gap_lengths_h'])} hours, up to {M['eval_gaps_per_series']} per
series, seed {M['seed']}), filled with each method and compared with the true value (mean absolute error, MAE, in the
unit of the pollutant; mean over the 8 pollutants):
{md(r.eval_wide)}
The most probable value by **linear interpolation** between the hour before and after has the smallest error in
{r.linear_best} of {r.n_best} pollutant × gap-length combinations (`tables/p3_method_evaluation.csv`): pollution
changes slowly from hour to hour, so the neighbouring hours carry more information than any long-term mean. The mean of
the same class beats the attribute mean, but loses to interpolation, so it is not used. {exceptions_text}

**Does this hold for real gaps?** Real gaps often happen next to instrument failures, where the data may be less calm.
The same hidden values are therefore split by context: segments within {M['near_real_gap_h']} hours of a real
gap versus quiet periods (MAE, mean over all pollutants):
{md(r.context)}
{context_text}

Order (real missing values only; structural gaps never):
1. **Most probable value from other attributes (physical formula):** NOx = NO2 + 1.53·NO in the same hour, when NO2
   and NO are measured and valid. Check of the formula (hours where all three are measured): {md(det)}
   The mean error ({det['mean_absolute_error'][0]:.2f} µg/m³) is negligible compared with the NOx mean
   ({det['mean_NOx'][0]:.1f} µg/m³). It is not applied before November 2019, because NO and NO2 are null there.
2. **Gap ≤ {M['max_gap_h']}h inside the series: most probable value by linear interpolation** (the winner of the
   evaluation above).
3. **Gap > {M['max_gap_h']}h or at the edge of the series: ignored (stays null).** Imputing whole days would invent
   data and artificially reduce the variance; at the edge there is no value on both sides.
{md(r.methods)}
{figs['methods']}
Why values stay null: {md(r.why_left)}

### 4.4 Checks
0 imputed values that are negative or above the physical limit; 0 imputed structural gaps; 0 measured values changed.
Distribution before/after:
{md(sens)}
The largest change of the mean is {sens['mean_change_pct'].abs().max():.1f}% and of the standard deviation
{sens['std_change_pct'].abs().max():.1f}%: imputation does not distort the distribution.
**Complete rows** in wide format (station × hour where every pollutant the station measures has a value):
{r.complete_pct['before']}% → {r.complete_pct['after']}%. Output: `data/gold/hourly_long/`.
"""
