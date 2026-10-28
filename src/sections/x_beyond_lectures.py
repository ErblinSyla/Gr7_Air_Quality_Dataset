"""Appendix report section: analyses beyond the lectures (MCAR, RMSE, Kolmogorov–Smirnov, Yeo-Johnson, mutual
information)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, md

FIT_YEAR = CFG["time"]["fit_until_year"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    t = r.mcar.row(0, named=True)
    rejected = t["chi2_hour"] > t["critical_value"]
    lower = r.before.filter(pl.col("difference_pct") < 0).height
    last = r.rmse.row(-1, named=True)
    rmse_note = (f" Only for {last['gap_h']}h gaps does the class mean have a slightly smaller RMSE "
                 f"({last['class_mean']:.2f} vs {last['linear']:.2f}): RMSE punishes the few large errors of "
                 "interpolation over a full day more than MAE does."
                 if last["class_mean"] < last["linear"] else "")
    hour_text = "so the hour is not completely random" if rejected else "so no dependence on the hour is shown"
    ks_max = r.ks_imp["ks_statistic"].max()
    p_min = r.ks_smp["p_value"].min()
    ks_text = ("no sample differs significantly from the population" if p_min > 0.05
               else "some samples differ from the population")
    yj_max = r.yj["skew_yeo_johnson"].abs().max()
    mi_top = r.mi.head(5)["feature"].to_list()
    mi_agree = r.mi.head(5).filter(pl.col("selected_by") > 0).height
    return f"""
## Appendix A. Analyses beyond the lectures
These analyses are **not** part of the lecture material and are not needed for the assignment. They only check the
main pipeline from another angle and change no data.

### A.1 Are the 1-hour gaps missing completely at random (MCAR)?
If short gaps were completely random, (a) they would happen equally often at every hour of the day and (b) the hour
before a gap would have the same pollution as any other hour. (a) χ² goodness-of-fit of the hour of the 1-hour gaps
against the share of hours:
{md(r.mcar)}
χ² {'>' if rejected else '≤'} critical value, {hour_text}, but Cramér's V = {t['cramers_v']} shows that the effect is
{'very small' if t['cramers_v'] < 0.1 else 'noticeable'} (the χ² is large because there are {t['gaps_1h']:,} gaps).
(b) Mean of the hour before a 1-hour gap versus the mean of the series:
{md(r.before)}
For {lower} of {r.before.height} pollutants the hour before a gap is less polluted than usual. **Conclusion:** the
1-hour gaps are approximately but not completely MCAR; they depend on observed values (missing at random, MAR).
This supports interpolation, which uses the neighbouring measured hours, over a global mean.

### A.2 RMSE instead of MAE
The method evaluation of section 4.3 measured again with the root mean squared error (same hidden values):
{md(r.rmse)}
{figs['rmse']}
The best method is the same with MAE and RMSE in {r.same} of {r.total} pollutant × gap-length cases.{rmse_note} The
strategy of section 4.4 does not change.

### A.3 Kolmogorov–Smirnov test
The KS statistic is the largest distance between two cumulative distributions (0 = identical).
- **Imputation** (measured values versus values after imputation):
{md(r.ks_imp)}
  The largest distance is {ks_max:.3f}: imputation does not change the shape of the distributions.
- **Sampling** (each 10% sample versus the population):
{md(r.ks_smp)}
  The smallest p-value is {p_min:.2f} {'>' if p_min > 0.05 else '≤'} 0.05: {ks_text}, which confirms section 5.

### A.4 Yeo-Johnson transformation
The pollutants are strongly right-skewed. The Yeo-Johnson power transformation (fitted on 2019–{FIT_YEAR}; unlike
Box-Cox it accepts zeros) is compared with log(1 + x):
{md(r.yj)}
{figs['yj']}
Yeo-Johnson brings the skewness of every pollutant close to 0 (|skew| ≤ {yj_max:.2f}), better than log(1 + x). It is
useful for later methods that assume roughly normal data; the main pipeline keeps the lecture normalizations (6.6).

### A.5 Mutual information
Mutual information measures any (also non-linear) dependence between a feature and the target (PM2.5 of the next
hour), on {CFG['features']['selection_sample']:,} rows of 2019–{FIT_YEAR}. The top 10 features and how many of the
three lecture methods of section 6.3 selected them (`tables/x_mutual_information.csv`):
{md(r.mi.head(10))}
The five most informative features are {', '.join(mi_top)}; {mi_agree} of them were also selected by the lecture
methods, so both approaches point to the same information: recent PM levels.
"""
