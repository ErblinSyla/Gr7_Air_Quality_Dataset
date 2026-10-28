"""p5 report section: feature selection and creation, discretization and binarization, normalization, PCA."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from utils import CFG, md

F, R, E, B = CFG["features"], CFG["reduction"], CFG["eaqi"], CFG["binarization"]
ALPHA = CFG["chi2"]["alpha"]
FIT_YEAR = CFG["time"]["fit_until_year"]


def render(r: SimpleNamespace, figs: dict[str, str]) -> str:
    sel, p, chi, si = r.sel, r.p, r.chi, r.smooth_info
    ev = p.ev
    zero_var = ", ".join(sel.variance.filter(~pl.col("passes_variance"))["feature"]) or "none"
    all3 = sel.summary.filter(pl.col("selected_by") == 3)["feature"].to_list()
    pm = r.pairs.filter((pl.col("a") == "PM10") & (pl.col("b") == "PM2_5"))
    pm_r = f"{pm['correlation'][0]:.3f}" if pm.height else "below the threshold"
    low_mm = r.norm.filter(pl.col("v_minmax") < 0.25)["pollutant"]
    fit_text = (f"the {p.n_fit:,} complete hours of 2019–{FIT_YEAR}, without the 2019 period when CO and SO2 are "
                "suspicious (p2)")
    chi_result = ("χ² > critical value → the independence hypothesis is rejected" if chi["chi2"] > chi["critical"]
                  else "χ² ≤ critical value → no relationship can be shown")
    return f"""
## 6. Feature selection and creation, discretization and binarization, transformation, dimensionality reduction

### 6.1 Wide format
The long table (p3) is turned into a wide table: one row = station × hour ({r.n_wide:,} rows), one column per pollutant
(value after imputation) and `<P>_obs` (the measured value, so lagged features use measurements only).

### 6.2 Feature subset selection
The lecture distinguishes **irrelevant** attributes (no information for the task, e.g. a student ID for predicting
GPA) and **redundant** attributes (they duplicate the information of other attributes, e.g. the price and the sales
tax).

**Source columns that are dropped** (evidence in p1, "uninformative columns"):
{md(r.dropped)}
**Variance filter** over the {sel.n_candidates} numeric features: every feature with variance 0 (constant, no
information) is removed. Features with variance 0: {zero_var} (`tables/p5_feature_variance.csv`).

**Redundant attributes: covariance and correlation.** For every pair of measured pollutants,
Cov(A, B) = E[(A − Ā)(B − B̄)] and r = Cov(A, B) / (σA·σB) are computed (`tables/p5_covariance_correlation.csv`).
A positive covariance means the two rise together; its size depends on the units, so the comparison uses
r ∈ [−1, 1].
{figs['correlation']}
Pairs with |r| > {R['corr_threshold']}:
{md(r.pairs)}
- **NOx is derivable:** NOx = NO2 + 1.53·NO (lecture: "derivable data", like age from the date of birth); the
  correlation between NOx and NO2 + 1.53·NO is r = {r.nox_r:.4f}. NOx is removed from PCA so it does not double the
  weight of NO/NO2. This is also why the NOx–NO pair is highly correlated.
- **PM10 and PM2.5** (r = {pm_r}) are both kept, because `pm_coarse` = PM10 − PM2.5 shows that PM10 also carries other
  information (dust).

**Heuristic search.** With d features there are 2^d subsets, so the three methods from the lecture are used, with the
measured PM2.5 of the next hour as the target and linear regression: the regression is fitted on {sel.n_sample:,}
random rows of 2019–{FIT_YEAR} and R² is measured on {sel.n_sample:,} rows of {FIT_YEAR + 1} it has not seen
(seed {F['seed']}), out of
{sel.n_rows:,} rows that have a value for all {sel.n_kept} features ({sel.stations} stations: XK0007A does not measure
NO and O3). Stopping rule: the improvement of R² is < {F['selection_min_gain']}.
- **Step-wise forward selection** (starts from the empty set and adds the best feature at each step):
{md(sel.forward)}
- **Step-wise backward elimination** (starts from all features and removes the worst one at each step; full table:
  `tables/p5_backward_elimination.csv`): {len(sel.remaining)} of {sel.n_kept} features remain, R² =
  {sel.r2_backward:.4f} (with all features: {sel.r2_all:.4f}).
- **Decision tree induction** (a tree of depth {F['tree_max_depth']}; features that do not appear in the tree are
  considered irrelevant): {sel.summary.filter('in_tree').height} features appear in the tree.
{md(sel.summary, 45)}
Forward selection reaches R² = {sel.r2_forward:.4f} with only {len(sel.chosen)} features, almost as much as with
all {sel.n_kept} ({sel.r2_all:.4f}). All three methods agree on: {', '.join(all3) if all3 else 'none'}. The
pollution of the next hour depends mainly on the pollution of the current and the most recent hours (persistence);
the day of the week and the weekend add almost nothing.

### 6.3 Feature creation
Following the lecture: **attribute extraction** (hour, day, month from the time) and **attribute construction**
(combining existing attributes, like BMI = weight / height²; discretization).
{md(r.created)}
Lagged features are computed within the same station, over the full hourly grid (shift 1 = exactly 1 hour earlier),
from measured values and without the current hour (`closed="left"`). Check: every `PM2_5_lag_1h` equals the measured
value of the previous hour exactly (0 errors).

### 6.4 Discretization of PM2.5
Discretization divides the range of a continuous attribute into intervals, and the values are replaced by the label of
their interval. Three methods are compared (unsupervised, top-down split):
{md(r.edges)}
{figs['classes']}
- **Equal width** (W = (B − A) / N): the strongly skewed distribution and the outliers put {r.first_bin:.1f}% of the
  hours in the first class, as the lecture warns.
- **Equal frequency:** balanced classes (about 20% each), but the edges have no health meaning.
- **EAQI = concept hierarchy:** numeric values are replaced by higher-level concepts ("good" … "extremely poor"),
  following the official EEA thresholds ({E['source']}). The classes are unbalanced but **interpretable** and
  comparable with the rest of Europe → chosen for analysis.
{md(r.eaqi)}
**Smoothing by binning** (binning for noise, lectures of weeks 4 and 5): the sorted values are split into
{F['n_bins']} equal-frequency bins; then every value is replaced by the bin mean or by the closest bin boundary. The
code is checked with the lecture example (4, 8, 9, 15 | 21, 21, 24, 25 | 26, 28, 29, 34 → means 9, 23, 29; boundaries
4, 4, 4, 15 | 21, 21, 25, 25 | 26, 26, 26, 34) before it is applied. For PM2.5:
{md(r.smooth)}
The standard deviation drops from {si['std']:.2f} to {si['std_means']:.2f} (means) and {si['std_bounds']:.2f}
(boundaries); the mean absolute smoothing error is {si['mae_means']:.2f} and {si['mae_bounds']:.2f} µg/m³. The last bin
holds the high-pollution episodes, where smoothing would remove exactly the peaks; so smoothing is shown as a method,
but the original values are kept for analysis (noise was handled in p2).

### 6.5 Binarization and encoding
- `pm10_exceed_eu` = 1 when the daily PM10 mean > {B['pm10_daily_limit']} µg/m³ (EU daily limit);
  `pm25_exceed_who` = 1 when the daily PM2.5 mean > {B['pm25_daily_who']} µg/m³ (WHO 2021 guideline).
  They are computed from **measured** daily means and only for valid days (≥ 18 hours, p4), because legal limits are
  assessed on measurements, not on imputed values.
{md(r.exceed, 12)}
- **One-hot** encoding of station, season and station type ({r.onehot_width} 0/1 columns): nominal categories have no
  order, so they cannot be encoded as 1, 2, 3.

**χ² test between two nominal attributes** (as in the lecture): is the heating season related to the daily PM2.5
exceedance? The values in brackets are the expected counts (row total × column total / grand total):
{md(r.chi_tab)}
χ² = {chi['chi2']:,.1f}, df = (2 − 1)·(2 − 1) = 1, critical value at the {ALPHA} level = {chi['critical']:.3f}.
{chi_result}: in the heating season {100 * chi['share'][0]:.0f}% of the days exceed the WHO guideline, outside it
{100 * chi['share'][1]:.0f}%.

### 6.6 Transformation: normalization
The pollutants are measured in different units and scales (CO in mg/m³ with values < 1, O3 up to hundreds of µg/m³);
without normalization, in distance-based methods (k-NN, clustering, PCA) the pollutant with the largest numbers would
dominate. The three methods from the lecture are applied to every pollutant (columns `<P>_minmax`, `<P>_z`, `<P>_dec`):
- **Min-max** to [0, 1]: v' = (v − min) / (max − min);
- **Z-score:** v' = (v − μ) / σ (how many standard deviations above/below the mean);
- **Decimal scaling:** v' = v / 10^j, where j is the smallest integer such that max|v'| < 1.

The formulas are checked with the lecture examples (73,600 → 0.716 with min-max on [12,000, 98,000]; → 1.225 with
z-score μ = 54,000, σ = 16,000; −986 and 917 → −0.986 and 0.917 with j = 3). Parameters and an example (v = P95 of the
pollutant):
{md(r.norm)}
{figs['normalization']}
All parameters (min, max, μ, σ, j, the equal-width and equal-frequency edges, the PCA mean and components) are
learned from 2019–{FIT_YEAR} only and then applied unchanged to {FIT_YEAR + 1}, so the last year stays unseen for
later analyses (no data leakage); in {FIT_YEAR + 1} a value can therefore fall slightly outside [0, 1].
Checks (2019–{FIT_YEAR}): every min-max column has min 0 and max 1, every z-score column has mean 0 and standard
deviation 1, every decimal value has |v'| < 1. Min-max is strongly affected by the maximum value (outliers):
for {', '.join(low_mm)} even
the P95 falls below 0.25, so most values are squeezed close to 0. Z-score is used for PCA (6.7), because it gives
every pollutant equal weight.

### 6.7 Dimensionality reduction (PCA)
The steps from the lecture: (1) z-score normalization of the 7 pollutants (without NOx, 6.2); (2) the eigenvectors of
the covariance matrix = the principal components; (3) every hour is expressed as a linear combination of them;
(4) sorting by variance; (5) the strong components are kept. PCA works with numeric data and complete rows only:
{p.n_complete:,} hours from {p.stations} stations (**XK0007A is excluded**, it does not measure NO and O3); values
after imputation, so hours with a single gap are not lost. The mean, standard deviation and components are computed
from {fit_text}.
{md(p.explained)}
{figs['scree']}
Loadings:
{md(p.loadings)}
{figs['biplot']}
**Interpretation:**
- **PC1 ({ev[0]:.0%})** = overall combustion pollution: PM2.5, PM10, NO, NO2 and CO load with the same sign, O3 with the
  opposite sign (O3 drops where NO consumes it). The mean PC1 score in the heating season is
  {p.scores[p.heat, 0].mean():.2f}, outside it {p.scores[~p.heat, 0].mean():.2f}.
- **PC2 ({ev[1]:.0%})** is dominated by SO2: a separate source (industry/power plants).
- **{p.n_90} components** are enough for {int(R['pca_variance'] * 100)}% of the variance, out of 7 pollutants.

Output: `data/gold/features_hourly.parquet` ({r.f.height:,} rows × {r.f.width} columns).
"""
