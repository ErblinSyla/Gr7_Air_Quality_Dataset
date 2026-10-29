## Appendix A. Analyses beyond the lectures
These analyses are **not** part of the lecture material and are not needed for the assignment. They only check the
main pipeline from another angle and change no data.

### A.1 Are the 1-hour gaps missing completely at random (MCAR)?
If short gaps were completely random, (a) they would happen equally often at every hour of the day and (b) the hour
before a gap would have the same pollution as any other hour. (a) χ² goodness-of-fit of the hour of the 1-hour gaps
against the share of hours:


| gaps_1h | chi2_hour | df | critical_value | cramers_v |
|---|---|---|---|---|
| 206042 | 22472.3 | 23 | 35.2 | 0.069 |


χ² > critical value, so the hour is not completely random, but Cramér's V = 0.069 shows that the effect is
very small (the χ² is large because there are 206,042 gaps).
(b) Mean of the hour before a 1-hour gap versus the mean of the series:


| pollutant | series_mean | mean_hour_before_gap | difference_pct |
|---|---|---|---|
| CO | 0.759 | 0.705 | -7.1 |
| NO | 9.301 | 8.686 | -6.6 |
| NO2 | 15.042 | 13.538 | -10.0 |
| NOx | 29.534 | 27.32 | -7.5 |
| O3 | 45.96 | 43.73 | -4.9 |
| PM10 | 23.423 | 19.917 | -15.0 |
| PM2_5 | 16.904 | 13.363 | -20.9 |
| SO2 | 9.99 | 8.186 | -18.1 |


For 8 of 8 pollutants the hour before a gap is less polluted than usual. **Conclusion:** the
1-hour gaps are approximately but not completely MCAR; they depend on observed values (missing at random, MAR).
This supports interpolation, which uses the neighbouring measured hours, over a global mean.

### A.2 RMSE instead of MAE
The method evaluation of section 4.3 measured again with the root mean squared error (same hidden values):


| gap_h | attribute_mean | class_mean | linear |
|---|---|---|---|
| 1 | 18.158 | 15.081 | 5.674 |
| 3 | 20.482 | 17.244 | 7.709 |
| 6 | 18.692 | 15.521 | 10.477 |
| 12 | 19.544 | 16.217 | 13.405 |
| 24 | 19.636 | 16.373 | 16.798 |



![x_rmse_by_gap](../figures/x_rmse_by_gap.png)

The best method is the same with MAE and RMSE in 37 of 40 pollutant × gap-length cases. Only for 24h gaps does the class mean have a slightly smaller RMSE (16.37 vs 16.80): RMSE punishes the few large errors of interpolation over a full day more than MAE does. The
strategy of section 4.3 does not change.

### A.3 Kolmogorov–Smirnov test
The KS statistic is the largest distance between two cumulative distributions (0 = identical).
- **Imputation** (measured values versus values after imputation):


| pollutant | ks_statistic |
|---|---|
| CO | 0.013 |
| NO | 0.012 |
| NO2 | 0.012 |
| NOx | 0.004 |
| O3 | 0.004 |
| PM10 | 0.004 |
| PM2_5 | 0.004 |
| SO2 | 0.022 |


  The largest distance is 0.022: imputation does not change the shape of the distributions.
- **Sampling** (each 10% sample versus the population):


| method | pollutant | ks_statistic | p_value |
|---|---|---|---|
| SRSWOR | PM2_5 | 0.003 | 0.89 |
| SRSWOR | PM10 | 0.002 | 0.895 |
| SRSWOR | NO2 | 0.002 | 0.979 |
| SRSWOR | O3 | 0.003 | 0.788 |
| SRSWR | PM2_5 | 0.005 | 0.13 |
| SRSWR | PM10 | 0.005 | 0.221 |
| SRSWR | NO2 | 0.003 | 0.654 |
| SRSWR | O3 | 0.004 | 0.561 |
| stratified | PM2_5 | 0.002 | 0.936 |
| stratified | PM10 | 0.003 | 0.606 |
| stratified | NO2 | 0.002 | 0.95 |
| stratified | O3 | 0.004 | 0.545 |


  The smallest p-value is 0.13 > 0.05: no sample differs significantly from the population, which confirms section 5.

### A.4 Yeo-Johnson transformation
The pollutants are strongly right-skewed. The Yeo-Johnson power transformation (fitted on 2019–2023; unlike
Box-Cox it accepts zeros) is compared with log(1 + x):


| pollutant | lambda | skew_original | skew_log1p | skew_yeo_johnson |
|---|---|---|---|---|
| SO2 | 0.068 | 4.1 | -0.16 | 0.0 |
| PM10 | 0.057 | 3.22 | -0.17 | 0.01 |
| PM2_5 | -0.055 | 3.38 | 0.15 | 0.0 |
| O3 | 0.519 | 0.53 | -1.22 | -0.15 |
| NO2 | 0.125 | 2.03 | -0.29 | -0.01 |
| NOx | -0.052 | 5.89 | 0.14 | 0.0 |
| CO | -1.203 | 2.84 | 1.13 | 0.14 |
| NO | -0.329 | 9.64 | 0.75 | 0.12 |



![x_yeo_johnson_pm25](../figures/x_yeo_johnson_pm25.png)

Yeo-Johnson brings the skewness of every pollutant close to 0 (|skew| ≤ 0.15), better than log(1 + x). It is
useful for later methods that assume roughly normal data; the main pipeline keeps the lecture normalizations (6.6).

### A.5 Mutual information
Mutual information measures any (also non-linear) dependence between a feature and the target (PM2.5 of the next
hour), on 50,000 rows of 2019–2023. The top 10 features and how many of the
three lecture methods of section 6.2 selected them (`tables/x_mutual_information.csv`):


| feature | mutual_information | selected_by |
|---|---|---|
| PM2_5_obs | 1.212 | 3 |
| PM10_obs | 0.825 | 1 |
| PM2_5_lag_1h | 0.821 | 3 |
| PM10_lag_1h | 0.608 | 1 |
| PM2_5_roll_mean_24h | 0.458 | 3 |
| PM10_roll_mean_24h | 0.376 | 1 |
| PM2_5_lag_24h | 0.291 | 1 |
| NOx_obs | 0.216 | 1 |
| PM10_lag_24h | 0.211 | 1 |
| month | 0.204 | 1 |


The five most informative features are PM2_5_obs, PM10_obs, PM2_5_lag_1h, PM10_lag_1h, PM2_5_roll_mean_24h; 5 of them were also selected by the lecture
methods, so both approaches point to the same information: recent PM levels.
