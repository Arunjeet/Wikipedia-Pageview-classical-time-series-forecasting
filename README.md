# Wikipedia Pageview Forecasting - Figma (2022-2025)

> Classical time series analysis and forecasting pipeline applied to Wikipedia daily pageview data for Figma. Covers the full methodology from stationarity testing through SARIMA model selection, cross-validated parameter search, and 365-day forward forecasting with inverse transform recovery.

---

## The Problem

Wikipedia pageview data for Figma exhibits three distinct regimes driven by a single external event - the Adobe acquisition announcement (September 2022). A single model cannot capture all three regimes equally. The challenge is to identify the dominant structure, apply appropriate transformations, select a model that generalises across regime boundaries, and produce credible forward forecasts.

---

## Results at a Glance

| Step | Method | Outcome |
|---|---|---|
| Decomposition | STL (period=7) | Weekly seasonality confirmed across all regimes |
| Stationarity | Ljung-Box + ADF + KPSS | Non-stationary confirmed, d=1 differencing sufficient |
| Variance stabilisation | Box-Cox transform | Spike-driven heteroskedasticity resolved |
| Outlier treatment | Winsorization (2.5% tails) | Train MSE reduced from 0.15 to 0.07 |
| Model selection | SARIMA(4,0,6)(1,0,1,7) | Best generaliser across all cross-validation splits |
| Final in-sample MSE | 0.0902 | Stable across all 4 final candidate models |
| Forecast | 365 days ahead | Steady decline with weekly seasonality preserved |

---

## File Structure

```
Wikipedia-Pageview-Forecasting/
├── analysis.ipynb      # Full analysis notebook - run top to bottom
├── requirements.txt    # Dependencies
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Update the data path in the first cell of the notebook

```python
file = r"path/to/pageviews-20220101-20250606.csv"
```

### 3. Run all cells top to bottom

The notebook is structured sequentially - each section builds on the previous. Run all cells in order for the full analysis and forecast output.

### Data

Download the pageview CSV from the Wikimedia API:
https://wikimedia.org/api/rest_v1/#/Pageviews_data

Select article: `Figma_(software)`, date range: 2022-01-01 to 2025-06-06, granularity: daily.

---

## Data

Wikipedia daily pageview counts for the Figma (software) article.
Date range: 1 January 2022 to 6 June 2025.
Source: Wikimedia Pageviews API - https://wikimedia.org/api/rest_v1/

---

## Regime Analysis

The series has three structurally distinct regimes:

### 1. Pre-spike (Jan 2022 - Aug 2022)
Clear upward trend with strong and regular weekly seasonality. Mean steadily rises, variance remains moderate. Residual noise is low and stable except for a minor spike in June 2022. High autocorrelation, especially at weekly lags. Well-behaved and forecast-friendly.

### 2. Post-spike decay (Sept 2022 - Dec 2023)
Triggered by the Adobe acquisition announcement. Trend sharply declines then flattens. Seasonality weakens and becomes less structured. Residuals capture the shock plus intermittent spikes in March and July 2023. Mean drops significantly while variance briefly remains high before stabilising.

### 3. Post-dip baseline (2024 - mid 2025)
Flat or slightly declining trend. Seasonality disappears almost entirely, dominated by low-magnitude noise. Mean is minimal, variance very low, autocorrelation virtually absent. Pattern suggests diminished user interest following the failed Adobe acquisition.

---

## Methodology

### Step 1 - Visual inspection and STL decomposition

STL decomposition with period=7 chosen over classical additive/multiplicative decomposition because LOESS smoothing handles the September 2022 spike as a localised anomaly without distorting the trend and seasonal estimates for surrounding periods. Period=7 captures dominant weekly usage patterns (weekday peaks, weekend troughs).

### Step 2 - Stationarity testing

Three complementary tests applied:

- Ljung-Box (lags=10): lb_stat=2129.06, p-value=0.0 - series is not white noise, significant autocorrelation confirmed
- ADF: statistic=-5.402, p-value=0.0 - reject H0, appears stationary at level
- KPSS (trend, ct): statistic=0.27, p-value=0.01 - reject H0, trend non-stationary
- KPSS (level, c): statistic=3.26, p-value=0.01 - reject H0, level non-stationary, mean changes over time

ADF and KPSS together confirm the series has a stochastic trend requiring differencing. ADF alone would have been misleading here.

### Step 3 - Making the series stationary

After first-order differencing: no trend, reduced seasonality, stable variance, mean near zero. ADF confirms stationarity post-differencing. Outliers remain but autocorrelation is minimal.

### Step 4 - Box-Cox transform and Winsorization

Box-Cox applied before differencing to stabilise variance across the spike regime. Winsorization (2.5% each tail) preferred over IQR clipping - less harsh, preserves distributional shape while reducing the influence of extreme values on SARIMA coefficient estimation.

Impact of winsorization on ARMA models:
- Before winsorization: Train MSE 0.15, Test MSE 0.19
- After winsorization: Train MSE 0.07, Test MSE 0.17

### Step 5 - ACF/PACF model identification

Two rounds of ACF/PACF analysis:

Round 1 (differenced only):
- ACF: sharp drop after lag 1 - MA(1) component
- PACF: significant spike at lag 1, tapering - AR(1) process
- Conclusion: SARIMA(1,1,1)(P,D,Q,7) as starting point

Round 2 (Box-Cox + differenced + winsorized):
- ACF: strong spike at lag 1, significant spikes at lags 7, 14, 21 - weekly seasonal MA
- PACF: dominant spike at lag 1, minor spikes at lags 7 and 14 - seasonal AR terms
- Conclusion: confirms SARIMA with S=7 and higher p/q orders warranted

### Step 6 - Parameter search

Non-seasonal AIC/BIC grid search over p,q in range(2,8):
- Best candidates: (4,0,7) highest AIC, (5,0,5) balanced AIC/BIC, (4,0,6) most stable and conservative

Seasonal grid search (P, 0, Q, 7) given non-seasonal order (4,0,6):
- Case 1 (no winsorization): best seasonal orders (3,0,0,7), (3,0,3,7), (3,0,2,7)
- Case 2 (winsorized): best seasonal orders (1,0,1,7) consistently selected

Note: Parameters differ between 80% and 100% training data due to the post-dip noise regime contaminating the full-data grid search.

---

## Model Comparison

### Baseline models (Box-Cox + diff, 80-20 split)

| Model | Train MSE | Test MSE | Notes |
|---|---|---|---|
| AR(14) | - | - | High MSE, no seasonality |
| MA(1) | - | - | Too simple, underfits |
| ARMA(4,7) | 0.15 | 0.19 | Cannot fully capture seasonality |
| ARMA(5,5) | 0.16 | 0.19 | Same limitation |
| ARMA(4,6) | 0.16 | 0.19 | Same limitation |
| SARIMA(4,0,7)(3,0,0,7) | 0.17 | 0.20 | Captures seasonality, high variance |
| SARIMA(5,0,5)(3,0,3,7) | 0.14 | 0.19 | Better, variance needs trimming |
| SARIMA(4,0,6)(3,0,2,7) | 0.14 | 0.19 | Better, variance needs trimming |

### Final models (Box-Cox + diff + winsorized, cross-validated)

| Model | In-sample MSE | Train MSE | Test MSE | Verdict |
|---|---|---|---|---|
| SARIMA(4,0,6)(3,0,2,7) | 0.0910 | 0.07 | 0.17 | Stable + good seasonality |
| SARIMA(5,0,5)(3,0,2,7) | 0.0896 | 0.07 | 0.17 | Similar, slightly higher AIC/BIC |
| SARIMA(4,0,6)(1,0,1,7) | 0.0902 | 0.07 | 0.17 | **SELECTED - best generaliser** |
| SARIMA(6,0,0)(1,0,1,7) | 0.0899 | 0.07 | 0.17 | Good post-spike, weak MA |

### Why SARIMA(4,0,6)(1,0,1,7) was selected

All four final models achieved similar in-sample MSE (~0.089-0.091). The selection was based on generalisation behaviour:

- SARIMA(4,0,6)(1,0,1,7) avoids overfitting to short-term noise with its simpler seasonal structure (1,0,1,7) versus (3,0,2,7)
- SARIMA(6,0,0)(1,0,1,7) performs well post-spike but its lack of MA components limits broader applicability
- SARIMA(4,0,6)(1,0,1,7) maintains reliable forecasts across different time periods and is robust beyond specific events like usage spikes
- (4,0,6) was consistently the most stable non-seasonal order across all cross-validation splits

### Cross-validation finding

80% training data outperforms 100% training data:
- 80% train, tested on full data: MSE 0.11
- 100% train, tested on full data: MSE 0.19

Reason: the model learns the trend and seasonality before and after the spike well. Post-spike it learns less due to increased noise. Beyond the dip it learns very little due to no trend, slight seasonality, and near-pure noise. Training on 80% boundary avoids the post-dip regime contaminating coefficient estimation.

---

## Forecast Conclusion

The SARIMA forecast for Figma pageviews over the next year (Jun 2025 - Jun 2026) shows a clear and steady decline in user activity with consistent weekly seasonality. The model captures the cyclical pattern of usage - higher engagement on weekdays, lower on weekends - while projecting an overall downward trend. By mid-2026 daily pageviews are expected to fall to very low levels.

This forecast suggests the current decline is not temporary but part of a broader structural trend following the failed Adobe acquisition. The absence of any predicted recovery implies continued diminishing user interest based on historical data alone.

Limitations: SARIMA does not account for external factors such as product updates, user acquisition efforts, or organisational shifts. This projection is a baseline scenario under stable conditions.

---

## Further Improvements

Classical SARIMA is inherently limited in handling abrupt structural shifts, irregular anomalies, and nonlinear dependencies. To move beyond this ceiling:

- SARIMA + XGBoost hybrid to capture residual nonlinear patterns
- LSTM for longer-range sequential dependencies
- Prophet with external regressors (product release dates, marketing events)
- Intervention modelling to explicitly model the September 2022 structural break

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Decomposition method | STL over classical | LOESS handles spike as localised anomaly, not global distortion |
| Decomposition period | 7 (weekly) | ACF shows dominant weekly autocorrelation, confirmed by STL |
| Stationarity tests | ADF + KPSS + Ljung-Box | Complementary null hypotheses - ADF alone would mislead here |
| Variance stabilisation | Box-Cox | Heteroskedasticity from spike regime |
| Outlier treatment | Winsorization over IQR | Less harsh, preserves distributional shape |
| Differencing order | d=1 | ADF confirms stationarity after single differencing |
| Seasonal period | S=7 | Weekly pageview cycle confirmed by STL and ACF lags 7, 14, 21 |
| Train split | 80% | Post-dip noise regime degrades fit if included; 80% MSE=0.11 vs 100% MSE=0.19 |
| Model selection | AIC/BIC + cross-validation | Information criterion for order search, CV for split validation |
| Final model | SARIMA(4,0,6)(1,0,1,7) | Best generaliser across splits, simpler seasonal structure avoids overfitting |

---

## Dependencies

```
pandas       >=1.5.0   -- time series data manipulation
numpy        >=1.23.0  -- numerical operations
matplotlib   >=3.6.0   -- plotting
statsmodels  >=0.14.0  -- STL, SARIMA, ADF, KPSS, Ljung-Box
scipy        >=1.10.0  -- Box-Cox, inv_boxcox, winsorize
scikit-learn >=1.2.0   -- MSE evaluation
pmdarima     >=2.0.0   -- auto_arima stepwise search
```

---

## Author

**Arunjeet Chakraborty**
MSc Data Science, University of Bristol (2024-2025)
[LinkedIn](https://www.linkedin.com/in/arunjeet) · [GitHub](https://github.com/Arunjeet)
