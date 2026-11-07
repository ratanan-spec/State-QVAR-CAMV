# Implementation audit dated 7 November 2026

Two discrepancies were found by reading the recovered fitting source and checking the data columns. The manuscript's old empirical outputs cannot be presented as validated results for the corrected one-step/check-loss-scale implementation.

1. **Likelihood parameterization.** The stated density is tau(1-tau)/sigma * exp[-rho_tau(u)/sigma]. PyMC AsymmetricLaplace uses kappa=sqrt(tau/(1-tau)) and density b/(kappa+1/kappa) * exp[-b*u*sign(u)*kappa**sign(u)]. Thus `b = sqrt(tau*(1-tau))/sigma`. Passing the mean training check loss directly as `b` does not implement the stated sigma. The corrected fit and forecast runners apply the conversion. Tests compare actual PyMC log densities with the check-loss expression at five quantiles.
2. **Time alignment.** The supplied data have `ret_lag1[t]=ret[t-1]` and `ret_lead1[t]=ret[t+1]`. The old fit selects the lag1 column, so its predictor and response are two calendar days apart. Version 0.2 instead selects current `ret[t]` to predict `ret[t+1]`. Its simulation generator is changed consistently from a two-step recursion to a one-step recursion.
3. **Terminal target.** The source return panel drops its last lead-aligned row. A locally reconstructed terminal return is taken from the already stored final lead value, not invented or newly collected. This lets target-date alignment retain the final available response. Raw market data are not distributed.
4. **Interpretation.** These checks fix implementation consistency. They do not establish convergence, unbiasedness, low RMSE, calibrated coverage, covariance robustness, or portfolio value. All dependent results must be recomputed. No thinning, resampling, relabeling, or removal of failing coordinates is used to manufacture an R-hat pass.
5. **Holdout.** Previously inspected dates cannot be made untouched retrospectively. There is no confirmatory CAMV performance claim in this release. New reserved data would be required for one.

PyMC source: https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.AsymmetricLaplace.html
