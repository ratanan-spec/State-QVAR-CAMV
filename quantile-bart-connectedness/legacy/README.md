# Archived implementation for audit only

The file in this directory preserves the recovered pre-audit fitting source. It passes the check-loss scale directly as PyMC `b`, and selects `ret_lag1[t]` for `ret_lead1[t]`. It therefore differs from the stated check-loss-scale, one-step implementation. Do not use its results as validated output of version 0.2. The corrected source is under `code/`.
