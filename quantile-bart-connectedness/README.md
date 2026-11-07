# Quantile BART connectedness experimental version 0.2

This directory supplies the corrected **one-step** quantile-BART finite-step connectedness implementation and a reproducible validation harness. It is experimental research software. The implementation audit found likelihood-scale and time-alignment discrepancies in the earlier working package. Legacy manuscript numbers do not validate the corrected implementation.

## Run

Use Python 3.12 in an isolated environment, then install `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python tests/test_parameterization.py
python code/download_panel.py --start 2018-05 --end 2026-10 --out local_data
python code/fit_qbart_connectedness_strict.py --data local_data/model_panel.csv --out results/q50 --tau 0.50 --target-date-cutoff 2025-02-19 --chains 4 --tune 6000 --draws 6000 --states 40
python code/run_recovery_validation.py --out results/recovery --replicates 200 --observations 1000 --tune 6000 --draws 6000
```

The repeated study uses all three DGPs and quantiles by default. It performs actual fits, preserves failed cells, and reports edge-specific errors. A generated protocol, successful unit checks, or a one-replication smoke run is not a completed recovery study.

Raw Binance data are not bundled. The downloader documents access and retains a retrieval manifest. Observe the data provider's access and redistribution terms.

## Scope and gates

The four-chain network and surface diagnostics must be checked on the exact stored draws. Numerical passing thresholds are R-hat <1.01 with adequate bulk ESS; successful algebra does not establish sampling coverage. Corrected empirical results are exploratory until their diagnostics and recovery tests pass. The repeated validation driver retains unconverged and failed fits and does not silently compute a primary success-only estimator.

The current main intervals condition on covariance. Full shared-block refits across quantiles are **not yet completed**; fixed-surface residual resampling is not an equivalent replacement. No fully validated forecast superiority or confirmatory portfolio result is claimed. The dates used previously for forecast analysis are not an untouched portfolio holdout.

See `docs/IMPLEMENTATION_AUDIT.md` and the completed results included with this release. A public code release does not imply that every methodological validation item has passed.

Software citation: Tanattrin Bunnag, Quantile BART Connectedness, experimental version 0.2, 2026. Cite the immutable Git commit for the exact version used. No journal publication DOI is claimed for this experimental method.
