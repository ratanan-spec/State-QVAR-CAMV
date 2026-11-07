#!/usr/bin/env python3
"""Fit strict-boundary Q-BART quantile models and forecast next-day returns.

Run once per tau (recommended grid: .05, .25, .50, .75, .95). Training rows
are selected by target date, not feature-row date, so no lead-1 response from
the first forecast day can enter the fit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import pymc_bart.bart as bart_module
from scipy.optimize import linprog


class _LocalManager:
    def list(self):
        return []


bart_module.Manager = _LocalManager
ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="model_panel.csv with date, *_ret, lag1, and lead1 columns")
    p.add_argument("--out", required=True, help="Output directory for this tau")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--train-fraction", type=float, default=.80)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--trees", type=int, default=20)
    p.add_argument("--alpha", type=float, default=.95)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=94017)
    args = p.parse_args()
    if not 0 < args.tau < 1 or not 0 < args.train_fraction < 1:
        raise ValueError("tau and train-fraction must lie strictly between 0 and 1")

    xcols = [f"{a}_ret" for a in ASSETS]
    ycols = [f"{a}_ret_lead1" for a in ASSETS]
    rcols = [f"{a}_ret" for a in ASSETS]
    data_path = Path(args.data).expanduser().resolve()
    raw = pd.read_csv(data_path)
    required = ["date", *xcols, *ycols, *rcols]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    raw["date"] = pd.to_datetime(raw.date, utc=True, errors="coerce")
    raw = raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    raw["target_date"] = raw["date"].shift(-1)
    # Audit the lead-1 construction against the actual return on target_date.
    date_to_ret = raw.set_index("date")[rcols]
    for asset, ycol, rcol in zip(ASSETS, ycols, rcols):
        expected = raw["target_date"].map(date_to_ret[rcol])
        mask = expected.notna() & raw[ycol].notna()
        err = np.max(np.abs(expected[mask].to_numpy(float) - raw.loc[mask, ycol].to_numpy(float)))
        if err > 1e-6:
            raise ValueError(f"{ycol} does not align with {rcol} on target_date; max error={err:g}")

    rows = raw[["date", "target_date", *xcols, *ycols]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    all_dates = np.sort(raw.date.dropna().unique())
    cut_i = int(args.train_fraction * len(all_dates))
    if cut_i <= 0 or cut_i >= len(all_dates):
        raise ValueError("train-fraction leaves no forecast period")
    cutoff = pd.Timestamp(all_dates[cut_i])
    train = rows.loc[rows.target_date < cutoff].reset_index(drop=True)
    forecast = rows.loc[rows.target_date >= cutoff].reset_index(drop=True)
    if train.empty or forecast.empty:
        raise ValueError("Target-date split produced an empty training or forecast set")
    if not (train.target_date.max() < cutoff <= forecast.target_date.min()):
        raise AssertionError("Target-date boundary is not strictly separated")
    if not (forecast.date < forecast.target_date).all():
        raise ValueError("Every forecast origin must precede its target date")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Xraw = train[xcols].to_numpy(float)
    xmean, xsd = Xraw.mean(axis=0), Xraw.std(axis=0, ddof=1)
    if np.any(~np.isfinite(xsd)) or np.any(xsd <= 0):
        raise ValueError("Training predictors have invalid standard deviations")
    X = (Xraw - xmean) / xsd
    Xdesign = np.column_stack([np.ones(len(X)), X])
    Xfraw = forecast[xcols].to_numpy(float)
    Xf = (Xfraw - xmean) / xsd
    Xfdesign = np.column_stack([np.ones(len(Xf)), Xf])

    fit_diags, output_frames = [], []
    versions = {"python": __import__("sys").version.split()[0], "pymc": pm.__version__,
                "pymc_bart": pmb.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    for target_i, (asset, ycol) in enumerate(zip(ASSETS, ycols)):
        yraw = train[ycol].to_numpy(float)
        ymean, ysd = float(yraw.mean()), float(yraw.std(ddof=1))
        y = (yraw - ymean) / ysd
        n, pcols = Xdesign.shape
        objective = np.r_[np.zeros(pcols), np.full(n, args.tau), np.full(n, 1 - args.tau)]
        constraints = np.hstack([Xdesign, np.eye(n), -np.eye(n)])
        qr = linprog(objective, A_eq=constraints, b_eq=y,
                     bounds=[(None, None)] * pcols + [(0, None)] * (2 * n), method="highs")
        if not qr.success:
            raise RuntimeError(f"Linear quantile offset failed for {asset}: {qr.message}")
        beta_q = qr.x[:pcols]
        base_fit = Xdesign @ beta_q
        residual = y - base_fit
        ald_scale = max(float(np.mean(np.maximum(args.tau * residual, (args.tau - 1) * residual))), .05)
        base_forecast = Xfdesign @ beta_q

        pred_chains = np.empty((args.chains, args.draws, len(forecast)), dtype=float)
        for chain_i in range(args.chains):
            bart_module.Manager = _LocalManager
            with pm.Model() as model:
                mu = pmb.BART("mu", X, residual, m=args.trees, alpha=args.alpha, beta=args.beta)
                pm.AsymmetricLaplace("obs", mu=base_fit + mu, b=np.sqrt(args.tau * (1 - args.tau)) / ald_scale, q=args.tau, observed=y)
                idata = pm.sample(tune=args.tune, draws=args.draws, chains=1, cores=1,
                                  compute_convergence_checks=False, progressbar=False,
                                  random_seed=args.seed + 100 * target_i + chain_i)
            sampler = pmb.utils._get_posterior_sampler(mu.owner.op)
            if len(sampler._chain_samplers) != 1:
                raise RuntimeError("Expected one retained BART sampler for each independent chain")
            raw_pred = sampler._chain_samplers[0].sample_posterior(Xf, list(range(args.draws)), None)
            pred_z = np.asarray(raw_pred).transpose(0, 2, 1)[:, :, 0] + base_forecast[None, :]
            pred_chains[chain_i] = pred_z * ysd + ymean
            del raw_pred, pred_z, sampler, idata, model, mu
            gc.collect()

        pred_idata = az.from_dict({"posterior": {"qforecast": pred_chains}})
        rhat = np.asarray(az.rhat(pred_idata, var_names=["qforecast"]).to_dataset()["qforecast"].values).reshape(-1)
        flat = pred_chains.reshape(-1, len(forecast))
        frame = pd.DataFrame({
            "origin_date": forecast.date.dt.strftime("%Y-%m-%d").to_numpy(),
            "target_date": forecast.target_date.dt.strftime("%Y-%m-%d").to_numpy(),
            "asset": asset, "tau": args.tau,
            "quantile_forecast": flat.mean(axis=0),
            "posterior_ci_lower": np.quantile(flat, .025, axis=0),
            "posterior_ci_upper": np.quantile(flat, .975, axis=0),
            "forecast_rhat": rhat,
        })
        output_frames.append(frame)
        fit_diags.append({"asset": asset, "tau": args.tau, "forecast_rows": len(forecast),
                          "forecast_rhat_max": float(np.nanmax(rhat)),
                          "forecast_rhat_p95": float(np.nanquantile(rhat, .95)),
                          "training_quantile_rows": len(train), "ald_scale": ald_scale})
        print(f"{asset} tau={args.tau:.2f}; forecast Rhat max={np.nanmax(rhat):.4f}", flush=True)
        del pred_chains, pred_idata, flat
        gc.collect()

    forecasts = pd.concat(output_frames, ignore_index=True)
    forecasts.to_csv(out / f"qbart_forecasts_tau_{args.tau:.2f}.csv", index=False)
    pd.DataFrame(fit_diags).to_csv(out / f"forecast_diagnostics_tau_{args.tau:.2f}.csv", index=False)
    metadata = {
        "model": "linear quantile offset + PyMC-BART residual correction with Asymmetric-Laplace likelihood",
        "tau": args.tau, "assets": ASSETS, "features": xcols,
        "training_origin_end": train.date.max().strftime("%Y-%m-%d"),
        "training_target_end": train.target_date.max().strftime("%Y-%m-%d"),
        "forecast_origin_start": forecast.date.min().strftime("%Y-%m-%d"),
        "forecast_target_start": forecast.target_date.min().strftime("%Y-%m-%d"),
        "forecast_target_end": forecast.target_date.max().strftime("%Y-%m-%d"),
        "target_date_cutoff": cutoff.strftime("%Y-%m-%d"),
        "training_rows": len(train), "forecast_rows_per_asset": len(forecast),
        "split_rule": "train only when target_date < cutoff; first forecast target_date >= cutoff",
        "lead_alignment_max_abs_error": "checked against actual returns on the next row date",
        "chains": args.chains, "draws": args.draws, "tune": args.tune, "trees": args.trees,
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": versions,
        "interpretation": "quantile-location forecast; not a conditional mean; aggregate across quantile levels for expected-return approximation",
    }
    (out / f"forecast_manifest_tau_{args.tau:.2f}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved Q-BART forecasts to {out.resolve()}")


if __name__ == "__main__":
    main()
