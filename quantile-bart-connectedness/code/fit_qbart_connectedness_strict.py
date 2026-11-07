"""Strict target-date fit for quantile-BART response-network connectedness.

This is a reproducible exploratory quantile-local connectedness experiment, not
a validated or structural-causality estimator. It fits one linear quantile offset
plus an asymmetric-Laplace BART residual correction per target, then applies a
symmetric finite-step secant, frozen-state recursion, and generalized FEVD with
a fixed training-sample residual covariance. Report all convergence diagnostics;
individual tail-surface coordinates and edges may remain above strict thresholds.
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


def cap_matrix(a: np.ndarray, cap: float) -> tuple[np.ndarray, float, float]:
    raw = float(np.max(np.abs(np.linalg.eigvals(a))))
    if raw <= cap:
        return a, raw, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if float(np.max(np.abs(np.linalg.eigvals(mid * a)))) <= cap:
            lo = mid
        else:
            hi = mid
    return lo * a, raw, lo


def local_gfevd(a: np.ndarray, sigma: np.ndarray, horizon: int) -> np.ndarray:
    """Generalized FEVD for the frozen local linearization at one state."""
    k = a.shape[0]
    power = np.eye(k)
    numerator = np.zeros((k, k))
    denominator = np.zeros(k)
    # Include h=0 so a zero local response still assigns own-shock variance.
    for _ in range(horizon):
        denominator += np.diag(power @ sigma @ power.T)
        for j in range(k):
            numerator[:, j] += (power @ sigma[:, j]) ** 2 / sigma[j, j]
        power = a @ power
    raw = numerator / np.maximum(denominator[:, None], 1e-14)
    row_sum = raw.sum(axis=1, keepdims=True)
    return np.divide(raw, row_sum, out=np.eye(k), where=row_sum > 1e-14)


def local_gfevd_batch(a: np.ndarray, sigma: np.ndarray, horizons: list[int]) -> dict[int, np.ndarray]:
    """Vectorized generalized FEVD for arrays shaped draw x state x k x k."""
    k = a.shape[-1]
    requested = set(int(h) for h in horizons)
    power = np.broadcast_to(np.eye(k), a.shape).copy()
    numerator = np.zeros((*a.shape[:-2], k, k), dtype=float)
    denominator = np.zeros((*a.shape[:-2], k), dtype=float)
    result = {}
    for step in range(1, max(requested) + 1):
        ps = power @ sigma
        denominator += np.einsum("...ij,...ij->...i", ps, power, optimize=True)
        for j in range(k):
            shock = power @ sigma[:, j]
            numerator[..., j] += shock**2 / sigma[j, j]
        if step in requested:
            raw = numerator / np.maximum(denominator[..., None], 1e-14)
            rows = raw.sum(axis=-1, keepdims=True)
            result[step] = np.divide(raw, rows, out=np.broadcast_to(np.eye(k), raw.shape).copy(), where=rows > 1e-14)
        power = a @ power
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to model_panel.csv (not bundled with this package)")
    parser.add_argument("--out", default="crypto_quantile_pilot/recovery_q50")
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--target-date-cutoff", default=None, help="Optional explicit UTC date; training rows require target_date strictly earlier")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--states", type=int, default=40)
    parser.add_argument("--step-sd", type=float, default=0.50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--sensitivity-h", type=float, nargs="*", default=None)
    parser.add_argument("--sensitivity-horizons", type=int, nargs="*", default=None)
    parser.add_argument("--sensitivity-states", type=int, nargs="*", default=None)
    parser.add_argument("--sensitivity-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=64017)
    args = parser.parse_args()
    if not 0 < args.tau < 1:
        raise ValueError("tau must lie strictly between zero and one")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xcols = [f"{a}_ret" for a in ASSETS]
    ycols = [f"{a}_ret_lead1" for a in ASSETS]
    rcols = [f"{a}_ret" for a in ASSETS]
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Input data file not found: {data_path}. Pass --data /path/to/model_panel.csv; "
            "the source data are intentionally not included in this ZIP."
        )
    raw_frame = pd.read_csv(data_path)
    raw_frame["date"] = pd.to_datetime(raw_frame["date"], utc=True, errors="coerce")
    raw_frame = raw_frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    raw_frame["target_date"] = raw_frame["date"].shift(-1)
    required_cols = ["date", *xcols, *ycols, *rcols]
    missing_cols = [col for col in required_cols if col not in raw_frame.columns]
    if missing_cols:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing_cols)
            + ". Expected date plus lag-1 predictors and lead-1 returns for BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, and ADAUSDT."
        )
    date_to_ret = raw_frame.set_index("date")[rcols]
    for asset, ycol, rcol in zip(ASSETS, ycols, rcols):
        expected = raw_frame["target_date"].map(date_to_ret[rcol])
        mask = expected.notna() & raw_frame[ycol].notna()
        err = np.max(np.abs(expected[mask].to_numpy(float) - raw_frame.loc[mask, ycol].to_numpy(float)))
        if err > 1e-6:
            raise ValueError(f"Lead alignment failed for {asset}; max error={err:g}")
    frame = raw_frame[["date", "target_date", *xcols, *ycols]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    all_dates = np.sort(raw_frame.date.dropna().unique())
    cut_i = int(args.train_fraction * len(all_dates))
    if args.target_date_cutoff:
        cutoff_date = pd.Timestamp(args.target_date_cutoff, tz="UTC")
        if cutoff_date <= pd.Timestamp(all_dates[0]) or cutoff_date >= pd.Timestamp(all_dates[-1]):
            raise ValueError("target-date-cutoff must lie inside the observed date range")
    else:
        if cut_i <= 0 or cut_i >= len(all_dates):
            raise ValueError("train-fraction leaves no training or holdout dates")
        cutoff_date = pd.Timestamp(all_dates[cut_i])
    # Correct supervised split: train on rows whose *response target date* is
    # strictly before the holdout. The last feature row before cutoff predicts
    # the first holdout return and must not enter the BART fit.
    train = frame.loc[frame.target_date < cutoff_date].reset_index(drop=True)
    if train.empty or train.target_date.max() >= cutoff_date:
        raise AssertionError("Target-date boundary leak detected")
    Xraw = train[xcols].to_numpy(float)
    xmean = Xraw.mean(axis=0)
    xsd = Xraw.std(axis=0, ddof=1)
    if np.any(~np.isfinite(xsd)) or np.any(xsd <= 0):
        bad = [xcols[i] for i in np.where((~np.isfinite(xsd)) | (xsd <= 0))[0]]
        raise ValueError("Predictors have zero or invalid training-sample standard deviation: " + ", ".join(bad))
    X = (Xraw - xmean) / xsd
    Xdesign = np.column_stack([np.ones(len(X)), X])

    # Evaluate the union of predeclared state grids once so h/H/state
    # sensitivity can reuse the same posterior surface draws.
    sensitivity_h = sorted(set([float(args.step_sd), *([float(v) for v in args.sensitivity_h] if args.sensitivity_h else [])]))
    sensitivity_horizons = sorted(set([int(args.horizon), *([int(v) for v in args.sensitivity_horizons] if args.sensitivity_horizons else [])]))
    sensitivity_state_counts = sorted(set([int(args.states), *([int(v) for v in args.sensitivity_states] if args.sensitivity_states else [])]))
    if any(v <= 0 for v in sensitivity_h) or any(v <= 0 for v in sensitivity_horizons):
        raise ValueError("Sensitivity h and horizon values must be positive")
    if args.sensitivity_draws < 1:
        raise ValueError("sensitivity-draws must be positive")
    if any(v <= 0 or v > len(X) for v in sensitivity_state_counts):
        raise ValueError("Sensitivity state counts must lie between 1 and the training sample size")
    state_ids_by_n = {n_states: np.linspace(0, len(X) - 1, n_states, dtype=int) for n_states in sensitivity_state_counts}
    state_ids = np.array(sorted(set(np.concatenate(list(state_ids_by_n.values())).tolist())), dtype=int)
    state_position = {int(row_id): idx for idx, row_id in enumerate(state_ids)}
    Xstate = X[state_ids]
    eval_blocks = [Xstate]
    index = {}
    next_row = len(Xstate)
    for h_value in sensitivity_h:
        for j in range(len(ASSETS)):
            xp, xm = Xstate.copy(), Xstate.copy()
            xp[:, j] += h_value
            xm[:, j] -= h_value
            eval_blocks.extend([xp, xm])
            index[(j, h_value, "+")] = np.arange(next_row, next_row + len(Xstate))
            next_row += len(Xstate)
            index[(j, h_value, "-")] = np.arange(next_row, next_row + len(Xstate))
            next_row += len(Xstate)
    Xeval = np.vstack(eval_blocks)
    Xeval_design = np.column_stack([np.ones(len(Xeval)), Xeval])

    # Per-target posteriors and chain-specific predictions.
    chain_derivs = np.zeros((len(ASSETS), args.chains, args.draws, len(Xstate), len(ASSETS), len(sensitivity_h)), dtype=np.float32)
    residual_matrix = np.zeros((len(train), len(ASSETS)))
    fit_diag = []
    surface_coord_diag = []
    versions = {"python": __import__("sys").version.split()[0], "pymc": pm.__version__,
                "pymc_bart": pmb.__version__, "numpy": np.__version__, "pandas": pd.__version__}

    for target_i, asset in enumerate(ASSETS):
        yraw = train[ycols[target_i]].to_numpy(float)
        ymean = float(yraw.mean())
        ysd = float(yraw.std(ddof=1))
        y = (yraw - ymean) / ysd
        # Fit a training-only linear quantile baseline. BART then learns the
        # nonlinear residual correction, which materially improves mixing on
        # the noisy crypto-return panel and keeps the quantile location model
        # centered near its data-supported value.
        n, p = Xdesign.shape
        objective = np.r_[np.zeros(p), np.full(n, args.tau), np.full(n, 1 - args.tau)]
        constraints = np.hstack([Xdesign, np.eye(n), -np.eye(n)])
        qr = linprog(
            objective, A_eq=constraints, b_eq=y,
            bounds=[(None, None)] * p + [(0, None)] * (2 * n), method="highs",
        )
        if not qr.success:
            raise RuntimeError(f"Linear quantile offset failed for {asset}: {qr.message}")
        beta_q = qr.x[:p]
        base_fit = Xdesign @ beta_q
        residual = y - base_fit
        ald_scale = max(float(np.mean(np.maximum(args.tau * residual, (args.tau - 1) * residual))), 0.05)
        base_eval = Xeval_design @ beta_q
        mu_chains = []
        chain_samplers = []
        for chain_i in range(args.chains):
            # Fit each chain in a fresh single-chain model. With cores=1, the
            # standard multi-chain call leaves only the last chain's tree
            # history on the BART Op, even though all chains appear in idata.
            # Independent one-chain fits preserve each chain's actual trees.
            bart_module.Manager = _LocalManager
            with pm.Model() as model:
                mu = pmb.BART("mu", X, residual, m=args.trees, alpha=args.alpha, beta=args.beta)
                pm.AsymmetricLaplace("obs", mu=base_fit + mu, b=np.sqrt(args.tau * (1 - args.tau)) / ald_scale, q=args.tau, observed=y)
                idata = pm.sample(
                    tune=args.tune, draws=args.draws, chains=1, cores=1,
                    compute_convergence_checks=False, progressbar=False,
                    random_seed=args.seed + 100 * target_i + chain_i,
                )
            mu_chains.append(np.asarray(idata.posterior.to_dataset()["mu"].values[0]))
            sampler = pmb.utils._get_posterior_sampler(mu.owner.op)
            if len(sampler._chain_samplers) != 1:
                raise RuntimeError("Expected one preserved BART tree history per independent chain")
            chain_samplers.append(sampler._chain_samplers[0])
            del idata, model, mu

        mu_values = np.stack(mu_chains, axis=0)
        qfit_values = mu_values + base_fit[None, None, :]
        qfit_idata = az.from_dict({"posterior": {"qfit": qfit_values}})
        rh = az.rhat(qfit_idata, var_names=["qfit"]).to_dataset()["qfit"].values
        ess = az.ess(qfit_idata, var_names=["qfit"], method="bulk").to_dataset()["qfit"].values
        surface_coord_diag.append(pd.DataFrame({
            "asset": asset,
            "tau": args.tau,
            "row_in_clean_panel": np.arange(len(train)),
            "date": train["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "rhat": np.asarray(rh).reshape(-1),
            "bulk_ess": np.asarray(ess).reshape(-1),
            "is_local_state": np.isin(np.arange(len(train)), state_ids),
        }))
        mu_bar = mu_values.mean(axis=(0, 1))
        full_fit = base_fit + mu_bar
        residual_matrix[:, target_i] = y - full_fit
        fit_diag.append({
            "asset": asset, "tau": args.tau,
            "qfit_rhat_median": float(np.nanmedian(rh)),
            "qfit_rhat_p95": float(np.nanquantile(rh, 0.95)),
            "qfit_rhat_max": float(np.nanmax(rh)),
            "qfit_bulk_ess_min": float(np.nanmin(ess)),
            "qfit_bulk_ess_median": float(np.nanmedian(ess)),
            "ald_scale_fixed_from_training_check_loss": ald_scale,
        })

        for chain_i, chain_sampler in enumerate(chain_samplers):
            # Preserve temporal order of posterior draws for valid MCMC
            # diagnostics; do not resample indices with replacement here.
            raw = chain_sampler.sample_posterior(Xeval, list(range(args.draws)), None)
            # PosteriorSampler returns draw x output x evaluation-row.
            draw = np.asarray(raw).transpose(0, 2, 1)[:, :, 0] + base_eval[None, :]
            for h_idx, h_value in enumerate(sensitivity_h):
                for sender_i in range(len(ASSETS)):
                    plus = draw[:, index[(sender_i, h_value, "+")]]
                    minus = draw[:, index[(sender_i, h_value, "-")]]
                    chain_derivs[target_i, chain_i, :, :, sender_i, h_idx] = (plus - minus) / (2 * h_value)
        print(f"Finished {asset} at tau={args.tau:.2f}; full quantile-surface Rhat p95="
              f"{fit_diag[-1]['qfit_rhat_p95']:.3f}", flush=True)
        # Long tail runs retain large posterior surfaces and BART tree histories.
        # Release equation-local objects before fitting the next target.
        del chain_samplers, mu_values, qfit_idata, rh, ess, mu_bar, full_fit, raw, draw, chain_sampler
        gc.collect()

    # Archive raw operators for downstream checks without refitting or scrambling draws.
    np.savez_compressed(out / "operator_draws.npz", derivatives=chain_derivs, residuals=residual_matrix, state_ids=state_ids, training_x=X, sensitivity_h=np.asarray(sensitivity_h))

    # Estimate fixed quantile-residual covariance from training data only.
    sigma = np.cov(residual_matrix, rowvar=False, ddof=1)
    sigma = (sigma + sigma.T) / 2
    min_eig = float(np.linalg.eigvalsh(sigma).min())
    if min_eig <= 1e-8:
        sigma += (1e-8 - min_eig + 1e-8) * np.eye(len(ASSETS))

    # Reuse each fitted surface draw over the declared h / state / horizon grid.
    sensitivity_rows, sensitivity_edge_rows = [], []
    baseline = None
    sensitivity_draw_ids = np.linspace(0, args.draws - 1, min(args.sensitivity_draws, args.draws), dtype=int)
    for h_idx, h_value in enumerate(sensitivity_h):
        # Vectorize the matrix cap and FEVD over draws and states. This keeps
        # the sensitivity grid computationally tractable without refitting BART.
        full_matrices = chain_derivs[:, :, :, :, :, h_idx].transpose(1, 2, 3, 0, 4).astype(float, copy=False)
        eig = np.linalg.eigvals(full_matrices)
        radius = np.max(np.abs(eig), axis=-1)
        shrink = np.minimum(1.0, np.divide(0.98, radius, out=np.ones_like(radius), where=radius > 0))
        capped = full_matrices * shrink[..., None, None]
        for n_states in sensitivity_state_counts:
            positions = np.array([state_position[int(i)] for i in state_ids_by_n[n_states]], dtype=int)
            rows_by_horizon = {}
            for horizon_value in sensitivity_horizons:
                is_baseline = (h_value == args.step_sd and n_states == args.states and horizon_value == args.horizon)
                draw_ids = np.arange(args.draws) if is_baseline else sensitivity_draw_ids
                # Preserve chain boundaries for R-hat on the baseline; sensitivity
                # intervals use the declared deterministic draw subsample.
                chain_edges = []
                chain_tci = []
                cap_count = 0
                operator_count = 0
                for c in range(args.chains):
                    a = capped[c, draw_ids][:, positions, :, :]
                    shares_by_h = local_gfevd_batch(a, sigma, [horizon_value])
                    shares = shares_by_h[horizon_value].mean(axis=1)
                    shares = np.divide(shares, shares.sum(axis=-1, keepdims=True), out=np.zeros_like(shares), where=shares.sum(axis=-1, keepdims=True) > 1e-14)
                    chain_edges.append(shares)
                    chain_tci.append(100 * (shares.sum(axis=(-1, -2)) - np.trace(shares, axis1=-2, axis2=-1)) / len(ASSETS))
                    cap_count += int(np.sum(shrink[c, draw_ids][:, positions] < 1.0 - 1e-12))
                    operator_count += int(len(draw_ids) * len(positions))
                tci_grid = np.stack(chain_tci)
                edges_grid = np.stack(chain_edges)
                row = {
                    "tau": args.tau, "step_sd": h_value, "states": n_states, "horizon": horizon_value,
                    "draws_per_chain": int(len(draw_ids)), "sensitivity_draw_subsample": not is_baseline,
                    "tci_mean_pct": float(tci_grid.mean()),
                    "tci_q025_pct": float(np.quantile(tci_grid, 0.025)),
                    "tci_q975_pct": float(np.quantile(tci_grid, 0.975)),
                    "stability_cap_activation_rate": float(cap_count / max(operator_count, 1)),
                }
                sensitivity_rows.append(row)
                for i, receiver in enumerate(ASSETS):
                    for j, sender in enumerate(ASSETS):
                        vals = edges_grid[:, :, i, j].ravel()
                        sensitivity_edge_rows.append({**row, "receiver": receiver, "sender": sender,
                            "edge_mean_pct": float(vals.mean()), "edge_q025_pct": float(np.quantile(vals, .025)),
                            "edge_q975_pct": float(np.quantile(vals, .975))})
                if is_baseline:
                    baseline = (tci_grid, edges_grid)
        del full_matrices, eig, radius, shrink, capped
    if baseline is None:
        raise RuntimeError("Baseline h/state/horizon combination missing from sensitivity grid")
    tci, edges = baseline
    # Baseline-only diagnostics retain the established output names.
    radii, shrinkages = [], []
    base_h_idx = sensitivity_h.index(float(args.step_sd))
    base_positions = np.array([state_position[int(i)] for i in state_ids_by_n[int(args.states)]], dtype=int)
    for c in range(args.chains):
        for d in range(args.draws):
            for s in base_positions:
                _, raw, shrink = cap_matrix(chain_derivs[:, c, d, s, :, base_h_idx], 0.98)
                radii.append(raw); shrinkages.append(shrink)

    net_idata = az.from_dict({"posterior": {"tci": tci, "edges": edges}})
    tci_rhat = float(az.rhat(net_idata, var_names=["tci"]).to_dataset()["tci"].values)
    tci_ess = float(az.ess(net_idata, var_names=["tci"], method="bulk").to_dataset()["tci"].values)
    edge_rhat = az.rhat(net_idata, var_names=["edges"]).to_dataset()["edges"].values
    edge_ess = az.ess(net_idata, var_names=["edges"], method="bulk").to_dataset()["edges"].values

    net_summary = pd.DataFrame([{
        "tau": args.tau, "tci_quantile_local_gfevd_mean_pct": float(tci.mean()),
        "tci_q025_pct": float(np.quantile(tci, 0.025)), "tci_q975_pct": float(np.quantile(tci, 0.975)),
        "tci_rhat": tci_rhat, "tci_bulk_ess": tci_ess,
        "edge_rhat_p95": float(np.nanquantile(edge_rhat, 0.95)), "edge_rhat_max": float(np.nanmax(edge_rhat)),
        "edge_bulk_ess_min": float(np.nanmin(edge_ess)),
        "edge_bulk_ess_median": float(np.nanmedian(edge_ess)),
        "median_raw_radius": float(np.median(radii)), "median_stability_shrink": float(np.median(shrinkages)),
        "states": args.states, "step_sd": args.step_sd, "horizon": args.horizon,
        "interpretation": "exploratory quantile-local GFEVD under frozen-state linearization and fixed residual covariance; predictive, not causal",
    }])
    edge_rows = []
    for i, receiver in enumerate(ASSETS):
        for j, sender in enumerate(ASSETS):
            vals = edges[:, :, i, j].ravel()
            edge_rows.append({
                "tau": args.tau, "receiver": receiver, "sender": sender,
                "mean_share": float(vals.mean()), "q025": float(np.quantile(vals, .025)),
                "q975": float(np.quantile(vals, .975)), "rhat": float(edge_rhat[i, j]),
                "bulk_ess": float(edge_ess[i, j]),
            })

    pd.DataFrame(fit_diag).to_csv(out / "fit_diagnostics.csv", index=False)
    coord_diag = pd.concat(surface_coord_diag, ignore_index=True)
    coord_diag.to_csv(out / "surface_coordinate_diagnostics.csv", index=False)
    coord_diag.sort_values("rhat", ascending=False).groupby("asset", as_index=False, group_keys=False).head(20).to_csv(
        out / "surface_worst_coordinates.csv", index=False
    )
    net_summary.to_csv(out / "network_diagnostics.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "edge_diagnostics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "sensitivity_grid.csv", index=False)
    pd.DataFrame(sensitivity_edge_rows).to_csv(out / "sensitivity_edges.csv", index=False)
    np.savez_compressed(out / "posterior_response_network_draws.npz", tci=tci, edges=edges)
    metadata = {
        "model": "separate linear-quantile-offset plus PyMC-BART residual-correction equations with asymmetric-Laplace working likelihood",
        "tau": args.tau, "assets": ASSETS, "features": xcols, "lag_order": 1,
        "input_file_name": data_path.name,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_rows": len(frame), "train_rows": len(train),
        "train_period": [str(train.date.iloc[0].date()), str(train.date.iloc[-1].date())],
        "train_target_period": [str(train.target_date.iloc[0].date()), str(train.target_date.iloc[-1].date())],
        "target_date_cutoff": str(cutoff_date.date()),
        "split_rule": "training rows require target_date < cutoff; no lead-1 response from holdout enters fit",
        "chains": args.chains, "draws_per_chain": args.draws, "tune_per_chain": args.tune,
        "states": args.states, "step_sd": args.step_sd, "horizon": args.horizon,
        "sensitivity_grid": {"h": sensitivity_h, "states": sensitivity_state_counts,
                             "horizons": sensitivity_horizons,
                             "posterior_draws_per_chain_for_nonbaseline_cells": min(args.sensitivity_draws, args.draws)},
        "trees": args.trees, "alpha": args.alpha, "beta": args.beta,
        "standardization": "training-sample mean and standard deviation for predictors and each outcome",
        "ald_scale": "check-loss scale sigma; PyMC b = sqrt(tau*(1-tau))/sigma",
        "operator": "symmetric finite-step secants of quantile BART at fixed training states; one-lag frozen recursion",
        "decomposition": "Pesaran-Shin generalized FEVD applied to the local operator and full quantile-residual covariance estimated on training data; h=0 included; shares row-normalized",
        "network_limit": "exploratory quantile-local predictive decomposition; fixed covariance and frozen-state approximation; not structural causality",
        "versions": versions,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nNETWORK SUMMARY\n", net_summary.to_string(index=False), flush=True)
    print("\nFIT DIAGNOSTICS\n", pd.DataFrame(fit_diag).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
