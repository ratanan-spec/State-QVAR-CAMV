"""Generate a known-truth DGP and optionally run strict Q-BART fits.

The generator and oracle truth calculation need NumPy/Pandas only. Passing
--run-fits additionally needs the exact PyMC/PyMC-BART/ArviZ stack required by
fit_qbart_connectedness_strict.py. The output is a Monte Carlo harness, not a
claim that estimation has passed; repeat count and MCMC length must be chosen
for available compute and a predeclared paper protocol.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
XCOLS = [f"{a}_ret" for a in ASSETS]
YCOLS = [f"{a}_ret_lead1" for a in ASSETS]
RCOLS = [f"{a}_ret" for a in ASSETS]
NORMAL_QZ = {0.05: -1.6448536269514729, 0.50: 0.0, 0.95: 1.6448536269514722}


def location(x: np.ndarray, dgp: str) -> np.ndarray:
    b = np.array([
        [.28, .08, .00, .00, .00], [.03, .25, .08, .00, .00],
        [.00, .04, .27, .07, .00], [.00, .00, .05, .24, .07],
        [.06, .00, .00, .04, .25],
    ])
    out = x @ b.T
    if dgp != "linear":
        for j in range(5):
            out[:, j] += .055*np.tanh(x[:, (j+1) % 5]) + .018*x[:, j]*x[:, (j+2) % 5]
    return out


def conditional_scale(x: np.ndarray, dgp: str) -> np.ndarray:
    s = np.empty_like(x)
    if dgp == "linear":
        return np.full_like(x, .16)
    for j in range(5):
        s[:, j] = .14 + .035/(1 + np.exp(-np.clip(x[:, j] + .25*x[:, (j+1) % 5], -20, 20)))
    return s


def quantile_z(tau: float, dgp: str) -> float:
    if dgp == "nonlinear_heavytail":
        df = 5.0
        return float(student_t.ppf(tau, df) * np.sqrt((df-2)/df))
    return NORMAL_QZ[tau]


def make_panel(n: int, seed: int, dgp: str) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    corr = .45 ** np.abs(np.subtract.outer(np.arange(5), np.arange(5)))
    chol = np.linalg.cholesky(corr)
    # One-step recursion: row t contains current r[t] and next target r[t+1].
    r = np.zeros((n+1, 5))
    r[0] = rng.normal(0, .15, size=5)
    for t in range(n):
        if dgp == "nonlinear_heavytail":
            df = 5.0
            z = (rng.standard_normal(5) @ chol.T) * np.sqrt((df-2)/rng.chisquare(df))
        else:
            z = rng.standard_normal(5) @ chol.T
        r[t+1] = location(r[t:t+1], dgp)[0] + conditional_scale(r[t:t+1], dgp)[0]*z
    rows = {"date": pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC")}
    for j, asset in enumerate(ASSETS):
        ret = r[:n, j].copy()
        lag = np.r_[np.nan, r[:n-1, j]]
        lead = r[1:n+1, j].copy()
        rows[f"{asset}_ret"] = ret
        rows[f"{asset}_ret_lag1"] = lag
        rows[f"{asset}_ret_lead1"] = lead
    return pd.DataFrame(rows), corr


def cap(a: np.ndarray, radius_cap: float = .98) -> np.ndarray:
    rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a * (min(1.0, radius_cap/rho) if rho > 0 else 1.0)


def gfevd(a: np.ndarray, sigma: np.ndarray, horizon: int) -> np.ndarray:
    k = len(a)
    power = np.eye(k)
    num, den = np.zeros((k, k)), np.zeros(k)
    for _ in range(horizon):
        den += np.diag(power @ sigma @ power.T)
        for shock in range(k):
            num[:, shock] += (power @ sigma[:, shock])**2 / sigma[shock, shock]
        power = a @ power
    raw = num / den[:, None]
    return raw / raw.sum(axis=1, keepdims=True)


def strict_train(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    raw = frame.copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw = raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    raw["target_date"] = raw["date"].shift(-1)
    clean = raw[["date", "target_date", *XCOLS, *YCOLS]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dates = np.sort(raw.date.dropna().unique())
    cutoff = pd.Timestamp(dates[int(train_fraction*len(dates))])
    train = clean.loc[clean.target_date < cutoff].reset_index(drop=True)
    if train.empty or train.target_date.max() >= cutoff:
        raise AssertionError("Strict target-date training boundary failed")
    return train, cutoff


def oracle_network(train: pd.DataFrame, tau: float, states_n: int, step: float, horizon: int, dgp: str) -> tuple[np.ndarray, float, dict]:
    xraw, yraw = train[XCOLS].to_numpy(float), train[YCOLS].to_numpy(float)
    xm, xs = xraw.mean(axis=0), xraw.std(axis=0, ddof=1)
    ym, ys = yraw.mean(axis=0), yraw.std(axis=0, ddof=1)
    xstd = (xraw-xm)/xs
    state_ids = np.linspace(0, len(train)-1, min(states_n, len(train)), dtype=int)
    state_raw = xraw[state_ids]
    qz = quantile_z(tau, dgp)
    mraw = location(xraw, dgp) + qz*conditional_scale(xraw, dgp)
    residual = (yraw-ym)/ys - (mraw-ym)/ys
    sigma = np.cov(residual, rowvar=False, ddof=1)
    sigma = (sigma+sigma.T)/2
    mineig = float(np.linalg.eigvalsh(sigma).min())
    if mineig <= 1e-8:
        sigma += (1e-8-mineig+1e-8)*np.eye(5)
    shares = []
    raw_radius = []
    for x0 in state_raw:
        z = (x0-xm)/xs
        a = np.zeros((5, 5))
        for i in range(5):
            xp, xm_step = z.copy(), z.copy()
            xp[i] += step
            xm_step[i] -= step
            raw_p, raw_m = xm + xs*xp, xm + xs*xm_step
            qp = (location(raw_p[None, :], dgp)[0] + qz*conditional_scale(raw_p[None, :], dgp)[0]-ym)/ys
            qm = (location(raw_m[None, :], dgp)[0] + qz*conditional_scale(raw_m[None, :], dgp)[0]-ym)/ys
            a[:, i] = (qp-qm)/(2*step)
        raw_radius.append(float(np.max(np.abs(np.linalg.eigvals(a)))))
        a = cap(a)
        shares.append(gfevd(a, sigma, horizon))
    phi = np.mean(shares, axis=0)
    phi /= phi.sum(axis=1, keepdims=True)
    tci = float(100*(phi.sum()-np.trace(phi))/5)
    info = {"train_rows": len(train), "states": len(state_ids), "target_cutoff": str(train.target_date.max().date()),
            "mean_raw_spectral_radius": float(np.mean(raw_radius)),
            "max_raw_spectral_radius": float(np.max(raw_radius)),
            "row_sum_max_error": float(np.max(np.abs(phi.sum(axis=1)-1)))}
    return phi, tci, info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/qbart_dgp_monte_carlo")
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--observations", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--dgp", choices=["linear", "nonlinear_gaussian", "nonlinear_heavytail"], default="nonlinear_gaussian")
    p.add_argument("--tau", type=float, nargs="+", default=[.05, .50, .95])
    p.add_argument("--train-fraction", type=float, default=.80)
    p.add_argument("--states", type=int, default=40)
    p.add_argument("--step-sd", type=float, default=.50)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--trees", type=int, default=20)
    p.add_argument("--run-fits", action="store_true", help="run PyMC-BART strict fits after generating each DGP")
    args = p.parse_args()
    if args.replicates < 1 or args.observations < 200:
        raise ValueError("replicates must be positive and observations must be at least 200")
    if any(t not in NORMAL_QZ for t in args.tau):
        raise ValueError("supported tau values: 0.05, 0.50, 0.95")
    if args.run_fits:
        missing = [m for m in ("pymc", "pymc_bart", "arviz") if importlib.util.find_spec(m) is None]
        if missing:
            raise SystemExit("Cannot run BART estimation; missing runtime dependencies: " + ", ".join(missing))
    root = Path(args.out).resolve()
    data_dir, truth_dir, fit_dir = root/"data", root/"truth", root/"fits"
    for d in (data_dir, truth_dir, fit_dir): d.mkdir(parents=True, exist_ok=True)
    fit_script = Path(__file__).with_name("fit_qbart_connectedness_strict.py")
    rows = []

    for rep in range(args.replicates):
        rep_seed = args.seed + rep*10007
        frame, corr = make_panel(args.observations, rep_seed, args.dgp)
        data_path = data_dir/f"dgp_rep_{rep+1:03d}.csv"
        frame.to_csv(data_path, index=False)
        train, cutoff = strict_train(frame, args.train_fraction)
        truth_rows = []
        for tau in args.tau:
            phi, tci, info = oracle_network(train, tau, args.states, args.step_sd, args.horizon, args.dgp)
            truth_path = truth_dir/f"truth_rep_{rep+1:03d}_tau_{tau:.2f}.csv"
            truth = pd.DataFrame([
                {"tau": tau, "receiver": ASSETS[i], "sender": ASSETS[j], "share": phi[i,j]}
                for i in range(5) for j in range(5)
            ])
            truth.to_csv(truth_path, index=False)
            info.update({"replicate": rep+1, "seed": rep_seed, "tau": tau,
                         "oracle_tci_pct": tci, "cutoff_date": str(cutoff.date()),
                         "innovation_correlation": corr.tolist(), "dgp": args.dgp})
            truth_rows.append(info)
            rows.append(info)
            if args.run_fits:
                run_out = fit_dir/f"rep_{rep+1:03d}_tau_{tau:.2f}"
                cmd = [sys.executable, str(fit_script), "--data", str(data_path), "--out", str(run_out),
                       "--tau", str(tau), "--train-fraction", str(args.train_fraction),
                       "--states", str(args.states), "--step-sd", str(args.step_sd),
                       "--horizon", str(args.horizon), "--chains", str(args.chains),
                       "--draws", str(args.draws), "--tune", str(args.tune),
                       "--trees", str(args.trees), "--seed", str(rep_seed + int(tau*1000))]
                subprocess.run(cmd, check=True)
                posterior = np.load(run_out/"posterior_response_network_draws.npz")
                edge_draws = posterior["edges"].reshape(-1, 5, 5)
                tci_draws = posterior["tci"].reshape(-1)
                est_edges = edge_draws.mean(axis=0)
                est_tci = float(tci_draws.mean())
                edge_lo, edge_hi = np.quantile(edge_draws, [.025, .975], axis=0)
                tci_lo, tci_hi = np.quantile(tci_draws, [.025, .975])
                edge_error = est_edges - phi
                tci_error = est_tci - tci
                metrics = {
                    "replicate": rep+1, "seed": rep_seed, "tau": tau,
                    "tci_bias_contribution": tci_error,
                    "tci_squared_error": tci_error**2,
                    "tci_coverage_95": bool(tci_lo <= tci <= tci_hi),
                    "tci_interval_width_95": float(tci_hi-tci_lo),
                    "edge_bias_contribution": float(np.mean(edge_error)),
                    "edge_squared_error_mean": float(np.mean(edge_error**2)),
                    "edge_abs_error_mean": float(np.mean(np.abs(edge_error))),
                    "edge_coverage_95": float(np.mean((edge_lo <= phi) & (phi <= edge_hi))),
                    "edge_interval_width_95_mean": float(np.mean(edge_hi-edge_lo)),
                    "estimated_tci_pct": est_tci, "oracle_tci_pct": tci,
                }
                rows[-1].update(metrics)
                network_diag = pd.read_csv(run_out/"network_diagnostics.csv").iloc[0]
                edge_diag = pd.read_csv(run_out/"edge_diagnostics.csv")
                rows[-1].update({"tci_rhat": float(network_diag.tci_rhat),
                                 "edge_rhat_max": float(edge_diag.rhat.max()),
                                 "edge_bulk_ess_min": float(edge_diag.bulk_ess.min())})

    summary = pd.DataFrame(rows)
    summary.to_csv(root/"monte_carlo_summary.csv", index=False)
    if args.run_fits and not summary.empty:
        def rmse_mcse(squared_error: pd.Series) -> float:
            rmse = float(np.sqrt(squared_error.mean()))
            if len(squared_error) < 2 or rmse == 0:
                return float("nan")
            return float(squared_error.std(ddof=1) / (2 * rmse * np.sqrt(len(squared_error))))

        aggregate_rows = []
        for (dgp, tau), group in summary.groupby(["dgp", "tau"], sort=True):
            nrep = int(len(group))
            tci_coverage = float(group.tci_coverage_95.mean())
            edge_coverage = float(group.edge_coverage_95.mean())
            aggregate_rows.append({
                "dgp": dgp, "tau": tau, "replications": nrep,
                "tci_bias": float(group.tci_bias_contribution.mean()),
                "tci_bias_mcse": float(group.tci_bias_contribution.std(ddof=1)/np.sqrt(nrep)) if nrep > 1 else float("nan"),
                "tci_rmse": float(np.sqrt(group.tci_squared_error.mean())),
                "tci_rmse_mcse_delta": rmse_mcse(group.tci_squared_error),
                "tci_coverage_95": tci_coverage,
                "tci_coverage_mcse": float(np.sqrt(tci_coverage*(1-tci_coverage)/nrep)),
                "tci_interval_width_95_mean": float(group.tci_interval_width_95.mean()),
                "edge_bias_mean": float(group.edge_bias_contribution.mean()),
                "edge_bias_mcse": float(group.edge_bias_contribution.std(ddof=1)/np.sqrt(nrep)) if nrep > 1 else float("nan"),
                "edge_rmse": float(np.sqrt(group.edge_squared_error_mean.mean())),
                "edge_rmse_mcse_delta": rmse_mcse(group.edge_squared_error_mean),
                "edge_coverage_95": edge_coverage,
                "edge_coverage_mcse": float(np.sqrt(edge_coverage*(1-edge_coverage)/nrep)),
                "edge_interval_width_95_mean": float(group.edge_interval_width_95_mean.mean()),
                "tci_convergence_rate": float((group.tci_rhat < 1.01).mean()),
                "edge_convergence_rate": float((group.edge_rhat_max < 1.01).mean()),
                "minimum_edge_bulk_ess_mean": float(group.edge_bulk_ess_min.mean()),
            })
        pd.DataFrame(aggregate_rows).to_csv(root/"monte_carlo_aggregate.csv", index=False)
    manifest = {
        "purpose": "known-truth finite-sample recovery harness for strict Q-BART connectedness",
        "fit_run_requested": bool(args.run_fits), "replicates": args.replicates,
        "observations_per_replicate": args.observations, "seed_rule": "base_seed + replicate_index*10007",
        "quantiles": args.tau, "train_fraction": args.train_fraction, "states": args.states,
        "step_sd": args.step_sd, "horizon": args.horizon, "chains": args.chains,
        "draws": args.draws, "tune": args.tune,
        "dgp": args.dgp,
        "dgp_definition": "stable five-variable recursion; linear or nonlinear location/scale; Gaussian or variance-standardized multivariate Student-t(df=5) innovations with cross-equation dependence",
        "interpretation": "When fit_run_requested is false, only synthetic data and oracle truth are produced; this is not an estimation recovery result.",
    }
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
