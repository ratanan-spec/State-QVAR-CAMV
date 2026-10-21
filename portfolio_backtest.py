from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf

from run_all import (
    BASELINE_HORIZON,
    EXPECTED_BLOCK_LENGTH,
    NODES,
    ROLLING_STEP,
    ROLLING_WINDOW,
    SEED,
    fit_qvar_connectedness,
    load_and_align,
    stationary_bootstrap,
)


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "sustainability_markets.xlsx"
OUT = ROOT / "outputs" / "portfolio"
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
TRANSACTION_COST_BPS = 10.0
CONNECTEDNESS_PENALTY = 0.05


def minimum_variance_weights(window: np.ndarray) -> np.ndarray:
    covariance = LedoitWolf().fit(window).covariance_
    n = covariance.shape[0]
    result = minimize(
        lambda w: float(w @ covariance @ w),
        np.repeat(1 / n, n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return result.x


def minimum_cvar_weights(window: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Long-only historical-CVaR portfolio solved as a linear program."""
    t, n = window.shape
    # Variables: n weights, VaR threshold z, and t nonnegative shortfalls u.
    objective = np.r_[np.zeros(n), 1.0, np.repeat(1.0 / (alpha * t), t)]
    aub = np.zeros((t, n + 1 + t))
    # u_s >= loss_s - z, with loss_s = -r_s'w.
    aub[:, :n] = -window
    aub[:, n] = -1.0
    aub[np.arange(t), n + 1 + np.arange(t)] = -1.0
    bub = np.zeros(t)
    aeq = np.zeros((1, n + 1 + t)); aeq[0, :n] = 1.0
    result = linprog(
        objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * t,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return result.x[:n]


def inverse_transmission_weights(window: pd.DataFrame, tau: float) -> tuple[np.ndarray, np.ndarray]:
    result = fit_qvar_connectedness(window, tau, BASELINE_HORIZON)
    # Larger TO means that shocks from the node explain more variance elsewhere.
    # Inverse-TO allocation reduces exposure to the dominant transmitters.
    score = np.maximum(result["TO"], 1e-6)
    weights = (1.0 / score) / np.sum(1.0 / score)
    return weights, result["TO"]


def connectedness_adjusted_minimum_variance(window: np.ndarray, to_values: np.ndarray) -> np.ndarray:
    """Long-only minimum variance with a prespecified transmitter penalty."""
    covariance = LedoitWolf().fit(window).covariance_
    score = np.maximum(to_values, 0.0)
    score = score / score.max() if score.max() > 0 else score
    scale = float(np.mean(np.diag(covariance)))
    n = covariance.shape[0]
    result = minimize(
        lambda w: float(w @ covariance @ w + CONNECTEDNESS_PENALTY * scale * (w @ score)),
        np.repeat(1 / n, n), method="SLSQP", bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Connectedness-adjusted optimization failed: {result.message}")
    return result.x


def performance_metrics(x: pd.Series, turnover: pd.Series) -> dict:
    x = x.dropna()
    wealth = (1.0 + x / 100.0).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    downside = x[x < 0].std(ddof=1)
    var = x.quantile(ALPHA)
    cvar = x[x <= var].mean()
    return {
        "Observations": len(x),
        "Annualized_Return_pct": ((wealth.iloc[-1] ** (252 / len(x))) - 1) * 100,
        "Annualized_Volatility_pct": x.std(ddof=1) * np.sqrt(252),
        "Sharpe_0rf": x.mean() / x.std(ddof=1) * np.sqrt(252),
        "Sortino_0rf": x.mean() / downside * np.sqrt(252) if downside > 0 else np.nan,
        "VaR_5pct": var,
        "CVaR_5pct": cvar,
        "Maximum_Drawdown_pct": drawdown.min() * 100,
        "Mean_Rebalance_Turnover": turnover.mean(),
        "Cumulative_Return_pct": (wealth.iloc[-1] - 1) * 100,
    }


def paired_risk_bootstrap(candidate: np.ndarray, benchmark: np.ndarray,
                          reps: int = 2000, expected_block: int = EXPECTED_BLOCK_LENGTH,
                          seed: int = SEED) -> dict:
    """Stationary-bootstrap differences in volatility and left-tail CVaR."""
    candidate, benchmark = np.asarray(candidate, float), np.asarray(benchmark, float)
    n, restart = len(candidate), 1.0 / expected_block
    rng = np.random.default_rng(seed)
    vol_diff, cvar_diff = np.empty(reps), np.empty(reps)
    for b in range(reps):
        idxs, idx = np.empty(n, dtype=int), int(rng.integers(0, n))
        for t in range(n):
            idxs[t] = idx
            idx = int(rng.integers(0, n)) if rng.random() < restart else (idx + 1) % n
        c, e = candidate[idxs], benchmark[idxs]
        vol_diff[b] = (np.std(c, ddof=1) - np.std(e, ddof=1)) * np.sqrt(252)
        cv, ev = np.quantile(c, ALPHA), np.quantile(e, ALPHA)
        cvar_diff[b] = np.mean(c[c <= cv]) - np.mean(e[e <= ev])
    return {
        "Volatility_difference": (np.std(candidate, ddof=1) - np.std(benchmark, ddof=1)) * np.sqrt(252),
        "Vol_CI_Lower": np.quantile(vol_diff, .025), "Vol_CI_Upper": np.quantile(vol_diff, .975),
        "CVaR_difference": np.mean(candidate[candidate <= np.quantile(candidate, ALPHA)])
                           - np.mean(benchmark[benchmark <= np.quantile(benchmark, ALPHA)]),
        "CVaR_CI_Lower": np.quantile(cvar_diff, .025), "CVaR_CI_Upper": np.quantile(cvar_diff, .975),
    }


def main():
    _, returns, states, _ = load_and_align(INPUT)
    strategies = ["EqualWeight", "MinimumVariance", "MinimumCVaR", "StateQVAR_CAMV"]
    daily = {s: pd.Series(index=returns.index, dtype=float) for s in strategies}
    weights_rows, turnover_rows = [], []
    previous = {s: None for s in strategies}

    endpoints = list(range(ROLLING_WINDOW, len(returns), ROLLING_STEP))
    for number, end in enumerate(endpoints, 1):
        train = returns.iloc[end - ROLLING_WINDOW:end]
        trade = returns.iloc[end:min(end + ROLLING_STEP, len(returns))]
        decision_date = returns.index[end - 1]
        historical_vix = states.loc[:decision_date, "VIX"]
        vix_threshold = historical_vix.quantile(.75)
        high_vix = bool(states.loc[decision_date, "VIX"] >= vix_threshold)
        tau = .05 if high_vix else .50

        w = {
            "EqualWeight": np.repeat(1 / len(NODES), len(NODES)),
            "MinimumVariance": minimum_variance_weights(train.to_numpy()),
            "MinimumCVaR": minimum_cvar_weights(train.to_numpy()),
        }
        _, to_values = inverse_transmission_weights(train, tau)
        w["StateQVAR_CAMV"] = connectedness_adjusted_minimum_variance(train.to_numpy(), to_values)

        for strategy in strategies:
            current = w[strategy]
            turnover = 0.0 if previous[strategy] is None else 0.5 * np.abs(current - previous[strategy]).sum()
            gross = pd.Series(trade.to_numpy() @ current, index=trade.index)
            net = gross.copy()
            if len(net):
                net.iloc[0] -= TRANSACTION_COST_BPS / 100.0 * turnover
            daily[strategy].loc[trade.index] = net
            turnover_rows.append({"Date": decision_date, "Strategy": strategy, "Turnover": turnover})
            for i, node in enumerate(NODES):
                weights_rows.append({"Date": decision_date, "Strategy": strategy,
                                     "Market": node, "Weight": current[i],
                                     "VIX_threshold_expanding": vix_threshold,
                                     "High_VIX": int(high_vix), "Selected_tau": tau})
            previous[strategy] = current
        if number % 40 == 0 or number == len(endpoints):
            print(f"Portfolio windows completed: {number}/{len(endpoints)}", flush=True)

    portfolio_returns = pd.DataFrame(daily).dropna(how="all")
    turnovers = pd.DataFrame(turnover_rows)
    weights = pd.DataFrame(weights_rows)
    metrics = []
    for strategy in strategies:
        metrics.append({"Strategy": strategy, **performance_metrics(
            portfolio_returns[strategy], turnovers.loc[turnovers.Strategy.eq(strategy), "Turnover"])})
    metrics = pd.DataFrame(metrics)

    # Paired stationary-bootstrap uncertainty relative to equal weight.
    inference = []
    benchmark = portfolio_returns["EqualWeight"].dropna()
    for strategy in strategies[1:]:
        candidate = portfolio_returns[strategy].loc[benchmark.index]
        loss_diff = (-candidate) - (-benchmark)
        mean_draws = stationary_bootstrap(loss_diff.to_numpy(), seed=SEED)
        # Negative mean loss difference favors the candidate.
        inference.append({
            "Strategy": strategy,
            "Mean_loss_difference_vs_EW": loss_diff.mean(),
            "CI_Lower": np.quantile(mean_draws, .025),
            "CI_Upper": np.quantile(mean_draws, .975),
            **paired_risk_bootstrap(candidate.to_numpy(), benchmark.to_numpy(), seed=SEED),
        })

    portfolio_returns.to_csv(OUT / "oos_portfolio_returns.csv", index_label="Date")
    weights.to_csv(OUT / "oos_portfolio_weights.csv", index=False)
    turnovers.to_csv(OUT / "oos_turnover.csv", index=False)
    metrics.to_csv(OUT / "oos_performance_metrics.csv", index=False)
    pd.DataFrame(inference).to_csv(OUT / "oos_paired_bootstrap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    wealth = (1 + portfolio_returns / 100).cumprod()
    wealth.plot(ax=axes[0], lw=1.2)
    axes[0].set_title("(a) Net out-of-sample wealth")
    axes[0].set_ylabel("Growth of 1 unit"); axes[0].set_xlabel(""); axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=7)
    metric_plot = metrics.set_index("Strategy")[["Annualized_Volatility_pct", "Maximum_Drawdown_pct"]]
    metric_plot.assign(Maximum_Drawdown_pct=-metric_plot.Maximum_Drawdown_pct).plot.bar(ax=axes[1])
    axes[1].set_title("(b) Risk and absolute drawdown")
    axes[1].set_ylabel("Percent"); axes[1].set_xlabel(""); axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(["Annualized volatility", "Absolute maximum drawdown"], frameon=False, fontsize=7)
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "figure_oos_portfolios.png", dpi=300); plt.close(fig)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()