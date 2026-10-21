"""Check that stored outputs reproduce the headline manuscript results.

The checks deliberately use tolerances rather than rounded display values.
If a reviewer-requested change alters an estimate, this script stops and lists
the affected headline result so that the manuscript can be updated safely.
"""

from pathlib import Path
import json
import math

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def close(label: str, actual: float, expected: float, tolerance: float = 0.002) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: expected {expected:.6f} ± {tolerance:.6f}, got {actual:.6f}"
        )
    print(f"PASS  {label}: {actual:.6f}")


def main() -> None:
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample"]["levels"] == 1441
    assert manifest["sample"]["returns"] == 1440
    assert manifest["sample"]["start"] == "2020-03-16"
    assert manifest["sample"]["end"] == "2025-12-31"

    static = pd.read_csv(OUT / "static_tci.csv").set_index("Quantile")
    close("TCI tau=0.05", static.loc[0.05, "TCI"], 34.068365)
    close("TCI tau=0.50", static.loc[0.50, "TCI"], 33.972831)
    close("TCI tau=0.95", static.loc[0.95, "TCI"], 34.160842)

    directional = pd.read_csv(OUT / "directional_connectedness.csv")
    cgw = directional[directional["Market"].eq("CGW")]
    bgrn = directional[directional["Market"].eq("BGRN")]
    assert (cgw["NET"] > 0).all(), "CGW must remain a transmitter at every quantile."
    assert (bgrn["NET"] < 0).all(), "BGRN must remain a receiver at every quantile."

    contrast = pd.read_csv(OUT / "tail_contrast_bootstrap.csv").set_index("Contrast")
    upper = contrast.loc["Upper_minus_Median"]
    close("Rolling upper-tail minus median", upper["Mean"], 0.672, 0.003)
    assert upper["CI_Lower"] > 0
    lower = contrast.loc["Lower_minus_Median"]
    assert lower["CI_Lower"] < 0 < lower["CI_Upper"]

    metrics = pd.read_csv(OUT / "portfolio" / "oos_performance_metrics.csv").set_index("Strategy")
    close("Equal-weight annualized volatility", metrics.loc["EqualWeight", "Annualized_Volatility_pct"], 13.215, 0.003)
    close("State-QVAR CAMV annualized volatility", metrics.loc["StateQVAR_CAMV", "Annualized_Volatility_pct"], 5.907, 0.003)
    close("State-QVAR CAMV maximum drawdown", metrics.loc["StateQVAR_CAMV", "Maximum_Drawdown_pct"], -13.220, 0.004)
    close("State-QVAR CAMV CVaR", metrics.loc["StateQVAR_CAMV", "CVaR_5pct"], -0.857, 0.003)

    required_figures = [
        OUT / "figures" / f"figure{i}_{name}.png"
        for i, name in [
            (1, "normalized_levels"), (2, "static_quantile"),
            (3, "rolling_tci"), (4, "regimes"),
            (5, "complete_networks"),
        ]
    ] + [OUT / "portfolio" / "figure_oos_portfolios.png"]
    missing = [str(path) for path in required_figures if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing figures: " + ", ".join(missing))

    print("\nAll manuscript-result checks passed.")


if __name__ == "__main__":
    main()