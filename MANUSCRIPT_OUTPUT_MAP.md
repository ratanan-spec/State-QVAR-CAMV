# Manuscript-to-output map

Use this map when responding to reviewers or extending the sample.

| Manuscript item | Generated source |
|---|---|
| Table 3: descriptive statistics | `outputs/return_descriptives.csv` |
| Table 4: correlations | `outputs/return_correlations.csv` |
| Table 5: TCI and NET | `outputs/static_tci.csv`, `outputs/directional_connectedness.csv` |
| Table 6: rolling tail inference | `outputs/tail_contrast_bootstrap.csv` |
| Table 7: VIX states | `outputs/vix_regime_summary.csv` |
| Table 8: cluster decomposition | `outputs/cluster_regime_decomposition.csv` |
| Table 9: portfolio performance | `outputs/portfolio/oos_performance_metrics.csv` |
| Table 10: paired bootstrap | `outputs/portfolio/oos_paired_bootstrap.csv` |
| Appendix Table A1 | `outputs/horizon_sensitivity.csv` |
| Appendix Tables A2–A4 | `outputs/quantile_gfevd.csv` |
| Appendix Table A5 | `outputs/diagnostics.csv` |
| Figures 1–5 | `outputs/figures/` |
| Figure 6 | `outputs/portfolio/figure_oos_portfolios.png` |

The consolidated workbook is `outputs/sustainability_qvar_results.xlsx`.
Run `python reproduce_all.py` to rebuild everything and validate the headline
numbers used in the abstract, results, and conclusion.