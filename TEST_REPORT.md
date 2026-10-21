# Release-candidate verification — 21 October 2026

- Python source syntax: PASS (compileall).
- Stored-results verification: PASS (`python verify_manuscript_results.py`).
- Reference outputs: TCI 34.068372 (0.05), 33.972831 (0.50), 34.160842 (0.95).
- CAMV volatility 5.906869%; max drawdown -13.220192%; 5% CVaR -0.857245%.
- Clean full rerun: NOT TESTED / BLOCKED because original market workbook is excluded.
- Data redistribution rights: NOT VERIFIED.
- MIT license: author-approved on 2026-10-21 for author-owned source code only; does not license market data, third-party content, or the published article.
- GitHub publishing: authorized by author on 2026-10-21. See repository history for publication status.
- Environment versions on verification host are recorded in `TEST_ENVIRONMENT.txt`; these are **not** independently validated reproduction versions.