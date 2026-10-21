"""One-command reproduction of every result reported in the manuscript."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("run_all.py")
    run("portfolio_backtest.py")
    run("verify_manuscript_results.py")
    print("\nAll analyses, tables, figures, and consistency checks completed.")